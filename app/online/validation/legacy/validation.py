"""Extracted from app.online.validator (mechanical move; behavior preserved)."""
from __future__ import annotations

import json
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from app.config import (
    MAX_PARALLEL_VALIDATOR_CALLS,
    VALIDATOR_BATCH_SIZE,
    VALIDATOR_TRACE_MAX_ATTEMPTS,
    REVALIDATION_OPERATIONAL_RETRIES,
    MAX_DUPLICATE_REPLACEMENT_ATTEMPTS_PER_SLOT,
    ASSESSMENT_CATALOG_VERSION,
    MAX_CONTENT_REPAIR_CALLS_PER_SLOT,
    MAX_FAILED_REPAIR_ATTEMPTS_PER_SLOT,
    MAX_COMPLETION_ATTEMPTS_PER_SLOT,
)
from app.llm.client import request_structured
from app.llm.factory import create_llm_client
from app.llm.schemas import (
    global_exam_review_response_model,
    repair_response_model,
    validator_response_model,
)
from app.logging_conf import get_logger
from app.online.eval_stats import record_first_validation, record_repair_outcome
from app.online.diagnostics import record_optional_failure
from app.online.assessment_catalog import (
    _apply_effective_variation,
    eligible_task_variation_candidates,
    is_pure_text_fitb_value,
    normalize_fitb_answer_key,
    variation_response_form_compatible,
)
from app.online.cross_model_duplicates import integrate_cross_model_duplicates
from app.online.models import (
    _normalize_definition,
    _normalize_equation,
    _normalize_essay,
    _normalize_mcq,
    _normalize_why,
    _normalize_true_false,
    _normalize_word_problem,
    authoritative_givens,
    normalize_fitb_item,
    normalize_text,
    numerical_instance_contract,
)
from app.offline.pipeline import PipelineCancelled

logger = get_logger("VALIDATOR")
from app.llm.math_strings import MATH_JSON_CONTRACT


from .. import runtime as _runtime
from ..constants import (
    ACTIONS, GENERAL_VALIDATOR_MAX_TOKENS, REPAIR_ACTIONS,
    STEM_QUANTITATIVE_VALIDATOR_MAX_TOKENS, VALIDATION_SCHEMA_VERSION,
    VALIDATOR_MAX_TOKENS, VALIDATOR_REASONING, VALIDATOR_TEMPERATURE,
)
from ..questions import (
    _allowed_canonical, _iter_questions,
)
from .global_review import (
    _deterministic_exam_verdicts,
)
from ..contracts import (
    STEM_QUANTITATIVE_VALIDATOR_RETRY_SYSTEM_PROMPT, STEM_QUANTITATIVE_VALIDATOR_SYSTEM_PROMPT, VALIDATOR_RETRY_SYSTEM_PROMPT, VALIDATOR_SYSTEM_PROMPT, _assigned_source_chunks, _catalog_task_contract, build_validator_prompt,
)

logger = get_logger("VALIDATION_LEGACY_VALIDATION")

def _sanitize_verdict(raw: Any, known: Set[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    qid = str(raw.get("question_id") or "").strip()
    if not qid or qid not in known:
        return None
    if not isinstance(raw.get("question_valid"), bool) or not isinstance(
        raw.get("answer_valid"), bool
    ):
        return None
    qv = raw["question_valid"]
    av = raw["answer_valid"]
    action = str(raw.get("action") or "").upper()
    if action not in ACTIONS:
        return None
    fields = raw.get("fields_to_fix")
    if not isinstance(fields, list):
        fields = []
    explicit = any(str(f).strip() for f in fields)
    if qv and av:
        # An explicit non-PASS action with fields is the model flagging a real
        # problem (e.g. FIX_OPTIONS for two correct options); do not collapse it.
        if not (explicit and action in ("FIX_OPTIONS", "FIX_ANSWER", "FIX_QUESTION", "FIX_QUESTION_AND_ANSWER")):
            action = "PASS"
    elif not qv and not av:
        action = "FIX_QUESTION_AND_ANSWER"
    elif not qv:
        action = "FIX_QUESTION"
    else:
        if action not in ("FIX_OPTIONS", "FIX_ANSWER"):
            action = "FIX_ANSWER"
    return {
        "question_id": qid,
        "question_valid": qv,
        "answer_valid": av,
        "action": action,
        "fields_to_fix": [str(f) for f in fields],
        "failure_code": str(raw.get("failure_code") or ("pass" if action == "PASS" else "other")),
        "expected_results": [
            str(value) for value in (raw.get("expected_results") or [])
            if str(value).strip()
        ],
        "reason": str(raw.get("reason") or "").strip(),
        "expected_fix": str(raw.get("expected_fix") or "").strip(),
        "solved_answer": raw.get("solved_answer"),
        "independent_solution": raw.get("independent_solution"),
        "certificate_valid": raw.get("certificate_valid"),
        "first_incorrect_step": (
            str(raw.get("first_incorrect_step") or "").strip() or None
        ),
    }


def _reconcile_simple_expected_answer(
    verdict: Dict[str, Any], entry: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve self-contradictory MCQ/TF verdicts from explicit expected answers."""
    if verdict.get("action") == "PASS" or not verdict.get("question_valid"):
        return verdict
    if verdict.get("certificate_valid") is False:
        # A matching answer label cannot erase a broken/missing quantitative
        # solution certificate or an incorrect intermediate calculation.
        return verdict
    qtype = str(entry.get("question_type") or "")
    item = entry.get("question") or {}
    expected = [str(value).strip() for value in verdict.get("expected_results") or []]
    matches = False
    if qtype == "mcq":
        stored = str(item.get("correct_answer") or "").strip().upper()
        matches = bool(stored) and any(
            (match := re.match(r"^\s*([A-D])(?:\b|\s*:)", value.upper()))
            and match.group(1) == stored
            for value in expected
        )
    elif qtype == "true_false":
        stored_raw = str(item.get("answer")).strip().casefold()
        truth_map = {
            "true": True, "false": False, "صح": True, "صحيح": True,
            "صواب": True, "خطأ": False, "خطا": False,
        }
        stored = truth_map.get(stored_raw)
        candidates: list[bool] = []
        for value in expected:
            folded = value.casefold()
            if re.search(r"\btrue\b", folded) or any(word in value for word in ("صح", "صحيح", "صواب")):
                candidates.append(True)
            if re.search(r"\bfalse\b", folded) or any(word in value for word in ("خطأ", "خطا")):
                candidates.append(False)
        matches = stored is not None and stored in candidates
    if matches:
        verdict = dict(verdict)
        verdict.update({
            "answer_valid": True,
            "action": "PASS",
            "fields_to_fix": [],
            "failure_code": "pass",
            "expected_fix": "",
        })
    return verdict


def validate_exam(
        exam: Dict[str, Any], client: Optional[Any] = None,
        question_ids: Optional[Iterable[str]] = None,
        plan_by_slot: Optional[Dict[str, Dict[str, Any]]] = None,
        source_by_chunk: Optional[Dict[str, str]] = None,
        validation_role: str = "general",
) -> Dict[str, Any]:
    """Validate every question in bounded batches, retrying missing IDs once.

    Every question with an ID receives either an actual validator/local verdict
    or an ``UNVALIDATED`` operational status. Missing coverage is never converted
    into a content defect.
    """
    model_number = exam.get("model_number") or 1
    quantitative = validation_role == "stem_quantitative"
    active_system_prompt = STEM_QUANTITATIVE_VALIDATOR_SYSTEM_PROMPT if quantitative else VALIDATOR_SYSTEM_PROMPT
    active_retry_prompt = STEM_QUANTITATIVE_VALIDATOR_RETRY_SYSTEM_PROMPT if quantitative else VALIDATOR_RETRY_SYSTEM_PROMPT
    questions = exam.get("questions") or {}
    selected_ids = (
        {str(question_id) for question_id in question_ids if str(question_id)}
        if question_ids is not None
        else None
    )
    warnings: List[str] = []

    # Deterministic structural problems never consume an LLM call.
    local_verdicts = _deterministic_exam_verdicts(
        exam, plan_by_slot, selected_ids
    )
    entries: List[Dict[str, Any]] = []
    for qtype, container, item in _iter_questions(questions):
        qid = item.get("question_id")
        if not qid:
            warnings.append(f"validate: question missing question_id; skipped")
            continue
        if selected_ids is not None and str(qid) not in selected_ids:
            continue
        if str(qid) in local_verdicts:
            continue
        slot = str(item.get("slot_id") or "")
        plan_item = (plan_by_slot or {}).get(slot)
        payload = dict(item)
        if qtype == "fill_in_the_blank" and isinstance(container, dict):
            payload["word_bank"] = container.get("word_bank") or []
        entries.append({
            "question_id": str(qid),
            "question_type": qtype,
            "question": payload,
            "task_contract": _catalog_task_contract(plan_item, item),
            "_source_chunks": _assigned_source_chunks(
                plan_item, source_by_chunk,
                list(item.get("effective_evidence_chunk_ids") or []) or None,
            ),
        })

    scope_ids = [*local_verdicts, *(entry["question_id"] for entry in entries)]
    logger.info(
        "Validation scope | model=%d | question_count=%d | question_ids=%s",
        model_number,
        len(scope_ids),
        scope_ids,
    )

    verdicts: Dict[str, Dict[str, Any]] = dict(local_verdicts)
    coverage: List[Dict[str, Any]] = []
    if entries:
        batches = [
            entries[offset: offset + _runtime.validator_batch_size()]
            for offset in range(0, len(entries), _runtime.validator_batch_size())
        ]

        def validate_batch(
            batch_index: int, batch: List[Dict[str, Any]],
        ) -> tuple[int, Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
            batch_warnings: List[str] = []
            batch_verdicts: Dict[str, Dict[str, Any]] = {}
            sent_ids = [entry["question_id"] for entry in batch]
            known_ids = set(sent_ids)
            returned_ids: List[str] = []
            unexpected_ids: List[str] = []
            duplicate_ids: List[str] = []
            entry_by_id = {entry["question_id"]: entry for entry in batch}

            def structural_errors(
                raw_result: Any, expected_ids: List[str], *, salvage: bool,
            ) -> tuple[str, list[dict[str, Any]]]:
                """Validate the exact response envelope and retain only proven rows.

                A malformed multi-question envelope may contain independently valid
                rows. Those rows are safe to retain; every unresolved ID is retried
                alone under the same strict one-verdict schema.
                """
                expected = set(expected_ids)
                if not isinstance(raw_result, dict):
                    return "root must be one JSON object", []
                rows = raw_result.get("verdicts")
                if not isinstance(rows, list):
                    return "root.verdicts must be an array", []
                raw_ids = [
                    str(row.get("question_id") or "").strip()
                    if isinstance(row, dict) else ""
                    for row in rows
                ]
                counts = {qid: raw_ids.count(qid) for qid in set(raw_ids) if qid}
                duplicate = sorted(qid for qid, count in counts.items() if count > 1)
                missing = sorted(expected - set(raw_ids))
                extra = sorted(set(raw_ids) - expected - {""})
                invalid_shape = sum(not isinstance(row, dict) for row in rows)
                problems: list[str] = []
                if len(rows) != len(expected_ids):
                    problems.append(
                        f"expected {len(expected_ids)} verdict(s), received {len(rows)}"
                    )
                if missing:
                    problems.append(f"missing question_ids: {missing}")
                if extra:
                    problems.append(f"extra question_ids: {extra}")
                if duplicate:
                    problems.append(f"duplicate question_ids: {duplicate}")
                if invalid_shape:
                    problems.append(f"{invalid_shape} verdict row(s) were not objects")

                response_model = validator_response_model(
                    expected_ids, quantitative=quantitative
                )
                try:
                    validated = response_model.model_validate(raw_result).model_dump()
                    return "", list(validated["verdicts"])
                except Exception as exc:
                    problems.append(f"schema validation failed: {exc}")

                if not salvage:
                    return "; ".join(problems), []

                valid_rows: list[dict[str, Any]] = []
                for qid in expected_ids:
                    if counts.get(qid) != 1:
                        continue
                    row = next((
                        value for value in rows
                        if isinstance(value, dict)
                        and str(value.get("question_id") or "").strip() == qid
                    ), None)
                    if row is None:
                        continue
                    try:
                        single_model = validator_response_model(
                            [qid], quantitative=quantitative
                        )
                        validated = single_model.model_validate(
                            {"verdicts": [row]}
                        ).model_dump()
                        valid_rows.append(validated["verdicts"][0])
                    except Exception:
                        continue
                return "; ".join(problems), valid_rows

            def consume(
                raw_result: Any,
                allowed_ids: Set[str],
                *,
                retry: bool = False,
                record_returned: bool = True,
            ) -> None:
                results = raw_result.get("verdicts") if isinstance(raw_result, dict) else None
                if not isinstance(results, list):
                    batch_warnings.append(
                        "validate: retry returned no usable verdict array"
                        if retry
                        else "validate: validator returned no usable verdict array"
                    )
                    return
                seen_in_response: Set[str] = set()
                for raw_verdict in results:
                    raw_id = (
                        str(raw_verdict.get("question_id") or "").strip()
                        if isinstance(raw_verdict, dict)
                        else ""
                    )
                    if record_returned:
                        returned_ids.append(raw_id)
                    if raw_id in seen_in_response:
                        if raw_id and raw_id not in duplicate_ids:
                            duplicate_ids.append(raw_id)
                        batch_warnings.append(
                            f"validate: duplicate verdict for {raw_id} ignored"
                        )
                        continue
                    seen_in_response.add(raw_id)
                    if not raw_id or raw_id not in allowed_ids:
                        if raw_id and raw_id not in unexpected_ids:
                            unexpected_ids.append(raw_id)
                        batch_warnings.append(
                            "validate: verdict with unknown/missing question_id rejected"
                        )
                        continue
                    v = _sanitize_verdict(raw_verdict, allowed_ids)
                    if v is None:
                        batch_warnings.append(
                            "validate: verdict with unknown/missing question_id rejected"
                        )
                        continue
                    if quantitative and (
                        not isinstance(v.get("independent_solution"), dict)
                        or not isinstance(v.get("certificate_valid"), bool)
                        or (
                            v.get("certificate_valid") is False
                            and not v.get("first_incorrect_step")
                            and not (
                                v.get("action") == "FIX_ANSWER"
                                and set(v.get("fields_to_fix") or []) == {
                                    "solution_certificate"
                                }
                                and str(v.get("failure_code") or "")
                                == "incomplete_solution"
                            )
                        )
                    ):
                        batch_warnings.append(
                            f"validate: quantitative verdict for {raw_id} omitted the "
                            "independent solution/certificate comparison; rejected"
                        )
                        continue
                    entry = entry_by_id.get(raw_id, {})
                    qtype = str(entry.get("question_type") or "")
                    if v.get("action") == "PASS":
                        v["fields_to_fix"] = []
                    else:
                        requested_fields = list(v.get("fields_to_fix") or [])
                        authorized_fields = sorted(_allowed_canonical(
                            qtype, v.get("fields_to_fix") or [], str(v.get("action") or "")
                        ))
                        if requested_fields and not authorized_fields:
                            batch_warnings.append(
                                f"validate: verdict for {raw_id} used non-canonical fields; rejected"
                            )
                            continue
                        v["fields_to_fix"] = authorized_fields
                    v = _reconcile_simple_expected_answer(v, entry)
                    # A retry never overwrites a verdict already accepted from
                    # the original batch response.
                    if v["question_id"] not in batch_verdicts:
                        batch_verdicts[v["question_id"]] = v

            active_max_tokens = (
                STEM_QUANTITATIVE_VALIDATOR_MAX_TOKENS
                if quantitative else GENERAL_VALIDATOR_MAX_TOKENS
            )
            batch_llm = client or _runtime.create_llm_client(
                local_reasoning=VALIDATOR_REASONING,
                stage=("quantitative_validator" if quantitative else "general_validator"),
            )
            initial_rejection_reason = ""
            initial_rows_already_recorded = False
            try:
                logger.info(
                    "Validator request | role=%s | batch_size=%d | question_ids=%s | reason=initial",
                    validation_role, len(sent_ids), sent_ids,
                )
                raw = _runtime.request_structured(
                    batch_llm,
                    build_validator_prompt(batch, quantitative=quantitative),
                    response_model=validator_response_model(
                        sent_ids, quantitative=quantitative
                    ),
                    schema_name="validator_verdicts",
                    agent=f"{validation_role}_validator",
                    operation=f"{validation_role}_validation",
                    item_count=len(batch),
                    expected_ids=sent_ids,
                    expected_id_field="question_id",
                    local_json=False,
                    system_prompt=active_system_prompt,
                    temperature=VALIDATOR_TEMPERATURE,
                    max_tokens=active_max_tokens,
                )
                initial_rejection_reason, usable_rows = structural_errors(
                    raw, sent_ids, salvage=True
                )
                if initial_rejection_reason:
                    initial_raw_rows = (
                        raw.get("verdicts") if isinstance(raw, dict) else []
                    )
                    initial_raw_ids = [
                        str(row.get("question_id") or "").strip()
                        if isinstance(row, dict) else ""
                        for row in initial_raw_rows or []
                    ]
                    returned_ids.extend(initial_raw_ids)
                    for raw_id in sorted(set(initial_raw_ids)):
                        if raw_id and initial_raw_ids.count(raw_id) > 1:
                            duplicate_ids.append(raw_id)
                        if raw_id and raw_id not in known_ids:
                            unexpected_ids.append(raw_id)
                    initial_rows_already_recorded = True
                    batch_warnings.append(
                        "validate: validator structured response rejected ("
                        f"{initial_rejection_reason})"
                    )
                raw = {"verdicts": usable_rows}
            except Exception as exc:
                logger.exception(
                    "Validator call failed | role=%s | question_ids=%s",
                    validation_role, sent_ids,
                )
                batch_warnings.append(
                    f"validate: validator LLM call failed ({exc}); no verdicts received"
                )
                initial_rejection_reason = str(exc)
                raw = {"verdicts": []}
            consume(
                raw, known_ids,
                record_returned=not initial_rows_already_recorded,
            )

            missing = [
                entry for entry in batch
                if entry["question_id"] not in batch_verdicts
            ]
            if missing:
                batch_warnings.append(
                    f"validate: {len(missing)} unresolved question(s); retrying each once"
                )
                rejection_reason = initial_rejection_reason or (
                    "the previous response did not provide a schema-valid verdict "
                    "for this question_id"
                )
                for missing_entry in missing:
                    missing_id = missing_entry["question_id"]
                    try:
                        logger.info(
                            "Validator request | role=%s | batch_size=1 | question_ids=%s | reason=revalidation:%s",
                            validation_role, [missing_id], rejection_reason,
                        )
                        raw2 = _runtime.request_structured(
                            batch_llm,
                            build_validator_prompt(
                                [missing_entry], quantitative=quantitative,
                                rejection_reason=rejection_reason,
                            ),
                            response_model=validator_response_model(
                                [missing_id], quantitative=quantitative
                            ),
                            schema_name="validator_missing_verdict",
                            agent=f"{validation_role}_validator",
                            operation=f"{validation_role}_revalidation",
                            item_count=1,
                            expected_ids=[missing_id],
                            expected_id_field="question_id",
                            local_json=False,
                            system_prompt=active_retry_prompt,
                            temperature=VALIDATOR_TEMPERATURE,
                            max_tokens=active_max_tokens,
                        )
                        retry_error, retry_rows = structural_errors(
                            raw2, [missing_id], salvage=False
                        )
                        if retry_error:
                            raise ValueError(retry_error)
                        consume(
                            {"verdicts": retry_rows}, {missing_id}, retry=True
                        )
                    except Exception as exc:
                        logger.exception(
                            "Validator retry failed | role=%s | question_ids=%s",
                            validation_role, [missing_id],
                        )
                        batch_warnings.append(
                            f"validate: validator retry failed for {missing_id} ({exc})"
                        )

            final_missing = [qid for qid in sent_ids if qid not in batch_verdicts]
            batch_coverage = {
                "sent_ids": sent_ids,
                "returned_ids": returned_ids,
                "matched_ids": [qid for qid in sent_ids if qid in batch_verdicts],
                "missing_ids": final_missing,
                "unexpected_ids": unexpected_ids,
                "duplicate_returned_ids": duplicate_ids,
            }
            return (
                batch_index, batch_verdicts, batch_warnings, batch_coverage
            )

        max_workers = min(_runtime.max_parallel_validator_calls(), len(batches))
        logger.info(
            "Parallel validator batches | role=%s | batches=%d | max_workers=%d",
            validation_role, len(batches), max_workers,
        )
        completed: dict[
            int, tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]
        ] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(copy_context().run, validate_batch, index, batch): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                index, batch_verdicts, batch_warnings, batch_coverage = future.result()
                completed[index] = (
                    batch_verdicts, batch_warnings, batch_coverage
                )

        # Completion order never affects the persisted report.
        for index in range(len(batches)):
            batch_verdicts, batch_warnings, batch_coverage = completed[index]
            verdicts.update(batch_verdicts)
            warnings.extend(batch_warnings)
            coverage.append(batch_coverage)

    all_pass = True
    ordered: List[Dict[str, Any]] = []
    for _, _, item in _iter_questions(questions):
        qid = str(item.get("question_id") or "")
        if not qid:
            continue
        if selected_ids is not None and qid not in selected_ids:
            continue
        v = verdicts.get(qid)
        if v is None:
            warnings.append(
                f"validate: no verdict returned for {qid} after retry; marked unvalidated"
            )
            v = {
                "question_id": qid,
                "question_valid": None,
                "answer_valid": None,
                "action": "UNVALIDATED",
                "fields_to_fix": [],
                "reason": "No validator verdict was returned after retry.",
                "expected_fix": "Retry validation; do not modify the question.",
            }
        if v["action"] != "PASS":
            all_pass = False
        v["validation_role"] = validation_role
        ordered.append(v)

    logger.info(
        "Exam validated | model=%d | questions=%d | invalid=%d | unvalidated=%d",
        model_number,
        len(ordered),
        sum(1 for v in ordered if v["action"] in REPAIR_ACTIONS),
        sum(1 for v in ordered if v["action"] == "UNVALIDATED"),
    )
    return {
        "model_number": model_number,
        "all_pass": all_pass,
        "verdicts": ordered,
        "warnings": warnings,
        "coverage": coverage,
    }


def _merge_validation_report(
    previous: Dict[str, Any], update: Dict[str, Any]
) -> Dict[str, Any]:
    """Replace revalidated verdicts by ID and retain every untouched verdict."""
    replacements = {
        str(verdict.get("question_id")): {
            **deepcopy(verdict), "evaluation_origin": "fresh"
        }
        for verdict in (update.get("verdicts") or [])
        if verdict.get("question_id")
    }
    merged_verdicts: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for verdict in previous.get("verdicts") or []:
        qid = str(verdict.get("question_id") or "")
        merged_verdicts.append(replacements.get(qid, {
            **deepcopy(verdict), "evaluation_origin": "historical_retained"
        }))
        if qid:
            seen.add(qid)
    for qid, verdict in replacements.items():
        if qid not in seen:
            merged_verdicts.append(verdict)

    merged = dict(previous)
    merged["model_number"] = update.get(
        "model_number", previous.get("model_number")
    )
    merged["verdicts"] = merged_verdicts
    merged["missing_slots"] = list(update.get("missing_slots", previous.get("missing_slots") or []))
    merged["missing_sections"] = list(
        update.get("missing_sections", previous.get("missing_sections") or [])
    )
    merged["all_pass"] = (
        all(verdict.get("action") == "PASS" for verdict in merged_verdicts)
        and not merged["missing_slots"] and not merged["missing_sections"]
    )
    merged["warnings"] = [
        *(previous.get("warnings") or []),
        *(update.get("warnings") or []),
    ]
    merged["coverage"] = [
        *(previous.get("coverage") or []),
        *(update.get("coverage") or []),
    ]
    for key in ("role_results", "role_fingerprints", "input_protection"):
        combined = deepcopy(previous.get(key) or {})
        combined.update(deepcopy(update.get(key) or {}))
        if combined:
            merged[key] = combined
    merged["validation_call_logs"] = [
        *(previous.get("validation_call_logs") or []),
        *(update.get("validation_call_logs") or []),
    ][-VALIDATOR_TRACE_MAX_ATTEMPTS:]
    return merged
