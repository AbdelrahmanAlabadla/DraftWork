"""Cross-model coordination built on the production Duplicate V2 judge.

Within-model duplicate review remains owned by :mod:`duplicate_pipeline`.  This
module runs only after every model has independently passed that review.  It
uses Planner identities to select a bounded set of cross-model pairs, while the
existing LLM judge evaluates the actual question and answer content.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from typing import Any

from app.online.assessment_catalog import (
    is_authorized_catalog_exhaustion_reuse,
    is_authorized_intentional_practice_reuse,
)
from app.online.duplicate_judge import review_duplicate_candidates


def _section_items(section: Any) -> list[dict[str, Any]]:
    values = section.get("items") if isinstance(section, dict) else section
    return [value for value in (values or []) if isinstance(value, dict)]


def _plan_index(plans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("slot_id") or ""): item
        for plan in plans
        for values in (plan.get("items") or {}).values()
        for item in values or []
        if item.get("slot_id")
    }


def _normalized(value: Any) -> str:
    return " ".join(
        re.findall(
            r"[\w]+",
            unicodedata.normalize("NFKC", str(value or "")).casefold(),
            flags=re.UNICODE,
        )
    )


def _question_content(question: dict[str, Any]) -> dict[str, Any]:
    return {
        key: question.get(key)
        for key in (
            "question", "statement", "term", "equation", "prompt", "options",
            "correct_answer", "answer", "answers", "reference_answer",
            "solution_steps", "final_answer", "solution_certificate", "key_points",
        )
        if key in question
    }


def _content_fingerprint(question: dict[str, Any]) -> str:
    normalized = {
        key: _normalized(value) if isinstance(value, str) else value
        for key, value in _question_content(question).items()
    }
    return hashlib.sha256(json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, default=str,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _identity_keys(plan: dict[str, Any], question: dict[str, Any]) -> set[tuple[str, str]]:
    values = {
        "task_id": plan.get("task_id") or question.get("task_id"),
        "variation_id": plan.get("selected_variation_id")
        or question.get("selected_variation_id"),
        "variation_group_id": plan.get("selected_variation_group_id")
        or question.get("selected_variation_group_id"),
        "effective_instance_id": plan.get("effective_instance_id")
        or question.get("effective_instance_id"),
    }
    return {(kind, str(value)) for kind, value in values.items() if value}


def _reuse_context(plan: dict[str, Any]) -> dict[str, Any] | None:
    if not (
        is_authorized_catalog_exhaustion_reuse(plan)
        or is_authorized_intentional_practice_reuse(plan)
    ):
        return None
    return {
        "status": "intentional_practice_variation",
        "authorized": True,
        "task_id": plan.get("task_id"),
        "selected_variation_id": plan.get("selected_variation_id"),
        "selected_variation_group_id": plan.get("selected_variation_group_id"),
        "required_instance_distinctions": deepcopy(
            plan.get("required_instance_distinctions") or []
        ),
        "numeric_values_may_change": not bool(plan.get("exact_source_instance")),
        "exact_source_instance": bool(plan.get("exact_source_instance")),
        "reference_slot_ids": list(plan.get("forced_duplicate_slot_ids") or []),
    }


def _accepted_questions(
    generated: list[dict[str, Any]], reports_by_model: dict[int, dict[str, Any]],
    plans: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    plan_by_slot = _plan_index(plans)
    rows: list[dict[str, Any]] = []
    by_cross_id: dict[str, dict[str, Any]] = {}
    for exam in generated:
        model_number = int(exam.get("model_number") or 0)
        passed = {
            str(value.get("question_id") or "")
            for value in (reports_by_model.get(model_number) or {}).get("verdicts") or []
            if value.get("action") == "PASS" and value.get("question_id")
        }
        for qtype, section in (exam.get("questions") or {}).items():
            for question in _section_items(section):
                question_id = str(question.get("question_id") or "")
                slot_id = str(question.get("slot_id") or "")
                if not question_id or question_id not in passed:
                    continue
                cross_id = f"m{model_number}::{question_id}"
                plan = plan_by_slot.get(slot_id) or {}
                copied = deepcopy(question)
                copied["question_id"] = cross_id
                context = _reuse_context(plan)
                if context:
                    copied["reuse_context"] = context
                row = {
                    "cross_question_id": cross_id,
                    "question_id": question_id,
                    "slot_id": slot_id,
                    "model_number": model_number,
                    "question_type": qtype,
                    "question": copied,
                    "plan": plan,
                    "identity_keys": _identity_keys(plan, question),
                    "content_fingerprint": _content_fingerprint(question),
                }
                rows.append(row)
                by_cross_id[cross_id] = row
    return rows, by_cross_id


def _candidate_pairs(rows: list[dict[str, Any]]) -> set[frozenset[str]]:
    by_identity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for identity in row["identity_keys"]:
            by_identity[identity].append(row)
    result: set[frozenset[str]] = set()
    for grouped in by_identity.values():
        for left, right in combinations(grouped, 2):
            if left["model_number"] == right["model_number"]:
                continue
            result.add(frozenset((
                left["cross_question_id"], right["cross_question_id"],
            )))
    return result


def _synthetic_exam(rows: list[dict[str, Any]]) -> dict[str, Any]:
    questions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        questions[row["question_type"]].append(deepcopy(row["question"]))
    return {"model_number": 0, "questions": dict(questions)}


def _duplicate_edges(
    review: dict[str, Any], by_cross_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pair_by_id = {
        str(value.get("pair_id") or ""): value
        for value in review.get("candidate_pairs") or []
    }
    edges: list[dict[str, Any]] = []
    for verdict in review.get("pair_verdicts") or []:
        pair_id = str(verdict.get("pair_id") or "")
        pair = pair_by_id.get(pair_id) or {}
        left = by_cross_id.get(str((pair.get("left") or {}).get("question_id") or ""))
        right = by_cross_id.get(str((pair.get("right") or {}).get("question_id") or ""))
        if not left or not right:
            continue
        classification = str(verdict.get("classification") or "")
        signals = set(pair.get("signals") or [])
        same_content = left["content_fingerprint"] == right["content_fingerprint"]
        same_instance_signals = bool(
            {"same_normalized_equation", "same_normalized_formula"} & signals
            and {"same_final_result", "same_answer_expression"} & signals
            and (
                "distinctive_repeated_givens" in signals
                or "near_identical_stem" in signals
            )
        )
        if same_content or same_instance_signals:
            classification = "cross_model_duplicate"
        elif classification == "duplicate":
            classification = "cross_model_duplicate"
        elif classification == "excessive_overlap":
            classification = "cross_model_excessive_overlap"
        if classification not in {
            "cross_model_duplicate", "cross_model_excessive_overlap",
        }:
            continue
        edges.append({
            "pair_id": pair_id,
            "left": left["cross_question_id"],
            "right": right["cross_question_id"],
            "classification": classification,
            "signals": sorted(signals),
            "same_content_fingerprint": same_content,
            "semantic_facts": {
                key: verdict.get(key)
                for key in (
                    "same_core_knowledge", "same_student_operation_or_method",
                    "same_target_or_result", "genuinely_new_reasoning",
                    "one_answer_substantially_gives_away_other",
                )
            },
        })
    return edges


def _conflict_components(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        adjacency[edge["left"]].add(edge["right"])
        adjacency[edge["right"]].add(edge["left"])
    components: list[set[str]] = []
    seen: set[str] = set()
    for node in sorted(adjacency):
        if node in seen:
            continue
        pending = [node]
        component: set[str] = set()
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(adjacency[current] - component)
        seen.update(component)
        components.append(component)
    return [{
        "nodes": sorted(component),
        "edges": [
            deepcopy(edge) for edge in edges
            if edge["left"] in component and edge["right"] in component
        ],
    } for component in components]


def integrate_cross_model_duplicates(
    state: dict[str, Any], generated: list[dict[str, Any]],
    reports_by_model: dict[int, dict[str, Any]], repair_manifest: list[dict[str, Any]],
    validated_models: list[int], warnings: list[Any],
) -> tuple[list[dict[str, Any]], list[int], list[Any], dict[str, Any]]:
    """Run or incrementally refresh the persisted cross-model duplicate scan."""
    num_models = int(state.get("num_models") or 1)
    prior = {
        "cross_model_duplicate_scan_complete": bool(
            state.get("cross_model_duplicate_scan_complete")
        ),
        "cross_model_duplicate_scan_revision": int(
            state.get("cross_model_duplicate_scan_revision") or 0
        ),
        "cross_model_duplicate_findings": deepcopy(
            state.get("cross_model_duplicate_findings") or []
        ),
        "cross_model_duplicate_fingerprints": dict(
            state.get("cross_model_duplicate_fingerprints") or {}
        ),
        "cross_model_duplicate_fact_cache": dict(
            state.get("cross_model_duplicate_fact_cache") or {}
        ),
        "cross_model_duplicate_last_review": deepcopy(
            state.get("cross_model_duplicate_last_review") or {}
        ),
    }
    if num_models <= 1:
        return repair_manifest, validated_models, warnings, {
            **prior, "cross_model_duplicate_scan_complete": True,
        }
    if set(validated_models) != set(range(1, num_models + 1)):
        return repair_manifest, validated_models, warnings, prior

    rows, by_cross_id = _accepted_questions(
        generated, reports_by_model, state.get("plans") or [],
    )
    fingerprints = {
        row["cross_question_id"]: row["content_fingerprint"] for row in rows
    }
    old_fingerprints = prior["cross_model_duplicate_fingerprints"]
    changed = {
        question_id for question_id, fingerprint in fingerprints.items()
        if old_fingerprints.get(question_id) != fingerprint
    }
    changed.update(set(old_fingerprints) - set(fingerprints))
    if prior["cross_model_duplicate_scan_complete"] and not changed:
        return repair_manifest, validated_models, warnings, prior

    forced_pairs = _candidate_pairs(rows)
    if prior["cross_model_duplicate_scan_complete"]:
        forced_pairs = {
            pair for pair in forced_pairs if set(pair) & changed
        }
    review_kwargs: dict[str, Any] = {}
    domain = str((state.get("assessment_profile") or {}).get("domain") or "")
    if domain:
        from app.llm.factory import create_review_client
        review_kwargs["client_factory"] = lambda: create_review_client(
            domain=domain, stage="duplicate_validator"
        )
    review = review_duplicate_candidates(
        _synthetic_exam(rows),
        plan_by_slot=_plan_index(state.get("plans") or []),
        forced_question_pairs=forced_pairs,
        changed_question_ids=(changed or None),
        fact_cache=prior["cross_model_duplicate_fact_cache"],
        forced_only=True,
        content_only_prompt=True,
        **review_kwargs,
    )
    edges = _duplicate_edges(review, by_cross_id)
    findings: list[dict[str, Any]] = []
    targets: dict[tuple[int, str], dict[str, Any]] = {}
    revision = prior["cross_model_duplicate_scan_revision"] + 1
    for group_number, component in enumerate(_conflict_components(edges), start=1):
        members = sorted(
            (by_cross_id[value] for value in component["nodes"]),
            key=lambda value: (
                # Preserve the original owner. A later Recovery replacement
                # has a higher assignment/question revision and is recovered
                # before an unchanged original, regardless of model number.
                int(value["plan"].get("assignment_revision") or 0),
                int(((state.get("slot_lifecycle") or {}).get(
                    value["slot_id"], {}
                )).get("question_revision") or 0),
                value["model_number"], value["slot_id"], value["question_id"],
            ),
        )
        edge_keys = {
            frozenset((edge["left"], edge["right"]))
            for edge in component["edges"]
        }
        # Build a deterministic maximal set of mutually non-conflicting
        # questions.  This preserves the earliest model for every actual edge
        # without needlessly replacing a later question that only conflicted
        # with another question already selected for recovery.
        retained: list[dict[str, Any]] = []
        duplicates: list[dict[str, Any]] = []
        references_by_duplicate: dict[str, list[dict[str, Any]]] = {}
        for member in members:
            conflicts = [
                reference for reference in retained
                if frozenset((
                    member["cross_question_id"], reference["cross_question_id"],
                )) in edge_keys
            ]
            if conflicts:
                duplicates.append(member)
                references_by_duplicate[member["cross_question_id"]] = conflicts
            else:
                retained.append(member)
        reference = retained[0]
        group_id = f"cross_group_r{revision}_{group_number}"
        finding = {
            "conflict_group_id": group_id,
            "task_id": reference["plan"].get("task_id"),
            "selected_variation_id": reference["plan"].get("selected_variation_id"),
            "selected_variation_group_id": reference["plan"].get(
                "selected_variation_group_id"
            ),
            "reference": {
                "model_number": reference["model_number"],
                "question_id": reference["question_id"],
                "slot_id": reference["slot_id"],
            },
            "duplicates": [{
                "model_number": value["model_number"],
                "question_id": value["question_id"],
                "slot_id": value["slot_id"],
            } for value in duplicates],
            "failure_code": "cross_model_duplicate",
            "reason": (
                "The actual question instance duplicates or excessively overlaps an "
                "earlier finalized model question."
            ),
            "required_change": (
                "Produce an independently valid, source-grounded question instance; "
                "wording-only changes do not resolve the conflict."
            ),
            "content_fingerprints": {
                value["question_id"]: value["content_fingerprint"] for value in members
            },
            "edges": deepcopy(component["edges"]),
            "scan_revision": revision,
            "recovery_status": "pending" if duplicates else "not_required",
        }
        findings.append(finding)
        for duplicate in duplicates:
            conflict_question_ids = [
                value["cross_question_id"]
                for value in references_by_duplicate[duplicate["cross_question_id"]]
            ]
            target = {
                "model_number": duplicate["model_number"],
                "question_id": duplicate["question_id"],
                "slot_id": duplicate["slot_id"],
                "question_type": duplicate["question_type"],
                "task_id": duplicate["plan"].get("task_id"),
                "validation_role": "duplicate",
                "validation_status": "FAIL",
                "status": "invalid",
                "failure_code": "cross_model_duplicate",
                "original_failure_code": "cross_model_duplicate",
                "failure_reason": finding["reason"],
                "repair_category": "duplicate_conflict",
                "repair_type": "replace_assignment",
                "duplicate_group_id": group_id,
                "conflicting_question_ids": [
                    value for value in conflict_question_ids
                ],
                "cross_model_conflict": deepcopy(finding),
                "verdict": {
                    "question_id": duplicate["question_id"],
                    "action": "FIX_QUESTION_AND_ANSWER",
                    "validation_status": "FAIL",
                    "validation_role": "duplicate",
                    "failure_code": "cross_model_duplicate",
                    "fields_to_fix": [],
                    "reason": finding["reason"],
                    "expected_fix": finding["required_change"],
                    "conflicting_question_ids": [
                        value for value in conflict_question_ids
                    ],
                    "required_fixes": [],
                },
            }
            targets[(duplicate["model_number"], duplicate["slot_id"])] = target
            lifecycle = (state.get("slot_lifecycle") or {}).get(duplicate["slot_id"])
            if isinstance(lifecycle, dict):
                lifecycle["local_validation_status"] = "PASS"
                lifecycle["cross_model_duplicate_status"] = "FAIL"
                lifecycle["final_acceptance_status"] = "PENDING_RECOVERY"

    retained_manifest = [
        value for value in repair_manifest
        if value.get("failure_code") != "cross_model_duplicate"
        and value.get("original_failure_code") != "cross_model_duplicate"
    ]
    retained_manifest.extend(targets.values())
    validated = sorted(
        set(validated_models) - {model for model, _slot in targets}
    )
    # Cross-model overlap remains internal quality telemetry.  It must never
    # become a user-facing exam warning or leak into PDF/DOCX metadata.
    return retained_manifest, validated, warnings, {
        "cross_model_duplicate_scan_complete": True,
        "cross_model_duplicate_scan_revision": revision,
        "cross_model_duplicate_findings": findings,
        "cross_model_duplicate_fingerprints": fingerprints,
        "cross_model_duplicate_fact_cache": review.get("fact_cache") or {},
        "cross_model_duplicate_last_review": {
            "candidate_pairs": deepcopy(review.get("candidate_pairs") or []),
            "pair_verdicts": deepcopy(review.get("pair_verdicts") or []),
            "similarity_distribution": deepcopy(
                review.get("similarity_distribution") or {}
            ),
            "changed_question_ids": sorted(changed),
            "scan_revision": revision,
        },
        "cross_model_duplicate_internal_count": len(targets),
    }
