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



VALIDATOR_SYSTEM_PROMPT = """You are the general, non-quantitative exam reviewer. Review each question independently. Do not perform duplicate detection or exam-wide coverage review.

Rules:
- Return one root JSON object with a `verdicts` array containing exactly one verdict per question_id.
- Use assigned_sources for facts, laws, formulas, definitions, terminology, and source-specific values. Neutral numbers explicitly stated as givens in a self-contained exercise are problem data and need not appear verbatim in the source.
- Apply the subject-aware unit policy: mathematics allows a dimensionally consistent unit change; physics and other sciences prefer the source unit and allow a change only when the conversion is derivable from stated givens or the source; exact_source_instance preserves the literal source value and unit. Never demand an invented conversion factor.
- Check clarity, grammar, grounding, planned question type and operation, difficulty fit, assessment guidance, and non-quantitative answer quality. If difficulty is wrong, repair the student-facing question to match the plan; never request changes to plan.difficulty or any Planner field.
- Verify difficulty from the cognitive work the student must actually perform against the assigned evidence and expected reasoning_depth/reasoning_requirements. Do not infer difficulty from wording length, large numbers, or reasoning_mode alone. For an adjusted task, independently reject a label-only change or any new operation/prerequisite unsupported by assigned_sources.
- Compare expected_catalog_task with the actual generated question. It must preserve the source-grounded concept/knowledge, genuinely fixed required_givens, complexity_boundaries, question type, difficulty, and evidence boundary. Ordinary Catalog/source example numbers are illustrative unless exact_source_instance is true or value_policy is immutable: a permitted new numerical instance is not task drift and must not be rejected merely because its numbers differ. The base operation, target/result, representation, or application may change only when one allowed_variations object explicitly names that target and operation and its stated conditions are satisfied. For ordinary non-exact exercises, coherent alternative numerical problem data are allowed without a separate flag; this never authorizes a new concept or method. When exact_source_instance is true, preserve the exact target and values. Being on the same broad topic is not enough. If the question changes to an unauthorized operation/target or crosses a complexity boundary, return failure_code=task_drift and describe the mismatch.
- Treat expected_catalog_task.selected_variation as the exact chosen assessment angle. Check the actual question, answer, and solution against its target, operation, reasoning, representation, application, and compatible question type. Its example values are illustrative unless exact_source_instance is true or value_policy is immutable; never require the generated instance to copy the example numbers. Use only effective_evidence_chunk_ids and the corresponding assigned_sources. Semantic consistency is required; exact wording matches are not.
- Check that required representations are included. Leave missing numerical/symbolic givens and calculations to the STEM quantitative validator.
- For MCQ, verify that the concise stem performs the assigned operation and does not quote a long passage or reveal the answer. Reject difficulty created only by long/confusing wording or irrelevant detail. Verify four parallel options, exactly one defensible best answer, and plausible same-concept distractors based on nearby concepts, misconceptions, reversed relationships, incomplete reasoning, or realistic mistakes. Reject irrelevant/ridiculous distractors, semantic-equivalent options, multiple defensible answers, and a correct option that stands out by length, detail, qualification, grammar, tone, or copied source wording. Reject unsupported facts, numbers, scenarios, and assumptions. Leave numerical/symbolic correctness to the STEM quantitative validator.
- Distinguish a good catalog task with a bad generated MCQ from a fundamentally unsuitable task. For a repairable generated item, request only the exact student-facing fields that need repair and keep the same task. If the assigned task cannot naturally support one clear best answer, three plausible competitors, and meaningful assessment of its target, use failure_code=mcq_task_unsuitable so allocation can select another task; do not repeatedly repair or weaken that task.
- Why must ask for causal reasoning, mechanisms, justification, or relationships and preserve a compatible reasoning task. Reject listing, definition, evidence-significance, or simple retrieval prompts labeled Why.
- Definition stores the bare source term or concept in term; the UI/export supplies the instruction to define it. Reject prompt text embedded in term, another operation, or an answer that does not directly and concisely define that target.
- Check True/False clarity; one-answer/one-blank FITB; relevant Why/Essay depth; and Equation versus meaningful Word Problem form. Equation must be equation-centered: the student works directly with a displayed mathematical or discipline-specific formal representation. Accept a short instruction, necessary givens/units, multiple equations, and supporting tables that directly serve the symbolic operation. Reject narrative Word Problems, extended prose explanations, or items whose symbolic content appears only in the answer. Do not use word count or symbol count alone. Require valid LaTeX with important expressions visually separate from instructions using display delimiters; require source-appropriate notation for Mathematics, Physics, Chemistry, and other STEM disciplines.
- For Equation and Word Problem, inspect task_operation, target, student_work_product, relationship, required_givens, and reasoning_mode together. Necessary premises and givens must be present, but the prompt must not perform the operation assigned to the student. A starting relationship is appropriate for rearrange or apply-a-provided-rule tasks. For derive, construct, formulate, combine, prove, balance, or select-the-governing-relationship tasks, reject a stem that supplies the final relationship, completed equation, exact reasoning step, or result being assessed; use failure_code=task_drift and request a question-field repair.
- Do not impose one universal Equation or Word Problem pattern. Verify that the student work follows the catalog operation rather than defaulting to rearrange-and-isolate or story-then-choose-a-formula. For a Word Problem, providing a rule is valid only when applying a provided rule is the assigned work; omit it when selecting or forming that relationship is assessed.
- Compare quantitative/application questions with assigned_sources. Reject copying a worked example's same scenario + values + requested result unless exact_source_instance is true. Reusing its supported method or constraints with a materially different, fully stated numerical instance is allowed for ordinary non-exact exercises. Do not treat coherent changed exercise numbers as task drift. Do not require invented facts or contexts.
- Choose the smallest repair: PASS, FIX_ANSWER, FIX_OPTIONS, FIX_QUESTION, or FIX_QUESTION_AND_ANSWER.
- fields_to_fix may name only exact canonical fields: question, statement, term, equation, options, correct_answer, reference_answer, solution_steps, final_answer, key_points, distractors, or solution_certificate. For Fill-in-the-Blank, request only question and/or distractors; its catalog answer and Word Bank are code-owned. Never use a generic answer alias and never name Planner metadata.
- Return only valid JSON using the required verdict schema."""

STEM_QUANTITATIVE_VALIDATOR_SYSTEM_PROMPT = """You are a quantitative STEM correctness validator using medium reasoning. Review only numerical and symbolic correctness. Do not judge difficulty, wording, grammar, style, formatting, duplication, coverage, Planner metadata, or whether neutral constructed givens appeared in the source.
For each question, first identify exactly what the student-facing request asks the student to produce. Validate everything quantitatively necessary for that requested output, and nothing beyond it. Do not require an additional quantity the question never asks for, do not automatically require every variable appearing in a system to be solved, do not make optional extra calculations mandatory, and do not redesign or improve the expected answer during validation. A break-even problem asking for the break-even customer count and profit at 60 customers does not also require an unrequested y value.
Then solve the student-facing problem independently without trusting its stored answer or solution_certificate. Record that independent calculation in independent_solution using checkable steps and a final answer. Inspect the stored solution_certificate and compare its givens, mathematical steps, option mapping, units, and final answer with your independent work. If they disagree, identify the first mathematically incorrect step in first_incorrect_step and set certificate_valid=false. Use a second verification method when practical: substitution, reverse calculation, dimensional analysis, constraint testing, recomputation, or checking every feasible candidate. State that check in independent_solution.verification_method.
Check equations, calculations, units, rounding, constraints, missing numerical or symbolic givens required by the actual request, all and only the requested outputs, solution completeness for those outputs, and whether the final answer solves exactly what was asked. For numerical or symbolic MCQ, solve first, verify exactly one option is correct, verify the stored answer points to it, and record matching_correct_option. For optimization, test only feasible candidates. For derivations, verify every step reaches the requested relationship. A missing or malformed certificate is an incomplete quantitative answer and must be repaired. An explicit solution_certificate=null means a previous question repair intentionally invalidated the old certificate; when the student-facing question and answer are otherwise correct, return FIX_ANSWER with fields_to_fix=["solution_certificate"], failure_code="incomplete_solution", and reason="previous solution certificate was invalidated after question repair and must be rebuilt for the current question". This certificate-only case has no mathematically incorrect step, so first_incorrect_step must be null.
Apply the subject-aware unit policy: do not flag a dimensionally valid unit change as a unit error in mathematics; in physics and other sciences require the conversion to be derivable from stated givens or the source; for exact_source_instance the literal source value and unit must be preserved. A contradictory source (mixed unit systems) is an evidence defect, not a student error.
Put the independently computed result in expected_results. Request only student-facing quantitative fields plus the internal solution_certificate. Never request a Planner-field edit. Do not report non-quantitative defects. Return one root JSON object with a `verdicts` array containing exactly one verdict per question_id and no other root fields."""

VALIDATOR_OUTPUT_SCHEMA = (
    '{\n'
    '  "verdicts": [\n'
    '  {\n'
    '    "question_id": "...",\n'
    '    "question_valid": true,\n'
    '    "answer_valid": false,\n'
    '    "action": "FIX_ANSWER",\n'
    '    "fields_to_fix": ["reference_answer"],\n'
    '    "failure_code": "incorrect_result",\n'
    '    "expected_results": ["the independently checked result"],\n'
    '    "reason": "...",\n'
    '    "expected_fix": "...",\n'
    '    "independent_solution": {"method": "...", "important_givens": ["..."], '
    '"calculation_steps": ["..."], "final_answer": "...", "units": "", '
    '"matching_correct_option": null, "verification_method": "..."},\n'
    '    "certificate_valid": true,\n'
    '    "first_incorrect_step": null\n'
    '  }\n'
    '  ]\n'
    '}'
)

QUANTITATIVE_VALIDATOR_OUTPUT_SCHEMA = (
    '{\n'
    '  "verdicts": [\n'
    '  {\n'
    '    "question_id": "...",\n'
    '    "question_valid": true,\n'
    '    "answer_valid": true,\n'
    '    "action": "PASS",\n'
    '    "fields_to_fix": [],\n'
    '    "failure_code": "pass",\n'
    '    "expected_results": ["the independently checked result"],\n'
    '    "reason": "...",\n'
    '    "expected_fix": "...",\n'
    '    "independent_solution": {"method": "...", "important_givens": ["..."], '
    '"calculation_steps": ["..."], "final_answer": "...", "units": "", '
    '"matching_correct_option": null, "verification_method": "..."},\n'
    '    "certificate_valid": true,\n'
    '    "first_incorrect_step": null\n'
    '  }\n'
    '  ]\n'
    '}'
)

VALIDATOR_RETRY_SYSTEM_PROMPT = VALIDATOR_SYSTEM_PROMPT + "\nA previous response omitted verdicts. Review every supplied question and never skip an ID."
STEM_QUANTITATIVE_VALIDATOR_RETRY_SYSTEM_PROMPT = STEM_QUANTITATIVE_VALIDATOR_SYSTEM_PROMPT + "\nA previous response omitted verdicts. Review every supplied question and never skip an ID."


def build_validator_prompt(
    entries: List[Dict[str, Any]], *, quantitative: bool = False,
    rejection_reason: str = "",
) -> str:
    prompt_entries: List[Dict[str, Any]] = []
    evidence_registry: Dict[str, str] = {}
    for entry in entries:
        cleaned = dict(entry)
        source_chunks = cleaned.pop("_source_chunks", {}) or {}
        if isinstance(source_chunks, dict):
            for chunk_id, content in source_chunks.items():
                if chunk_id and content:
                    evidence_registry.setdefault(str(chunk_id), str(content))
        cleaned["assigned_evidence_ids"] = [
            str(chunk_id) for chunk_id in source_chunks if chunk_id
        ] if isinstance(source_chunks, dict) else []
        prompt_entries.append(cleaned)
    evidence_payload = json.dumps(evidence_registry, ensure_ascii=False)
    payload = json.dumps(prompt_entries, ensure_ascii=False)
    retry_context = (
        "\n\n## Corrective retry\n"
        "The previous structured response was rejected by code for this exact reason:\n"
        f"{rejection_reason}\n"
        "Return a fresh verdict for only the supplied question ID and preserve the exact schema."
        if rejection_reason else ""
    )
    return (
        "Review every exam question below and return one structured verdict per "
        "question (same number of results as entries).\n"
        "Use the exact question_ids you were given and never invent new ones. Each entry is an "
        "isolated evidence boundary. Use only that entry's assigned_sources; never use another "
        "entry's evidence IDs to validate or rescue it. Source text is stored once in the "
        "shared evidence_registry; an entry may use only its assigned_evidence_ids.\n\n"
        "## Evidence registry\n"
        f"{evidence_payload}\n\n"
        "## Questions to review\n"
        f"{payload}\n\n"
        "## Verdict object format (use exactly these field names)\n"
        f"{QUANTITATIVE_VALIDATOR_OUTPUT_SCHEMA if quantitative else VALIDATOR_OUTPUT_SCHEMA}\n\n"
        "Return ONLY one JSON object with a `verdicts` array. No text before or after, "
        "no fences, and never return a bare root array."
        f"{retry_context}"
    )


def _plan_source_ids(plan_item: Optional[Dict[str, Any]]) -> List[str]:
    if not plan_item:
        return []
    values = list(plan_item.get("source_evidence_ids") or [
        plan_item.get("source_chunk_id"),
        *(plan_item.get("support_chunk_ids") or []),
    ])
    return list(dict.fromkeys(str(value) for value in values if value))


def _assigned_source_chunks(
    plan_item: Optional[Dict[str, Any]], source_by_chunk: Optional[Dict[str, str]],
    evidence_ids: Optional[List[str]] = None,
) -> Dict[str, str]:
    sources = source_by_chunk or {}
    return {
        chunk_id: sources[chunk_id]
        for chunk_id in (evidence_ids if evidence_ids is not None else _plan_source_ids(plan_item))
        if sources.get(chunk_id)
    }


def _plan_evidence_consistency_error(
    plan_item: Optional[Dict[str, Any]], source_by_chunk: Optional[Dict[str, str]],
) -> str | None:
    """Validate the materialized assignment before generation or repair calls."""
    if not plan_item:
        return "replacement plan item is missing"
    assigned = _plan_source_ids(plan_item)
    if not assigned:
        return "replacement plan has no assigned evidence"
    selected_id = str(plan_item.get("selected_variation_id") or "")
    variations = [
        *(plan_item.get("permitted_variations") or []),
        *(plan_item.get("allowed_variations") or []),
    ]
    selected = next((
        value for value in variations
        if selected_id and str(value.get("variation_id") or "") == selected_id
    ), None)
    variation_evidence = (
        list(dict.fromkeys(map(str, selected.get("evidence_chunk_ids") or [])))
        if selected is not None and selected.get("evidence_chunk_ids") is not None
        else assigned
    )
    if variation_evidence != assigned:
        return (
            "selected variation evidence does not match the materialized assignment: "
            f"variation={variation_evidence}, assigned={assigned}"
        )
    unavailable = [value for value in assigned if value not in (source_by_chunk or {})]
    if unavailable:
        return f"assigned replacement evidence is unavailable: {unavailable}"
    return None


def _catalog_task_contract(
    plan_item: Optional[Dict[str, Any]], question: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Expose the intended assessment task without relying on broad topic fit."""
    plan = plan_item or {}
    givens_contract = authoritative_givens(plan)
    value_contract = numerical_instance_contract(plan)
    contract = {
        "slot_id": plan.get("slot_id"),
        "task_id": plan.get("task_id"),
        "task_family_id": plan.get("task_family_id"),
        "coverage_id": plan.get("coverage_id"),
        "concept_id": plan.get("concept_id"),
        "relationship_id": plan.get("relationship_id"),
        "concept": plan.get("concept") or plan.get("relationship") or plan.get("knowledge_focus"),
        "knowledge_target": plan.get("knowledge_focus"),
        "student_operation": plan.get("task_operation") or plan.get("canonical_skill"),
        "target_or_result": plan.get("target"),
        "relationship": plan.get("relationship"),
        "reasoning_mode": plan.get("reasoning_mode"),
        "required_output": plan.get("student_work_product"),
        "required_givens": list(plan.get("required_givens") or []),
        "variable_givens": list(plan.get("variable_givens") or []),
        "allowed_variations": list(plan.get("allowed_variations") or []),
        "complexity_boundaries": list(plan.get("complexity_boundaries") or []),
        "exact_source_instance": bool(plan.get("exact_source_instance", False)),
        "scenario_family": plan.get("scenario_family"),
        "difficulty": plan.get("difficulty"),
        "assigned_evidence_ids": _plan_source_ids(plan),
        "selected_variation_id": (question or {}).get("selected_variation_id"),
        "selected_variation": (question or {}).get("selected_variation"),
        "selected_variation_group_id": plan.get("selected_variation_group_id"),
        "effective_instance_id": plan.get("effective_instance_id"),
        "effective_givens": list(givens_contract["givens"]),
        "source_variation_givens": list(givens_contract["source_givens"]),
        "uses_effective_givens": givens_contract["uses_effective_givens"],
        "numeric_adjustment_authorized": givens_contract["numeric_adjustment_authorized"],
        "value_policy": value_contract["value_policy"],
        "illustrative_values": list(value_contract["illustrative_values"]),
        "immutable_values": list(value_contract["immutable_values"]),
        "effective_evidence_chunk_ids": list(
            (question or {}).get("effective_evidence_chunk_ids") or _plan_source_ids(plan)
        ),
    }
    if any(key in plan for key in (
        "reasoning_depth", "reasoning_requirements", "prerequisites",
        "requested_difficulty", "actual_difficulty", "difficulty_design",
        "difficulty_adaptation",
    )):
        contract.update({
            "reasoning_depth": plan.get("reasoning_depth"),
            "reasoning_requirements": list(plan.get("reasoning_requirements") or []),
            "prerequisites": list(plan.get("prerequisites") or []),
            "requested_difficulty": plan.get("requested_difficulty"),
            "actual_difficulty": plan.get("actual_difficulty"),
            "difficulty_adjustment": (
                plan.get("difficulty_design") or plan.get("difficulty_adaptation")
            ),
        })
    if plan.get("fitb_entry_id"):
        contract.update({
            "fitb_entry_id": plan.get("fitb_entry_id"),
            "fitb_answer": plan.get("fitb_answer"),
            "clue_basis": plan.get("clue_basis"),
        })
    return contract


from app.llm.math_strings import MATH_JSON_CONTRACT

REPAIR_SYSTEM_PROMPT = (
    "You are the general, non-quantitative exam-question repairer. For each item, fix ONLY "
    "the wording, clarity, grounding, type-rule, difficulty-fit, or conceptual-quality defect "
    "listed in fields_to_fix. Calculations, equations, units, numerical givens, and "
    "numerical/symbolic answers normally belong to the STEM quantitative repairer; when "
    "fields_to_fix authorizes them (for example a copied_source_example finding where numeric "
    "variation is permitted), recompute them together with the revised question instead of "
    "preserving stale values. For exact_source_instance tasks restore the exact source value and "
    "unit rather than inventing a conversion. Rules:\n"
    "- Return the EXACT same question_id for each repaired question; never change it.\n"
    "- Do NOT change fields that are not listed in fields_to_fix.\n"
    "- Do NOT rewrite a question that is fine; preserve every valid field verbatim.\n"
    "- Only repair the specific problem described in reason/expected_fix.\n"
    "- Each item is an isolated evidence boundary. When plan and source_chunks are supplied, use "
    "only that item's source_chunks and preserve its assessment guidance, task_operation, "
    "relationship, target, reasoning_mode, and difficulty. Never borrow evidence from another item.\n"
    "- For Fill-in-the-Blank, return only stem, distractor_1, and distractor_2. The selected "
    "fitb_entry_id and answer are authoritative and immutable; code rebuilds the Word Bank.\n"
    "- A Why repair must remain causal and use Why/علّل/لماذا/ما سبب/فسّر سبب wording; never "
    "replace it with listing, definition, evidence-significance, or simple retrieval.\n"
    "- A Definition repair stores only the bare source term or concept in term and must return a "
    "concise direct definition. Never put instruction wording or another operation in term.\n"
    "- Conceptual Multiple Choice (when options are in fields_to_fix): evaluate the stem "
    "and source evidence, then evaluate the option set. There must be EXACTLY one correct "
    "option. If two or more options are correct, keep one and rewrite only the "
    "conflicting distractor(s) so exactly one correct answer remains. If the real "
    "correct answer is missing from the options, replace one incorrect option with "
    "the real correct answer. After changing the options, update the stored answer "
    "to point to the one correct option. Preserve every already-valid option; change "
    "the minimum number of options needed; never regenerate the whole MCQ randomly; "
    "keep the question text unchanged unless the validator explicitly marked the "
    "question itself as broken.\n"
    "- For FIX_OPTIONS, preserve the valid stem and correct-answer meaning unless explicitly "
    "marked broken. Treat every distractor listed in rejected_distractors as rejected content: "
    "discard it and write a NEW replacement; never keep it unchanged or merely move it to a "
    "different A-D label. Generate replacements from the assigned concept, evidence, operation, "
    "and task. Before returning, self-check that every distractor is same-topic, plausible to a "
    "partially informed student, definitely incorrect, and meaningfully competes with the correct "
    "answer, while exactly one answer remains defensible.\n"
    "- MCQ distractors must compete on the assigned concept and remain parallel in length, grammar, "
    "detail, specificity, and tone. Replace irrelevant or implausible distractors with grounded "
    "misconceptions or incomplete/reversed reasoning; never add unsupported facts or scenarios.\n"
    "- For an Equation or Word Problem wording/type repair, preserve all quantitative givens, "
    "solution steps, and final answer verbatim unless a listed non-quantitative field must change.\n"
    "- When repairing an Equation or Word Problem question field, preserve the exact assigned "
    "operation. Give a starting relation for rearrange/apply-provided-rule work, but do not reveal "
    "the final relationship, completed equation, governing rule, reasoning step, or result when "
    "deriving, constructing, formulating, balancing, proving, or selecting it is the assessed work.\n"
    "- An Equation repair must remain equation-centered and use valid LaTeX. Keep short instructions "
    "separate from displayed equations, systems, relationships, reactions, or supporting tables. "
    "Do not turn it into a narrative Word Problem or reveal a relationship the student must derive.\n"
    "- Return one root JSON object with a `repairs` array containing one repair patch per "
    "input item, in any order. Each object is "
    '{"question_id":"...","patch":{only the exact fields_to_fix}}. '
    "Never return the complete question. When the response schema requests repair_metadata, include it and no other field.\n"
    "- Return ONLY valid JSON. No fences, no extra text.\n\n" + MATH_JSON_CONTRACT
)

STEM_QUANTITATIVE_REPAIR_SYSTEM_PROMPT = """You are a quantitative STEM question repairer using medium reasoning. Repair every authorized field listed in fields_to_fix. Quantitative fields include numbers, equations, calculations, units, missing numerical or symbolic givens, solution_steps, final_answer, and numerical or symbolic MCQ options or correct_answer. When required_fixes also contains a General Validator finding, address its authorized wording, option, or structure field in the same coherent patch; do not discard either validator's finding. Also replace the internal solution_certificate so it exactly documents the repaired mathematics. Preserve the catalog task and every field not authorized by fields_to_fix. Add only premises and givens needed by the assigned operation; never add the final relationship, completed equation, governing rule, reasoning step, or result when producing or selecting it is the student work being assessed. Keep Equation items equation-centered and in valid LaTeX, with concise instructions separated from display equations and no narrative Word Problem conversion.
Independently solve the repaired problem from scratch. Treat expected_results and expected_fix as diagnostic hypotheses, not authority. Verify by a second method when practical, such as substitution, reverse calculation, constraint testing, conservation, dimensional analysis, or independent recomputation. For optimization, use only feasible candidates. For MCQ, verify exactly one option is correct and put its label in solution_certificate.matching_correct_option. Keep solution_steps, final_answer, options, stored answer, and solution_certificate consistent. Never edit Planner metadata or address wording, difficulty, duplication, or coverage.
Return one root JSON object with a `repairs` array. Each item must include question_id and patch; when the response schema requests repair_metadata, include it with modified_fields, changes, and targeted_failure_codes. Never return the complete question or any other unrequested field. Return only valid JSON.""" + "\n\n" + MATH_JSON_CONTRACT

def build_repair_prompt(
    payload: List[Dict[str, Any]], *, include_metadata: bool = False,
) -> str:
    block = json.dumps(payload, ensure_ascii=False, indent=None)
    metadata = (
        ',"repair_metadata":{"modified_fields":["field"],"changes":'
        '[{"field":"field","previous_value":"JSON value before",'
        '"new_value":"JSON value after","reason":"why this fixes the finding"}],'
        '"targeted_failure_codes":["code"]}'
        if include_metadata else ""
    )
    return (
        "Repair the following questions. Each item tells you exactly what is broken "
        "and what to fix. Change only the listed fields and preserve the intended assessment task. Always keep "
        "question_id unchanged.\n\n"
        "## Items to repair\n"
        f"{block}\n\n"
        "## Repair object format\n"
        '{"repairs":[{"question_id":"the unchanged ID","patch":'
        "{exactly every field in fields_to_fix and no other fields}"
        f"{metadata}" + "}]}\n\n"
        "Return ONLY one JSON object with a `repairs` array. No text before or after, "
        "no fences, and never return a bare root array."
    )
