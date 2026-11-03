from __future__ import annotations

import json
import hashlib
import logging
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from itertools import combinations
from typing import Any, Literal

from pydantic import Field, create_model

from app import config
from app.llm.client import request_structured
from app.config import DUPLICATE_VALIDATOR_BATCH_SIZE, MAX_PARALLEL_DUPLICATE_VALIDATOR_CALLS
from app.llm.factory import create_llm_client
from app.llm.schemas import StrictOutputModel
from app.online.semantic_keys import formula_semantic_key

logger = logging.getLogger("DUPLICATE_VALIDATOR")

DUPLICATE_VALIDATOR_PROMPT_VERSION = "semantic-facts-v6-metadata-candidates"
SEMANTIC_FACT_FIELDS = (
    "same_core_knowledge",
    "same_student_operation_or_method",
    "same_target_or_result",
    "genuinely_new_reasoning",
    "one_answer_substantially_gives_away_other",
)


class DuplicatePairVerdict(StrictOutputModel):
    pair_id: str = Field(min_length=1)
    same_core_knowledge: Literal["yes", "no"]
    same_student_operation_or_method: Literal["yes", "no"]
    same_target_or_result: Literal["yes", "no"]
    genuinely_new_reasoning: Literal["yes", "no"]
    one_answer_substantially_gives_away_other: Literal["yes", "no"]


class DuplicateValidatorResponseError(ValueError):
    """A pair-review response cannot safely establish duplicate-free content."""


def _validated_batch_verdicts(
    verdicts: list[dict[str, Any]], expected_pair_ids: list[str],
) -> list[dict[str, Any]]:
    """Require one well-typed semantic-fact row for every requested pair.

    Provider schema validation normally enforces this boundary.  Keeping the
    check here protects direct/mock provider adapters as well: omitting an
    uncertain pair must block validation, never silently mean acceptable reuse.
    """
    expected = set(expected_pair_ids)
    received: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for value in verdicts:
        if not isinstance(value, dict):
            raise DuplicateValidatorResponseError("Duplicate Validator returned a non-object verdict row.")
        pair_id = value.get("pair_id")
        if not isinstance(pair_id, str) or pair_id not in expected:
            raise DuplicateValidatorResponseError(
                f"Duplicate Validator returned unknown pair_id={pair_id!r}."
            )
        if pair_id in received:
            raise DuplicateValidatorResponseError(
                f"Duplicate Validator returned duplicate pair_id={pair_id!r}."
            )
        invalid = [field for field in SEMANTIC_FACT_FIELDS
                   if value.get(field) not in {"yes", "no"}]
        if invalid:
            raise DuplicateValidatorResponseError(
                f"Duplicate Validator returned invalid semantic facts for {pair_id}: {', '.join(invalid)}."
            )
        received.add(pair_id)
        normalized.append(value)
    missing = sorted(expected - received)
    if missing:
        raise DuplicateValidatorResponseError(
            "Duplicate Validator omitted verdicts for " + ", ".join(missing) + "."
        )
    return normalized


def _literal(values: list[str]):
    unique = tuple(dict.fromkeys(values))
    return Literal.__getitem__(unique) if unique else str


def duplicate_response_model(pair_ids: list[str]):
    item = create_model(
        "ActualQuestionDuplicatePairVerdict",
        __base__=DuplicatePairVerdict,
        pair_id=(_literal(pair_ids), ...),
    )
    return create_model(
        "ActualQuestionDuplicateReviewResponse",
        __base__=StrictOutputModel,
        verdicts=(list[item], Field(min_length=len(pair_ids), max_length=len(pair_ids))),
    )


def _section_items(section: Any) -> list[dict[str, Any]]:
    values = section.get("items") if isinstance(section, dict) else section
    return [item for item in (values or []) if isinstance(item, dict)]


def _stem(qtype: str, item: dict[str, Any]) -> str:
    value = (
        item.get("statement") or item.get("question") or item.get("term")
        or item.get("equation") or item.get("prompt") or ""
    )
    if qtype == "mcq" and isinstance(item.get("options"), dict):
        options = " | ".join(f"{label}: {text}" for label, text in item["options"].items())
        return f"{value}\nOptions: {options}".strip()
    return str(value).strip()


def _answer(qtype: str, item: dict[str, Any]) -> str:
    certificate = item.get("solution_certificate") or {}
    certificate_steps = certificate.get("calculation_steps") or [] if isinstance(certificate, dict) else []
    certificate_text = "\n".join([
        *(str(step) for step in certificate_steps if str(step).strip()),
        *([f"Certificate final: {certificate.get('final_answer')}"]
          if isinstance(certificate, dict) and certificate.get("final_answer") else []),
    ])
    if qtype == "mcq":
        label = str(item.get("correct_answer") or "").strip()
        option = (item.get("options") or {}).get(label, "") if isinstance(item.get("options"), dict) else ""
        base = f"{label}: {option}".strip(": ")
        return "\n".join(value for value in (base, certificate_text) if value)
    if qtype == "true_false":
        value = item.get("answer")
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value) if value is not None else ""
    if qtype == "fill_in_the_blank":
        return " | ".join(
            str(value).strip() for value in item.get("answers") or []
            if str(value).strip()
        )
    if qtype in {"equation", "word_problem"}:
        steps = item.get("solution_steps") or []
        if not isinstance(steps, list):
            steps = [steps]
        final = item.get("final_answer") or item.get("answer") or ""
        return "\n".join([*(str(step) for step in steps if str(step).strip()), f"Final: {final}"])
    if qtype == "essay":
        points = item.get("key_points") or []
        return "\n".join(str(point) for point in points) if isinstance(points, list) else str(points)
    base = str(item.get("reference_answer") or item.get("answer") or item.get("final_answer") or "").strip()
    return "\n".join(value for value in (base, certificate_text) if value)


def actual_question_registry(
    exam: dict[str, Any], plan_by_slot: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    plan_by_slot = plan_by_slot or {}
    registry: list[dict[str, Any]] = []
    for qtype, section in (exam.get("questions") or {}).items():
        for item in _section_items(section):
            qid = str(item.get("question_id") or "")
            if not qid:
                continue
            stem = _stem(qtype, item)
            answer = _answer(qtype, item)
            plan = plan_by_slot.get(str(item.get("slot_id") or "")) or {}
            registry.append({
                "question_id": qid, "slot_id": str(item.get("slot_id") or ""),
                "question_type": qtype, "stem": stem, "answer": answer,
                "reuse_context": item.get("reuse_context") or plan.get("reuse_context"),
                "task_id": plan.get("task_id") or item.get("task_id"),
                "selected_variation_id": plan.get("selected_variation_id") or item.get("selected_variation_id"),
                "selected_variation_group_id": plan.get("selected_variation_group_id") or item.get("selected_variation_group_id"),
                "effective_instance_id": item.get("effective_instance_id"),
                "concept": plan.get("concept") or plan.get("concept_to_test"),
                "idea_to_test": plan.get("idea_to_test") or plan.get("concept_to_test"),
                "task_operation": plan.get("task_operation") or plan.get("canonical_skill"),
                "target": plan.get("target"),
                "student_work_product": plan.get("student_work_product") or plan.get("expected_work_product"),
                "reasoning_mode": plan.get("reasoning_mode"),
                "effective_givens": list(plan.get("effective_givens") or plan.get("required_givens") or []),
            })
    return registry


def _normalized(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold(), flags=re.UNICODE))


def _tokens(value: Any) -> set[str]:
    return set(_normalized(value).split())


def _jaccard(left: Any, right: Any) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def _formula_keys(value: Any) -> set[tuple[str, str]]:
    text = str(value or "")
    candidates = [text, *re.split(r"[\n|;؛,]", text)]
    for left, right in re.findall(
        r"([\wΔδ]+(?:_[\w]+)?)\s*=\s*"
        r"([\wΔδ]+(?:\s*[*×/+−\-^]\s*[\wΔδ().]+)*)",
        text,
        flags=re.UNICODE,
    ):
        candidates.append(f"{left}={right}")
    return {key for candidate in candidates if (key := formula_semantic_key(candidate))}


def _number_signature(value: Any) -> tuple[str, ...]:
    values = re.findall(
        r"(?<![\w.])[+\-−]?\d+(?:\.\d+)?(?:\s*[x×]\s*10\s*\^?\s*[+\-−]?\d+)?",
        str(value or ""),
    )
    return tuple(sorted(value.replace(" ", "").replace("−", "-") for value in values))


def _equation_signatures(value: Any) -> set[str]:
    text = str(value or "").replace("−", "-").replace("\\(", "").replace("\\)", "")
    signatures = set()
    for match in re.findall(r"[^\n|؛,،؟]{0,80}[=≤≥<>][^\n|؛,،؟]{0,80}", text):
        compact = re.sub(r"[^\w=≤≥<>+\-*/^.]", "", match.casefold(), flags=re.UNICODE)
        if len(compact) >= 3:
            signatures.add(compact)
    return signatures


def _answer_core(value: Any) -> str:
    text = re.sub(r"^(?:[A-D]\s*:|Final\s*:)", "", str(value or "").strip(), flags=re.I)
    return _normalized(text)


def _same_meaningful_result(left: Any, right: Any) -> bool:
    left_core, right_core = _answer_core(left), _answer_core(right)
    if not left_core or left_core != right_core:
        return False
    # Boolean labels and other generic one-token outcomes are common across an
    # exam and provide no useful duplicate evidence by themselves.
    generic = {"true", "false", "yes", "no", "صح", "صحيح", "خطأ"}
    return left_core not in generic and len(left_core) >= 2


def shortlist_duplicate_pairs(
    exam: dict[str, Any],
    forced_question_pairs: set[frozenset[str]] | None = None,
    plan_by_slot: dict[str, dict[str, Any]] | None = None,
    changed_question_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Shortlist pairs without loading an embedding model or calling a provider."""
    items = actual_question_registry(exam, plan_by_slot)
    if len(items) < 2:
        return [], {"question_count": len(items), "possible_pair_count": 0,
                    "candidate_count": 0, "embedding_calls": 0}
    pair_rows: list[dict[str, Any]] = []
    evaluated = 0
    for left_index, right_index in combinations(range(len(items)), 2):
        left, right = items[left_index], items[right_index]
        if changed_question_ids is not None and not (
            {left["question_id"], right["question_id"]} & changed_question_ids
        ):
            continue
        evaluated += 1
        signals = []
        if frozenset((left["question_id"], right["question_id"])) in (
            forced_question_pairs or set()
        ):
            signals.append("forced_catalog_reuse")
        if left.get("task_id") and left.get("task_id") == right.get("task_id"):
            signals.append("shared_task_id")
        if (left.get("selected_variation_id") and
                left.get("selected_variation_id") == right.get("selected_variation_id")):
            signals.append("shared_variation_id")
        if (
            left.get("selected_variation_group_id")
            and left.get("selected_variation_group_id")
            == right.get("selected_variation_group_id")
        ):
            signals.append("shared_variation_group")
        operation_match = bool(
            _normalized(left.get("task_operation"))
            and _normalized(left.get("task_operation")) == _normalized(right.get("task_operation"))
        )
        target_match = bool(
            _normalized(left.get("target")) and (
                _normalized(left.get("target")) == _normalized(right.get("target"))
                or _jaccard(left.get("target"), right.get("target")) >= 0.72
            )
        )
        product_match = bool(
            _normalized(left.get("student_work_product"))
            and _normalized(left.get("student_work_product"))
            == _normalized(right.get("student_work_product"))
        )
        if operation_match and (target_match or product_match):
            signals.append("matching_operation_and_target")
        if (operation_match and _normalized(left.get("concept"))
                and _normalized(left.get("concept")) == _normalized(right.get("concept"))):
            signals.append("matching_concept_and_operation")
        if _formula_keys(f"{left['stem']}\n{left['answer']}") & _formula_keys(f"{right['stem']}\n{right['answer']}"):
            signals.append("same_normalized_formula")
        equation_overlap = _equation_signatures(left["stem"]) & _equation_signatures(right["stem"])
        if equation_overlap:
            signals.append("same_normalized_equation")
        answer_left, answer_right = _normalized(left["answer"]), _normalized(right["answer"])
        if len(answer_left) >= 4 and answer_left == answer_right and _same_meaningful_result(
            left["answer"], right["answer"]
        ):
            signals.append("same_answer_expression")
        if _same_meaningful_result(left["answer"], right["answer"]):
            signals.append("same_final_result")
        if _jaccard(left["stem"], right["stem"]) >= 0.72:
            signals.append("near_identical_stem")
        givens_left, givens_right = _number_signature(left["stem"]), _number_signature(right["stem"])
        if len(givens_left) >= 2 and givens_left == givens_right:
            signals.append("distinctive_repeated_givens")
        else:
            left_numbers, right_numbers = set(givens_left), set(givens_right)
            shared_numbers = left_numbers & right_numbers
            if len(shared_numbers) >= 3 and len(shared_numbers) / max(1, min(len(left_numbers), len(right_numbers))) >= 0.75:
                signals.append("important_repeated_numbers")
        if signals:
            qids = sorted((left["question_id"], right["question_id"]))
            pair_rows.append({
                "pair_id": f"{qids[0]}::{qids[1]}", "left": left, "right": right,
                "signals": sorted(set(signals)),
            })
    candidates = sorted(pair_rows, key=lambda row: row["pair_id"])
    scores = {
        "question_count": len(items),
        "possible_pair_count": len(items) * (len(items) - 1) // 2,
        "evaluated_pair_count": evaluated,
        "candidate_count": len(candidates),
        "screening_policy": "metadata-and-deterministic-v2",
        "embedding_calls": 0,
    }
    logger.info("Duplicate candidate screening | %s", json.dumps(scores, sort_keys=True))
    return candidates, scores


def _prompt(candidates: list[dict[str, Any]]) -> str:
    registry: dict[str, dict[str, Any]] = {}
    pairs = []
    for pair in candidates:
        for side in ("left", "right"):
            item = pair[side]
            registry[item["question_id"]] = {
                "question_type": item["question_type"], "stem": item["stem"],
                "complete_correct_answer_and_solution": item["answer"],
                "reuse_context": item.get("reuse_context"),
                "task_id": item.get("task_id"),
                "selected_variation_id": item.get("selected_variation_id"),
                "selected_variation_group_id": item.get("selected_variation_group_id"),
                "effective_instance_id": item.get("effective_instance_id"),
                "concept": item.get("concept"), "idea_to_test": item.get("idea_to_test"),
                "task_operation": item.get("task_operation"), "target": item.get("target"),
                "student_work_product": item.get("student_work_product"),
                "reasoning_mode": item.get("reasoning_mode"),
                "effective_givens": item.get("effective_givens") or [],
            }
        pairs.append({
            "pair_id": pair["pair_id"], "left_question_id": pair["left"]["question_id"],
            "right_question_id": pair["right"]["question_id"],
        })
    return """Analyze the actual student-facing questions and complete correct answers/solutions.
An optional reuse_context states an intended distinction. It is context only: verify that
the actual questions realize it, and never treat intentional reuse as an automatic pass. Do not infer
missing Planner metadata. Do not choose duplicate/excessive/acceptable labels. Return exactly
five factual yes/no fields for every pair, using these definitions consistently:
- same_core_knowledge: yes only when both require the same atomic fact, law, definition, relationship,
  or mathematical structure. A shared chapter or broad topic is not enough.
- same_student_operation_or_method: operation means what the student actually does. Return yes only
  when the student performs the same cognitive operation with essentially the same required method.
  Solving, verifying, modeling, deriving, interpreting, and identifying differ when each requires its
  own work. Explicit elimination and substitution differ.
- same_target_or_result: target means what the student is asked to produce or conclude. Return yes for
  the same requested unknown, conclusion, relationship, classification, proof target, or decision.
  A different numerical answer does not make the target different, and an incidental equal answer is
  not enough to make targets the same.
- genuinely_new_reasoning: reasoning means the process, strategy, interpretation, or solution method
  needed to reach the target. Return yes only when the second item requires a materially different
  one. Changing only numbers, coordinates, constants, names, units, surface wording, or scenario
  details is no: the same procedure remains the same reasoning. An explicitly different method can
  supply new reasoning, while method sameness is still reported independently above.
- one_answer_substantially_gives_away_other: yes only when seeing either complete answer supplies
  the exact requested conclusion, final value, or complete derivation needed by the other and makes
  that other item's intended assessment work substantially trivial. A shared prerequisite equation,
  useful intermediate concept, or background relationship is not enough by itself.

Calibration:
- equality modeling versus inequality modeling normally has a different target and new reasoning;
- elimination versus substitution has a different method and new reasoning when each is assessed;
- solving versus verifying the same equations differs in operation, but the solved answer can give
  away verification;
- different systems that incidentally have the same result do not share a target for that reason alone;
- when the same equations appear and one item exposes the exact requested solution, give-away is yes;
  a shared prerequisite equation or useful intermediate relationship is allowed when the assessed
  target, operation, or reasoning differs.
- inferring an unknown starting quantity from an intermediate state and then deriving a general
  equation requires a different operation and new reasoning from directly translating stated fixed
  and variable charges into an equation, even though both answers are equations;
- Q1 `Check whether (0,0) satisfies the system` and Q2 `Check whether (-4,0) satisfies the same
  system` have same knowledge=yes, same operation=yes, same target=yes, new reasoning=no, and
  give-away=no. Different test points and results do not create a new reasoning process.

The pairs were already selected for inspection. You receive no similarity or discovery evidence because
those signals must not influence these semantic facts. Return one root JSON object with a `verdicts`
array containing exactly one analysis for every pair_id and no other root fields. Never return a bare
root array. Each verdict contains pair_id plus exactly the five factual fields defined above.

Question registry:
""" + json.dumps(registry, ensure_ascii=False) + "\n\nPairs:\n" + json.dumps(pairs, ensure_ascii=False)


def deterministic_pair_classification(
    analysis: dict[str, Any], *, authorized_practice_variation: bool = False,
) -> str:
    yes = lambda key: str(analysis.get(key) or "").casefold() == "yes"
    same_knowledge = yes("same_core_knowledge")
    same_operation = yes("same_student_operation_or_method")
    same_target = yes("same_target_or_result")
    new_reasoning = yes("genuinely_new_reasoning")
    gives_away = yes("one_answer_substantially_gives_away_other")
    if same_knowledge and same_operation and same_target and not new_reasoning:
        if authorized_practice_variation and not gives_away:
            return "acceptable_practice_variation"
        return "duplicate"
    # A give-away is actionable only when the two items still overlap in the
    # assessed work. Shared prerequisites are common in STEM exams and must not
    # become duplicates merely because one answer contains a useful equation.
    if gives_away and (same_target or same_operation):
        return "excessive_overlap"
    if same_knowledge and not new_reasoning:
        return "excessive_overlap"
    return "acceptable_practice_variation" if authorized_practice_variation else "acceptable_reuse"


def _fact_cache_key(pair: dict[str, Any], client: Any) -> str:
    def content(value: Any) -> str:
        return " ".join(
            unicodedata.normalize("NFKC", str(value or "")).casefold().split()
        )

    contents = []
    for side in ("left", "right"):
        item = pair[side]
        contents.append(json.dumps({
            "question_type": item.get("question_type"),
            "stem": content(item.get("stem")),
            "answer": content(item.get("answer")),
            "reuse_context": item.get("reuse_context"),
            "task_id": item.get("task_id"),
            "selected_variation_id": item.get("selected_variation_id"),
            "selected_variation_group_id": item.get("selected_variation_group_id"),
            "concept": content(item.get("concept")),
            "idea_to_test": content(item.get("idea_to_test")),
            "task_operation": content(item.get("task_operation")),
            "target": content(item.get("target")),
            "student_work_product": content(item.get("student_work_product")),
            "reasoning_mode": item.get("reasoning_mode"),
            "effective_givens": item.get("effective_givens") or [],
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    model = str(
        getattr(client, "model", None)
        or getattr(client, "model_name", None)
        or type(client).__name__
    )
    material = "\n".join([
        DUPLICATE_VALIDATOR_PROMPT_VERSION,
        "duplicate-screen-v3-metadata-deterministic",
        "pair-ledger-v3-job-local",
        model,
        *sorted(contents),
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _question_value(item: dict[str, Any]) -> float:
    depth = {"word_problem": 6, "equation": 5, "essay": 5, "why": 4,
             "mcq": 3, "definition": 2, "true_false": 1, "fill_in_the_blank": 1}.get(
        item["question_type"], 2
    )
    return depth + min(len(item["answer"]) / 1000.0, 0.5)


def build_duplicate_groups(candidates: list[dict[str, Any]], verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair = {pair["pair_id"]: pair for pair in candidates}
    conflicts = [v for v in verdicts if v.get("classification") in {"duplicate", "excessive_overlap"}
                 and v.get("pair_id") in by_pair]
    adjacency: dict[str, set[str]] = defaultdict(set)
    edges: list[dict[str, Any]] = []
    item_by_id: dict[str, dict[str, Any]] = {}
    for verdict in conflicts:
        pair = by_pair[verdict["pair_id"]]
        left, right = pair["left"]["question_id"], pair["right"]["question_id"]
        item_by_id[left], item_by_id[right] = pair["left"], pair["right"]
        adjacency[left].add(right); adjacency[right].add(left)
        same = {
            "same_knowledge": verdict.get("same_core_knowledge") == "yes",
            "same_operation": verdict.get("same_student_operation_or_method") == "yes",
            "same_target": verdict.get("same_target_or_result") == "yes",
            "same_work_product": verdict.get("same_student_operation_or_method") == "yes",
            "new_reasoning": verdict.get("genuinely_new_reasoning") == "yes",
            "answer_giveaway": verdict.get("one_answer_substantially_gives_away_other") == "yes",
        }
        labels = [name.removeprefix("same_") for name in ("same_knowledge", "same_operation", "same_target") if same[name]]
        if same["answer_giveaway"]:
            labels.append("answer giveaway")
        must_differ = [name for name in ("operation", "target", "work_product", "reasoning")
                       if name in {"operation", "target"} or not same["new_reasoning"]]
        edges.append({
            "left": left, "right": right, "classification": verdict["classification"],
            **same,
            "conflict_reason": "Overlap in " + ", ".join(labels or ["assessed work"]),
            "what_must_be_different": list(dict.fromkeys(must_differ)),
        })
    groups, visited = [], set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack, component = [start], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node); visited.add(node); stack.extend(adjacency[node] - component)
        component_edges = [edge for edge in edges if edge["left"] in component and edge["right"] in component]
        nodes = sorted(component)
        if len(nodes) <= 12:
            covers = []
            for size in range(1, len(nodes) + 1):
                covers = [subset for subset in combinations(nodes, size)
                          if all(edge["left"] in subset or edge["right"] in subset for edge in component_edges)]
                if covers:
                    break
            replacements = min(
                covers,
                key=lambda subset: sum(_question_value(item_by_id[qid]) for qid in subset),
            )
        else:
            # Bounded deterministic 2-approximation for pathological groups.
            uncovered = {(edge["left"], edge["right"]) for edge in component_edges}
            selected: set[str] = set()
            while uncovered:
                left, right = min(uncovered)
                preferred = min((left, right), key=lambda qid: (_question_value(item_by_id[qid]), qid))
                selected.add(preferred)
                uncovered = {edge for edge in uncovered if preferred not in edge}
            replacements = tuple(sorted(selected))
        groups.append({
            "group_id": f"group_{len(groups) + 1}", "question_ids": nodes,
            "edges": component_edges,
            "keep_question_ids": [qid for qid in nodes if qid not in replacements],
            "replace_question_ids": list(replacements),
        })
    return groups


def review_duplicate_candidates(
    exam: dict[str, Any], plan_by_slot: dict[str, dict[str, Any]] | None = None,
    client: Any | None = None,
    fact_cache: dict[str, dict[str, str]] | None = None,
    forced_question_pairs: set[frozenset[str]] | None = None,
    changed_question_ids: set[str] | None = None,
    client_factory: Any | None = None,
) -> dict[str, Any]:
    candidates, distribution = shortlist_duplicate_pairs(
        exam, forced_question_pairs, plan_by_slot, changed_question_ids,
    )
    cache = dict(fact_cache or {})
    if not candidates:
        return {"candidate_pairs": [], "pair_verdicts": [], "duplicate_groups": [],
                "similarity_distribution": distribution, "usage": {},
                "fact_cache": cache, "fact_cache_hits": 0, "fact_cache_misses": 0}
    explicit_client = client
    client = client or (client_factory() if client_factory else create_llm_client())
    cached_by_pair: dict[str, dict[str, Any]] = {}
    uncached: list[dict[str, Any]] = []
    cache_key_by_pair: dict[str, str] = {}
    for pair in candidates:
        key = _fact_cache_key(pair, client)
        cache_key_by_pair[pair["pair_id"]] = key
        facts = cache.get(key)
        if isinstance(facts, dict) and all(field in facts for field in SEMANTIC_FACT_FIELDS):
            cached_by_pair[pair["pair_id"]] = {
                "pair_id": pair["pair_id"],
                **{field: facts[field] for field in SEMANTIC_FACT_FIELDS},
            }
        else:
            uncached.append(pair)
    fresh_by_pair: dict[str, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    if uncached:
        # Each response extracts independent pairwise facts. Classification and
        # conflict grouping remain deterministic after all batches are merged,
        # so splitting here cannot remove cross-pair reasoning.
        batches = [
            uncached[index:index + DUPLICATE_VALIDATOR_BATCH_SIZE]
            for index in range(0, len(uncached), DUPLICATE_VALIDATOR_BATCH_SIZE)
        ]

        def review_batch(batch_index: int, batch: list[dict[str, Any]]):
            batch_ids = [pair["pair_id"] for pair in batch]
            batch_client = explicit_client or (
                client_factory() if client_factory else create_llm_client()
            )
            raw = request_structured(
                batch_client, _prompt(batch),
                response_model=duplicate_response_model(batch_ids),
                schema_name="actual_question_duplicate_judgment",
                agent="duplicate_validator",
                operation="duplicate_validation",
                item_count=len(batch), expected_ids=batch_ids,
                expected_id_field="pair_id",
                system_prompt=(
                    "Extract only the five requested pairwise semantic facts using the supplied definitions. "
                    "Never assign a duplicate, overlap, or reuse label."
                ), temperature=0.0,
                max_tokens=min(16000, max(4096, len(batch) * 260)),
            )
            verdicts = _validated_batch_verdicts(
                list((raw or {}).get("verdicts") or []), batch_ids,
            )
            return batch_index, verdicts, (
                getattr(batch_client, "last_usage", None) or {}
            )

        completed: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
        workers = min(MAX_PARALLEL_DUPLICATE_VALIDATOR_CALLS, 50, len(batches))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(copy_context().run, review_batch, index, batch): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_index, verdicts, batch_usage = future.result()
                completed[batch_index] = (verdicts, batch_usage)
        usage = {"batches": []}
        for batch_index in range(len(batches)):
            verdicts, batch_usage = completed[batch_index]
            for item in verdicts:
                fresh_by_pair[str(item.get("pair_id"))] = item
            usage["batches"].append(batch_usage)
    ordered = []
    for pair in candidates:
        pair_id = pair["pair_id"]
        raw_analysis = cached_by_pair.get(pair_id) or fresh_by_pair.get(pair_id)
        if raw_analysis is None:
            continue
        analysis = dict(raw_analysis)
        has_all_facts = all(field in analysis for field in SEMANTIC_FACT_FIELDS)
        facts = {field: str(analysis.get(field) or "no") for field in SEMANTIC_FACT_FIELDS}
        if pair_id in fresh_by_pair and has_all_facts:
            cache[cache_key_by_pair[pair_id]] = facts
        analysis = {"pair_id": pair_id, **facts}
        authorized = all(
            bool(pair[side].get("reuse_context", {}).get("authorized"))
            for side in ("left", "right")
            if pair[side].get("reuse_context")
        ) and any(pair[side].get("reuse_context") for side in ("left", "right"))
        analysis["classification"] = deterministic_pair_classification(
            analysis, authorized_practice_variation=authorized,
        )
        analysis["authorized_practice_variation"] = authorized
        ordered.append(analysis)
    return {"candidate_pairs": candidates, "pair_verdicts": ordered,
            "duplicate_groups": build_duplicate_groups(candidates, ordered),
            "similarity_distribution": distribution,
            "usage": usage, "fact_cache": cache,
            "fact_cache_hits": len(cached_by_pair),
            "fact_cache_misses": len(uncached)}
