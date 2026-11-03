"""Read-only 40-question mocked end-to-end evaluation for DraftWork.

This module is an *evaluation harness*, not a production change. It drives the
real validator / repair / regeneration / replacement / revalidation / duplicate
/ assembly orchestration and mocks only external LLM boundaries.
"""

from __future__ import annotations

import hashlib
import json
import threading
from copy import deepcopy

import pytest

import app.llm.client as llm_client
from app.online import (
    difficulty_adjustment, duplicate_judge, exam_builder, multi_repair,
    multi_validator, validator,
)
from app.online import duplicate_pipeline
from app.online.exam_builder import assemble_exams_node
from app.online.multi_validator import clear_job_validation_cache, validate_exam_multi
from app.online.validator import (
    _manifest_from_report,
    repair_invalid_questions,
    validate_generated_questions,
)

# ---------------------------------------------------------------------------
# Fixture: 40 realistic Physics/Mathematics questions.
# ---------------------------------------------------------------------------

QTYPE_BY_ID = {
    **{q: "mcq" for q in ("Q1", "Q2", "Q3", "Q5", "Q6", "Q7", "Q9", "Q10", "Q16", "Q17")},
    **{q: "true_false" for q in ("Q4", "Q8", "Q12", "Q13", "Q18")},
    **{q: "fill_in_the_blank" for q in ("Q19", "Q21", "Q22", "Q24", "Q25")},
    **{q: "definition" for q in ("Q23", "Q27", "Q28", "Q29")},
    **{q: "why" for q in ("Q15", "Q31", "Q32", "Q33")},
    **{q: "equation" for q in ("Q11", "Q30", "Q36", "Q37")},
    **{q: "word_problem" for q in ("Q14", "Q26", "Q38", "Q39")},
    **{q: "essay" for q in ("Q20", "Q34", "Q35", "Q40")},
}
QIDS = sorted(QTYPE_BY_ID, key=lambda value: int(value[1:]))

DEFECTS = {
    "Q2": [("answer_correctness", "incorrect_answer", ["correct_answer"])],
    "Q3": [
        ("answer_correctness", "ambiguous_or_structural", ["question"]),
        ("answer_correctness", "multiple_correct_options", ["options"]),
        ("answer_correctness", "incorrect_answer", ["correct_answer"]),
    ],
    "Q5": [("answer_correctness", "weak_distractors", ["options"])],
    "Q8": [("answer_correctness", "incorrect_answer", ["answer"])],
    "Q11": [("quantitative", "incorrect_calculation", ["solution_steps", "final_answer"])],
    "Q14": [("quantitative", "unit_inconsistency", ["solution_steps", "final_answer"])],
    "Q17": [("task_alignment", "wrong_givens", ["question"])],
    "Q20": [("difficulty", "reasoning_complexity_mismatch", ["question"])],
    "Q23": [("grounding", "unsupported", ["term"])],
    "Q26": [("difficulty", "reasoning_complexity_mismatch", ["question"])],
    "Q30": [("task_alignment", "visual_dependency", ["equation"])],
    "Q35": [("difficulty", "difficulty_mismatch", ["question"])],
}

SOURCE_TEXT = {
    "Q1": "Newton's second law states that net force equals mass multiplied by acceleration: F = ma.",
    "Q2": "For a net force F = ma, a 10 kg object accelerating at 3 m/s2 experiences 30 N.",
    "Q3": "Newton's second law F = ma gives 6 N for a 2 kg mass accelerating at 3 m/s2.",
    "Q4": "Near Earth's surface gravitational acceleration is approximately 9.8 m/s2.",
    "Q5": "Force is measured in newtons, the SI unit of force.",
    "Q6": "Newton's second law is expressed by the equation F = ma.",
    "Q7": "Average speed equals distance divided by time: v = d / t.",
    "Q8": "In a vacuum all objects fall with the same acceleration regardless of mass.",
    "Q9": "Speed is a scalar quantity because it has magnitude but no direction.",
    "Q10": "Gravitational potential energy is mgh; 2 kg at 5 m gives 100 J.",
    "Q11": "For m = 4 kg and a = 2 m/s2 the net force is F = ma = 8 N.",
    "Q12": "Work and energy are measured in joules.",
    "Q13": "Momentum is a vector quantity equal to mass times velocity.",
    "Q14": "For m = 5 kg and a = 4 m/s2 the net force is 20 N.",
    "Q15": "A falling object accelerates because gravity exerts a net downward force.",
    "Q16": "Electric current is measured in amperes.",
    "Q17": "A 3 kg mass under a 12 N net force accelerates at 4 m/s2.",
    "Q18": "Kinetic energy depends on speed, so an object at rest has zero kinetic energy.",
    "Q19": "Isaac Newton formulated the three laws of motion.",
    "Q20": "Newton's third law says the swimmer pushes water backward and water pushes the swimmer forward with an equal and opposite force.",
    "Q21": "Acceleration is the rate of change of velocity.",
    "Q22": "A stretched spring stores elastic potential energy.",
    "Q23": "Inertia is the tendency of an object to resist changes in motion; inertia is not a force.",
    "Q24": "Momentum is the product of mass and velocity.",
    "Q25": "Friction is the resistance that opposes motion between surfaces in contact.",
    "Q26": "Final speed under constant acceleration is v = u + at.",
    "Q27": "Momentum is the product of an object's mass and velocity.",
    "Q28": "Work is the product of force and displacement in the direction of the force.",
    "Q29": "Power is the rate at which work is done.",
    "Q30": "Newton's second law relates net force, mass, and acceleration; no free-body diagram is supplied.",
    "Q31": "When walking, friction provides the forward reaction force on the foot.",
    "Q32": "A net force is required to change a moving car's momentum.",
    "Q33": "مقاومة الهواء تعاكس حركة الجسم الساقط وتقلل تسارعه.",
    "Q34": "عند سقوط جسم تتحول طاقة الوضع إلى طاقة حركية.",
    "Q35": "A roller coaster converts potential energy to kinetic energy and back while friction dissipates energy.",
    "Q36": "For u = 0, a = 3 m/s2 and t = 4 s, v = u + at = 12 m/s.",
    "Q37": "Power is work divided by time; 240 J in 8 s gives 30 W.",
    "Q38": "Momentum p = mv; 3 kg at 4 m/s gives 12 kg m/s.",
    "Q39": "كمية الحركة تساوي الكتلة مضروبة في السرعة.",
    "Q40": "Safety belts increase the stopping time, reducing the average force for the same change in momentum.",
    "chunk-Q30-replacement": "The kinematic velocity equation v = u + at relates initial velocity, acceleration, and time.",
}

ARABIC_IDS = {"Q33", "Q34", "Q39"}


def _slot(qid: str) -> str:
    qtype = QTYPE_BY_ID[qid]
    peers = [value for value in QIDS if QTYPE_BY_ID[value] == qtype]
    return f"m1_{qtype}_{peers.index(qid) + 1}"


def _variation_id(qid: str) -> str:
    return f"var-{qid.lower()}"


def build_fixture() -> tuple[dict, dict, dict, list]:
    questions: dict[str, object] = {}
    sections: dict[str, list] = {qtype: [] for qtype in set(QTYPE_BY_ID.values())}
    plan_sections: dict[str, list] = {qtype: [] for qtype in sections}
    sources = {
        f"chunk-{qid}": text for qid, text in SOURCE_TEXT.items() if qid.startswith("Q")
    }
    sources["chunk-Q30-replacement"] = SOURCE_TEXT["chunk-Q30-replacement"]

    def add(qid: str, payload: dict) -> None:
        qtype = QTYPE_BY_ID[qid]
        base = {
            "question_id": qid,
            "slot_id": _slot(qid),
            "selected_variation_id": _variation_id(qid),
        }
        sections[qtype].append(base | payload)

    # MCQ ------------------------------------------------------------------
    mcq = {
        "Q1": ("A 4 kg block accelerates at 2 m/s². What net force acts on it?",
               {"A": "8 N", "B": "2 N", "C": "6 N", "D": "16 N"}, "A"),
        "Q2": ("A 10 kg object accelerates at 3 m/s². What net force acts on it?",
               {"A": "13 N", "B": "30 N", "C": "3.3 N", "D": "300 N"}, "A"),
        "Q3": ("What is the force?",
               {"A": "6 N", "B": "5 N", "C": "9 N", "D": "12 N"}, "C"),
        "Q5": ("Which quantity is measured in newtons?",
               {"A": "Force", "B": "Blue", "C": "Banana", "D": "Monday"}, "A"),
        "Q6": ("Which equation expresses Newton's second law?",
               {"A": "F = ma", "B": "E = mc²", "C": "v = d/t", "D": "P = IV"}, "A"),
        "Q7": ("A car travels 120 m in 30 s. What is its average speed?",
               {"A": "4 m/s", "B": "90 m/s", "C": "150 m/s", "D": "3600 m/s"}, "A"),
        "Q9": ("Which of these is a scalar quantity?",
               {"A": "Speed", "B": "Velocity", "C": "Acceleration", "D": "Force"}, "A"),
        "Q10": ("A 2 kg ball is lifted 5 m (g = 10 m/s²). What is its gain in potential energy?",
                {"A": "100 J", "B": "10 J", "C": "40 J", "D": "25 J"}, "A"),
        "Q16": ("What is the SI unit of electric current?",
                {"A": "Ampere", "B": "Volt", "C": "Ohm", "D": "Watt"}, "A"),
        "Q17": ("A 3 kg mass experiences a 12 N net force. Find its acceleration.",
                {"A": "4 m/s²", "B": "36 m/s²", "C": "0.25 m/s²", "D": "15 m/s²"}, "A"),
    }
    for qid, (stem, options, answer) in mcq.items():
        add(qid, {"question": stem, "options": options, "correct_answer": answer})

    # True/False (production normalizer emits strings) --------------------
    tf = {
        "Q4": ("The acceleration due to gravity near Earth's surface is approximately 9.8 m/s².", "True"),
        "Q8": ("In a vacuum, a heavy object falls faster than a light object.", "True"),
        "Q12": ("Work is measured in joules.", "True"),
        "Q13": ("Momentum is a scalar quantity.", "False"),
        "Q18": ("An object at rest has zero kinetic energy.", "True"),
    }
    for qid, (statement, answer) in tf.items():
        add(qid, {"statement": statement, "answer": answer})

    # FITB ----------------------------------------------------------------
    fitb = {
        "Q19": ("The scientist who formulated the three laws of motion is ___.", ["Newton"], ["Einstein", "Galileo"]),
        "Q21": ("The rate of change of velocity is called ___.", ["acceleration"], ["speed", "momentum"]),
        "Q22": ("The energy stored in a stretched spring is ___ energy.", ["potential"], ["kinetic", "thermal"]),
        "Q24": ("The product of mass and velocity is ___.", ["momentum"], ["impulse", "force"]),
        "Q25": ("The resistance opposing motion between surfaces is ___.", ["friction"], ["tension", "gravity"]),
    }
    for qid, (stem, answers, distractors) in fitb.items():
        add(qid, {"question": stem, "answers": answers, "distractors": distractors})

    # Definition ----------------------------------------------------------
    definitions = {
        "Q23": ("inertia", "Inertia is the force that keeps objects moving."),
        "Q27": ("momentum", "The product of an object's mass and velocity."),
        "Q28": ("work", "The product of force and displacement in the direction of the force."),
        "Q29": ("power", "The rate at which work is done."),
    }
    for qid, (term, reference) in definitions.items():
        add(qid, {"term": term, "reference_answer": reference})

    # Why -----------------------------------------------------------------
    whys = {
        "Q15": ("Why does a falling object accelerate?", "Gravity exerts a net force that changes its velocity."),
        "Q31": ("Why is friction useful when walking?", "Friction provides the forward reaction force on the foot."),
        "Q32": ("Why does a moving car require a force to stop?", "A net force is needed to change its momentum."),
        "Q33": ("لماذا يزداد زمن سقوط الجسم في وجود مقاومة الهواء؟",
                "لأن مقاومة الهواء تعاكس الحركة وتقلل تسارع الجسم."),
    }
    for qid, (question, reference) in whys.items():
        add(qid, {"question": question, "reference_answer": reference})

    # Equation ------------------------------------------------------------
    equations = {
        "Q11": ("F = ma",
                ["Substitute m = 4 kg and a = 2 m/s².", "F = 4 + 2 = 6 N"], "6 N"),
        "Q30": ("Draw the free-body diagram and find the net force.",
                ["Use the diagram."], "10 N"),
        "Q36": ("v = u + at",
                ["u = 0, a = 3 m/s², t = 4 s", "v = 0 + 3 × 4 = 12 m/s"], "12 m/s"),
        "Q37": ("P = W / t",
                ["W = 240 J, t = 8 s", "P = 240 / 8 = 30 W"], "30 W"),
    }
    for qid, (equation, steps, final) in equations.items():
        add(qid, {"equation": equation, "solution_steps": steps, "final_answer": final})

    # Word problems -------------------------------------------------------
    word_problems = {
        "Q14": ("A 5 kg box accelerates at 4 m/s². Find the net force.",
                ["F = ma = 5 + 4"], "9 pounds"),
        "Q26": ("A cyclist accelerates from rest at 2 m/s² for 6 s. Find the final speed.",
                ["v = u + at = 0 + 2 × 6 = 12 m/s"], "12 m/s"),
        "Q38": ("A 3 kg cart moves at 4 m/s. Find its momentum.",
                ["p = mv = 3 × 4 = 12"], "12 kg·m/s"),
        "Q39": ("يتحرك جسم بسرعة 5 م/ث وكتلته 2 كجم. احسب كمية الحركة.",
                ["p = mv = 2 × 5 = 10"], "10 كجم·م/ث"),
    }
    for qid, (question, steps, final) in word_problems.items():
        add(qid, {"question": question, "solution_steps": steps, "final_answer": final})

    # Essay ---------------------------------------------------------------
    essays = {
        "Q20": ("Explain how Newton's third law applies when a swimmer pushes water backward.",
                "The swimmer pushes water backward and the water pushes the swimmer forward with an equal and opposite force.",
                ["third law", "equal and opposite"]),
        "Q34": ("اشرح كيف يتحول شكل الطاقة عند سقوط جسم.",
                "تتحول طاقة الوضع إلى طاقة حركية مع زيادة السرعة.",
                ["طاقة الوضع", "طاقة حركية"]),
        "Q35": ("Describe the energy transformations in a roller coaster.",
                "Potential energy becomes kinetic energy on descent and returns to potential energy on ascent; friction dissipates some energy.",
                ["potential", "kinetic", "dissipation"]),
        "Q40": ("Explain why safety belts reduce injury in a collision using impulse and momentum.",
                "Belts increase stopping time, reducing the average force for the same momentum change.",
                ["impulse", "momentum", "stopping time"]),
    }
    for qid, (question, reference, key_points) in essays.items():
        add(qid, {"question": question, "reference_answer": reference, "key_points": key_points})

    fitb_items = sections.pop("fill_in_the_blank")
    bank = list(dict.fromkeys(
        value for item in fitb_items for value in [*item["answers"], *item["distractors"]]
    ))
    questions["fill_in_the_blank"] = {"items": fitb_items, "word_bank": bank}
    questions.update(sections)

    plans_by_slot: dict[str, dict] = {}
    for qid in QIDS:
        qtype = QTYPE_BY_ID[qid]
        slot_id = _slot(qid)
        chunk_id = f"chunk-{qid}"
        quantitative = qtype in {"equation", "word_problem"}
        variation = {
            "variation_id": _variation_id(qid),
            "question_types": [qtype],
            "evidence_chunk_ids": [chunk_id],
            "target": SOURCE_TEXT[qid][:80],
            "operation": "calculate" if quantitative else "explain",
            "givens": ["given values"] if quantitative else [],
            "reasoning_mode": "calculation" if quantitative else "conceptual_reasoning",
            "student_work_product": "worked result" if quantitative else "written explanation",
        }
        plan = {
            "slot_id": slot_id,
            "question_type": qtype,
            "task_id": f"task-{qid.lower()}",
            "task_family_id": f"family-{qid.lower()}",
            "coverage_id": f"coverage-{qid.lower()}",
            "concept_id": f"concept-{qid.lower()}",
            "relationship_id": None,
            "source_chunk_id": chunk_id,
            "support_chunk_ids": [],
            "evidence_chunk_ids": [chunk_id],
            "selected_variation_id": _variation_id(qid),
            "selected_variation_group_id": f"vg-{qid.lower()}",
            "permitted_variations": [variation],
            "concept": "mechanics",
            "knowledge_focus": f"concept behind {qid}",
            "student_work_product": variation["student_work_product"],
            "idea_to_test": "apply the assigned physics concept",
            "target": variation["target"],
            "task_operation": variation["operation"],
            "required_givens": list(variation["givens"]),
            "given_conditions": [],
            "effective_givens": list(variation["givens"]),
            "complexity_boundaries": [],
            "difficulty": "medium",
            "requested_difficulty": "hard" if qid in {"Q20", "Q35"} else "medium",
            "actual_difficulty": "easy" if qid in {"Q20", "Q35"} else "medium",
            "reasoning_mode": variation["reasoning_mode"],
            "reasoning_depth": "medium",
            "reasoning_requirements": ["show the relevant relationship"],
            "requires_quantitative_validation": quantitative,
        }
        plans_by_slot[slot_id] = plan
        plan_sections[qtype].append(plan)

    exam = {"model_number": 1, "questions": questions}
    plan_list = [{"model_number": 1, "items": plan_sections}]
    sources["chunk-Q30-replacement"] = SOURCE_TEXT["chunk-Q30-replacement"]
    return exam, plans_by_slot, sources, plan_list


CATALOG = {
    "selected_chunk_ids": ["chunk-Q30-replacement", "chunk-Q26"],
    "assessment_profile": {"question_types": {}},
    "tasks": [{
        "task_id": "task-replacement-kinematics",
        "task_family_id": "family-replacement-kinematics",
        "coverage_id": "coverage-replacement-kinematics",
        "concept_id": "concept-replacement-kinematics",
        "relationship_id": None,
        "evidence_chunk_ids": ["chunk-Q30-replacement"],
        "allowed_question_types": ["equation"],
        "primary_question_types": ["equation"],
        "text_answerable": True,
        "native_difficulty": "medium",
        "supported_difficulties": ["easy", "medium", "hard", "mix"],
        "task_operation": "calculate",
        "target": "final velocity using v = u + at",
        "student_work_product": "worked numerical result",
        "reasoning_mode": "calculation",
        "required_givens": ["initial velocity", "acceleration", "time"],
        "concept": "kinematics",
        "knowledge_anchor": "kinematic equations",
        "scenario_family": "linear motion",
        "complexity_boundaries": [],
        "variable_givens": [],
        "requires_quantitative_validation": True,
    }, {
        "task_id": "task-replacement-motion",
        "task_family_id": "family-replacement-motion",
        "coverage_id": "coverage-replacement-motion",
        "concept_id": "concept-replacement-motion",
        "relationship_id": None,
        "evidence_chunk_ids": ["chunk-Q26"],
        "allowed_question_types": ["word_problem"],
        "primary_question_types": ["word_problem"],
        "text_answerable": True,
        "native_difficulty": "medium",
        "supported_difficulties": ["easy", "medium", "hard", "mix"],
        "task_operation": "calculate",
        "target": "final velocity under constant acceleration",
        "student_work_product": "worked numerical result",
        "reasoning_mode": "calculation",
        "required_givens": ["initial velocity", "acceleration", "time"],
        "concept": "kinematics",
        "knowledge_anchor": "kinematic equations",
        "scenario_family": "linear motion",
        "complexity_boundaries": [],
        "variable_givens": [],
        "requires_quantitative_validation": True,
    }],
}


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

def _payload(prompt: str) -> dict:
    return json.loads(prompt.split("<VALIDATION_DATA>\n", 1)[1].split("\n</VALIDATION_DATA>", 1)[0])


def _repair_payload(prompt: str) -> dict:
    return json.loads(prompt.split("<REPAIR_DATA>\n", 1)[1].split("\n</REPAIR_DATA>", 1)[0])


def _json_value(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class Harness:
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.exam, self.plans, self.sources, self.plan_list = build_fixture()
        self.validator_calls: list[dict] = []
        self.repair_calls: list[dict] = []
        self.generator_prompts: list[dict] = []
        self.difficulty_adjustment_prompts: list[dict] = []
        self.selector_calls: list[dict] = []
        self.cache_stats: list[dict] = []
        self.active_calls = 0
        self.peak_calls = 0
        self.lock = threading.Lock()
        self.q26_difficulty_revalidation_failures = 0
        self.difficulty_timeouts_remaining = 0
        self.disable_all_pass = False
        self.repair_behaviour = "normal"
        self.validator_exception_role = None
        self.omit_issue_checks_role = None
        self.q26_revalidation_fail = True
        self._install()

    # -- instrumentation -------------------------------------------------
    def _enter(self):
        with self.lock:
            self.active_calls += 1
            self.peak_calls = max(self.peak_calls, self.active_calls)

    def _leave(self):
        with self.lock:
            self.active_calls -= 1

    def _install(self) -> None:
        original_validate = multi_validator.validate_exam_multi

        def wrapped_validate(*args, **kwargs):
            result = original_validate(*args, **kwargs)
            stats = dict(result.get("cache_stats") or {})
            self.cache_stats.append(stats)
            return result

        self.monkeypatch.setattr(multi_validator, "validate_exam_multi", wrapped_validate)
        self.monkeypatch.setattr(multi_validator, "request_structured", self.validator_request)
        self.monkeypatch.setattr(multi_repair, "request_structured", self.repair_request)
        self.monkeypatch.setattr(exam_builder, "request_structured", self.generator_request)
        self.monkeypatch.setattr(
            difficulty_adjustment, "request_structured", self.difficulty_adjustment_request,
        )
        self.monkeypatch.setattr(llm_client, "request_structured", self.selector_request)
        self.monkeypatch.setattr(duplicate_judge, "request_structured", self.duplicate_judge_request)
        self.monkeypatch.setattr(multi_validator.time, "sleep", lambda _seconds: None)
        self.monkeypatch.setattr(multi_repair.time, "sleep", lambda _seconds: None)
        self.monkeypatch.setattr(multi_validator._RATE_LIMITER, "acquire", lambda *a, **k: None)
        self.monkeypatch.setattr(multi_repair._RATE_LIMITER, "acquire", lambda *a, **k: None)

        class FakeClient:
            provider_name = "mock"
            model = "mock-model"
            last_usage = {}

        self.monkeypatch.setattr(
            "app.llm.factory.create_review_client", lambda **_kwargs: FakeClient()
        )
        self.monkeypatch.setattr(
            "app.llm.factory.create_llm_client", lambda **_kwargs: FakeClient()
        )
        self.monkeypatch.setattr(
            "app.llm.factory.create_duplicate_replacement_client", lambda **_kwargs: FakeClient()
        )

    # -- validator boundary ----------------------------------------------
    def validator_request(self, _client, prompt, **_kwargs):
        self._enter()
        try:
            payload = _payload(prompt)
            self.validator_calls.append(payload)
            role = payload["validator"]
            if role == self.validator_exception_role:
                raise TimeoutError(f"mocked {role} transport timeout")
            if role == "difficulty" and self.difficulty_timeouts_remaining > 0:
                self.difficulty_timeouts_remaining -= 1
                raise TimeoutError("mocked difficulty validator timeout")
            rows = []
            for row in payload["questions"]:
                qid = str(row["question_id"])
                revalidation = row.get("revalidation") or {}
                original = list(revalidation.get("original_failure_findings") or [])
                current_payload = row.get("question_and_answer") or {}
                original_entry = next((
                    item
                    for section in self.exam.get("questions", {}).values()
                    for item in (
                        section.get("items", []) if isinstance(section, dict) else section
                    )
                    if str(item.get("question_id") or "") == qid
                ), {})
                candidate_changed = bool(
                    qid != "Q8"
                    and current_payload and original_entry
                    and current_payload != original_entry
                )
                issues = []
                if revalidation:
                    if (
                        qid == "Q26"
                        and role == "difficulty"
                        and self.q26_revalidation_fail
                        and self.q26_difficulty_revalidation_failures == 0
                    ):
                        self.q26_difficulty_revalidation_failures += 1
                        issues = [{
                            "failure_code": "reasoning_complexity_mismatch",
                            "fields_to_fix": ["question"],
                            "reason": "The repaired candidate still does not demand the requested complexity.",
                            "expected_fix": "Generate a replacement with the required reasoning depth.",
                        }]
                elif not self.disable_all_pass and not candidate_changed:
                    issues = [
                        {
                            "failure_code": code,
                            "fields_to_fix": list(fields),
                            "reason": f"Mocked {code} for {qid}.",
                            "expected_fix": "Correct the defect while preserving the assigned task.",
                        }
                        for issue_role, code, fields in DEFECTS.get(qid, [])
                        if issue_role == role
                    ]
                status = "FAIL" if issues else "PASS"
                checks = []
                if revalidation:
                    if role == self.omit_issue_checks_role and original:
                        checks = []
                    else:
                        checks = [{
                            "issue_id": issue["issue_id"],
                            "resolved": status == "PASS",
                            "reason": (
                                "The staged candidate resolves the original issue."
                                if status == "PASS" else "The original defect remains."
                            ),
                            "remaining_defect": "" if status == "PASS" else "unresolved",
                        } for issue in original]
                rows.append({
                    "question_id": qid,
                    "status": status,
                    "issues": issues,
                    "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
                    "observed_difficulty": "easy" if role == "difficulty" else None,
                    "calculation_evidence": ["mocked independent check"] if role == "quantitative" else [],
                    "original_issue_checks": checks,
                })
            return {"validator": role, "verdicts": rows}
        finally:
            self._leave()

    # -- repair boundary --------------------------------------------------
    def repair_request(self, _client, prompt, **_kwargs):
        self._enter()
        try:
            raw = _repair_payload(prompt)
            self.repair_calls.append(raw)
            agent = raw["repair_agent"]
            repairs = []
            for entry in raw["questions"]:
                qid = str(entry["question_id"])
                current = entry["current_question"]
                fields = list(entry["editable_fields"])
                if self.repair_behaviour == "unauthorized":
                    patch = {field: self._field_value(qid, field, current) for field in fields}
                    patch["question"] = "Unauthorized rewrite of the stem."
                elif self.repair_behaviour == "no_change":
                    patch = {field: current.get(field) for field in fields}
                elif self.repair_behaviour == "bad_metadata":
                    patch = {field: self._field_value(qid, field, current) for field in fields}
                else:
                    patch = {field: self._field_value(qid, field, current) for field in fields}
                changes = [{
                    "field": field,
                    "previous_value": _json_value(current.get(field)),
                    "new_value": _json_value(value),
                    "reason": f"{agent} repair corrected {field} for {qid}.",
                } for field, value in patch.items()]
                if self.repair_behaviour == "bad_metadata" and changes:
                    changes[0]["previous_value"] = _json_value("deliberately wrong previous value")
                repairs.append({
                    "question_id": qid,
                    "patch": patch,
                    "repair_metadata": {
                        "modified_fields": list(patch),
                        "changes": changes,
                        "targeted_failure_codes": sorted({
                            str(issue.get("failure_code") or "") for issue in entry["findings"]
                        }),
                    },
                })
            return {"repairs": repairs}
        finally:
            self._leave()

    def _field_value(self, qid: str, field: str, current: dict):
        if field == "correct_answer":
            return "B" if str(current.get(field) or "").upper() == "A" else "A"
        if field == "answer":
            return "True" if str(current.get("answer")) != "True" else "False"
        if field == "options":
            return {"A": "8 N", "B": "2 N", "C": "6 N", "D": "16 N"}
        if field == "distractors":
            return ["joule", "watt"]
        if field == "question":
            if qid == "Q26":
                return ("A motorcycle accelerates uniformly from 5 m/s at 3 m/s² for 5 s. "
                        "Determine the final velocity.")
            return str(current.get(field) or "") + " State the relationship and show the required reasoning."
        if field == "statement":
            return str(current.get(field) or "") + " This statement is physically testable."
        if field == "term":
            return "momentum" if qid == "Q23" else str(current.get(field) or "term")
        if field == "equation":
            return "F = ma"
        if field == "solution_steps":
            return ["Substitute the assigned givens.", "Compute the consistent result with correct units."]
        if field == "final_answer":
            return "20 N" if qid == "Q14" else "8 N"
        if field == "reference_answer":
            return "A source-grounded answer that satisfies the assigned task."
        if field == "key_points":
            return ["source-grounded point", "required reasoning"]
        if field == "solution_certificate":
            return {
                "method": "direct calculation",
                "important_givens": ["m", "a"],
                "calculation_steps": ["F = ma"],
                "final_answer": "8 N",
                "units": "N",
                "matching_correct_option": None,
                "verification_method": "substitution",
            }
        return f"corrected {field}"

    # -- generator boundary ----------------------------------------------
    def difficulty_adjustment_request(self, _client, prompt, **_kwargs):
        payload = json.loads(prompt)
        self.difficulty_adjustment_prompts.append(payload)
        task = payload["task"]
        slot_id = task["immutable_identifiers"]["slot_id"]
        requested = task["requested_difficulty"]
        current = list(task["reasoning_profile"].get("reasoning_requirements") or [])
        recovery_attempt = 1 + sum(
            1 for prior in self.difficulty_adjustment_prompts[:-1]
            if (prior.get("task") or {}).get("immutable_identifiers", {}).get("slot_id")
            == slot_id
        )
        revised = [
            *current,
            f"mocked {requested} cognitive-demand adaptation attempt {recovery_attempt}",
        ]
        evidence_ids = list((payload.get("assigned_evidence") or {}).keys())
        return {
            "slot_id": slot_id,
            "adjustment_succeeded": True,
            "achieved_difficulty": requested,
            "adjusted_fields": {
                "reasoning_requirements": revised,
                "prerequisites": None,
                "given_conditions": None,
                "required_givens": None,
                "idea_to_test": None,
                "strategy": "Apply a source-preserving mocked difficulty adaptation.",
            },
            "evidence_support": evidence_ids,
            "evidence_justification": "The assigned fixture evidence supports the task.",
            "reason": "The assignment now exposes the requested cognitive demand.",
        }

    def generator_request(self, _client, prompt, **kwargs):
        self._enter()
        try:
            expected = list(kwargs.get("expected_ids") or [])
            schema = str(kwargs.get("schema_name") or "")
            qtype = schema.replace("_questions", "")
            self.generator_prompts.append({
                "qtype": qtype,
                "expected_ids": expected,
                "prompt": prompt,
                "system_prompt": kwargs.get("system_prompt") or "",
            })
            questions = []
            for slot_id in expected:
                if qtype == "equation":
                    questions.append({
                        "slot_id": slot_id,
                        "equation": "s = u t + 0.5 a t^2",
                        "solution_steps": ["u = 5 m/s, a = 3 m/s², t = 5 s", "s = 25 + 37.5 = 62.5 m"],
                        "final_answer": "62.5 m",
                    })
                elif qtype == "word_problem":
                    generation_number = sum(
                        1 for prior in self.generator_prompts
                        if slot_id in (prior.get("expected_ids") or [])
                    )
                    questions.append({
                        "slot_id": slot_id,
                        "question": ("A train accelerates uniformly from 4 m/s at 2 m/s² for 7 s. "
                                     "Calculate its final speed and justify the chosen kinematic relation. "
                                     f"Recovery version {generation_number}."),
                        "solution_steps": ["v = u + at", "v = 4 + 2 × 7 = 18 m/s"],
                        "final_answer": "18 m/s",
                    })
                elif qtype == "essay":
                    questions.append({
                        "slot_id": slot_id,
                        "question": (
                            "Explain the assigned source-supported relationship and justify "
                            "each required reasoning step."
                        ),
                        "reference_answer": (
                            "A source-grounded explanation satisfying the revised cognitive "
                            "requirements."
                        ),
                        "key_points": ["source-grounded point", "required reasoning"],
                    })
                else:
                    raise AssertionError(f"unexpected regeneration type: {qtype}")
            return {"questions": questions}
        finally:
            self._leave()

    # -- replacement selector boundary -----------------------------------
    def selector_request(self, _client, prompt, **kwargs):
        self._enter()
        try:
            if "Choose the first semantically suitable candidate" not in prompt:
                raise AssertionError("unexpected request on app.llm.client boundary")
            context = json.loads(prompt.split("\n", 1)[1])
            self.selector_calls.append(context)
            slot = str(context.get("slot_to_replace") or "")
            task_id = "task-replacement-motion" if "word_problem" in slot else "task-replacement-kinematics"
            return {
                "slot_to_replace": context["slot_to_replace"],
                "conflicts_with": [],
                "conflict_reason": "mock evaluation",
                "replacement_task_id": task_id,
                "replacement_fitb_entry_id": None,
                "why_this_candidate_resolves_the_conflict": "It is an unused compatible task.",
                "checked_against_active_slots": True,
                "rejected_candidates": [],
            }
        finally:
            self._leave()

    # -- duplicate judge boundary ----------------------------------------
    def duplicate_judge_request(self, _client, prompt, **kwargs):
        self._enter()
        try:
            self.duplicate_calls = getattr(self, "duplicate_calls", []) + [prompt]
            expected = list(kwargs.get("expected_ids") or [])
            return {"verdicts": [{
                "pair_id": pair_id,
                "same_core_knowledge": "no",
                "same_student_operation_or_method": "no",
                "same_target_or_result": "no",
                "genuinely_new_reasoning": "yes",
                "one_answer_substantially_gives_away_other": "no",
            } for pair_id in expected]}
        finally:
            self._leave()

    # -- orchestration helpers -------------------------------------------
    def initial_state(self, job_id: str) -> dict:
        report = validate_exam_multi(
            self.exam,
            plan_by_slot=self.plans,
            source_by_chunk=self.sources,
            domain="STEM",
            cache_scope=job_id,
        )
        clear_job_validation_cache(job_id)
        report = validate_exam_multi(
            self.exam,
            plan_by_slot=self.plans,
            source_by_chunk=self.sources,
            domain="STEM",
            cache_scope=job_id,
        )
        return {
            "job_id": job_id,
            "generated_exams": [deepcopy(self.exam)],
            "plans": deepcopy(self.plan_list),
            "retrieved_chunks": [
                {"child_id": key, "content": value} for key, value in self.sources.items()
            ],
            "assessment_profile": {"domain": "STEM", "question_types": {}},
            "assessment_catalog": deepcopy(CATALOG),
            "difficulty": "mix",
            "repair_manifest": _manifest_from_report(report, self.exam, self.plans),
            "validation_reports": [report],
            "validated_models": [],
            "question_repair_attempts": {},
            "warnings": [],
            "invalid_questions": [],
        }


def _fake_embed(texts, max_length: int = 512):
    vectors = []
    for text in texts:
        vector = [0.0] * 64
        for token in str(text).casefold().split():
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            vector[int(digest, 16) % 64] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        vectors.append([value / norm for value in vector])
    return vectors


def _verdict_by_id(report: dict) -> dict:
    return {str(v.get("question_id")): v for v in report.get("verdicts") or []}


def _unresolved(state: dict) -> list[dict]:
    return [
        item for item in state.get("repair_manifest") or []
        if item.get("status") not in {"resolved", "exhausted", "terminal_failure", "validation_blocked"}
    ]


def _run_until_settled(state: dict, harness: Harness, max_rounds: int = 8) -> dict:
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        if not state.get("repair_candidates") and not _unresolved(state):
            break
        repaired = repair_invalid_questions(state)
        state = {**state, **repaired}
        completed = validate_generated_questions(state)
        state = {**state, **completed}
    return state


def _final_questions(state: dict) -> list[dict]:
    items = []
    for exam in state.get("generated_exams") or []:
        for section in (exam.get("questions") or {}).values():
            values = section.get("items") if isinstance(section, dict) else section
            items.extend(values or [])
    return items


# ---------------------------------------------------------------------------
# Main 40-question evaluation
# ---------------------------------------------------------------------------

def test_40q_mocked_end_to_end_pipeline(monkeypatch):
    h = Harness(monkeypatch)
    job_id = "readonly-40q-main"
    clear_job_validation_cache(job_id)
    state = h.initial_state(job_id)
    initial_report = state["validation_reports"][0]
    initial = _verdict_by_id(initial_report)

    statuses = {qid: initial[qid]["validation_status"] for qid in QIDS}
    initial_pass = {qid for qid, value in statuses.items() if value == "PASS"}
    initial_tf_input_invalid = {
        qid for qid, value in statuses.items() if value == "INPUT_INVALID"
    }

    # Legacy string values are normalized at the validation boundary and no
    # valid True/False item is rejected for its stored representation.
    assert initial_tf_input_invalid == set(), statuses

    assert len(state["repair_manifest"]) == 12, sorted(
        item["question_id"] for item in state["repair_manifest"]
    )

    state = _run_until_settled(state, h)

    # --- Q3: three dependent repairs on one question --------------------
    history = state.get("validation_repair_history") or []
    q3_steps = [
        event for event in history
        if event.get("question_id") == "Q3" and event.get("repair_action") == "repair_question"
    ]
    q3_repair_logs = [
        log for log in state.get("repair_call_logs") or []
        if log.get("question_id") == "Q3"
    ]
    q3 = _verdict_by_id(
        next(report for report in state["validation_reports"] if report["model_number"] == 1)
    ).get("Q3")
    assert q3 and q3["action"] == "PASS", (q3, q3_steps, q3_repair_logs)
    q3_statuses = [step.get("status") for step in q3_steps]
    assert q3_statuses and all(status == "accepted" for status in q3_statuses), q3_steps

    q3_repair_objects = [
        call for call in h.repair_calls
        if any(row.get("question_id") == "Q3" for row in call.get("questions") or [])
    ]
    q3_agents = [call["repair_agent"] for call in q3_repair_objects]
    assert q3_agents[:3] == ["question", "options", "answer"], q3_agents

    # --- Q26: regeneration through the real Generator node --------------
    q26_events = [
        event for event in history if event.get("question_id") == "Q26"
    ]
    assert any(event.get("status") == "rejected" for event in q26_events), q26_events
    q26_generation = [
        prompt for prompt in h.generator_prompts if _slot("Q26") in prompt["expected_ids"]
    ]
    assert q26_generation, "Q26 regeneration did not reach the generator"
    q26_prompt = q26_generation[-1]["prompt"]
    assert "assigned sources" in q26_prompt.lower() or "source_chunk_id" in q26_prompt
    q26_adjustments = [
        call for call in h.difficulty_adjustment_prompts
        if (call.get("task") or {}).get("immutable_identifiers", {}).get("slot_id")
        == _slot("Q26")
    ]
    assert q26_adjustments, "Q26 did not reach targeted Difficulty Adjustment"
    assert "reasoning_complexity_mismatch" in json.dumps(q26_adjustments[-1])

    # --- Q30: catalog replacement through selector + generator ----------
    q30_generation = [
        prompt for prompt in h.generator_prompts if _slot("Q30") in prompt["expected_ids"]
    ]
    assert q30_generation, "Q30 catalog replacement did not reach the generator"
    q30_prompt = q30_generation[-1]["prompt"]
    assert "v = u + at" in q30_prompt or "chunk-Q30-replacement" in q30_prompt
    assert h.selector_calls, "replacement selector was never consulted"
    print("READONLY_Q30_DEBUG", json.dumps({
        "history": [
            event for event in history if event.get("question_id") == "Q30"
        ],
        "selector_slots": [call.get("slot_to_replace") for call in h.selector_calls],
        "generator_calls": [
            {"qtype": call["qtype"], "expected_ids": call["expected_ids"]}
            for call in h.generator_prompts
        ],
    }, ensure_ascii=False, default=str))
    q30_plan = None
    for plan in state["plans"]:
        for values in (plan.get("items") or {}).values():
            for item in values:
                if item.get("slot_id") == _slot("Q30"):
                    q30_plan = item
    assert q30_plan and q30_plan.get("task_id") == "task-replacement-kinematics", q30_plan

    # --- Final assembly --------------------------------------------------
    assembled = assemble_exams_node(state)
    final_items = _final_questions(assembled)
    final_ids = {item["question_id"] for item in final_items}
    final_report = _verdict_by_id(
        next(report for report in assembled.get("validation_reports") or state["validation_reports"]
             if report["model_number"] == 1)
    )

    assert {"Q4", "Q8", "Q12", "Q13", "Q18"}.issubset(final_ids), final_ids
    assert len(final_items) == 40, sorted(final_ids)
    assert all(final_report[qid]["action"] == "PASS" for qid in final_ids)

    # No unresolved question may be rendered.
    unresolved_rendered = [
        qid for qid in final_ids
        if (final_report.get(qid) or {}).get("action") != "PASS"
    ]
    assert not unresolved_rendered, unresolved_rendered

    # Batch/concurrency invariants.
    all_calls = h.validator_calls
    assert all(len(call["questions"]) <= 4 for call in all_calls)
    assert max(len(call["questions"]) for call in all_calls) <= 4
    assert h.peak_calls <= 50

    # Cache is job-local.
    assert all(stats.get("scope") == job_id for stats in h.cache_stats if stats.get("scope"))
    assert any(stats.get("hits", 0) > 0 for stats in h.cache_stats), h.cache_stats

    telemetry = {
        "questions": 40,
        "initial_pass": len(initial_pass),
        "initial_fail": 40 - len(initial_pass),
        "true_false_input_invalid": sorted(initial_tf_input_invalid),
        "repair_manifest_targets": len(state.get("repair_manifest") or []),
        "validator_calls": len(h.validator_calls),
        "validator_calls_by_agent": {
            role: sum(call.get("validator") == role for call in h.validator_calls)
            for role in sorted({str(call.get("validator")) for call in h.validator_calls})
        },
        "repair_batches": len(h.repair_calls),
        "repair_calls_by_agent": {
            agent: sum(call.get("repair_agent") == agent for call in h.repair_calls)
            for agent in sorted({str(call.get("repair_agent")) for call in h.repair_calls})
        },
        "generator_calls": len(h.generator_prompts),
        "selector_calls": len(h.selector_calls),
        "max_validator_batch": max(len(call["questions"]) for call in all_calls),
        "peak_concurrency": h.peak_calls,
        "cache_stats_initial": h.cache_stats[0] if h.cache_stats else {},
        "cache_stats_last": h.cache_stats[-1] if h.cache_stats else {},
        "final_questions": len(final_items),
        "q26_events": [event.get("status") for event in q26_events],
        "q3_agents": q3_agents,
    }
    print("READONLY_40Q_TELEMETRY " + json.dumps(telemetry, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# True/False contract inconsistency (focused reproduction)
# ---------------------------------------------------------------------------

def test_true_false_answer_contract_is_canonical_and_legacy_compatible(monkeypatch):
    from app.online.multi_validator import protect_question

    task = {"source_chunk_id": "c1", "question_type": "true_false"}
    source = {"c1": "The sky appears blue."}
    string_question = {
        "question_id": "tf-string", "slot_id": "s1",
        "statement": "The sky appears blue.", "answer": "True",
    }
    bool_question = {
        "question_id": "tf-bool", "slot_id": "s2",
        "statement": "The sky appears blue.", "answer": False,
    }

    string_result = protect_question("true_false", string_question, task, source)
    bool_result = protect_question("true_false", bool_question, task, source)

    assert string_result.errors == []
    assert string_result.question["answer"] is True
    assert bool_result.errors == []
    assert bool_result.question["answer"] is False

    from app.online.validator import _local_verdict

    false_verdict = _local_verdict("true_false", bool_question, None)
    assert false_verdict is None


# ---------------------------------------------------------------------------
# Q35 operational revalidation: timeout, retry isolation, exhaustion
# ---------------------------------------------------------------------------

def _single_essay_state(harness: Harness, job_id: str, qid: str = "Q35") -> dict:
    exam, plans, sources, plan_list = build_fixture()
    slot = _slot(qid)
    only_plan = plans[slot]
    only_item = next(
        item for section in exam["questions"].values()
        for item in (section.get("items") if isinstance(section, dict) else section)
        if item.get("question_id") == qid
    )
    exam = {"model_number": 1, "questions": {"essay": [deepcopy(only_item)]}}
    plan_by_slot = {slot: only_plan}
    plan_list = [{"model_number": 1, "items": {"essay": [deepcopy(only_plan)]}}]
    report = validate_exam_multi(
        exam, plan_by_slot=plan_by_slot, source_by_chunk=sources,
        domain="STEM", cache_scope=job_id,
    )
    state = {
        "job_id": job_id, "generated_exams": [deepcopy(exam)], "plans": plan_list,
        "retrieved_chunks": [{"child_id": k, "content": v} for k, v in sources.items()],
        "assessment_profile": {"domain": "STEM", "question_types": {}},
        "assessment_catalog": deepcopy(CATALOG), "difficulty": "mix",
        "repair_manifest": _manifest_from_report(report, exam, plan_by_slot),
        "validation_reports": [report], "validated_models": [],
        "question_repair_attempts": {}, "warnings": [], "invalid_questions": [],
    }
    return state


def test_q35_difficulty_timeout_retry_isolation_and_eventual_commit(monkeypatch):
    h = Harness(monkeypatch)
    job_id = "readonly-q35-retry"
    state = _single_essay_state(h, job_id)
    # Force the initial validator to fail difficulty so repair stages a candidate.
    state["repair_manifest"][0]["verdict"]["required_fixes"] = [{
        "validator": "difficulty", "failure_code": "difficulty_mismatch",
        "fields_to_fix": ["question"], "reason": "too easy", "expected_fix": "deepen",
        "issue_id": "Q35-I-mock",
    }]
    state["repair_manifest"][0]["verdict"]["failure_code"] = "difficulty_mismatch"
    state["repair_manifest"][0]["verdict"]["fields_to_fix"] = ["question"]
    state["repair_manifest"][0]["verdict"]["action"] = "FIX_QUESTION"
    state["repair_manifest"][0]["validation_role"] = "difficulty"
    repaired = repair_invalid_questions(state)
    state = {**state, **repaired}
    assert state["repair_candidates"], state.get("warnings")
    repair_calls_before = len(h.repair_calls)
    attempts_before = dict(state.get("question_repair_attempts") or {})

    # First revalidation: only difficulty times out (three provider attempts).
    h.difficulty_timeouts_remaining = 3
    completed = validate_generated_questions(state)
    state = {**state, **completed}
    assert state["repair_candidates"], "candidate should await retry"
    progress = state.get("revalidation_progress") or {}
    assert progress, "operational progress was not recorded"
    record = next(iter(progress.values()))
    assert record["candidate_status"] == "PENDING_RETRY", record
    assert record["pending_roles"] == ["difficulty"], record
    assert (state.get("question_repair_attempts") or {}) == attempts_before
    assert len(h.repair_calls) == repair_calls_before, "repair agent must not be re-called"
    grounding_verdicts = [
        call for call in h.validator_calls
        if call["validator"] == "grounding"
        and any(row["question_id"] == "Q35" for row in call["questions"])
    ]
    assert grounding_verdicts

    # Second revalidation: difficulty recovers; cached PASS verdicts are reused.
    completed = validate_generated_questions(state)
    state = {**state, **completed}
    active_report = next(
        report for report in state["validation_reports"] if report["model_number"] == 1
    )
    verdict = _verdict_by_id(active_report)["Q35"]
    assert verdict["action"] == "PASS", (verdict, state.get("warnings"))
    assert not state.get("repair_candidates")
    assert len(h.repair_calls) == repair_calls_before
    assert (state.get("question_repair_attempts") or {}) == attempts_before
    assert state.get("revalidation_progress") == {}


def test_q35_operational_retry_exhaustion_blocks_candidate(monkeypatch):
    h = Harness(monkeypatch)
    job_id = "readonly-q35-exhaust"
    state = _single_essay_state(h, job_id)
    state["repair_manifest"][0]["verdict"]["required_fixes"] = [{
        "validator": "difficulty", "failure_code": "difficulty_mismatch",
        "fields_to_fix": ["question"], "reason": "too easy", "expected_fix": "deepen",
        "issue_id": "Q35-I-mock",
    }]
    state["repair_manifest"][0]["verdict"]["failure_code"] = "difficulty_mismatch"
    state["repair_manifest"][0]["verdict"]["fields_to_fix"] = ["question"]
    state["repair_manifest"][0]["verdict"]["action"] = "FIX_QUESTION"
    state["repair_manifest"][0]["validation_role"] = "difficulty"
    state = {**state, **repair_invalid_questions(state)}
    assert state["repair_candidates"], {
        "warnings": state.get("warnings"),
        "manifest": state.get("repair_manifest"),
        "history": state.get("validation_repair_history"),
    }

    h.difficulty_timeouts_remaining = 100
    for _ in range(10):
        if not state.get("repair_candidates"):
            break
        state = {**state, **validate_generated_questions(state)}

    blocked = state.get("blocked_revalidation_candidates") or []
    assert blocked, state.get("revalidation_progress")
    assert blocked[0]["candidate_status"] == "VALIDATION_BLOCKED"
    assert not state.get("repair_candidates")
    target = next(item for item in state["repair_manifest"] if item["question_id"] == "Q35")
    assert target["status"] == "validation_blocked"


# ---------------------------------------------------------------------------
# Failure safety
# ---------------------------------------------------------------------------

def _single_mcq_state(harness: Harness, job_id: str) -> dict:
    exam, plans, sources, plan_list = build_fixture()
    slot = _slot("Q2")
    item = next(
        item for section in exam["questions"].values()
        for item in (section.get("items") if isinstance(section, dict) else section)
        if item.get("question_id") == "Q2"
    )
    exam = {"model_number": 1, "questions": {"mcq": [deepcopy(item)]}}
    plan_by_slot = {slot: plans[slot]}
    plan_list = [{"model_number": 1, "items": {"mcq": [deepcopy(plans[slot])]}}]
    report = validate_exam_multi(
        exam, plan_by_slot=plan_by_slot, source_by_chunk=sources,
        domain="STEM", cache_scope=job_id,
    )
    return {
        "job_id": job_id, "generated_exams": [deepcopy(exam)], "plans": plan_list,
        "retrieved_chunks": [{"child_id": k, "content": v} for k, v in sources.items()],
        "assessment_profile": {"domain": "STEM", "question_types": {}},
        "assessment_catalog": deepcopy(CATALOG), "difficulty": "mix",
        "repair_manifest": _manifest_from_report(report, exam, plan_by_slot),
        "validation_reports": [report], "validated_models": [],
        "question_repair_attempts": {}, "warnings": [], "invalid_questions": [],
    }


def test_repair_rejects_unauthorized_fields(monkeypatch):
    h = Harness(monkeypatch)
    state = _single_mcq_state(h, "readonly-unauthorized")
    h.repair_behaviour = "unauthorized"
    repaired = repair_invalid_questions(state)
    assert not repaired["repair_candidates"], repaired.get("warnings")
    assert any(
        "unauthorized" in warning or "rejected" in warning
        or "Extra inputs are not permitted" in warning
        for warning in repaired["warnings"]
    ), repaired["warnings"]
    target = next(item for item in repaired["repair_manifest"] if item["question_id"] == "Q2")
    assert target["status"] != "resolved"


def test_repair_normalizes_inaccurate_diagnostic_metadata(monkeypatch):
    h = Harness(monkeypatch)
    state = _single_mcq_state(h, "readonly-bad-metadata")
    h.repair_behaviour = "bad_metadata"
    repaired = repair_invalid_questions(state)
    assert repaired["repair_candidates"], repaired.get("warnings")
    metadata = repaired["repair_candidates"][0]["repair_metadata"]
    assert metadata["modified_fields"]
    assert any(
        change.get("previous_value") != "deliberately wrong previous value"
        for change in metadata["final_structural_diff"]
    )


def test_missing_original_issue_resolution_blocks_commit(monkeypatch):
    h = Harness(monkeypatch)
    state = _single_mcq_state(h, "readonly-missing-check")
    state = {**state, **repair_invalid_questions(state)}
    assert state["repair_candidates"], state.get("warnings")
    original = deepcopy(state["generated_exams"][0]["questions"]["mcq"][0])
    h.omit_issue_checks_role = "answer_correctness"
    completed = validate_generated_questions(state)
    assert completed["repair_candidates"], "candidate must not commit without issue checks"
    assert completed["generated_exams"][0]["questions"]["mcq"][0] == original
    progress = completed.get("revalidation_progress") or {}
    assert progress, completed.get("warnings")


def test_validator_exception_and_malformed_output_never_commit(monkeypatch):
    h = Harness(monkeypatch)
    state = _single_mcq_state(h, "readonly-validator-exception")
    state = {**state, **repair_invalid_questions(state)}
    assert state["repair_candidates"]
    original = deepcopy(state["generated_exams"][0]["questions"]["mcq"][0])
    h.validator_exception_role = "grounding"
    completed = validate_generated_questions(state)
    assert completed["repair_candidates"], "operational failure should retry, not commit"
    assert completed["generated_exams"][0]["questions"]["mcq"][0] == original


def test_revalidation_new_defect_keeps_original_and_regenerates(monkeypatch):
    h = Harness(monkeypatch)
    # Q26 is a word problem and uses the real generator on escalation.
    exam, plans, sources, plan_list = build_fixture()
    slot = _slot("Q26")
    item = next(
        item for section in exam["questions"].values()
        for item in (section.get("items") if isinstance(section, dict) else section)
        if item.get("question_id") == "Q26"
    )
    only_exam = {"model_number": 1, "questions": {"word_problem": [deepcopy(item)]}}
    report = validate_exam_multi(
        only_exam, plan_by_slot={slot: plans[slot]}, source_by_chunk=sources,
        domain="STEM", cache_scope="readonly-regen",
    )
    state = {
        "job_id": "readonly-regen", "generated_exams": [deepcopy(only_exam)],
        "plans": [{"model_number": 1, "items": {"word_problem": [deepcopy(plans[slot])]}}],
        "retrieved_chunks": [{"child_id": k, "content": v} for k, v in sources.items()],
        "assessment_profile": {"domain": "STEM", "question_types": {}},
        "assessment_catalog": deepcopy(CATALOG), "difficulty": "mix",
        "repair_manifest": _manifest_from_report(report, only_exam, {slot: plans[slot]}),
        "validation_reports": [report], "validated_models": [],
        "question_repair_attempts": {}, "warnings": [], "invalid_questions": [],
    }
    state = {**state, **repair_invalid_questions(state)}
    assert state["repair_candidates"]
    original = deepcopy(state["generated_exams"][0]["questions"]["word_problem"][0])
    state = {**state, **validate_generated_questions(state)}
    # First candidate is rejected as a new lateral defect; original remains.
    assert state["generated_exams"][0]["questions"]["word_problem"][0] == original
    assert any(
        event.get("status") == "rejected"
        for event in state.get("validation_repair_history") or []
    ), state.get("validation_repair_history")
    state = {**state, **repair_invalid_questions(state)}
    assert any(_slot("Q26") in call["expected_ids"] for call in h.generator_prompts)
    assert state["repair_candidates"], {
        "warnings": state.get("warnings"),
        "manifest": state.get("repair_manifest"),
        "history": state.get("validation_repair_history"),
    }
    state = {**state, **validate_generated_questions(state)}
    committed = state["generated_exams"][0]["questions"]["word_problem"][0]
    assert committed["question"] != original["question"]
    report_by_id = _verdict_by_id(
        next(report for report in state["validation_reports"] if report["model_number"] == 1)
    )
    assert report_by_id["Q26"]["action"] == "PASS"


# ---------------------------------------------------------------------------
# Duplicate validation after repair
# ---------------------------------------------------------------------------

def test_duplicate_detection_runs_after_repair(monkeypatch):
    h = Harness(monkeypatch)
    exam = {
        "model_number": 1,
        "questions": {"mcq": [
            {
                "question_id": "D1", "slot_id": "m1_mcq_1",
                "question": "What is the net force on a 2 kg mass accelerating at 3 m/s²?",
                "options": {"A": "6 N", "B": "5 N", "C": "7 N", "D": "9 N"},
                "correct_answer": "A",
            },
            {
                "question_id": "D2", "slot_id": "m1_mcq_2",
                "question": "What is the net force on a 2 kg mass accelerating at 3 m/s²?",
                "options": {"A": "6 N", "B": "5 N", "C": "7 N", "D": "9 N"},
                "correct_answer": "A",
            },
        ]},
    }
    report = {
        "model_number": 1,
        "verdicts": [
            {"question_id": qid, "action": "PASS", "validation_status": "PASS"}
            for qid in ("D1", "D2")
        ],
        "all_pass": True,
    }
    # Force the judge to report semantic duplication for every candidate pair.
    def duplicate_yes(_client, prompt, **kwargs):
        return {"verdicts": [{
            "pair_id": pair_id,
            "same_core_knowledge": "yes",
            "same_student_operation_or_method": "yes",
            "same_target_or_result": "yes",
            "genuinely_new_reasoning": "no",
            "one_answer_substantially_gives_away_other": "yes",
        } for pair_id in kwargs.get("expected_ids") or []]}

    monkeypatch.setattr(duplicate_judge, "request_structured", duplicate_yes)
    state = {
        "job_id": "readonly-duplicates", "generated_exams": [exam],
        "plans": [{"model_number": 1, "items": {"mcq": [
            {"slot_id": "m1_mcq_1", "question_type": "mcq", "task_id": "t1"},
            {"slot_id": "m1_mcq_2", "question_type": "mcq", "task_id": "t2"},
        ]}}],
        "validation_reports": [report], "assessment_profile": {"domain": "STEM"},
        "retrieved_chunks": [], "warnings": [], "invalid_questions": [],
    }
    manifest, validated, _warnings, result = validator._integrate_duplicate_validation(
        state, [exam], state["plans"], {1: report}, [], [1], [],
    )
    assert validated == []
    assert manifest and manifest[0]["validation_role"] == "duplicate"
    groups = result.get("duplicate_reports", [])[-1].get("duplicate_groups") or []
    assert {qid for group in groups for qid in group.get("question_ids") or []} == {"D1", "D2"}
