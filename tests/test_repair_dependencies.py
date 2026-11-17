from __future__ import annotations

from copy import deepcopy

from app.online import multi_repair
from app.online.multi_repair import (
    RepairStep, _apply_step, _stale_dependent_error, build_repair_route,
    repair_question_specialized,
)
from app.online.multi_validator import (
    ProtectedQuestion, _compatibility_verdict, dependent_fields_for,
)


def _cert(final, *, option="A", units=""):
    return {
        "method": "deterministic check",
        "important_givens": [],
        "calculation_steps": [f"source result: {final}"],
        "final_answer": final,
        "units": units,
        "matching_correct_option": option,
        "verification_method": "recompute",
    }


def _issue(role, code, fields, *, expected="fix", dependent=None):
    issue = {
        "validator": role,
        "validation_status": "FAIL",
        "failure_code": code,
        "fields_to_fix": fields,
        "dependent_fields_to_update": [],
        "reason": f"{code} detected",
        "expected_fix": expected,
    }
    if dependent is not None:
        issue["dependent_fields_to_update"] = dependent
    return issue


def _verdict(*issues):
    return {
        "question_id": "q1",
        "validation_status": "FAIL",
        "action": "FIX_QUESTION_AND_ANSWER",
        "failure_code": issues[0]["failure_code"],
        "fields_to_fix": sorted({f for i in issues for f in i["fields_to_fix"]}),
        "required_fixes": list(issues),
        "repair_instruction_conflicts": [],
    }


def _protected(qtype, question):
    return ProtectedQuestion(
        question_id=question["question_id"],
        question_type=qtype,
        slot_id=question["slot_id"],
        question=question,
        task={"task_id": "synthetic-task"},
        evidence={},
        word_bank=[],
        sanitization_events=[],
        errors=[],
        requires_quantitative=True,
    )


def _run(monkeypatch, *, question, qtype, verdict, patches):
    calls = []

    def fake_call_step(**kwargs):
        agent = kwargs["agent"]
        calls.append(agent)
        return ({
            "question_id": verdict["question_id"],
            "patch": deepcopy(patches[agent]),
            "repair_metadata": {"targeted_failure_codes": [verdict["failure_code"]]},
        }, None, {"input_fingerprint": agent, "output_fingerprint": agent})

    monkeypatch.setattr(multi_repair, "_call_step", fake_call_step)
    exam = {"questions": {qtype: [question]}}
    result = repair_question_specialized(
        exam=exam, verdict=verdict, qtype=qtype,
        plan_item={"slot_id": question["slot_id"]},
        source_chunks={}, domain="STEM", client_factory=lambda _agent: object(),
    )
    return result, question, calls


def test_validator_derives_missing_dependency_fields_and_reason():
    question = {
        "question_id": "q1", "slot_id": "s1", "equation": "2x + 3 = 11",
        "solution_steps": ["2x = 14", "x = 7"], "final_answer": "7",
        "solution_certificate": _cert("7"),
    }
    issue = _issue("quantitative", "incorrect_calculation", ["solution_steps"])
    issue.pop("dependent_fields_to_update")
    verdict = _compatibility_verdict(
        _protected("equation", question),
        {"quantitative": {"status": "FAIL", "issues": [issue]}},
        {"action": "PASS"},
    )
    fixed = verdict["required_fixes"][0]
    assert fixed["fields_to_fix"] == ["solution_steps"]
    assert fixed["dependent_fields_to_update"] == [
        "final_answer", "solution_certificate",
    ]
    assert fixed["dependency_reason"]
    assert verdict["dependent_fields_to_update"] == [
        "final_answer", "solution_certificate",
    ]


def test_validator_corrects_an_incorrect_dependency_list():
    question = {
        "question_id": "q1", "slot_id": "s1", "equation": "2x + 3 = 11",
        "solution_steps": ["2x = 14", "x = 7"], "final_answer": "7",
        "solution_certificate": _cert("7"),
    }
    issue = _issue(
        "quantitative", "incorrect_calculation", ["solution_steps"],
        dependent=["question", "correct_answer", "solution_certificate"],
    )
    verdict = _compatibility_verdict(
        _protected("equation", question),
        {"quantitative": {"status": "FAIL", "issues": [issue]}},
        {"action": "PASS"},
    )
    fixed = verdict["required_fixes"][0]
    assert fixed["dependent_fields_to_update"] == [
        "final_answer", "solution_certificate",
    ]


def test_why_semantically_equal_prose_certificate_is_not_stale():
    before = {
        "question": "Why does a helium balloon rise in air?",
        "reference_answer": "Helium is denser than air, so the balloon rises.",
        "solution_certificate": {
            "method": "density", "important_givens": [], "calculation_steps": ["density"],
            "final_answer": "helium is denser than air", "units": "",
            "matching_correct_option": None, "verification_method": "compare density",
        },
    }
    after = {
        **before,
        "reference_answer": (
            "Helium is less dense than air, so the surrounding air is denser than the "
            "balloon and the buoyant force pushes it upward."
        ),
        "solution_certificate": {
            "method": "density", "important_givens": ["helium is less dense than air"],
            "calculation_steps": ["compare the densities"],
            "final_answer": (
                "Helium is less dense than air, so the surrounding air's buoyant force "
                "pushes the balloon upward."
            ),
            "units": "", "matching_correct_option": None,
            "verification_method": "compare the stated densities",
        },
    }
    assert _stale_dependent_error("why", before, after) is None
    contradicted = {**after, "solution_certificate": {
        **after["solution_certificate"], "final_answer": "helium is denser than air",
    }}
    assert _stale_dependent_error("why", before, contradicted) is not None


def test_mcq_certificate_dependencies_include_options_and_key():
    issue = _issue(
        "quantitative", "significant_figures_error", ["solution_certificate"],
    )
    issue.pop("dependent_fields_to_update")
    route = build_repair_route("mcq", _verdict(issue))
    assert [step.agent for step in route["steps"]] == ["options", "solution", "answer"]
    fields = {step.agent: set(step.fields) for step in route["steps"]}
    assert fields["options"] == {"options"}
    assert fields["answer"] == {"correct_answer"}
    assert fields["solution"] == {"solution_certificate"}


def test_task_alignment_question_only_patch_ignores_certificate_normalization():
    question = {
        "question_id": "q1", "slot_id": "s1", "question": "Define efficiency.",
        "reference_answer": "Efficiency is useful output divided by input.",
        "key_points": ["useful output", "input"],
        "solution_certificate": {
            "method": "direct calculation", "important_givens": [],
            "calculation_steps": ["useful output / input"],
            "final_answer": "efficiency ratio", "units": "",
            "matching_correct_option": "", "verification_method": "recompute",
        },
    }
    issue = _issue("task_alignment", "wrong_variation", ["question"])
    step = RepairStep(
        agent="task_alignment", issues=(issue,), fields=("question",),
    )
    repaired, _metadata, error = _apply_step("essay", question, step, {
        "question_id": "q1",
        "patch": {"question": "Compare the effort of two machines at equal resistance."},
        "repair_metadata": {
            "modified_fields": ["question"],
            "changes": [{
                "field": "question", "previous_value": "Define efficiency.",
                "new_value": "Compare the effort of two machines at equal resistance.",
                "reason": "Restore the required comparison.",
            }],
            "targeted_failure_codes": ["wrong_variation"],
        },
    })
    assert error is None, error
    assert repaired["question"].startswith("Compare")
    assert repaired["solution_certificate"]["matching_correct_option"] == ""


def test_missing_dependency_list_is_derived_at_router_boundary():
    issue = _issue("quantitative", "incorrect_calculation", ["solution_steps"])
    issue.pop("dependent_fields_to_update")
    route = build_repair_route("equation", _verdict(issue))
    solution = next(step for step in route["steps"] if step.agent == "solution")
    assert solution.fields == ("final_answer", "solution_certificate", "solution_steps")


def test_router_routes_dependencies_to_owners_not_task_alignment():
    issue = _issue(
        "answer_correctness", "answer_inconsistent", ["solution_certificate"],
        dependent=["options", "correct_answer"],
    )
    route = build_repair_route("mcq", _verdict(issue))
    assert [step.agent for step in route["steps"]] == ["options", "solution", "answer"]
    task_alignment = next(
        (step for step in route["steps"] if step.agent == "task_alignment"), None,
    )
    assert task_alignment is None
    assert "solution_certificate" not in set(
        field for step in route["steps"] if step.agent == "task_alignment" for field in step.fields
    )


def test_noop_options_step_continues_to_solution_and_answer(monkeypatch):
    question = {
        "question_id": "q1", "slot_id": "s1", "question": "How many pieces?",
        "options": {"A": "6", "B": "6.0", "C": "6.00", "D": "60"},
        "correct_answer": "C", "solution_certificate": _cert("6.00 pieces", option="C"),
    }
    issue = _issue(
        "quantitative", "significant_figures_error", ["solution_certificate"],
        dependent=["options", "correct_answer"],
    )
    verdict = _verdict(issue)
    patches = {
        "options": {"options": deepcopy(question["options"])},
        "solution": {"solution_certificate": _cert("6.0 pieces", option="B")},
        "answer": {"correct_answer": "B"},
    }
    result, repaired, calls = _run(
        monkeypatch, question=question, qtype="mcq", verdict=verdict, patches=patches,
    )
    assert calls == ["options", "solution", "answer"]
    assert result["repaired_ids"] == ["q1"]
    assert result["route_status"] == "COMPLETE"
    assert repaired["correct_answer"] == "B"
    assert repaired["solution_certificate"]["final_answer"] == "6.0 pieces"


def test_live_mcq_calculation_continues_after_already_correct_option(monkeypatch):
    question = {
        "question_id": "q1", "slot_id": "s1",
        "question": "A 12 N force moves an object 4 m. How much work is done?",
        "options": {"A": "48 J", "B": "46 J", "C": "52 J", "D": "40 J"},
        "correct_answer": "A", "solution_certificate": _cert("46 J", units="J"),
    }
    issue = _issue(
        "quantitative", "incorrect_calculation", ["solution_certificate"],
        dependent=["options", "correct_answer"],
    )
    patches = {
        "options": {"options": deepcopy(question["options"])},
        "solution": {"solution_certificate": _cert("48 J", units="J")},
        "answer": {"correct_answer": "A"},
    }
    result, repaired, calls = _run(
        monkeypatch, question=question, qtype="mcq",
        verdict=_verdict(issue), patches=patches,
    )
    assert calls == ["options", "solution", "answer"]
    assert result["repaired_ids"] == ["q1"]
    assert result["route_status"] == "COMPLETE"
    assert repaired["correct_answer"] == "A"
    assert repaired["solution_certificate"]["final_answer"] == "48 J"


def test_already_correct_dependent_field_can_return_noop(monkeypatch):
    question = {
        "question_id": "q1", "slot_id": "s1", "question": "How many pieces?",
        "options": {"A": "6", "B": "6.0", "C": "6.00", "D": "60"},
        "correct_answer": "C", "solution_certificate": _cert("6.0 pieces", option="C"),
    }
    issue = _issue(
        "quantitative", "significant_figures_error", ["solution_certificate"],
        dependent=["options", "correct_answer"],
    )
    verdict = _verdict(issue)
    new_options = {**question["options"], "C": "6.0"}
    patches = {
        "options": {"options": new_options},
        "solution": {"solution_certificate": deepcopy(question["solution_certificate"])},
        "answer": {"correct_answer": "C"},
    }
    result, repaired, calls = _run(
        monkeypatch, question=question, qtype="mcq", verdict=verdict, patches=patches,
    )
    assert calls == ["options", "solution", "answer"]
    assert result["repaired_ids"] == ["q1"]
    assert result["route_status"] == "COMPLETE"
    assert repaired["options"]["C"] == "6.0"
    assert repaired["solution_certificate"] == question["solution_certificate"]


def test_live_equation_failure_synchronizes_certificate(monkeypatch):
    question = {
        "question_id": "q1", "slot_id": "s1", "equation": "2x + 3 = 11",
        "solution_steps": ["2x = 14", "x = 7"], "final_answer": "7",
        "solution_certificate": _cert("7"),
    }
    issue = _issue(
        "quantitative", "incorrect_calculation", ["solution_steps"],
        dependent=["final_answer", "solution_certificate"],
    )
    patches = {"solution": {
        "solution_steps": ["2x = 8", "x = 4"],
        "final_answer": "4",
        "solution_certificate": _cert("4"),
    }}
    result, repaired, calls = _run(
        monkeypatch, question=question, qtype="equation",
        verdict=_verdict(issue), patches=patches,
    )
    assert calls == ["solution"]
    assert result["repaired_ids"] == ["q1"]
    assert repaired["final_answer"] == "4"
    assert repaired["solution_certificate"]["final_answer"] == "4"
    assert "x = 7" not in repaired["solution_steps"]


def test_live_word_problem_failure_synchronizes_steps_and_certificate(monkeypatch):
    question = {
        "question_id": "q1", "slot_id": "s1",
        "question": "A car travels 60 km in 1.5 h. What is its average speed?",
        "solution_steps": ["60 / 1.5 = 30"], "final_answer": "30 km/h",
        "solution_certificate": _cert("30 km/h", units="km/h"),
    }
    issue = _issue(
        "quantitative", "incorrect_result", ["final_answer"],
        dependent=["solution_steps", "solution_certificate"],
    )
    patches = {"solution": {
        "solution_steps": ["60 / 1.5 = 40"],
        "final_answer": "40 km/h",
        "solution_certificate": _cert("40 km/h", units="km/h"),
    }}
    result, repaired, calls = _run(
        monkeypatch, question=question, qtype="word_problem",
        verdict=_verdict(issue), patches=patches,
    )
    assert calls == ["solution"]
    assert result["repaired_ids"] == ["q1"]
    assert repaired["final_answer"] == "40 km/h"
    assert repaired["solution_certificate"]["final_answer"] == "40 km/h"
    assert "30" not in " ".join(repaired["solution_steps"])


def test_live_essay_task_alignment_dependency_chain(monkeypatch):
    question = {
        "question_id": "q1", "slot_id": "s1", "question": "Define efficiency.",
        "reference_answer": "Efficiency is useful output divided by input.",
        "key_points": ["useful output", "input"],
        "solution_certificate": _cert("efficiency ratio"),
    }
    issue = _issue(
        "task_alignment", "wrong_variation", ["question"],
        dependent=["reference_answer", "key_points", "solution_certificate"],
    )
    patches = {
        "task_alignment": {
            "question": "Compare the effort force of two machines at equal resistance.",
        },
        "solution": {
            "solution_certificate": _cert("The lower-effort machine has the larger mechanical advantage."),
        },
        "answer": {
            "reference_answer": "For equal resistance, the machine with larger mechanical advantage needs less effort.",
            "key_points": ["equal resistance", "effort", "mechanical advantage"],
        },
    }
    result, repaired, calls = _run(
        monkeypatch, question=question, qtype="essay",
        verdict=_verdict(issue), patches=patches,
    )
    assert calls == ["task_alignment", "answer", "solution"]
    assert result["repaired_ids"] == ["q1"]
    assert "Compare" in repaired["question"]
    assert repaired["reference_answer"]
    assert repaired["solution_certificate"]
