from __future__ import annotations

import json
from copy import deepcopy

from app.online import multi_repair, multi_validator
from app.online.multi_validator import validate_exam_multi
from app.online.validator import (
    _manifest_from_report,
    repair_invalid_questions,
    validate_generated_questions,
)


def _question(qtype: str, index: int) -> dict:
    base = {"question_id": f"q-{qtype}-{index}", "slot_id": f"s-{qtype}-{index}"}
    if qtype == "mcq":
        return base | {"question": "What is 2 + 2?" if index != 1 else "2+2?",
                       "options": {"A": "4", "B": "3" if index != 1 else "4", "C": "5", "D": "6"},
                       "correct_answer": "A" if index != 1 else "B"}
    if qtype == "true_false":
        return base | {"statement": "Gravity attracts masses.", "answer": True}
    if qtype == "fill_in_the_blank":
        answers = ("gravity", "attraction", "weight")
        distractors = (("mass", "force"), ("inertia", "energy"), ("speed", "volume"))
        return base | {"question": "___ is the requested source term.", "answers": [answers[index - 1]],
                       "distractors": list(distractors[index - 1])}
    if qtype == "definition":
        return base | {"term": "mass", "reference_answer": "The amount of matter."}
    if qtype == "why":
        return base | {"question": "Why do objects fall?", "reference_answer": "Gravity attracts them."}
    if qtype == "equation":
        return base | {"equation": "2 + 2", "solution_steps": ["Add the values."], "final_answer": "4"}
    if qtype == "word_problem":
        return base | {"question": "Two apples plus two apples?", "solution_steps": ["2 + 2 = 4"], "final_answer": "4"}
    return base | {"question": "Explain gravity.", "reference_answer": "Gravity attracts masses.",
                   "key_points": ["attraction", "mass"]}


def _fixture():
    qtypes = ("mcq", "true_false", "fill_in_the_blank", "definition", "why", "equation", "word_problem", "essay")
    questions, plans, sources = {}, {}, {}
    plan_sections = {qtype: [] for qtype in qtypes}
    for qtype in qtypes:
        items = [_question(qtype, index) for index in range(1, 4)]
        if qtype == "fill_in_the_blank":
            questions[qtype] = {"items": items, "word_bank": [
                value for item in items for value in [*item["answers"], *item["distractors"]]
            ]}
        else:
            questions[qtype] = items
        for item in items:
            slot = item["slot_id"]
            quant = qtype in {"equation", "word_problem"}
            plan = {"slot_id": slot, "question_type": qtype, "task_id": f"task-{slot}",
                    "source_chunk_id": f"chunk-{slot}", "support_chunk_ids": [],
                    "concept": "gravity" if not quant else "addition", "knowledge_focus": "source fact",
                    "idea_to_test": "apply the assigned concept", "target": "correct result",
                    "task_operation": "calculate" if quant else "explain",
                    "required_givens": ["2 and 2"] if quant else [], "given_conditions": [],
                    "difficulty": "medium", "reasoning_mode": "calculation" if quant else "conceptual_reasoning",
                    "requires_quantitative_validation": quant}
            plans[slot] = plan
            plan_sections[qtype].append(plan)
            sources[f"chunk-{slot}"] = "Two plus two is four. Gravity attracts masses. Mass is matter."
    return ({"model_number": 1, "questions": questions}, plans, sources,
            [{"model_number": 1, "items": plan_sections}])


def test_24_question_validation_repair_revalidation_commit(monkeypatch):
    exam, plans, sources, plan_list = _fixture()
    target = "q-mcq-1"
    phase = {"value": "initial"}
    validator_calls: list[tuple[str, tuple[str, ...]]] = []
    repair_calls: list[str] = []

    def validator_request(_client, prompt, **_kwargs):
        payload = json.loads(prompt.split("<VALIDATION_DATA>\n", 1)[1].split("\n</VALIDATION_DATA>", 1)[0])
        role = payload["validator"]
        validator_calls.append((role, tuple(row["question_id"] for row in payload["questions"])))
        verdicts = []
        for row in payload["questions"]:
            qid = row["question_id"]
            if phase["value"] == "initial" and role == "answer_correctness" and qid == target:
                issues = [
                    {"failure_code": "ambiguous_or_structural", "fields_to_fix": ["question"],
                     "reason": "The stem is too terse.", "expected_fix": "Clarify the stem."},
                    {"failure_code": "incorrect_answer", "fields_to_fix": ["correct_answer"],
                     "reason": "B is not the intended unique key.", "expected_fix": "Select A."},
                ]
                status = "FAIL"
            else:
                issues, status = [], "PASS"
            original = ((row.get("revalidation") or {}).get("original_failure_findings") or [])
            checks = [{"issue_id": issue["issue_id"], "resolved": True,
                       "reason": "The final candidate resolves this defect.", "remaining_defect": ""}
                      for issue in original]
            verdicts.append({"question_id": qid, "status": status, "issues": issues,
                "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
                "observed_difficulty": "medium" if role == "difficulty" else None,
                "calculation_evidence": ["2 + 2 = 4"] if role == "quantitative" else [],
                "original_issue_checks": checks})
        return {"validator": role, "verdicts": verdicts}

    def repair_request(_client, prompt, **kwargs):
        payload = json.loads(prompt.split("<REPAIR_DATA>\n", 1)[1].split("\n</REPAIR_DATA>", 1)[0])["questions"][0]
        agent, current = payload["repair_agent"], payload["current_question"]
        repair_calls.append(agent)
        if agent == "question":
            patch = {"question": "What is 2 + 2?"}
        elif agent == "options":
            patch = {"options": {"A": "4", "B": "3", "C": "5", "D": "6"}}
        else:
            patch = {"correct_answer": "A"}
        changes = []
        for field, value in patch.items():
            changes.append({"field": field,
                            "previous_value": json.dumps(current.get(field), ensure_ascii=False, sort_keys=True),
                            "new_value": json.dumps(value, ensure_ascii=False, sort_keys=True),
                            "reason": f"{agent} repair corrected {field}."})
        codes = sorted({issue["failure_code"] for issue in payload["findings"]})
        return {"repairs": [{"question_id": target, "patch": patch,
            "repair_metadata": {"modified_fields": list(patch), "changes": changes,
                                "targeted_failure_codes": codes}}]}

    monkeypatch.setattr(multi_validator, "request_structured", validator_request)
    monkeypatch.setattr(multi_repair, "request_structured", repair_request)
    monkeypatch.setattr("app.online.duplicate_judge.request_structured", lambda _client, _prompt, **kwargs: {
        "verdicts": [{
            "pair_id": pair_id, "same_core_knowledge": "no",
            "same_student_operation_or_method": "no", "same_target_or_result": "no",
            "genuinely_new_reasoning": "yes",
            "one_answer_substantially_gives_away_other": "no",
        } for pair_id in kwargs.get("expected_ids") or []]
    })
    monkeypatch.setattr("app.llm.factory.create_review_client", lambda **_kwargs: object())
    monkeypatch.setattr(multi_validator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(multi_repair.time, "sleep", lambda _seconds: None)

    initial = validate_exam_multi(exam, plan_by_slot=plans, source_by_chunk=sources,
                                  domain="STEM", cache_scope="mock-24")
    assert len(initial["verdicts"]) == 24
    failures = [(value["question_id"], value["failure_code"], value.get("reason"))
                for value in initial["verdicts"] if value["action"] != "PASS"]
    assert len(failures) == 1, failures
    manifest = _manifest_from_report(initial, exam, plans)
    assert len(manifest) == 1
    assert all(issue.get("issue_id") for issue in manifest[0]["required_fixes"])

    state = {"job_id": "mock-24", "generated_exams": [deepcopy(exam)], "plans": plan_list,
             "retrieved_chunks": [{"child_id": key, "content": value} for key, value in sources.items()],
             "assessment_profile": {"domain": "STEM", "question_types": {}},
             "repair_manifest": manifest, "validation_reports": [initial], "validated_models": [],
             "question_repair_attempts": {}, "warnings": []}
    staged = repair_invalid_questions(state)
    assert repair_calls == ["question", "options", "answer"], staged.get("warnings")
    candidate = staged["repair_candidates"][0]
    assert len(candidate["repair_metadata"]["steps"]) == 3
    assert [change["agent"] for change in candidate["repair_metadata"]["changes"]] == [
        "question", "options", "answer",
    ]

    phase["value"] = "revalidation"
    completed = validate_generated_questions({**state, **staged})
    assert not completed.get("repair_manifest"), (
        completed.get("blocked_revalidation_candidates"), completed.get("revalidation_progress")
    )
    committed = completed["generated_exams"][0]["questions"]["mcq"][0]
    assert committed["question"] == "What is 2 + 2?"
    assert len(set(committed["options"].values())) == 4
    assert committed["options"][committed["correct_answer"]] == "4"
    assert completed["repair_candidates"] == []
    assert max(len(ids) for _role, ids in validator_calls) <= 4
    by_role = {role: sum(1 for called_role, _ids in validator_calls if called_role == role)
               for role in multi_validator.ROLES}
    print("mock_24_metrics", {"questions": 24, "validator_requests": len(validator_calls),
          "validator_requests_by_role": by_role, "repair_requests_by_role": repair_calls,
          "max_batch_size": max(len(ids) for _role, ids in validator_calls),
          "initial_cache": initial["cache_stats"], "revalidation_outcome": "committed"})
