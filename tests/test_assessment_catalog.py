from __future__ import annotations

from types import SimpleNamespace

from app.online import assessment_catalog as catalog


def _profile(chunk_id: str) -> dict:
    question_types = {}
    for qtype in (
        "mcq", "fill_in_the_blank", "true_false", "definition", "why", "why",
        "equation", "word_problem", "essay",
    ):
        question_types[qtype] = {
            "support": "supported",
            "task_operations": ["explain"],
            "form_guidance": "Use a grounded text task.",
            "evidence_chunk_ids": [chunk_id],
        }
    return {
        "subject": "Physics",
        "domain": "STEM",
        "domains": ["physics"],
        "educational_level": "secondary",
        "assessment_style": "grounded",
        "question_types": question_types,
        "unsupported_visual_tasks": [],
    }


def test_catalog_compiles_once_then_reuses_exact_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog.config, "ASSESSMENT_CATALOG_CACHE_DIR", str(tmp_path))
    calls = []
    fake_client = SimpleNamespace(last_usage={"input_tokens": 120, "output_tokens": 30})
    routed = []
    monkeypatch.setattr(
        catalog, "create_assessment_compiler_client",
        lambda *, domain_type: routed.append(domain_type) or fake_client,
    )

    def fake_request(_client, prompt, **_kwargs):
        calls.append(prompt)
        return {
            "assessment_profile": _profile("c1"),
            "tasks": [{
                "knowledge_anchor": "work equals force times displacement along force",
                "topic": "work", "concept": "mechanical work",
                "relationship": "W = F d cos theta", "task_operation": "explain",
                "student_work_product": "explain the relationship", "target": "meaning of work",
                "reasoning_mode": "conceptual_reasoning",
                "supported_difficulties": ["medium"],
                "allowed_question_types": ["why"],
                "evidence_chunk_ids": ["c1"], "required_givens": [],
                "scenario_family": None, "text_answerable": True,
            }],
        }

    monkeypatch.setattr(catalog, "request_structured", fake_request)
    state = {
        "processed_document_key": "processed-abc", "document_id": "doc-1",
        "retrieved_chunks": [{"child_id": "c1", "content": "Work is W = Fd cos theta."}],
        "planner_context": "[source_chunk_id=c1] Work is W = Fd cos theta.",
        "document_language": "en",
        "domain_type": "stem", "subject": "mathematics",
    }
    first = catalog.prepare_assessment_catalog(state)
    second = catalog.prepare_assessment_catalog(state)

    assert len(calls) == 1
    assert routed == ["stem"]
    assert first["assessment_catalog_cache_hit"] is False
    assert second["assessment_catalog_cache_hit"] is True
    assert second["planner_context"] == ""
    assert first["assessment_catalog"]["tasks"][0]["task_id"].startswith("task_")
    assert first["assessment_catalog"]["assessment_profile"]["domain"] == "STEM"
    assert "assessment_profile.subject: mathematics" in calls[0]
    assert "do not reclassify or override" in calls[0]
    assert first["assessment_catalog_key"] == second["assessment_catalog_key"]


def test_catalog_key_changes_with_selection_processed_version_and_logic(monkeypatch):
    one = catalog.catalog_cache_key("processed-a", ["c2", "c1"])
    assert one == catalog.catalog_cache_key("processed-a", ["c1", "c2"])
    assert one != catalog.catalog_cache_key("processed-a", ["c1"])
    assert one != catalog.catalog_cache_key("processed-b", ["c1", "c2"])
    monkeypatch.setattr(catalog.config, "ASSESSMENT_CATALOG_VERSION", "catalog-next")
    assert one != catalog.catalog_cache_key("processed-a", ["c1", "c2"])


def test_finalizer_builds_separate_fitb_bank_and_allocates_entry_ids():
    entries = [
        {"answer": answer, "evidence_chunk_id": "c1", "clue_basis": clue}
        for answer, clue in (
            ("covalent bond", "bond formed by sharing electrons"),
            ("ionic bond", "bond formed by electron transfer"),
            ("catalyst", "substance that speeds a reaction"),
            ("reactant", "starting substance in a reaction"),
            ("product", "substance formed by a reaction"),
        )
    ]
    finalized = catalog._finalize_catalog(
        {"assessment_profile": _profile("c1"), "tasks": [], "fitb_bank": entries},
        key="catalog", processed_document_key="processed", selected_chunk_ids=["c1"],
        usage=None,
    )

    assert finalized["tasks"] == []
    assert len(finalized["fitb_bank"]) == 5
    plans, issues = catalog.allocate_catalog_plan(
        finalized, [("fill_in_the_blank", 5)], 1, "easy"
    )
    allocated = plans[0]["items"]["fill_in_the_blank"]
    assert issues == []
    assert len(allocated) == 5
    assert len({item["fitb_entry_id"] for item in allocated}) == 5
    assert all(item["task_id"] is None for item in allocated)


def test_finalizer_uses_fitb_bank_not_task_compatibility_for_profile_support():
    task = {
        "knowledge_anchor": "explain a broad theme", "topic": "history",
        "concept": "theme", "relationship": None, "task_operation": "explain_causes",
        "student_work_product": "write an explanation", "target": "several causes",
        "reasoning_mode": "conceptual_reasoning",
        "allowed_question_types": ["fill_in_the_blank"],
        "evidence_chunk_ids": ["c1"], "required_givens": [],
        "scenario_family": None, "text_answerable": True,
    }
    finalized = catalog._finalize_catalog(
        {"assessment_profile": _profile("c1"), "tasks": [task], "fitb_bank": [{
            "answer": "historical theme", "evidence_chunk_id": "c1",
            "clue_basis": "central idea repeated through a historical account",
        }]},
        key="catalog", processed_document_key="processed", selected_chunk_ids=["c1"],
        usage=None,
    )

    assert finalized["tasks"] == []
    profile = finalized["assessment_profile"]["question_types"]["fill_in_the_blank"]
    assert profile["support"] == "supported"
    assert profile["task_operations"] == ["identify_term_from_text_clue"]
    assert profile["evidence_chunk_ids"] == ["c1"]


def test_fitb_bank_ids_aliases_and_pure_text_rules_are_code_owned():
    raw = {
        "assessment_profile": _profile("c1"),
        "tasks": [],
        "fitb_bank": [
            {"answer": "catalyst", "evidence_chunk_id": "c1",
             "clue_basis": "substance that speeds a reaction"},
            {"answer": "Catalyst", "evidence_chunk_id": "c1",
             "clue_basis": "substance that increases reaction rate"},
            {"answer": "NaCl", "evidence_chunk_id": "c1",
             "clue_basis": "chemical formula for table salt"},
            {"answer": "twelve mol", "evidence_chunk_id": "c1",
             "clue_basis": "amount used in a calculation"},
        ],
    }
    finalized = catalog._finalize_catalog(
        raw, key="catalog", processed_document_key="processed",
        selected_chunk_ids=["c1"], usage=None,
    )

    assert len(finalized["fitb_bank"]) == 2
    first, alias = finalized["fitb_bank"]
    assert first["fitb_entry_id"].startswith("fitb_")
    assert first["normalized_answer"] == "catalyst"
    assert first["used"] is False and first["usage_count"] == 0
    assert alias["alias_of_fitb_entry_id"] == first["fitb_entry_id"]
    assert alias["duplicate_normalized_answer"] is True
    assert catalog.is_pure_text_fitb_value("The reaction helper is ________.", allow_blank=True)
    assert not catalog.is_pure_text_fitb_value("Use NaCl to fill ________.", allow_blank=True)
    assert not catalog.is_pure_text_fitb_value("The answer is 12 mol.", allow_blank=True)
    assert not catalog.is_pure_text_fitb_value("Calculate the ________.", allow_blank=True)
    assert not catalog.is_pure_text_fitb_value("Use the law to find ________.", allow_blank=True)


def test_catalog_prompt_defines_semantic_suitability_for_active_types():
    prompt = catalog._catalog_prompt(
        "source", "English", ["c1"], domain_type="stem", subject="Physics"
    )
    for phrase in (
        "one defensible best answer",
        "word, term, name, concept, or short phrase",
        "cause, reason, justification, mechanism, or relationship",
        "symbolic manipulation, derivation, balancing, solving",
        "grounded verbal quantitative scenario",
        "bounded multi-point synthesis",
    ):
        assert phrase in prompt
    assert "concept + narrow target + student operation + expected work product" in prompt
    assert "identify a term from its description versus state its definition" in prompt
    assert "Do not target a fixed task count" in prompt


def test_finalizer_does_not_invent_mcq_or_semantic_types():
    task = {
        "knowledge_anchor": "causal explanation", "topic": "science",
        "concept": "mechanism", "relationship": "cause and effect",
        "task_operation": "explain_process", "student_work_product": "explain",
        "target": "mechanism", "reasoning_mode": "conceptual_reasoning",
        "allowed_question_types": ["why"], "evidence_chunk_ids": ["c1"],
        "required_givens": [], "scenario_family": None, "text_answerable": True,
    }
    finalized = catalog._finalize_catalog(
        {"assessment_profile": _profile("c1"), "tasks": [task]},
        key="catalog", processed_document_key="processed", selected_chunk_ids=["c1"],
        usage=None,
    )
    assert finalized["tasks"][0]["allowed_question_types"] == ["why"]


def test_atomic_task_identity_changes_with_work_product_target_or_reasoning():
    base = {
        "knowledge_anchor": "wavelength", "topic": "waves", "concept": "wavelength",
        "relationship": None, "task_operation": "assess_wavelength",
        "student_work_product": "term", "target": "identify wavelength",
        "reasoning_mode": "conceptual_recall", "allowed_question_types": ["mcq"],
        "evidence_chunk_ids": ["c1"], "required_givens": [], "scenario_family": None,
        "text_answerable": True,
    }
    raw = {"assessment_profile": _profile("c1"), "tasks": [
        base,
        {**base, "student_work_product": "definition", "target": "state wavelength meaning",
         "allowed_question_types": ["definition"]},
        {**base, "reasoning_mode": "conceptual_reasoning", "target": "explain wavelength relation",
         "student_work_product": "explanation", "allowed_question_types": ["why"]},
    ]}
    finalized = catalog._finalize_catalog(
        raw, key="catalog", processed_document_key="processed", selected_chunk_ids=["c1"],
        usage=None,
    )
    assert len({task["task_id"] for task in finalized["tasks"]}) == 3
    assert len({task["task_family_id"] for task in finalized["tasks"]}) == 3
    assert len({task["coverage_id"] for task in finalized["tasks"]}) == 1


def test_catalog_and_plan_semantics_share_the_same_canonical_identity():
    task = {
        "knowledge_anchor": "photon energy from a level transition",
        "topic": "quantum physics", "concept": "photon energy",
        "relationship": "E = h nu", "task_operation": "calculate_energy",
        "student_work_product": "calculate emitted photon energy",
        "target": "emitted photon energy", "reasoning_mode": "calculation",
        "allowed_question_types": ["word_problem"],
        "evidence_chunk_ids": ["c1"], "required_givens": ["frequency"],
        "scenario_family": "energy-level transition", "text_answerable": True,
    }
    catalog_identity = catalog.canonical_task_ids(task)
    plan_identity = catalog.canonical_task_ids({
        **task,
        "knowledge_focus": task["knowledge_anchor"],
        "knowledge_anchor": None,
    })

    assert catalog_identity is not None
    assert plan_identity == catalog_identity


def test_allocator_prefers_unused_concepts_before_second_operation_on_used_concept():
    def task(task_id, coverage, operation, qtypes):
        return {
            "task_id": task_id, "task_family_id": f"family_{task_id}",
            "coverage_id": coverage, "concept_id": coverage,
            "knowledge_anchor": coverage, "topic": "waves", "concept": coverage,
            "relationship": None, "task_operation": operation,
            "student_work_product": operation, "target": operation,
            "reasoning_mode": "conceptual_recall", "supported_difficulties": ["easy"],
            "allowed_question_types": qtypes, "evidence_chunk_ids": ["c1"],
            "required_givens": [], "scenario_family": None, "text_answerable": True,
        }
    tasks = [
        task("wave_identify", "wavelength", "identify_term", ["mcq"]),
        task("wave_define", "wavelength", "state_definition", ["definition"]),
        task("photon_define", "photon", "state_definition", ["definition"]),
    ]
    plans, issues = catalog.allocate_catalog_plan(
        {"tasks": tasks}, [("mcq", 1), ("definition", 1)], 1, "easy"
    )
    chosen = {
        qtype: [item["task_id"] for item in items]
        for qtype, items in plans[0]["items"].items()
    }
    assert issues == []
    assert chosen == {"mcq": ["wave_identify"], "definition": ["photon_define"]}


def test_catalog_cache_identity_changes_with_compiler_provider_model_and_version():
    base = {"provider": "openai", "model": "gpt-6-luna", "version": "compiler-v2"}
    key = catalog.catalog_cache_key("processed", ["c1"], base)
    compiler_key = catalog.compiler_cache_key("processed", ["c1"], base)
    for changed in (
        {**base, "provider": "deepseek"},
        {**base, "model": "gpt-6-luna-next"},
        {**base, "version": "compiler-v3"},
    ):
        assert key != catalog.catalog_cache_key("processed", ["c1"], changed)
        assert compiler_key != catalog.compiler_cache_key("processed", ["c1"], changed)


def test_stored_non_stem_metadata_routes_compiler_to_deepseek_path(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog.config, "ASSESSMENT_CATALOG_CACHE_DIR", str(tmp_path))
    routed = []
    fake_client = SimpleNamespace(last_usage={})
    monkeypatch.setattr(
        catalog, "create_assessment_compiler_client",
        lambda *, domain_type: routed.append(domain_type) or fake_client,
    )
    profile = _profile("c1")
    profile.update({"subject": "Wrong Model Guess", "domain": "STEM"})
    monkeypatch.setattr(catalog, "request_structured", lambda *_args, **_kwargs: {
        "assessment_profile": profile,
        "tasks": [{
            "knowledge_anchor": "causes of industrialization", "topic": "history",
            "concept": "industrial revolution", "relationship": None,
            "task_operation": "explain_causes",
            "student_work_product": "explain historical causes", "target": "industrialization",
            "reasoning_mode": "conceptual_reasoning",
            "allowed_question_types": ["why"], "evidence_chunk_ids": ["c1"],
            "required_givens": [], "scenario_family": None, "text_answerable": True,
        }],
    })
    result = catalog.prepare_assessment_catalog({
        "processed_document_key": "processed-history",
        "retrieved_chunks": [{"child_id": "c1", "content": "Industrialization history."}],
        "planner_context": "[source_chunk_id=c1] Industrialization history.",
        "document_language": "en", "domain_type": "non_stem", "subject": "history",
    })
    assert routed == ["non_stem"]
    stored = result["assessment_catalog"]["assessment_profile"]
    assert stored["domain"] == "NON_STEM"
    assert stored["subject"] == "history"


def test_catalog_shares_relationship_identity_but_keeps_distinct_atomic_targets():
    task = {
        "knowledge_anchor": "energy formula", "topic": "quantum", "concept": "energy",
        "task_operation": "apply_equation", "student_work_product": "write equation",
        "target": "energy", "reasoning_mode": "symbolic_manipulation",
        "allowed_question_types": ["mcq"], "evidence_chunk_ids": ["c1"],
        "required_givens": [], "scenario_family": None, "text_answerable": True,
    }
    finalized = catalog._finalize_catalog(
        {
            "assessment_profile": _profile("c1"),
            "tasks": [
                {**task, "relationship": "E_photon = hν", "target": "photon energy"},
                {**task, "relationship": "E_كم = hν", "target": "quantum energy"},
            ],
        },
        key="catalog", processed_document_key="processed", selected_chunk_ids=["c1"],
        usage=None,
    )

    relationships = [entry["relationship_id"] for entry in finalized["tasks"]]
    families = [entry["task_family_id"] for entry in finalized["tasks"]]
    assert len(relationships) == 2
    assert len(set(relationships)) == 1
    assert len(set(families)) == 2


def test_allocator_never_reuses_one_catalog_task_with_metadata_renaming():
    base = {
        "knowledge_anchor": "definition of work", "topic": "work", "concept": "work",
        "relationship": "W = Fd cos theta", "task_operation": "recall",
        "student_work_product": "state meaning", "target": "work", "reasoning_mode": "conceptual_recall",
        "supported_difficulties": ["easy"], "allowed_question_types": ["mcq", "definition"],
        "evidence_chunk_ids": ["c1"], "required_givens": [], "scenario_family": None,
        "text_answerable": True, "coverage_id": "knowledge_work", "relationship_id": "rel_work",
    }
    tasks = [
        {**base, "task_id": "task_work"},
        {**base, "task_id": "task_power", "coverage_id": "knowledge_power",
         "knowledge_anchor": "definition of power", "concept": "power", "target": "power"},
    ]
    plans, issues = catalog.allocate_catalog_plan(
        {"tasks": tasks}, [("mcq", 1), ("definition", 1)], 1, "easy"
    )
    chosen = [item["task_id"] for values in plans[0]["items"].values() for item in values]
    assert issues == []
    assert len(chosen) == len(set(chosen)) == 2


def test_replacement_cannot_select_the_current_task_again():
    tasks = []
    for suffix in ("current", "unused"):
        tasks.append({
            "task_id": f"task_{suffix}",
            "task_family_id": f"family_{suffix}",
            "coverage_id": f"coverage_{suffix}",
            "knowledge_anchor": suffix,
            "topic": "physics",
            "concept": suffix,
            "relationship": suffix,
            "task_operation": "identify_formula",
            "student_work_product": "write a formula",
            "target": suffix,
            "reasoning_mode": "symbolic_manipulation",
            "supported_difficulties": ["medium"],
            "allowed_question_types": ["equation"],
            "evidence_chunk_ids": ["c1"],
            "required_givens": [],
            "scenario_family": None,
            "text_answerable": True,
        })
    current = catalog.materialize_plan_item(tasks[0], "m1_equation_1", "medium")
    plans = [{"model_number": 1, "items": {"equation": [current]}}]

    replacement = catalog.unused_catalog_replacement(
        {"tasks": tasks}, plans, "m1_equation_1", "equation", "medium"
    )

    assert replacement is not None
    assert replacement["task_id"] == "task_unused"
    assert replacement["task_family_id"] == "family_unused"


def test_replacement_uses_unused_operation_on_used_concept_before_task_reuse():
    base = {
        "concept_id": "concept_wave", "knowledge_anchor": "wave relation",
        "topic": "waves", "concept": "wave relation", "relationship": "c = lambda nu",
        "reasoning_mode": "symbolic_manipulation", "supported_difficulties": ["medium"],
        "allowed_question_types": ["equation"], "evidence_chunk_ids": ["c1"],
        "required_givens": [], "scenario_family": None, "text_answerable": True,
    }
    identify = {
        **base, "task_id": "identify", "task_family_id": "family_identify",
        "coverage_id": "coverage_wave", "task_operation": "identify_relationship",
        "student_work_product": "relationship", "target": "wave relation",
    }
    explain = {
        **base, "task_id": "explain", "task_family_id": "family_explain",
        "coverage_id": "coverage_wave",
    }
    explain.update({
        "task_operation": "explain_relationship",
        "student_work_product": "explanation",
        "target": "inverse relationship",
    })
    current = catalog.materialize_plan_item(identify, "s_current", "medium")
    plans = [{"model_number": 1, "items": {"equation": [current]}}]

    replacement = catalog.unused_catalog_replacement(
        {"tasks": [identify, explain]}, plans, "s_current", "equation", "medium"
    )

    assert replacement is not None
    assert replacement["task_id"] == "explain"


def _equation_task(task_id: str, family: str, coverage: str) -> dict:
    return {
        "task_id": task_id, "task_family_id": family, "coverage_id": coverage,
        "concept_id": f"concept_{task_id}", "knowledge_anchor": task_id,
        "topic": "algebra", "concept": task_id, "relationship": "linear model",
        "task_operation": "write_equation", "student_work_product": "equation",
        "target": task_id, "reasoning_mode": "symbolic_manipulation",
        "supported_difficulties": ["medium"], "allowed_question_types": ["equation"],
        "evidence_chunk_ids": ["c1"], "required_givens": [],
        "scenario_family": None, "text_answerable": True,
    }


def test_catalog_exhaustion_prefers_unused_task_even_when_family_is_used():
    current_task = _equation_task("current", "family_current", "coverage_current")
    occupied_task = _equation_task("occupied", "family_shared", "coverage_shared")
    unused_same_family = _equation_task("unused", "family_shared", "coverage_unused")
    plans = [{"model_number": 1, "items": {"equation": [
        catalog.materialize_plan_item(current_task, "s_current", "medium"),
        catalog.materialize_plan_item(occupied_task, "s_occupied", "medium"),
    ]}}]

    replacement = catalog.catalog_exhaustion_replacement(
        {"tasks": [current_task, occupied_task, unused_same_family]},
        plans, "s_current", "equation", "medium",
    )

    assert replacement["task_id"] == "unused"
    assert replacement["catalog_replacement_stage"] == "unused_task_unused_concept"
    assert replacement.get("reused_due_to_catalog_exhaustion") is not True
    assert "s_occupied" in replacement["forced_duplicate_slot_ids"]


def test_duplicate_replacement_prefers_unused_concept_before_used_concept():
    current = _equation_task("current", "family_current", "coverage_current")
    current["concept_id"] = "concept_current"
    occupied = _equation_task("occupied", "family_occupied", "coverage_occupied")
    occupied["concept_id"] = "concept_used"
    unused_on_used_concept = _equation_task(
        "unused_used_concept", "family_new_operation", "coverage_new_operation"
    )
    unused_on_used_concept["concept_id"] = "concept_used"
    unused_concept = _equation_task(
        "unused_concept", "family_unused_concept", "coverage_unused_concept"
    )
    unused_concept["concept_id"] = "concept_unused"
    plans = [{"model_number": 1, "items": {"equation": [
        catalog.materialize_plan_item(current, "s_replace", "medium"),
        catalog.materialize_plan_item(occupied, "s_existing", "medium"),
    ]}}]

    replacement = catalog.catalog_exhaustion_replacement(
        {"tasks": [current, occupied, unused_on_used_concept, unused_concept]},
        plans, "s_replace", "equation", "medium",
    )

    assert replacement["task_id"] == "unused_concept"
    assert replacement["catalog_replacement_stage"] == "unused_task_unused_concept"


def test_duplicate_replacement_ranks_unused_tasks_by_active_exam_breadth():
    current = _equation_task("current", "family_current", "coverage_current")
    repeated = _equation_task("a_repeated", "family_repeated", "coverage_shared")
    broader = _equation_task("z_broader", "family_broader", "coverage_shared")
    repeated.update({
        "concept_id": "concept_shared", "task_operation": "repeated_operation",
        "target": "repeated_target", "reasoning_mode": "repeated_reasoning",
    })
    broader.update({
        "concept_id": "concept_shared", "task_operation": "unused_operation",
        "target": "unused_target", "reasoning_mode": "unused_reasoning",
    })
    active_overlap = _equation_task(
        "active_overlap", "family_overlap", "coverage_other"
    )
    active_overlap.update({
        "concept_id": "concept_other", "task_operation": "repeated_operation",
        "target": "repeated_target", "reasoning_mode": "repeated_reasoning",
    })
    removed_overlap = _equation_task(
        "removed_overlap", "family_removed", "coverage_other"
    )
    removed_overlap.update({
        "concept_id": "concept_other", "task_operation": "unused_operation",
        "target": "unused_target", "reasoning_mode": "unused_reasoning",
    })
    plans = [{"model_number": 1, "items": {"equation": [
        catalog.materialize_plan_item(current, "s_replace", "medium"),
        catalog.materialize_plan_item(active_overlap, "s_active", "medium"),
        catalog.materialize_plan_item(removed_overlap, "s_removed", "medium"),
    ]}}]

    replacement = catalog.catalog_exhaustion_replacement(
        {"tasks": [current, repeated, broader]}, plans,
        "s_replace", "equation", "medium",
        active_slot_ids={"s_replace", "s_active"},
    )

    assert replacement["task_id"] == "z_broader"
    assert replacement["task_reused"] is False


def test_duplicate_replacement_uses_other_used_task_before_original_task():
    task_a = _equation_task("task_a", "family_a", "coverage_a")
    task_b = _equation_task("task_b", "family_b", "coverage_b")
    plans = [{"model_number": 1, "items": {"equation": [
        catalog.materialize_plan_item(task_a, "s_replace", "medium"),
        catalog.materialize_plan_item(task_b, "s_existing", "medium"),
    ]}}]

    replacement = catalog.catalog_exhaustion_replacement(
        {"tasks": [task_a, task_b]}, plans, "s_replace", "equation", "medium",
        forced_duplicate_slot_ids=["s_conflict"],
        forced_duplicate_question_ids=["q_conflict"],
    )

    assert replacement["catalog_replacement_stage"] == "reused_task_after_catalog_exhaustion"
    assert replacement["reused_due_to_catalog_exhaustion"] is True
    assert replacement["task_id"] == "task_b"
    assert replacement["reuse_reason"] == "no_unused_compatible_task"
    assert replacement["forced_duplicate_slot_ids"] == ["s_existing", "s_conflict"]
    assert replacement["forced_duplicate_question_ids"] == ["q_conflict"]
    assert "numbers alone do not qualify" in replacement["reuse_rule"]
    inventory = replacement["task_inventory"]
    assert inventory["all_unused_compatible_count"] == 1
    assert inventory["eligible_unused_after_exclusions_count"] == 1
    assert inventory["remaining_unused_compatible_count"] == 1


def test_duplicate_replacement_ranks_used_tasks_by_active_exam_coverage():
    current = _equation_task("current", "family_current", "coverage_current")
    crowded = _equation_task("a_crowded", "family_crowded", "coverage_crowded")
    sparse = _equation_task("z_sparse", "family_sparse", "coverage_sparse")
    active_overlap = _equation_task("active_overlap", "family_overlap", "coverage_crowded")
    removed_overlap = _equation_task("removed_overlap", "family_removed", "coverage_sparse")
    other_model_overlap = _equation_task("other_overlap", "family_other", "coverage_sparse")
    plans = [
        {"model_number": 1, "items": {"equation": [
            catalog.materialize_plan_item(current, "m1_replace", "medium"),
            catalog.materialize_plan_item(crowded, "m1_crowded", "medium"),
            catalog.materialize_plan_item(sparse, "m1_sparse", "medium"),
            catalog.materialize_plan_item(active_overlap, "m1_active_overlap", "medium"),
            catalog.materialize_plan_item(removed_overlap, "m1_removed", "medium"),
        ]}},
        {"model_number": 2, "items": {"equation": [
            catalog.materialize_plan_item(other_model_overlap, "m2_other", "medium"),
        ]}},
    ]

    replacement = catalog.catalog_exhaustion_replacement(
        {"tasks": [current, crowded, sparse]}, plans,
        "m1_replace", "equation", "medium",
        active_slot_ids={"m1_replace", "m1_crowded", "m1_sparse", "m1_active_overlap"},
    )

    assert replacement["task_id"] == "z_sparse"
    assert replacement["catalog_replacement_stage"] == "reused_task_after_catalog_exhaustion"
    inventory = replacement["task_inventory"]
    assert inventory["compatible_used_task_ids"] == ["a_crowded", "z_sparse"]
    assert inventory["compatible_unused_task_ids"] == ["current"]


def test_duplicate_replacement_uses_original_task_only_after_alternatives_excluded():
    current = _equation_task("current", "family_current", "coverage_current")
    alternative = _equation_task("alternative", "family_alternative", "coverage_alternative")
    plans = [{"model_number": 1, "items": {"equation": [
        catalog.materialize_plan_item(current, "s_replace", "medium"),
        catalog.materialize_plan_item(alternative, "s_existing", "medium"),
    ]}}]

    replacement = catalog.catalog_exhaustion_replacement(
        {"tasks": [current, alternative]}, plans,
        "s_replace", "equation", "medium",
        excluded_task_ids={"alternative"},
        forced_duplicate_slot_ids=["s_conflict"],
        forced_duplicate_question_ids=["q_conflict"],
    )

    assert replacement["task_id"] == "current"
    assert replacement["catalog_replacement_stage"] == "regenerated_original_task_after_duplicate"
    assert replacement["reused_due_to_catalog_exhaustion"] is False
    assert replacement["forced_duplicate_slot_ids"] == ["s_conflict"]
    assert replacement["forced_duplicate_question_ids"] == ["q_conflict"]


def test_duplicate_replacement_uses_operation_target_family_and_reasoning_breadth():
    current = _equation_task("current", "family_current", "coverage_current")
    repeated = _equation_task("a_repeated", "family_repeated", "coverage_shared")
    broader = _equation_task("z_broader", "family_broader", "coverage_shared")
    repeated.update({
        "concept_id": "concept_shared", "task_operation": "repeat_operation",
        "target": "repeat_target", "reasoning_mode": "repeat_reasoning",
    })
    broader.update({
        "concept_id": "concept_shared", "task_operation": "broader_operation",
        "target": "broader_target", "reasoning_mode": "broader_reasoning",
    })
    overlap = _equation_task("overlap", "family_repeated", "coverage_other")
    overlap.update({
        "concept_id": "concept_other", "task_operation": "repeat_operation",
        "target": "repeat_target", "reasoning_mode": "repeat_reasoning",
    })
    plans = [{"model_number": 1, "items": {"equation": [
        catalog.materialize_plan_item(current, "s_replace", "medium"),
        catalog.materialize_plan_item(repeated, "s_repeated", "medium"),
        catalog.materialize_plan_item(broader, "s_broader", "medium"),
        catalog.materialize_plan_item(overlap, "s_overlap", "medium"),
    ]}}]

    replacement = catalog.catalog_exhaustion_replacement(
        {"tasks": [current, repeated, broader]}, plans,
        "s_replace", "equation", "medium",
        active_slot_ids={"s_replace", "s_repeated", "s_broader", "s_overlap"},
    )

    assert replacement["task_id"] == "z_broader"


def test_explicitly_excluded_original_task_falls_back_to_other_used_task():
    task_a = _equation_task("task_a", "family_a", "coverage_a")
    task_b = _equation_task("task_b", "family_b", "coverage_b")
    plans = [{"model_number": 1, "items": {"equation": [
        catalog.materialize_plan_item(task_a, "s_replace", "medium"),
        catalog.materialize_plan_item(task_b, "s_existing", "medium"),
    ]}}]

    replacement = catalog.catalog_exhaustion_replacement(
        {"tasks": [task_a, task_b]}, plans, "s_replace", "equation", "medium",
        excluded_task_ids={"task_a"},
    )

    assert replacement["catalog_replacement_stage"] == "reused_task_after_catalog_exhaustion"
    assert replacement["task_id"] == "task_b"
    inventory = replacement["task_inventory"]
    assert inventory["all_unused_compatible_count"] == 1
    assert inventory["eligible_unused_after_exclusions_count"] == 0
    assert inventory["excluded_compatible_task_ids"] == ["task_a"]


def test_fitb_replacement_prefers_unused_answer_separately_from_task_state():
    current = catalog.canonical_fitb_entry({
        "answer": "Lyman series", "evidence_chunk_id": "c1",
        "clue_basis": "hydrogen series ending at the ground level",
    }, {"c1"})
    unused_answer = catalog.canonical_fitb_entry({
        "answer": "Paschen series", "evidence_chunk_id": "c1",
        "clue_basis": "hydrogen series ending at the third level",
    }, {"c1"})
    plans = [{"model_number": 1, "items": {"fill_in_the_blank": [
        catalog.materialize_fitb_plan_item(current, "m1_fill_in_the_blank_1", "medium")
    ]}}]

    replacement = catalog.catalog_exhaustion_replacement(
        {"tasks": [], "fitb_bank": [current, unused_answer]}, plans,
        "m1_fill_in_the_blank_1", "fill_in_the_blank", "medium",
        used_fitb_answer_keys={"سلسلة ليمان", "Balmer series"},
    )

    assert replacement["fitb_entry_id"] == unused_answer["fitb_entry_id"]
    assert replacement["task_id"] is None
    assert replacement["fitb_entry_reused"] is False
    inventory = replacement["task_inventory"]
    # Replacing the current slot frees its former entry while reserving the new one.
    assert inventory["compatible_unused_count"] == 1
    assert catalog.normalize_fitb_answer_key("سلسلة ليمان") == "lyman"
    assert catalog.normalize_fitb_answer_key("Lyman series") == "lyman"


def test_fitb_replacement_reuses_least_used_entry_after_bank_exhaustion():
    current = catalog.canonical_fitb_entry({
        "answer": "Lyman series", "evidence_chunk_id": "c1",
        "clue_basis": "hydrogen series ending at the ground level",
    }, {"c1"})
    alternate = catalog.canonical_fitb_entry({
        "answer": "Paschen series", "evidence_chunk_id": "c1",
        "clue_basis": "hydrogen series ending at the third level",
    }, {"c1"})
    plans = [{"model_number": 1, "items": {
        "fill_in_the_blank": [
            catalog.materialize_fitb_plan_item(current, "m1_fill_in_the_blank_1", "medium"),
            catalog.materialize_fitb_plan_item(alternate, "m1_fill_in_the_blank_2", "medium"),
        ],
    }}]

    replacement = catalog.catalog_exhaustion_replacement(
        {"tasks": [], "fitb_bank": [current, alternate]}, plans,
        "m1_fill_in_the_blank_1", "fill_in_the_blank", "medium",
        used_fitb_answer_keys={"Lyman series", "Balmer series"},
    )

    assert replacement["fitb_entry_id"] == alternate["fitb_entry_id"]
    assert replacement["fitb_entry_reused"] is True
    assert replacement["catalog_replacement_stage"] == "reused_fitb_entry_after_bank_exhaustion"
