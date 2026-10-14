from __future__ import annotations

import json

from typing import Any

SYSTEM_PROMPT = (
    "You are an assessment-aware exam question generator for teachers. "
    "Your task is to realize each supplied assessment-catalog task as a high-quality "
    "student-facing question.\n\n"

    "=== SOURCE AND GROUNDING RULES ===\n"
    "- Each plan entry contains its own assigned evidence boundary. Use only that evidence "
    "for source facts, laws, formulas, definitions, terminology, and domain claims.\n"
    "- Neutral constructed numbers and givens are allowed for a self-contained quantitative "
    "problem when the catalog task and assessment guidance permit calculation or modeling. "
    "They are problem data, not claims about the source, and must not be presented as source facts.\n"
    "- Preserve any plan givens marked as required or fixed. Never invent a missing scientific "
    "constant, factual value, law, condition, or source-specific example.\n\n"

    "=== QUESTION GENERATION RULES ===\n"
    "- Generate completely NEW questions based on the document content.\n"
    "- Do NOT copy existing questions from the document.\n"
    "- Do NOT simply change a few words from sentences in the document.\n"
    "- Convert explanations and concepts into exam questions that test student understanding.\n"
    "- Follow the assigned catalog task exactly. Preserve its concept/knowledge target, student "
    "operation, target/result, reasoning mode, required output/work product, required givens, and "
    "assigned evidence. Being on the same broad topic is not sufficient. For example, interpreting "
    "a common solution region must not become checking whether one point satisfies a system.\n"
    "- You may vary permitted wording, scenario details, and neutral constructed givens, but never "
    "silently change what knowledge, operation, result, reasoning, or output the student is assessed on.\n"
    "- Recall is legitimate for Definition, Fill-in-the-Blank, and recall-oriented True/False "
    "tasks when the plan explicitly requests it. Do not inflate such tasks into unrelated reasoning.\n"
    "- If task_reused is true, inspect previous_task_questions (including rejected/conflict roles) "
    "and previous_task_plan_items, then follow suggested_assessment_angle and satisfy every "
    "required_instance_distinction through a different target, required "
    "method, representation, constraint structure, direction of reasoning, or application that "
    "requires independent work. Different wording, names, scenarios, or numbers alone do not count.\n"
    "- Match the stored educational level and assessment style.\n\n"

    "=== QUESTION CLARITY RULES ===\n"
    "- Every question must be self-contained.\n"
    "- A student should understand the question without seeing the document context.\n"
    "- Do not refer to the source material inside questions.\n"
    "- Never use phrases such as:\n"
    "  * 'According to the context'\n"
    "  * 'According to the document'\n"
    "  * 'Based on the provided information'\n"
    "  * 'Based on the text'\n"
    "  * 'In the passage'\n"
    "  * 'From the given context'\n"
    "- Never use phrases that describe the source of information, including "
    "'according to the analysis', 'as shown in the figure', 'as discussed above', "
    "'the following example shows', or similar phrases.\n"
    "- Do not create reading-comprehension style questions.\n"
    "- Do not mention that the information came from a PDF, document, chapter, or context.\n\n"

    "=== ANSWER QUALITY RULES ===\n"
    "- Correct answers must follow from the assigned evidence and the givens stated in the question.\n"
    "- Avoid ambiguous questions with multiple possible correct answers.\n"
    "- Avoid overly easy questions where the answer can be guessed from the options.\n"
    "- Avoid overly specific questions about small details unless they represent an important concept.\n"
    "- For every quantitative item, solve it completely before returning it. Verify that all "
    "symbols and necessary givens are defined, every equation and calculation is valid, constraints "
    "are respected, units and rounding are correct, and the final answer solves exactly what was asked.\n"
    "- For a quantitative MCQ, independently solve the stem and verify that exactly one option "
    "matches the result and every distractor is wrong for the stated givens.\n"
    "- Every quantitative or symbolic STEM question, including a quantitative MCQ, "
    "must include an internal solution_certificate object. It is not student-facing. Use exactly "
    "these fields: method, important_givens, calculation_steps, final_answer, units, "
    "matching_correct_option, and verification_method. Set matching_correct_option to the verified "
    "A/B/C/D label for MCQ and null otherwise. The certificate must show checkable equations and "
    "calculations, not generic claims. verification_method must describe a second check such as "
    "substitution, reverse calculation, constraint testing, or independent recomputation.\n"
    "- Self-check every question and answer before output.\n\n"

    "=== OUTPUT RULES ===\n"
    "- Generate EXACTLY the requested number of questions.\n"
    "- Return ONLY valid JSON.\n"
    "- Do not include markdown fences.\n"
    "- Do not include explanations before or after the JSON.\n"
    "- Every required field must exist and contain a non-empty value."
)

_OUTPUT_DIRECTIVE = (
    "\nReturn ONLY the raw JSON object. Do NOT wrap it in ```json fences and do NOT "
    "add any text before or after the JSON.\n"
    "Prefer compact JSON: put every field and question on as few lines as possible, "
    "and separate array items with commas.\n"
)

_DIFFICULTY_EASY = (
    "Generate EASY questions. Test one clear idea with a direct, familiar task, usually "
    "one main step, and strong clues or an obvious method where appropriate. "
    "Easy quantitative questions may still require solving; do not turn them into "
    "recall questions merely because the requested difficulty is easy."
)

_DIFFICULTY_MEDIUM = (
    "Generate MEDIUM questions. Require the student to choose or apply the right idea, "
    "interpret information, rearrange a relationship, or complete 2-3 connected steps. Use less "
    "scaffolding than easy questions, without adding unsupported concepts."
)

_DIFFICULTY_HARD = (
    "Generate HARD questions. Require supported multi-step reasoning: select among "
    "relationships, combine related laws or concepts, perform several justified "
    "transformations, make an inference, derive or prove, compare alternatives, analyze an "
    "error or representation, or use an intermediate result to reach a "
    "conclusion. "
    "Do not make a question 'hard' merely by using confusing wording; the difficulty "
    "must come from the reasoning required to answer it."
)

_DIFFICULTY_MIX = (
    "Aim for a balanced mixture of EASY, MEDIUM, and HARD questions when the selected source "
    "supports those reasoning levels. Assign each slot its honest reasoning demand. Never "
    "relabel a simple task as hard merely to balance counts."
)

_DIFFICULTY_TARGET = {
    "easy": _DIFFICULTY_EASY,
    "medium": _DIFFICULTY_MEDIUM,
    "hard": _DIFFICULTY_HARD,
    "mix": _DIFFICULTY_MIX,
}


def _difficulty_block(difficulty: str) -> str:
    directive = _DIFFICULTY_TARGET.get(difficulty, _DIFFICULTY_MIX)
    return f"## Difficulty Target (follow this strictly)\n{directive}"


def _planned_difficulty_block(
    difficulty: str,
    planned_items: list[dict[str, Any]] | None,
) -> str:
    if not planned_items:
        return _difficulty_block(difficulty)
    return (
        "## Difficulty Target (follow this strictly)\n"
        f"The teacher requested {difficulty.upper()} difficulty. For every planned slot, the "
        "slot's current difficulty, actual_difficulty, reasoning_depth, and concrete "
        "reasoning_requirements are authoritative, including fixed-difficulty exams. Generate "
        "the cognitive demand recorded on that slot using only its assigned evidence. When a "
        "fallback slot remains at a different genuine difficulty, do not silently relabel, "
        "simplify, or intensify it to match the overall request."
    )


_LANGUAGE_NAMES = {
    "en": "English",
    "ar": "Arabic",
}


def _language_block(language: str | None) -> str:
    """Output-language instruction. The document context itself already carries
    the source language; this makes the requirement explicit so scaffolding
    (questions, answers, options, terms) matches it."""
    if not language or language == "en":
        return ""
    name = _LANGUAGE_NAMES.get(language, language)
    return (
        f"## Output Language (follow this strictly)\n"
        f"Write ALL generated content in {name}: every question, statement, option, "
        f"answer, reference answer, key point, and Word Bank term must be in {name}. "
        f"Keep the JSON field NAMES exactly as specified (they stay English), but all "
        f"field VALUES must be written in {name}. Preserve technical terms, acronyms, "
        f"and formulas from the source as they appear.\n\n"
    )


def _version_note(model_number: int) -> str:
    return (
        f"This question set is Exam Model #{model_number} "
        "of a set of several exam versions generated from the SAME source material. "
        "Make sure these questions are DIFFERENT from the questions produced for other "
        "exam models: do not repeat or reword the same questions. The models may cover "
        "the same broad concepts using different grounded facts, ideas, relationships, "
        "examples, or angles. Remain fully grounded in the selected source content so "
        "each exam version is distinct."
    )

MCQ_RULES = (
    "- Test exactly the assigned finalized catalog task. Preserve its concept, operation, target, "
    "reasoning_mode, difficulty, and evidence; do not substitute a simpler recall task.\n"
    "- Before writing the item, decide whether the task naturally supports one defensible best "
    "answer and three plausible competing alternatives. If it does not, return that slot in "
    "unsuitable_tasks with failure_code mcq_task_unsuitable and a precise reason. Do not fabricate "
    "facts, scenarios, numbers, or complexity to fill the slot.\n"
    "- Keep the stem concise. Include only information needed to solve it. Never copy a long source "
    "passage and then ask the student to repeat an answer already stated in the stem.\n"
    "- Each question has EXACTLY four options (A, B, C, D) and EXACTLY one correct answer. "
    "No 'all of the above', no 'none of the above'.\n"
    "- Distractors must be plausible and knowledge-based — they should represent "
    "common misconceptions or related-but-incorrect concepts, not obviously wrong "
    "or random unrelated facts.\n"
    "- Correct answer letter (A/B/C/D) must be evenly distributed across a batch: "
    "no single letter may appear more than 50% of the time.\n"
    "- Options must use parallel, readable forms. Numeric and symbolic options may be concise values "
    "or expressions rather than complete sentences.\n"
    "- Keep options reasonably parallel in length, grammar, detail, specificity, and tone. The "
    "correct option must not stand out by being longer, more qualified, or the only option that "
    "copies source wording. No two options may be semantically equivalent or defensibly correct.\n"
    "- The question must be answerable from the provided context.\n"
    "- Follow the planned reasoning_mode. For quantitative or symbolic MCQs, require the "
    "student to solve or reason before choosing; do not replace the planned operation with "
    "a definition, factual-recall, why, or how question.\n"
    "- Distractors must be close enough in value, meaning, form, or reasoning that the student "
    "must solve or understand the question. No distractor may also be scientifically or "
    "mathematically valid for the stem.\n"
    "- For calculation and symbolic MCQs, build distractors from distinct plausible student "
    "errors such as a wrong operation, sign, unit conversion, formula, ratio, substitution, "
    "an intermediate result, or algebraic step. Do not use random numbers.\n"
    "- For quantitative tasks, independently solve the assigned task and verify givens, equations, "
    "units, constraints, and the solution certificate before constructing options. The correct "
    "option, solution, certificate, and answer letter must agree.\n"
    "- Difficulty must come from the assigned reasoning: direct recognition for Easy; meaningful "
    "distinction, inference, application, or connected steps for Medium; deeper multi-concept "
    "reasoning, error analysis, derivation, or non-obvious interpretation for Hard. Never create "
    "difficulty through long wording, irrelevant details, or unsupported calculations.\n"
    "- logical_proof_reasoning MCQs may test a valid proof step, contradiction, induction "
    "step, logical equivalence, or counterexample; they must not ask for a full constructed proof.\n"
)

MCQ_SCHEMA = (
    '{\n'
    '  "questions": [\n'
    '    {\n'
    '      "slot_id": "m1_mcq_1",\n'
    '      "question": "the question text",\n'
    '      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},\n'
    '      "correct_answer": "B",\n'
    '      "solution_certificate": null\n'
    '    }\n'
    '  ],\n'
    '  "unsuitable_tasks": [\n'
    '    {"slot_id": "m1_mcq_2", "failure_code": "mcq_task_unsuitable", '
    '"reason": "why this task cannot support one best answer and three plausible alternatives"}\n'
    '  ]\n'
    '}'
)

MCQ_EXAMPLE = (
    '{\n'
    '  "questions": [\n'
    '    {\n'
    '      "slot_id": "m1_mcq_1",\n'
    '      "question": "Which metric is the harmonic mean of precision and recall?",\n'
    '      "options": {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"},\n'
    '      "correct_answer": "B"\n'
    '    }\n'
    '  ],\n'
    '  "unsuitable_tasks": []\n'
    '}'
)

TRUE_FALSE_RULES = (
    "- Each statement must be unambiguously TRUE or FALSE according to the context.\n"
    "- Each statement must be a single, clear claim with exactly one True/False answer.\n"
    "- Avoid double negatives and vague qualifiers such as 'often', 'always', 'may', "
    "or 'can' unless the context explicitly supports them.\n"
    '- Every question object MUST include an "answer" field with the exact string '
    '"True" or "False" (capitalized).\n'
    "- Across a batch of two or more statements, use both True and False whenever the planned "
    "claims permit it; do not make every statement have the same answer.\n"
    "- Do NOT omit the answer field, or the question will be rejected.\n"
)

TRUE_FALSE_SCHEMA = (
    '{\n'
    '  "questions": [\n'
    '    {\n'
    '      "slot_id": "m1_true_false_1",\n'
    '      "statement": "the factual statement",\n'
    '      "answer": "True"\n'
    '    }\n'
    '  ]\n'
    '}'
)

TRUE_FALSE_EXAMPLE = (
    '{\n'
    '  "questions": [\n'
    '    {\n'
    '      "slot_id": "m1_true_false_1",\n'
    '      "statement": "A confusion matrix can be used to evaluate a classifier.",\n'
    '      "answer": "True"\n'
    '    },\n'
    '    {\n'
    '      "slot_id": "m1_true_false_2",\n'
    '      "statement": "Precision is defined as TP / (TP + FN).",\n'
    '      "answer": "False"\n'
    '    }\n'
    '  ]\n'
    '}'
)

WHY_SCHEMA = (
    '{\n'
    '  "questions": [\n'
    '    {\n'
    '      "slot_id": "m1_why_1",\n'
    '      "question": "the question text",\n'
    '      "reference_answer": "a short reference answer",\n'
    '      "solution_certificate": null\n'
    '    }\n'
    '  ]\n'
    '}'
)

WHY_RULES = (
    "- Every question must genuinely require cause/effect reasoning, justification, a mechanism, "
    "or an explanation of a relationship.\n"
    "- Preserve the planned reasoning operation and assess deeper understanding rather than retrieval.\n"
    "- Valid forms include Why...?, Give the reason..., Justify..., and Arabic علّل، لماذا، ما سبب، "
    "or فسّر سبب.\n"
    "- Never drift into mention/list, definition, comparison, meaning, evidence-significance, or "
    "general clarification.\n"
    '- Provide a concise "reference_answer" stating the requested reason directly.\n'
)
WHY_EXAMPLE = (
    '{"questions":[{"slot_id":"m1_why_1","question":"Why must a believer remain mindful of God?",'
    '"reference_answer":"Because awareness that God knows outward and inward actions encourages obedience and restraint."}]}'
)

DEFINITION_RULES = (
    "- Select one important term, concept, law, principle, or named idea from the source.\n"
    '- The "term" field is only the source term or concept to define, for example "kinetic energy". '
    "Do not put instruction wording such as Define or What is meant by in this field.\n"
    '- Provide a concise, accurate "reference_answer" that defines the term using only the '
    "selected source content.\n"
    "- Prefer terms whose meaning is explicitly supported by the source and avoid vague labels.\n"
    "- Definition is intentionally a recall form; do not add an application, importance claim, or "
    "extended explanation unless the plan explicitly requests it.\n"
)

DEFINITION_SCHEMA = (
    '{\n'
    '  "questions": [\n'
    '    {"slot_id": "m1_definition_1", "term": "kinetic energy", '
    '"reference_answer": "The energy an object has due to its motion."}\n'
    '  ]\n'
    '}'
)

DEFINITION_EXAMPLE = DEFINITION_SCHEMA

EQUATION_RULES = (
    '- The "equation" field is a complete equation-centered prompt. Put a short instruction first, '
    'then put important equations, systems, reactions, and formal relationships on separate display '
    'lines using valid LaTeX delimiters `\\[ ... \\]`. Use inline `\\( ... \\)` only for short '
    'expressions that naturally belong in a sentence. Because this is JSON, escape every LaTeX '
    'backslash correctly.\n'
    "- Follow the entry's assessment guidance and task_operation exactly. The selected material "
    "determines whether Equation means balancing or completing a reaction, solving or factoring an "
    "expression, deriving or proving a relationship, calculation, or another equation-centered task.\n"
    "- Include a clear explicit instruction matching task_operation and target. Supply every premise "
    "needed by that operation from the entry's assigned evidence.\n"
    "- Decide what to reveal from task_operation. For rearrange/apply-a-provided-relation tasks, "
    "give the required starting relation. For derive, construct, formulate, combine, prove, balance, "
    "or select-the-relationship tasks, give only the supported premises and givens; do not state the "
    "final relationship, completed equation, exact derivation, or result the student must produce.\n"
    "- A shared schema example is output-shape guidance only. Never turn it into a universal "
    "rearrange-and-isolate template, and never substitute a Physics-style operation for the assigned "
    "operation in another subject.\n"
    "- Keep the student work centered on forming, solving, manipulating, balancing, completing, or "
    "deriving an equation. Use no narrative wrapper unless context is essential to the planned operation.\n"
    "- Supporting tables are allowed when they directly support evaluating, comparing, or solving "
    "the displayed relationships. Keep their columns aligned and mathematical entries in LaTeX.\n"
    "- Use source-appropriate notation: Mathematics equations/systems/expressions; Physics symbolic "
    "relationships with concise values and units; Chemistry reactions and chemical or quantitative "
    "relations; and each other STEM discipline's own formal representation. Never force all subjects "
    "into an algebra template.\n"
    "- Use valid LaTeX for fractions, exponents, roots, inequalities, systems, units, and reactions: "
    "for example `\\frac{a}{b}`, `x^{2}`, `\\sqrt{x}`, `\\begin{cases}...\\end{cases}`, "
    "`\\mathrm{m\\,s^{-1}}`, and `\\mathrm{Fe}+\\mathrm{O_2}\\rightarrow\\mathrm{Fe_2O_3}`. "
    "Do not use Markdown bracket pseudo-delimiters or raw Unicode layout as a substitute for LaTeX.\n"
    "- Do not turn the task into a Word Problem. If the student must extract relevant quantities from "
    "a meaningful situation and interpret the result in that situation, it belongs in Word Problem.\n"
    '- Provide ordered "solution_steps" appropriate to the planned operation and a separate '
    '"final_answer" in the form expected by the material.\n'
)

EQUATION_SCHEMA = (
    '{\n'
    '  "questions": [\n'
    '    {"slot_id": "m1_equation_1", '
    '"equation": "Solve the system:\\n\\\\[\\\\begin{cases}2x+y=7\\\\\\\\x-y=2\\\\end{cases}\\\\]", '
    '"solution_steps": ["Add the equations to obtain 3x = 9.", '
    '"So x = 3; substitute to obtain y = 1."], "final_answer": "(3, 1)", '
    '"solution_certificate": {"method": "elimination", "important_givens": '
    '["2x + y = 7", "x - y = 2"], "calculation_steps": ["3x = 9", "x = 3", '
    '"y = 1"], "final_answer": "(3, 1)", "units": "", '
    '"matching_correct_option": null, "verification_method": "Substitute x=3 and y=1 into both equations."}}\n'
    '  ]\n'
    '}'
)

EQUATION_EXAMPLE = EQUATION_SCHEMA

WORD_PROBLEM_RULES = (
    "- Follow the entry's assessment guidance and task_operation exactly. The selected material "
    "determines what applied or narrative form is natural for this Word Problem.\n"
    "- Write a self-contained, meaningful quantitative or application situation whose requested work "
    "matches the assigned operation and target. Applying a provided rule, selecting a governing "
    "relationship, forming a model, calculating, comparing, and interpreting are different tasks; "
    "require only the work assigned by the catalog.\n"
    "- A superficial name or story wrapped around an already-written equation is not a Word Problem.\n"
    "- Source facts and relationships must be supported by assigned evidence. Neutral constructed "
    "problem data are allowed when the task permits them and must all be stated explicitly.\n"
    "- Include all information necessary for the planned operation without giving away reasoning "
    "that the profile says the student should perform. If selecting or forming the governing "
    "relationship is assessed, do not provide it. If applying a provided rule is assessed, providing "
    "that rule is valid. Never state the exact intermediate inference or result being assessed.\n"
    "- Worked examples establish supported methods and constraints. Do not copy the same scenario + "
    "values + requested result from an assigned source unless the catalog target explicitly requires "
    "that exact source instance. Any constructed givens must remain evidence-consistent and fully "
    "stated; do not invent unsupported scientific facts or contexts.\n"
    '- Provide ordered "solution_steps" appropriate to the planned operation and a separate '
    '"final_answer" in the expected form.\n'
)

WORD_PROBLEM_SCHEMA = (
    '{\n'
    '  "questions": [\n'
    '    {"slot_id": "m1_word_problem_1", '
    '"question": "A 5 kg object accelerates at 4 m/s². Calculate the force acting on it.", '
    '"solution_steps": ["Use F = ma", "F = 5 kg × 4 m/s²", "F = 20 N"], '
    '"final_answer": "20 N", "solution_certificate": {"method": "apply F = ma", '
    '"important_givens": ["m = 5 kg", "a = 4 m/sÂ²"], "calculation_steps": '
    '["F = 5 Ã— 4", "F = 20 N"], "final_answer": "20 N", "units": "N", '
    '"matching_correct_option": null, "verification_method": "Check dimensions kgÂ·m/sÂ² = N."}}\n'
    '  ]\n'
    '}'
)

WORD_PROBLEM_EXAMPLE = WORD_PROBLEM_SCHEMA

ESSAY_RULES = (
    "- Ask an open-ended question requiring synthesis of several supported points, structured comparison, "
    "argument, or extended explanation. A single brief causal justification belongs in Why instead.\n"
    '- Provide a "reference_answer" (a model essay outline of several sentences) that is fully '
    "grounded in the context.\n"
    '- Provide a "key_points" array (3-6 short bullet expectations) that a good answer should '
    "cover, in order of importance.\n"
    "- The question must be answerable from the provided context and must not require external "
    "knowledge.\n"
)

ESSAY_SCHEMA = (
    '{\n'
    '  "questions": [\n'
    '    {\n'
    '      "slot_id": "m1_essay_1",\n'
    '      "question": "the essay question text",\n'
    '      "reference_answer": "a model reference answer of several sentences",\n'
    '      "key_points": ["point 1", "point 2", "point 3"]\n'
    '    }\n'
    '  ]\n'
    '}'
)

ESSAY_EXAMPLE = (
    '{\n'
    '  "questions": [\n'
    '    {\n'
    '      "slot_id": "m1_essay_1",\n'
    '      "question": "Compare how precision and recall behave as the decision threshold '
    'changes, and explain the trade-off.",\n'
    '      "reference_answer": "Raising the threshold makes the classifier more selective: '
    'fewer items are predicted positive, which raises precision but drops recall. Lowering '
    'the threshold captures more positives, raising recall but pulling down precision. The '
    'trade-off means no single threshold optimizes both unless the classifier is perfect.",\n'
    '      "key_points": ["threshold controls selectivity", "higher threshold raises '
    'precision", "lower threshold raises recall", "the two cannot both be maximized on '
    'imperfect models"]\n'
    '    }\n'
    '  ]\n'
    '}'
)

FILL_IN_THE_BLANK_ITEMS_RULES = (
    "- Write exactly {count} numbered fill-in-the-blank items.\n"
    "- Use ONE blank per item by default, marked with the underscore sequence: ________.\n"
    "- A blank may be marked with a single run of underscores (do not add extra markers).\n"
    "- Use exactly ONE blank in every item. The current architecture assigns exactly one answer per slot.\n"
    "- The selected catalog answer is authoritative and is supplied only as context. Do not "
    "return, rewrite, or substitute it.\n"
    "- Questions must be self-contained and grounded only in the selected source content.\n"
    "- The question must be pure text: no numbers, units, equations, formulas, mathematical or "
    "chemical symbolic expressions, calculations, or supplied laws.\n"
    "- Return two candidate shared-bank distractors with each item. Every candidate must be "
    "same-topic and plausible, but definitely wrong for EVERY blank and every fixed answer in "
    "this request. Candidates must be distinct and cannot be aliases of any fixed answer. Code "
    "will select exactly two shared distractors after the final active item set is known.\n"
)

FILL_IN_THE_BLANK_ITEMS_SCHEMA = (
    '{\n'
    '  "items": [\n'
    '    {\n'
    '      "slot_id": "m1_fill_in_the_blank_1",\n'
    '      "stem": "sentence with ________ for a blank",\n'
    '      "distractor_1": "related wrong term 1",\n'
    '      "distractor_2": "related wrong term 2"\n'
    '    }\n'
    '  ]\n'
    '}'
)

FILL_IN_THE_BLANK_ITEMS_EXAMPLE = (
    '{\n'
    '  "items": [\n'
    '    {\n'
    '      "slot_id": "m1_fill_in_the_blank_1",\n'
    '      "stem": "The measure of the surface enclosed by a boundary is called ________.",\n'
    '      "distractor_1": "perimeter",\n'
    '      "distractor_2": "volume"\n'
    '    }\n'
    '  ]\n'
    '}'
)

# --- Exam planning prompts --------------------------------------------------

PLANNER_SYSTEM_PROMPT = (
    "You are an assessment-aware exam planner. In this single call, work in two internal phases. "
    "Phase 1 analyzes the selected material: infer its domain or domains and educational level; "
    "inspect concepts, operations, worked examples, exercises, and question patterns; decide how "
    "each DraftWork question type naturally applies; and identify tasks that depend on unavailable "
    "visuals. Phase 2 builds the requested exam plan from that analysis.\n\n"
    "Return one compact root-level assessment_profile and the exam plans. The profile describes "
    "only assessment form, style, level, support, canonical task operations, and evidence chunk "
    "IDs. It must not contain reusable textbook facts, equations, reaction content, numeric values, "
    "or answers. Cover mcq, fill_in_the_blank, true_false, definition, why, equation, "
    "word_problem, and essay even when some are unsupported.\n\n"
    "For every type, set support to supported, limited, or unsupported; list source-appropriate "
    "task_operations in English snake_case; give one concise form_guidance sentence; and list the "
    "selected evidence_chunk_ids that demonstrate support. Unsupported types use empty operation "
    "and evidence arrays. Source exercises may inform form and level, but question facts remain "
    "bounded to each plan item's source_chunk_id plus support_chunk_ids.\n\n"
    "Do not assume a universal meaning for any question type. Derive Equation and Word Problem "
    "behavior from this material's examples, exercises, operations, and conventions. Operations "
    "such as balance_equation, complete_reaction, derive_relationship, prove_relationship, "
    "solve_equation, factor_expression, simplify_expression, calculate_from_scenario, "
    "stoichiometric_conversion, explain_cause, and compare_concepts are examples rather than a "
    "closed taxonomy. Treat visual-dependent tasks as unsupported unless fully answerable from "
    "textual or tabular information that can be included in the question.\n\n"
    "If a requested type cannot support the requested number of distinct grounded tasks, return a "
    "planning_issues entry with requested_count and supported_count. Do not invent support or "
    "substitute another type. Plan only supported_count items for that type.\n\n"
    "Each plan item requires its exact requested slot_id, source_chunk_id, support_chunk_ids, topic, "
    "concept, canonical_skill, knowledge_focus, student_work_product, relationship (or null), "
    "target, task_operation, given_conditions, "
    "scenario_family (or null), required_givens, text_answerable, reasoning_mode, difficulty, and "
    "idea_to_test. task_operation is "
    "the precise disciplinary action and must exactly match one listed in that type's profile. "
    "reasoning_mode is the broader cognitive mode. relationship is a stable source-grounded law, "
    "rule, theorem, equation family, or null. target is a concise canonical noun phrase without "
    "scenario-specific values or objects.\n\n"
    "Difficulty metadata is explicit. requested_difficulty records the teacher request; "
    "actual_difficulty records the level the task genuinely satisfies and normally equals difficulty. "
    "For an ordinary task, base_catalog_task_id and base_task_family_id may be null, "
    "difficulty_adaptation is null, and difficulty_fallback_reason is null. A catalog-backed "
    "difficulty adaptation preserves those base IDs and uses the supplied adaptation object. Never "
    "claim actual_difficulty=hard from a label alone.\n\n"
    "canonical_skill is a concise stable skill label independent of names, materials, story wrappers, "
    "knowledge_focus is the exact atomic fact, principle, relationship, or skill being assessed, "
    "written as a stable canonical label independent of question type and wording. "
    "student_work_product is the observable thing the student must produce or do, such as "
    "select_term, judge_claim, state_definition, explain_relationship, derive_formula, "
    "calculate_angle, or calculate_refractive_index. "
    "and numeric values. target names the output category independently of the specific entity or "
    "material (for example, 'angle of refraction', not 'angle in glass'). given_conditions lists the important supplied "
    "quantities and conditions that define the task. scenario_family is a stable setup label, without "
    "decorative wording, or null for a non-scenario task. required_givens lists everything the student "
    "must receive in the generated stem. Set text_answerable true only when the full task can be asked "
    "and answered without an unavailable image, graph, table, diagram, figure, or visual label. If it "
    "is false, do not plan the slot; report a planning issue instead.\n\n"
    "Use only selected chunk IDs. Add support chunks only when genuinely necessary. Every required "
    "fact, condition, relationship, quantity, and scenario detail must be supported by the assigned "
    "chunks. idea_to_test describes the skill rather than supplying question facts.\n\n"
    "Before returning, perform an internal uniqueness audit across every slot and question type. "
    "Group tasks by knowledge_focus. Sharing a knowledge focus is allowed when the student operation, "
    "target, and work product are genuinely different, such as identifying a term from its description "
    "versus stating its definition. Two tasks are repetition when they assess the same knowledge focus "
    "with the same effective operation, target, and work product, even if their question types differ. "
    "Reusing a relationship is allowed only when student_work_product, task_operation, target, and "
    "reasoning_mode are all meaningfully different and source-supported. If there are not enough "
    "distinct grounded atomic tasks, "
    "return a planning_issues shortfall instead of filling slots with repetition. "
    "Across all question types in an exam, compare knowledge_focus, student_work_product, "
    "canonical_skill, task_operation, relationship or "
    "concept, target, given_conditions, and scenario_family. Treat the same operation + relationship + "
    "target as repetition even when values, materials, or question type differ. A different type or story wrapper does "
    "not make the same underlying problem distinct. Do not reuse the same formula/concept, target, "
    "important values/conditions, and setup to ask a slightly different step. Prefer changing the "
    "skill, operation, target, or relationship rather than merely changing numbers. Keep models "
    "equivalent while assigning distinct "
    "grounded ideas. Difficulty reflects reasoning depth: easy is direct, medium requires selection "
    "or connected steps, and hard requires genuinely deeper multi-step reasoning, inference, proof, "
    "comparison, or error analysis.\n\n"
    "Allowed reasoning_mode values are conceptual_recall, conceptual_reasoning, calculation, "
    "symbolic_manipulation, law_or_theorem_application, derivation_relationship, "
    "logical_proof_reasoning, representation_interpretation, comparison_prediction, error_analysis, "
    "and estimation_reasonableness. Return only valid compact JSON matching the supplied schema."
)

PLANNER_SCHEMA_EXAMPLE = (
    '{\n'
    '  "assessment_profile": {"subject": "normalized_subject", "domain": "STEM", '
    '"domains": ["..."], "educational_level": "...", '
    '"assessment_style": "...", "question_types": {\n'
    '    "mcq": {"support": "supported", "task_operations": ["analyze_concept"], "form_guidance": "...", "evidence_chunk_ids": ["child-id"]},\n'
    '    "fill_in_the_blank": {"support": "supported", "task_operations": ["recall_term"], "form_guidance": "...", "evidence_chunk_ids": ["child-id"]},\n'
    '    "true_false": {"support": "supported", "task_operations": ["evaluate_statement"], "form_guidance": "...", "evidence_chunk_ids": ["child-id"]},\n'
    '    "definition": {"support": "supported", "task_operations": ["define_term"], "form_guidance": "...", "evidence_chunk_ids": ["child-id"]},\n'
    '    "why": {"support": "supported", "task_operations": ["explain_cause"], "form_guidance": "...", "evidence_chunk_ids": ["child-id"]},\n'
    '    "equation": {"support": "limited", "task_operations": ["derive_relationship"], "form_guidance": "...", "evidence_chunk_ids": ["child-id"]},\n'
    '    "word_problem": {"support": "supported", "task_operations": ["calculate_from_scenario"], "form_guidance": "...", "evidence_chunk_ids": ["child-id"]},\n'
    '    "essay": {"support": "unsupported", "task_operations": [], "form_guidance": "...", "evidence_chunk_ids": []}\n'
    '  }, "unsupported_visual_tasks": []},\n'
    '  "planning_issues": [],\n'
    '  "exams": [\n'
    '    {\n'
    '      "model_number": 1,\n'
    '      "items": {\n'
    '        "mcq": [\n'
    '          {"slot_id": "m1_mcq_1", "source_chunk_id": "child-id-from-source", "support_chunk_ids": [], "topic": "Refraction", '
    '"concept": "Refraction", "canonical_skill": "apply refraction relationship", '
    '"knowledge_focus": "snell_law_unknown_refractive_index", '
    '"student_work_product": "calculate_refractive_index", "relationship": "Snell\'s law", '
    '"target": "refractive index", "task_operation": "calculate_from_scenario", '
    '"given_conditions": ["incidence angle", "refraction angle"], "scenario_family": "light crossing media", '
    '"required_givens": ["incidence angle", "refraction angle"], "text_answerable": true, '
    '"reasoning_mode": "calculation", '
    '"difficulty": "medium", "idea_to_test": "Determine refractive index by selecting '
    'and rearranging Snell\'s law."}\n'
    '        ]\n'
    '      }\n'
    '    }\n'
    '  ]\n'
    '}'
)

PLANNER_OUTPUT_DIRECTIVE = (
    "\nReturn ONLY the raw JSON object. Do NOT wrap it in ```json fences and do NOT "
    "add any text before or after the JSON.\nPrefer compact JSON.\n"
)


def build_planner_prompt(
    num_models: int,
    tasks: list[tuple[str, int]],
    planner_context: str,
    difficulty: str = "mix",
    language: str = "en",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the one planning call (all models)."""
    task_lines = "\n".join(f"  - {count} {qtype}" for qtype, count in tasks)
    slot_lines = "\n".join(
        f"  - m{model_number}_{qtype}_{index}"
        for model_number in range(1, num_models + 1)
        for qtype, count in tasks
        for index in range(1, count + 1)
    )
    lang_block = _language_block(language).replace(
        "ALL generated content", "the 'topic', 'concept', and 'idea_to_test' values"
    ).replace("field VALUES must be written in", "values must be written in") \
        if language != "en" else ""
    user_prompt = (
        f"Plan exam questions for {num_models} exam model(s) generated from the same "
        f"selected source content.\n\n"

        f"## Requested question counts (apply EXACTLY to EVERY exam model)\n{task_lines}\n\n"

        f"## Required slot IDs\n{slot_lines}\n\n"

        f"{lang_block}"

        f"## Requested difficulty\n{difficulty}\n"
        "For a fixed difficulty, assign that value to every slot. For mix, distribute easy, "
        "medium, and hard as evenly as the slot count permits. Difficulty must describe "
        "reasoning depth and scaffolding, not wording length or larger numbers.\n\n"

        f"## Selected source content (titles + full selected context)\n{planner_context}\n\n"

        f"Produce a plan for EVERY exam model. Each model's plan must contain the requested "
        f"per-type counts unless a structured planning issue reports a grounded shortfall. "
        f"The concepts must be distributed so the "
        f"exam versions remain equivalent. Reuse broad concepts when appropriate, but "
        f"assign different grounded ideas or angles across models.\n\n"

        f"## Required JSON format (use exactly these field names)\n{PLANNER_SCHEMA_EXAMPLE}"
        f"\n\nThe 'exams' array must contain exactly {num_models} entries, one per "
        f"exam model. Each model has an 'items' object whose requested question-type "
        f"arrays contain slot_id / source_chunk_id / support_chunk_ids / topic / concept / "
        f"canonical_skill / knowledge_focus / student_work_product / relationship / target / "
        f"task_operation / given_conditions / "
        f"scenario_family / required_givens / text_answerable / reasoning_mode / difficulty / "
        f"idea_to_test. Use only "
        f"source_chunk_id values printed in the selected content.\n"
        f"{PLANNER_OUTPUT_DIRECTIVE}"
    )
    return PLANNER_SYSTEM_PROMPT, user_prompt


def build_prompt(
    question_type: str,
    count: int,
    context: str,
    difficulty: str = "mix",
    model_number: int = 1,
    planned_items: list[dict[str, Any]] | None = None,
    language: str = "en",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the given question type."""
    if question_type == "mcq":
        rules, schema, example = MCQ_RULES, MCQ_SCHEMA, MCQ_EXAMPLE
        type_name = "Multiple Choice (MCQ)"
    elif question_type == "true_false":
        rules, schema, example = TRUE_FALSE_RULES, TRUE_FALSE_SCHEMA, TRUE_FALSE_EXAMPLE
        type_name = "True/False"
    elif question_type == "definition":
        rules, schema, example = DEFINITION_RULES, DEFINITION_SCHEMA, DEFINITION_EXAMPLE
        type_name = "Definition"
    elif question_type == "why":
        rules, schema, example = WHY_RULES, WHY_SCHEMA, WHY_EXAMPLE
        type_name = "Why / Give a Reason"
    elif question_type == "equation":
        rules, schema, example = EQUATION_RULES, EQUATION_SCHEMA, EQUATION_EXAMPLE
        type_name = "Equation"
    elif question_type == "word_problem":
        rules, schema, example = WORD_PROBLEM_RULES, WORD_PROBLEM_SCHEMA, WORD_PROBLEM_EXAMPLE
        type_name = "Word Problem"
    elif question_type == "essay":
        rules, schema, example = ESSAY_RULES, ESSAY_SCHEMA, ESSAY_EXAMPLE
        type_name = "Essay"
    else:
        raise ValueError(f"Unsupported question type: {question_type}")

    if planned_items:
        plan_lines = "\n".join(
            _format_planned_item(i, it)
            for i, it in enumerate(planned_items, start=1)
        )
        plan_block = (
            f"## Question Plan\n"
            f"Generate EXACTLY one question per entry below. Each entry is an isolated evidence "
            f"boundary: use only that entry's assigned_sources and never borrow facts, equations, "
            f"values, conditions, or examples from another entry. Follow its assessment_guidance, "
            f"task_operation, relationship, target, reasoning_mode, reasoning_depth, "
            f"reasoning_requirements, idea, and difficulty. The per-slot difficulty controls "
            f"each question for fixed and mixed requests. "
            f"Do not deviate from the planned concepts and "
            f"do not introduce unplanned concepts.\n{plan_lines}\n\n"
        )
    else:
        plan_block = ""

    isolated_entries = bool(planned_items) and all(
        isinstance(item.get("assigned_sources"), dict) for item in planned_items or []
    )
    source_context = (
        "Evidence is attached separately to every plan entry below."
        if isolated_entries else context
    )
    user_prompt = (
        f"Create exactly {count} {type_name} exam question(s).\n\n"
        "Use each plan entry's assigned evidence as its knowledge boundary. "
        "Do not copy source wording and do not refer to the source inside questions. "
        "Then follow the Question Plan (if present), the Question Type Rules, and "
        "the Required JSON Format at the end.\n\n"

        f"## Selected Source Content\n{source_context}\n\n"

        f"{plan_block}"

        f"## Question Type Rules\n{rules}\n\n"

        f"## Required JSON Format (use exactly these field names)\n{schema}\n\n"

        f"## Example of a valid output\n{example}\n\n"

        f"{_planned_difficulty_block(difficulty, planned_items)}\n\n"

        f"{_language_block(language)}"

        f"## Exam Version\n{_version_note(model_number)}\n\n"

        f"## Task\nGenerate exactly {count} {type_name} exam question(s) following "
        f"the rules and the enforced JSON Schema. Copy every slot_id exactly from "
        f"the question plan.\n\n"

        f"{_OUTPUT_DIRECTIVE}"
    )
    return SYSTEM_PROMPT, user_prompt


def _format_planned_item(index: int, item: dict[str, Any]) -> str:
    relationship = item.get("relationship")
    relationship_text = "null" if relationship is None else str(relationship)
    return (
        f"{index}. slot_id={item.get('slot_id', '')} | "
        f"task_id={item.get('task_id', '')} | "
        f"fitb_entry_id={item.get('fitb_entry_id', '')} | "
        f"task_family_id={item.get('task_family_id', '')} | "
        f"coverage_id={item.get('coverage_id', '')} | "
        f"concept_id={item.get('concept_id', '')} | "
        f"relationship_id={item.get('relationship_id')} | "
        f"source_chunk_id={item.get('source_chunk_id', '')} | topic={item.get('topic', '')} | "
        f"support_chunk_ids={item.get('support_chunk_ids', [])} | "
        f"concept={item.get('concept', item.get('topic', ''))} | "
        f"canonical_skill={item.get('canonical_skill', '')} | "
        f"knowledge_focus={item.get('knowledge_focus', '')} | "
        f"student_work_product={item.get('student_work_product', '')} | "
        f"relationship={relationship_text} | target={item.get('target', '')} | "
        f"task_operation={item.get('task_operation', '')} | "
        f"given_conditions={item.get('given_conditions', [])} | "
        f"scenario_family={item.get('scenario_family')} | "
        f"required_givens={item.get('required_givens', [])} | "
        f"text_answerable={item.get('text_answerable')} | "
        f"reasoning_mode={item.get('reasoning_mode', '')} | "
        f"reasoning_depth={item.get('reasoning_depth', '')} | "
        f"reasoning_requirements={item.get('reasoning_requirements', [])} | "
        f"prerequisites={item.get('prerequisites', [])} | "
        f"requested_difficulty={item.get('requested_difficulty', '')} | "
        f"difficulty={item.get('difficulty', '')} | "
        f"actual_difficulty={item.get('actual_difficulty', item.get('difficulty', ''))} | "
        f"difficulty_adaptation={json.dumps(item.get('difficulty_adaptation'), ensure_ascii=False)} | "
        f"difficulty_fallback_reason={item.get('difficulty_fallback_reason')} | "
        f"idea_to_test={item.get('idea_to_test', item.get('concept_to_test', ''))} | "
        f"task_reused={bool(item.get('task_reused'))} | "
        f"fitb_entry_reused={bool(item.get('fitb_entry_reused'))} | "
        f"fitb_answer={item.get('fitb_answer')} | "
        f"normalized_fitb_answer={item.get('normalized_fitb_answer')} | "
        f"clue_basis={item.get('clue_basis')} | "
        f"reuse_reason={item.get('reuse_reason')} | "
        f"catalog_replacement_stage={item.get('catalog_replacement_stage')} | "
        f"fitb_answer_reused={bool(item.get('fitb_answer_reused'))} | "
        f"required_instance_distinctions={item.get('required_instance_distinctions', [])} | "
        f"suggested_assessment_angle={item.get('suggested_assessment_angle')} | "
        f"reuse_justification={item.get('reuse_justification')} | "
        f"task_inventory={json.dumps(item.get('task_inventory', {}), ensure_ascii=False)} | "
        f"previous_task_plan_items={json.dumps(item.get('previous_task_plan_items', []), ensure_ascii=False)} | "
        f"previous_task_questions={json.dumps(item.get('previous_task_questions', []), ensure_ascii=False)} | "
        f"assessment_guidance={json.dumps(item.get('assessment_guidance', {}), ensure_ascii=False)} | "
        f"assigned_sources={json.dumps(item.get('assigned_sources', {}), ensure_ascii=False)}"
    )


def _plan_block(planned_items: list[dict[str, Any]] | None) -> str:
    if not planned_items:
        return ""
    plan_lines = "\n".join(
        _format_planned_item(i, it)
        for i, it in enumerate(planned_items, start=1)
    )
    return (
        "## Question Plan\n"
        "Each entry is an isolated evidence boundary; never use another entry's assigned_sources. "
        "The metadata controls assessment guidance, task_operation, concept, relationship, target, "
        "reasoning mode, and difficulty. Use the compiler-selected FITB answer for each slot. "
        "When task_reused or fitb_entry_reused is true, use previous_task_questions and "
        "previous_task_plan_items to create a substantially different "
        "assessment angle; context_role identifies the rejected duplicate, its conflict, or earlier "
        "items from a reused task. Treat all of them as negative examples. Changing wording, names, "
        "or numbers alone is insufficient. Follow suggested_assessment_angle and every structured "
        "required_instance_distinction as mandatory constraints. "
        f"Do not introduce concepts unrelated to the plan.\n{plan_lines}\n\n"
    )


def _entry_bounded_context(
    planned_items: list[dict[str, Any]] | None, fallback: str
) -> str:
    if planned_items and all(
        isinstance(item.get("assigned_sources"), dict) for item in planned_items
    ):
        return "Evidence is attached separately to every plan entry below."
    return fallback


def build_fitb_items_prompt(
    count: int,
    answer_by_slot: dict[str, str],
    context: str,
    difficulty: str = "mix",
    model_number: int = 1,
    planned_items: list[dict[str, Any]] | None = None,
    language: str = "en",
) -> tuple[str, str]:
    """Stage 2: write each blank around its already-approved answer."""
    rules = FILL_IN_THE_BLANK_ITEMS_RULES.format(count=count)
    assigned_line = json.dumps(answer_by_slot, ensure_ascii=False)
    user_prompt = (
        f"Create exactly {count} numbered Fill-in-the-Blank items.\n\n"
        "Use each plan entry's assigned evidence as its knowledge boundary. Write each sentence "
        "only from that slot's evidence.\n\n"

        f"## Selected Source Content\n{_entry_bounded_context(planned_items, context)}\n\n"

        f"{_plan_block(planned_items)}"

        f"## Fixed answer for each slot\n{assigned_line}\n\n"
        "- Create the blank sentence around the fixed answer.\n"
        "- Return only stem, distractor_1, and distractor_2 for each slot.\n"
        "- Never return or replace the answer. Code will assemble the fixed answer and word bank.\n\n"
        "- Distractor candidates must be wrong for all fixed answers listed above, because the "
        "final section uses one shared pair for every blank.\n\n"

        f"## Item Rules\n{rules}\n\n"

        f"## Required JSON Format (use exactly these field names)\n"
        f"{FILL_IN_THE_BLANK_ITEMS_SCHEMA}\n\n"

        f"## Example of a valid output\n{FILL_IN_THE_BLANK_ITEMS_EXAMPLE}\n\n"

        f"{_planned_difficulty_block(difficulty, planned_items)}\n\n"

        f"{_language_block(language)}"

        f"## Exam Version\n{_version_note(model_number)}\n\n"

        f"## Task\nGenerate exactly {count} items using their fixed per-slot answers. "
        "Copy every slot_id exactly from the plan.\n\n"

        f"{_OUTPUT_DIRECTIVE}"
    )
    return SYSTEM_PROMPT, user_prompt



def _objective_bundle_json_format(
    mcq_planned: list[dict[str, Any]],
    tf_planned: list[dict[str, Any]],
) -> str:
    """Render the one canonical dynamic envelope enforced by the response model."""
    payload: dict[str, Any] = {}
    if mcq_planned:
        payload["mcq"] = {
            "questions": [{
                "slot_id": str(mcq_planned[0].get("slot_id") or "m1_mcq_1"),
                "question": "the MCQ question",
                "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
                "correct_answer": "B",
                "solution_certificate": None,
            }],
            "unsuitable_tasks": [],
        }
    if tf_planned:
        payload["true_false"] = {
            "questions": [{
                "slot_id": str(tf_planned[0].get("slot_id") or "m1_true_false_1"),
                "statement": "the factual statement",
                "answer": "True",
                "solution_certificate": None,
            }],
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_obj_bundled_prompt(
    planned: dict[str, list[dict[str, Any]]],
    context: str,
    difficulty: str = "mix",
    model_number: int = 1,
    feedback: str = "",
    language: str = "en",
) -> tuple[str, str]:
    """Build one exact-slot prompt for MCQ and True/False questions."""
    mcq_planned = planned.get("mcq") or []
    tf_planned = planned.get("true_false") or []
    mcq_count = len(mcq_planned)
    tf_count = len(tf_planned)
    required_json_format = _objective_bundle_json_format(mcq_planned, tf_planned)
    requested_sections = []
    if mcq_count:
        requested_sections.append(f"`mcq` ({mcq_count} questions)")
    if tf_count:
        requested_sections.append(f"`true_false` ({tf_count} statements)")
    requested_description = " and ".join(requested_sections)
    mcq_nesting_rule = (
        "The `unsuitable_tasks` array belongs inside the `mcq` object. Never return "
        "`unsuitable_tasks` at the top level. Always include `mcq.unsuitable_tasks`, using "
        "an empty array when no MCQ task is unsuitable. Even when this request contains only "
        "MCQs, keep the outer `mcq` object. "
        if mcq_count else ""
    )

    plan_lines = []
    for qtype, label in (
        ("mcq", "Multiple Choice"),
        ("true_false", "True/False"),
    ):
        items = planned.get(qtype) or []
        if not items:
            continue
        block = f"### {label} ({len(items)} items)\n" + "\n".join(
            _format_planned_item(i, it)
            for i, it in enumerate(items, start=1)
        )
        plan_lines.append(block)
    plan_block = (
        "## Question Plan\nGenerate EXACTLY the planned counts below in their "
        "respective sections. Treat each entry as an isolated evidence boundary and never use "
        "another entry's assigned_sources. Follow every slot's assessment_guidance, "
        "task_operation, relationship, target, reasoning_mode, reasoning_depth, "
        "reasoning_requirements, idea_to_test, and difficulty. The per-slot difficulty "
        "controls each question for fixed and mixed requests. Test ONLY the stated topics and concepts; do not "
        "introduce unplanned concepts.\n\n"
        + "\n\n".join(plan_lines)
        if plan_lines
        else ""
    )

    user_prompt = (
        f"Create the requested objective section(s) of an exam in ONE response: "
        f"{requested_description}.\n\n"
        "Use each plan entry's assigned evidence as its knowledge boundary. "
        "Do not copy source wording and do not refer to the source inside questions.\n\n"

        f"## Selected Source Content\n{_entry_bounded_context([*mcq_planned, *tf_planned], context)}\n\n"

        f"{plan_block}\n\n"

        f"## Multiple Choice Rules\n{MCQ_RULES}\n\n"
        f"## True/False Rules\n{TRUE_FALSE_RULES}\n\n"
        "Every returned question must copy its slot_id exactly from the plan.\n\n"

        "## Required JSON Format (use exactly this nesting)\n"
        f"{required_json_format}\n\n"
        f"{mcq_nesting_rule}Omit a question-type section only when its planned "
        "count is zero and the enforced schema omits it.\n\n"

        f"{_planned_difficulty_block(difficulty, [*mcq_planned, *tf_planned])}\n\n"

        f"{_language_block(language)}"

        f"## Exam Version\n{_version_note(model_number)}\n\n"

        f"## Task\nReturn one JSON object with exactly these requested root sections: "
        f"{requested_description}. Follow the enforced JSON Schema.\n\n"

        f"{feedback}"

        f"{_OUTPUT_DIRECTIVE}"
    )
    return SYSTEM_PROMPT, user_prompt


def build_validation_repair_prompt(
    expected_label: str,
    rejected_items: list[Any],
    schema: str,
    context: str,
    difficulty: str = "mix",
    model_number: int = 1,
    language: str = "en",
) -> tuple[str, str]:
    """Repair ONLY the structurally-invalid pieces of rejected JSON.

    The rejected output is sent back unchanged plus the expected schema and rules;
    the model must fix the invalid fields/structure, not invent a new question.
    """
    rejected_block = "\n".join(
        (json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item))
        for item in rejected_items
    )
    user_prompt = (
        "Repair the following rejected JSON element(s) so they match the expected schema.\n\n"
        "IMPORTANT:\n"
        "- Do NOT generate a new question.\n"
        "- Do NOT change valid fields unless strictly required to fix the schema error.\n"
        "- Keep each question's content identical; only fix fields/types/structure that are "
        "invalid.\n"
        "- Return the SAME number of elements as provided.\n\n"

        f"## Selected Source Content\n{context}\n\n"

        f"## Rejected JSON to repair\n{rejected_block}\n\n"

        f"## Expected schema (register of what must be valid)\n{schema}\n\n"

        f"{_language_block(language)}"

        f"## Exam Version\n{_version_note(model_number)}\n\n"

        f"## Task\nReturn ONLY the repaired JSON object matching the expected schema, "
        f"preserving all valid content.\n\n"

        f"{_OUTPUT_DIRECTIVE}"
    )
    return SYSTEM_PROMPT, user_prompt
