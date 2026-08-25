from __future__ import annotations

import json

from typing import Any

SYSTEM_PROMPT = (
    "You are an expert university exam question generator for teachers. "
    "Your task is to create high-quality original exam questions using only "
    "the knowledge contained in the provided document context.\n\n"

    "=== SOURCE AND GROUNDING RULES ===\n"
    "- The document context is the ONLY source of information.\n"
    "- Use only facts, concepts, explanations, examples, definitions, and results "
    "that appear in the provided context.\n"
    "- Never invent information, numbers, examples, terminology, or explanations.\n"
    "- Do not use external knowledge even if you know the topic.\n"
    "- Every question and answer must be supported by the provided context.\n\n"

    "=== QUESTION GENERATION RULES ===\n"
    "- Generate completely NEW questions based on the document content.\n"
    "- Do NOT copy existing questions from the document.\n"
    "- Do NOT simply change a few words from sentences in the document.\n"
    "- Convert explanations and concepts into exam questions that test student understanding.\n"
    "- Prefer questions that test reasoning, interpretation, comparison, explanation, "
    "and understanding of concepts.\n"
    "- Avoid questions that only test memorization of an isolated sentence.\n"
    "- Do NOT create duplicate questions that test the exact same concept, fact, "
    "or definition. Each question must test a distinct concept.\n"
    "- It IS allowed to create multiple questions from the same section or topic, "
    "as long as they ask about different aspects and have different answers. "
    "Never repeat the same question or give the same answer twice.\n"
    "- Questions should resemble questions written by a university professor.\n"
    "- Distribute the questions across the provided sections/subsections instead "
    "of clustering them on a single topic.\n\n"

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
    "- Correct answers must be clearly supported by the document context.\n"
    "- Every fact, number, term, and example used in a question, its answer, and "
    "its options must appear in or be directly inferable from the Document "
    "Knowledge Context. Do NOT introduce facts that are absent from the context.\n"
    "- Avoid ambiguous questions with multiple possible correct answers.\n"
    "- Avoid overly easy questions where the answer can be guessed from the options.\n"
    "- Avoid overly specific questions about small details unless they represent an important concept.\n"
    "- Before returning the JSON, self-check every question: it must be answerable "
    "from the context, its answer/distractors must be correct and distinct, and no "
    "two questions may test the same concept. Fix any question that fails before output.\n\n"

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
    "Generate EASY questions. They should mainly test basic recall, definitions, "
    "direct facts, simple understanding, and straightforward relationships. Avoid "
    "unnecessarily complex scenarios and avoid deep multi-step reasoning."
)

_DIFFICULTY_MEDIUM = (
    "Generate MEDIUM questions. They should mainly test understanding, interpretation, "
    "and application of concepts, plus moderate reasoning and relationships between "
    "concepts. More challenging than direct recall, but they should not require "
    "advanced analysis."
)

_DIFFICULTY_HARD = (
    "Generate HARD questions. They should emphasize deeper reasoning, analysis, "
    "comparison, applying concepts to scenarios, distinguishing between similar "
    "concepts, and multi-step understanding when supported by the source material. "
    "Do not make a question 'hard' merely by using confusing wording; the difficulty "
    "must come from the reasoning required to answer it."
)

_DIFFICULTY_MIX = (
    "Mix the difficulty of the questions so that the exam contains a balanced "
    "mixture of EASY, MEDIUM, and HARD questions. Distribute them across the three "
    "difficulty levels as evenly as the total number of questions allows. Do not "
    "generate everything at a single level."
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


def _version_note(model_number: int) -> str:
    return (
        f"This question set is Exam Model #{model_number} "
        "of a set of several exam versions generated from the SAME source material. "
        "Make sure these questions are DIFFERENT from the questions produced for other "
        "exam models: do not reuse, repeat, or reword the same questions or test the "
        "exact same concepts that other models cover. Remain fully grounded in the "
        "same selected source content, but vary the question topics and phrasing so "
        "each exam version is distinct."
    )

MCQ_RULES = (
    "- Each question has EXACTLY four options (A, B, C, D) and EXACTLY one correct answer.\n"
    "- Distractors must be plausible but clearly wrong.\n"
    "- Make the four options clearly distinct and mutually exclusive so exactly one is correct.\n"
    "- Distractors must be factually wrong and clearly different from the correct answer; "
    "do not make them near-identical rewordings, simple negations, or synonyms of one another.\n"
    "- The question must be answerable from the provided context.\n"
)

MCQ_SCHEMA = (
    '{\n'
    '  "questions": [\n'
    '    {\n'
    '      "question": "the question text",\n'
    '      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},\n'
    '      "correct_answer": "B"\n'
    '    }\n'
    '  ]\n'
    '}'
)

MCQ_EXAMPLE = (
    '{\n'
    '  "questions": [\n'
    '    {\n'
    '      "question": "Which metric is the harmonic mean of precision and recall?",\n'
    '      "options": {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"},\n'
    '      "correct_answer": "B"\n'
    '    }\n'
    '  ]\n'
    '}'
)

TRUE_FALSE_RULES = (
    "- Each statement must be unambiguously TRUE or FALSE according to the context.\n"
    "- Each statement must be a single, clear claim with exactly one True/False answer.\n"
    "- Avoid double negatives and vague qualifiers such as 'often', 'always', 'may', "
    "or 'can' unless the context explicitly supports them.\n"
    '- Every question object MUST include an "answer" field with the exact string '
    '"True" or "False" (capitalized).\n'
    "- Do NOT omit the answer field, or the question will be rejected.\n"
)

TRUE_FALSE_SCHEMA = (
    '{\n'
    '  "questions": [\n'
    '    {\n'
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
    '      "statement": "A confusion matrix can be used to evaluate a classifier.",\n'
    '      "answer": "True"\n'
    '    },\n'
    '    {\n'
    '      "statement": "Precision is defined as TP / (TP + FN).",\n'
    '      "answer": "False"\n'
    '    }\n'
    '  ]\n'
    '}'
)

SHORT_ANSWER_RULES = (
    "- Ask 'why', 'explain', or 'how' questions that require a short written answer.\n"
    '- Provide a concise "reference_answer" (2-4 sentences) that directly answers '
    "the question and is fully grounded in the context.\n"
)

SHORT_ANSWER_SCHEMA = (
    '{\n'
    '  "questions": [\n'
    '    {\n'
    '      "question": "the question text",\n'
    '      "reference_answer": "a short reference answer"\n'
    '    }\n'
    '  ]\n'
    '}'
)

SHORT_ANSWER_EXAMPLE = (
    '{\n'
    '  "questions": [\n'
    '    {\n'
    '      "question": "Explain why precision and recall often trade off against each other.",\n'
    '      "reference_answer": "Precision focuses on how many selected items are relevant, while '
    'recall focuses on how many relevant items are selected. Raising the decision threshold '
    'improves precision but typically lowers recall, and vice versa."\n'
    '    }\n'
    '  ]\n'
    '}'
)

ESSAY_RULES = (
    "- Ask an open-ended question that requires a structured, multi-sentence answer.\n"
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

FILL_IN_THE_BLANK_BANK_RULES = (
    "- Choose exactly {count} distinct correct answer TERMS from the source content, one per "
    "numbered blank item by default.\n"
    "- Every term must be a single word or a short phrase that appears in or is directly "
    "supported by the source content.\n"
    "- Prefer unique, non-repeating terms so the Word Bank is useful and not redundant.\n"
    "- Also choose EXACTLY 2 distractor words: plausible and related to the source, believable "
    "as answers, but that are NOT correct for any blank.\n"
    "- Do not invent unrelated terms merely to create distractors; they must be grounded in the "
    "source.\n"
    "- Distractors must never be a valid answer for one of the blank sentences.\n"
    '- Return a "correct_terms" array and a "distractors" array with exactly 2 entries.\n'
)

FILL_IN_THE_BLANK_BANK_SCHEMA = (
    '{\n'
    '  "correct_terms": ["term 1", "term 2", "term 3"],\n'
    '  "distractors": ["distractor 1", "distractor 2"]\n'
    '}'
)

FILL_IN_THE_BLANK_BANK_EXAMPLE = (
    '{\n'
    '  "correct_terms": ["area", "perimeter", "length", "width", "diagonal"],\n'
    '  "distractors": ["square", "triangle"]\n'
    '}'
)

FILL_IN_THE_BLANK_ITEMS_RULES = (
    "- Write exactly {count} numbered fill-in-the-blank items.\n"
    "- Use ONE blank per item by default, marked with the underscore sequence: ________.\n"
    "- A blank may be marked with a single run of underscores (do not add extra markers).\n"
    "- An item may contain TWO blanks only when it genuinely improves the question; never force "
    "two blanks.\n"
    "- NEVER use more than 2 blanks in an item.\n"
    '- Each item must include an "answers" array with the exact Word Bank terms that fill the '
    "blank(s), in left-to-right order.\n"
    "- Every answer MUST be one of the provided Word Bank entries.\n"
    "- Do not introduce any term that is not in the Word Bank.\n"
    "- Use every correct term from the Word Bank in at least one blank; the 2 distractors must "
    "never be used.\n"
    "- Questions must be self-contained and grounded only in the selected source content.\n"
)

FILL_IN_THE_BLANK_ITEMS_SCHEMA = (
    '{\n'
    '  "items": [\n'
    '    {\n'
    '      "question": "sentence with ________ for a blank",\n'
    '      "answers": ["term"]\n'
    '    },\n'
    '    {\n'
    '      "question": "sentence with ________ and ________ for two blanks",\n'
    '      "answers": ["term a", "term b"]\n'
    '    }\n'
    '  ]\n'
    '}'
)

FILL_IN_THE_BLANK_ITEMS_EXAMPLE = (
    '{\n'
    '  "items": [\n'
    '    {\n'
    '      "question": "To calculate the ________ of a rectangle, multiply its length by its "'
    'width.",\n'
    '      "answers": ["area"]\n'
    '    },\n'
    '    {\n'
    '      "question": "The ________ and ________ of a rectangle are used to compute its area.",\n'
    '      "answers": ["length", "width"]\n'
    '    }\n'
    '  ]\n'
    '}'
)

# --- Exam planning prompts --------------------------------------------------

PLANNER_SYSTEM_PROMPT = (
    "You are an expert university exam planning assistant.\n"
    "Your ONLY job is to decide WHAT each exam question should test.\n\n"
    "You do NOT write actual questions, answers, options, or reference answers. "
    "You only choose, for each planned question:\n"
    "  - question_type   (mcq | true_false | fill_in_the_blank | short_answer | essay)\n"
    "  - topic           (the source section the question belongs to)\n"
    "  - concept_to_test (the specific concept the question will assess)\n\n"
    "The exam will later be generated by a separate question-generation model.\n\n"

    "=== PLANNING GOALS ===\n"
    "- Decide what every question should test from the available source content.\n"
    "- Distribute the requested number of questions across the requested question "
    "types for EVERY exam model.\n"
    "- Make the exam VERSIONS meaningfully different: prefer different concepts "
    "across models rather than the same concept reworded.\n"
    "- Reduce concept repetition between models. Try to spread different concepts "
    "across the models when the source material has enough variety.\n"
    "- If the source material is too limited to avoid all repetition, some overlap "
    "is acceptable, but avoid unnecessary duplication.\n\n"

    "=== CONCEPT DISTINCTNESS RULES ===\n"
    "- Two plan items count as duplicates if they test basically the SAME concept, "
    "even when worded differently. For example 'what inductive reasoning means' "
    "and 'definition of inductive reasoning' are effectively the same concept.\n"
    "- AVOID these across different exam models.\n"
    "- Do not merely rename the same concept for different models.\n"
    "- Only reuse a concept across models when the source cannot support distinct "
    "concepts for every model.\n\n"

    "=== OUTPUT RULES ===\n"
    "- Return ONLY valid JSON.\n"
    "- Do not include markdown fences.\n"
    "- Do not include text before or after the JSON.\n"
    "- Every planned item must include non-empty question_type, topic, and "
    "concept_to_test.\n"
    "- For EVERY exam model, the total number of planned items of a given "
    "question_type must exactly match the requested count for that question_type.\n"
    "- The 'exams' array must contain exactly one entry per exam model.\n\n"

    "Use the planner JSON schema below exactly."
)

PLANNER_SCHEMA_EXAMPLE = (
    '{\n'
    '  "exams": [\n'
    '    {\n'
    '      "model_number": 1,\n'
    '      "questions": [\n'
    '        {\n'
    '          "question_type": "mcq",\n'
    '          "topic": "Malaria and blood types",\n'
    '          "concept_to_test": "Why type O blood is less affected by severe malaria"\n'
    '        }\n'
    '      ]\n'
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
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the one planning call (all models)."""
    task_lines = "\n".join(f"  - {count} {qtype}" for qtype, count in tasks)
    user_prompt = (
        f"Plan exam questions for {num_models} exam model(s) generated from the same "
        f"selected source content.\n\n"

        f"## Requested question counts (apply EXACTLY to EVERY exam model)\n{task_lines}\n\n"

        f"## Selected source content (titles + short snippets)\n{planner_context}\n\n"

        f"Produce a plan for EVERY exam model. Each model's plan must contain exactly "
        f"the requested per-type counts, and the concepts must be distributed so the "
        f"exam versions differ meaningfully. Distribute different concepts across the "
        f"models where the source material allows.\n\n"

        f"## Required JSON format (use exactly these field names)\n{PLANNER_SCHEMA_EXAMPLE}"
        f"\n\nThe 'exams' array must contain exactly {num_models} entries, one per "
        f"exam model, each with non-empty question_type / topic / concept_to_test. "
        f"question_type must be one of: mcq, true_false, fill_in_the_blank, "
        f"short_answer, essay.\n"
        f"{PLANNER_OUTPUT_DIRECTIVE}"
    )
    return PLANNER_SYSTEM_PROMPT, user_prompt


def build_plan_repair_prompt(
    previous_output: str,
    errors: list[str],
    planner_context: str,
) -> str:
    """User prompt: send the previous planner output + exact failures to fix."""
    error_lines = "\n".join(f"- {e}" for e in errors)
    return (
        "The previous exam plan was partially invalid. Fix ONLY the broken parts. "
        "Keep every already-valid model and every already-valid planned item "
        "EXACTLY as they are. Do not rewrite valid entries or reorder them.\n\n"
        f"## Validation errors to fix\n{error_lines}\n\n"
        "Fix these precisely:\n"
        "- If a model is missing or there are too many models: fix the model count.\n"
        "- If a question_type has too many items: remove the extra items from the "
        "END of that type's list.\n"
        "- If a question_type has too few items: ADD exactly the missing number of "
        "NEW items of that type. Inspect the existing items first and do NOT "
        "duplicate any existing concept (avoid equivalent concepts).\n"
        "- If an item is missing a field: fill in the missing field(s).\n"
        "- If two models reuse the same concept where the source allows distinct "
        "concepts: replace one with a different concept.\n\n"
        f"## Selected source content (titles + short snippets)\n{planner_context}\n\n"
        f"## Previous planner output to repair\n{previous_output}\n\n"
        "Return ONLY the corrected raw JSON with the exact same schema as before "
        "(an exams array with model_number, and per model a questions array of "
        'entries with question_type, topic, and concept_to_test). No fences, no '
        "extra text."
    )


def build_prompt(
    question_type: str,
    count: int,
    context: str,
    difficulty: str = "mix",
    model_number: int = 1,
    planned_items: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the given question type."""
    if question_type == "mcq":
        rules, schema, example = MCQ_RULES, MCQ_SCHEMA, MCQ_EXAMPLE
        type_name = "Multiple Choice (MCQ)"
    elif question_type == "true_false":
        rules, schema, example = TRUE_FALSE_RULES, TRUE_FALSE_SCHEMA, TRUE_FALSE_EXAMPLE
        type_name = "True/False"
    elif question_type == "essay":
        rules, schema, example = ESSAY_RULES, ESSAY_SCHEMA, ESSAY_EXAMPLE
        type_name = "Essay"
    else:
        rules, schema, example = SHORT_ANSWER_RULES, SHORT_ANSWER_SCHEMA, SHORT_ANSWER_EXAMPLE
        type_name = "Short Answer"

    if planned_items:
        plan_lines = "\n".join(
            f"{i}. topic={it.get('topic', '')} | concept_to_test={it.get('concept_to_test', '')}"
            for i, it in enumerate(planned_items, start=1)
        )
        plan_block = (
            f"## Question Plan\n"
            f"Generate EXACTLY one question per planned item below, testing ONLY the "
            f"stated topic and concept. Do not deviate from the planned concepts and "
            f"do not introduce unplanned concepts.\n{plan_lines}\n\n"
        )
    else:
        plan_block = ""

    user_prompt = (
        f"Create exactly {count} {type_name} exam question(s).\n\n"
        "Read the full selected source content below FIRST. It is the ONLY knowledge "
        "source for every question, answer, True/False decision, and MCQ option. "
        "Do not copy its wording and do not refer to it inside the questions. "
        "Then follow the Question Plan (if present), the Question Type Rules, and "
        "the Required JSON Format at the end.\n\n"

        f"## Selected Source Content\n{context}\n\n"

        f"{plan_block}"

        f"## Question Type Rules\n{rules}\n\n"

        f"## Required JSON Format (use exactly these field names)\n{schema}\n\n"

        f"## Example of a valid output\n{example}\n\n"

        f"{_difficulty_block(difficulty)}\n\n"

        f"## Exam Version\n{_version_note(model_number)}\n\n"

        f"## Task\nGenerate exactly {count} {type_name} exam question(s) following "
        f"the rules and the required JSON format above.\n\n"

        f"{_OUTPUT_DIRECTIVE}"
    )
    return SYSTEM_PROMPT, user_prompt


def _plan_block(planned_items: list[dict[str, Any]] | None) -> str:
    if not planned_items:
        return ""
    plan_lines = "\n".join(
        f"{i}. topic={it.get('topic', '')} | concept_to_test={it.get('concept_to_test', '')}"
        for i, it in enumerate(planned_items, start=1)
    )
    return (
        f"## Question Plan\n"
        f"The concepts below guide which terms and ideas to use. Select answer terms that "
        f"test these concepts. Do not introduce concepts unrelated to the plan.\n{plan_lines}\n\n"
    )


def build_fitb_bank_prompt(
    count: int,
    context: str,
    difficulty: str = "mix",
    model_number: int = 1,
    planned_items: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Stage 1: choose the correct answer terms + exactly 2 distractors."""
    rules = FILL_IN_THE_BLANK_BANK_RULES.format(count=count)
    user_prompt = (
        f"Create exactly {count} correct answer terms for a Fill-in-the-Blank section.\n\n"
        "Read the full selected source content below FIRST. It is the ONLY knowledge source "
        "for the terms. Choose terms that are grounded in the source.\n\n"

        f"## Selected Source Content\n{context}\n\n"

        f"{_plan_block(planned_items)}"

        f"## Word Bank Rules\n{rules}\n\n"

        f"## Required JSON Format (use exactly these field names)\n"
        f"{FILL_IN_THE_BLANK_BANK_SCHEMA}\n\n"

        f"## Example of a valid output\n{FILL_IN_THE_BLANK_BANK_EXAMPLE}\n\n"

        f"{_difficulty_block(difficulty)}\n\n"

        f"## Exam Version\n{_version_note(model_number)}\n\n"

        f"## Task\nCreate exactly {count} correct terms plus exactly 2 distractors.\n\n"

        f"{_OUTPUT_DIRECTIVE}"
    )
    return SYSTEM_PROMPT, user_prompt


def build_fitb_items_prompt(
    count: int,
    word_bank: list[str],
    context: str,
    difficulty: str = "mix",
    model_number: int = 1,
) -> tuple[str, str]:
    """Stage 2: write numbered items using ONLY the fixed, already-shuffled Word Bank."""
    rules = FILL_IN_THE_BLANK_ITEMS_RULES.format(count=count)
    bank_line = " · ".join(word_bank)
    user_prompt = (
        f"Create exactly {count} numbered Fill-in-the-Blank items.\n\n"
        "Read the full selected source content below FIRST. It is the ONLY knowledge source "
        "for the questions.\n\n"

        f"## Selected Source Content\n{context}\n\n"

        f"## Fixed Word Bank (use ONLY these entries)\n{bank_line}\n\n"
        "- Every answer must be one of these entries.\n"
        "- Use every entry except the two distractors in at least one blank.\n"
        "- Never introduce a term that is not in the Word Bank.\n\n"

        f"## Item Rules\n{rules}\n\n"

        f"## Required JSON Format (use exactly these field names)\n"
        f"{FILL_IN_THE_BLANK_ITEMS_SCHEMA}\n\n"

        f"## Example of a valid output\n{FILL_IN_THE_BLANK_ITEMS_EXAMPLE}\n\n"

        f"{_difficulty_block(difficulty)}\n\n"

        f"## Exam Version\n{_version_note(model_number)}\n\n"

        f"## Task\nGenerate exactly {count} numbered items using only the fixed Word Bank.\n\n"

        f"{_OUTPUT_DIRECTIVE}"
    )
    return SYSTEM_PROMPT, user_prompt



OBJ_BUNDLED_SCHEMA = (
    '{\n'
    '  "mcq": {\n'
    '    "questions": [\n'
    '      {"question": "the MCQ question", "options": {"A": "...", "B": "...", '
    '"C": "...", "D": "..."}, "correct_answer": "B"}\n'
    '    ]\n'
    '  },\n'
    '  "true_false": {\n'
    '    "questions": [\n'
    '      {"statement": "the factual statement", "answer": "True"}\n'
    '    ]\n'
    '  },\n'
    '  "fill_in_the_blank": {\n'
    '    "word_bank": ["term 1", "term 2", "term 3", "distractor 1", "distractor 2"],\n'
    '    "items": [\n'
    '      {"question": "sentence with ________ for a blank", "answers": ["term 1"]}\n'
    '    ]\n'
    '  }\n'
    '}'
)

OBJ_BUNDLED_EXAMPLE = (
    '{\n'
    '  "mcq": {\n'
    '    "questions": [\n'
    '      {"question": "Which metric is the harmonic mean of precision and recall?", '
    '       "options": {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"}, '
    '       "correct_answer": "B"}\n'
    '    ]\n'
    '  },\n'
    '  "true_false": {\n'
    '    "questions": [\n'
    '      {"statement": "A confusion matrix can be used to evaluate a classifier.", '
    '"answer": "True"}\n'
    '    ]\n'
    '  },\n'
    '  "fill_in_the_blank": {\n'
    '    "word_bank": ["area", "perimeter", "length", "width", "square"],\n'
    '    "items": [\n'
    '      {"question": "To calculate the ________ of a rectangle, multiply its length by its '
    'width.", "answers": ["area"]}\n'
    '    ]\n'
    '  }\n'
    '}'
)


def build_obj_bundled_prompt(
    planned: dict[str, list[dict[str, Any]]],
    context: str,
    difficulty: str = "mix",
    model_number: int = 1,
    feedback: str = "",
) -> tuple[str, str]:
    """Build ONE prompt that returns MCQ + True/False + Fill-in-the-Blank together.

    ``planned`` maps qtype -> list of {topic, concept_to_test} plan items for the
    three objective types (the still-missing items on a retry). The FITB count is
    derived from its own plan list; the Word Bank must hold that many correct
    terms plus exactly 2 distractors. ``feedback`` lists already-accepted
    questions the model must not repeat.
    """
    mcq_planned = planned.get("mcq") or []
    tf_planned = planned.get("true_false") or []
    fitb_planned = planned.get("fill_in_the_blank") or []
    mcq_count = len(mcq_planned)
    tf_count = len(tf_planned)
    fitb_count = len(fitb_planned)

    plan_lines = []
    for qtype, label in (
        ("mcq", "Multiple Choice"),
        ("true_false", "True/False"),
        ("fill_in_the_blank", "Fill-in-the-Blank"),
    ):
        items = planned.get(qtype) or []
        if not items:
            continue
        block = f"### {label} ({len(items)} items)\n" + "\n".join(
            f"{i}. topic={it.get('topic', '')} | concept_to_test={it.get('concept_to_test', '')}"
            for i, it in enumerate(items, start=1)
        )
        plan_lines.append(block)
    plan_block = (
        "## Question Plan\nGenerate EXACTLY the planned counts below in their "
        "respective sections. Test ONLY the stated topics and concepts; do not "
        "introduce unplanned concepts.\n\n"
        + "\n\n".join(plan_lines)
        if plan_lines
        else ""
    )

    bank_rules = FILL_IN_THE_BLANK_BANK_RULES.format(count=fitb_count)
    items_rules = FILL_IN_THE_BLANK_ITEMS_RULES.format(count=fitb_count)

    user_prompt = (
        "Create the objective sections of an exam in ONE response: Multiple "
        f"Choice ({mcq_count}), True/False ({tf_count}), and Fill-in-the-Blank "
        f"({fitb_count} items with a shared Word Bank).\n\n"
        "Read the full selected source content below FIRST. It is the ONLY knowledge "
        "source for every question, answer, option, True/False decision, and FITB term. "
        "Do not copy its wording and do not refer to it inside the questions.\n\n"

        f"## Selected Source Content\n{context}\n\n"

        f"{plan_block}\n\n"

        f"## Multiple Choice Rules\n{MCQ_RULES}\n\n"
        f"## True/False Rules\n{TRUE_FALSE_RULES}\n\n"
        f"## Word Bank Rules\n{bank_rules}\n\n"
        f"## Fill-in-the-Blank Item Rules\n{items_rules}\n\n"

        f"## Required JSON Format (use exactly these field names)\n"
        f"{OBJ_BUNDLED_SCHEMA}\n\n"

        f"## Example of a valid output\n{OBJ_BUNDLED_EXAMPLE}\n\n"

        f"{_difficulty_block(difficulty)}\n\n"

        f"## Exam Version\n{_version_note(model_number)}\n\n"

        f"## Task\nReturn one JSON object with three keys: 'mcq' ({mcq_count} questions), "
        f"'true_false' ({tf_count} statements), and 'fill_in_the_blank' (a 'word_bank' of "
        f"exactly {fitb_count} correct terms plus 2 distractors, and {fitb_count} numbered "
        f"'items' answered ONLY from that Word Bank, distractors never used).\n\n"

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

        f"## Exam Version\n{_version_note(model_number)}\n\n"

        f"## Task\nReturn ONLY the repaired JSON array matching the expected schema, "
        f"preserving all valid content.\n\n"

        f"{_OUTPUT_DIRECTIVE}"
    )
    return SYSTEM_PROMPT, user_prompt
