from __future__ import annotations

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
    "- Questions should resemble questions written by a university professor.\n\n"

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
    "- Avoid ambiguous questions with multiple possible correct answers.\n"
    "- Avoid overly easy questions where the answer can be guessed from the options.\n"
    "- Avoid overly specific questions about small details unless they represent an important concept.\n\n"

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

MCQ_RULES = (
    "- Each question has EXACTLY four options (A, B, C, D) and EXACTLY one correct answer.\n"
    "- Distractors must be plausible but clearly wrong.\n"
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
    '- Provide a concise "reference_answer" (2-4 sentences) grounded in the context.\n'
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


def build_prompt(
    question_type: str, count: int, context: str
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the given question type."""
    if question_type == "mcq":
        rules, schema, example = MCQ_RULES, MCQ_SCHEMA, MCQ_EXAMPLE
        type_name = "Multiple Choice (MCQ)"
    elif question_type == "true_false":
        rules, schema, example = TRUE_FALSE_RULES, TRUE_FALSE_SCHEMA, TRUE_FALSE_EXAMPLE
        type_name = "True/False"
    else:
        rules, schema, example = SHORT_ANSWER_RULES, SHORT_ANSWER_SCHEMA, SHORT_ANSWER_EXAMPLE
        type_name = "Short Answer"

    user_prompt = (
        f"Create exactly {count} {type_name} exam question(s).\n\n"
        "The questions must be generated from the concepts contained in the document "
        "context below. The context is only the knowledge source; do not copy its wording "
        "or refer to it in the questions.\n\n"

        f"## Question Type Rules\n{rules}\n\n"

        f"## Required JSON Format (use exactly these field names)\n{schema}\n\n"

        f"## Example of a valid output\n{example}\n\n"

        f"## Document Knowledge Context\n{context}\n\n"

        f"{_OUTPUT_DIRECTIVE}"
    )
    return SYSTEM_PROMPT, user_prompt
