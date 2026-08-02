from __future__ import annotations

import json
import re


class JSONExtractionError(ValueError):
    pass


REPAIR_SYSTEM_PROMPT = (
    "You are a JSON repair assistant. Fix the provided broken JSON so it is "
    "valid and parseable. Preserve the original content, structure, and meaning. "
    "Change only syntax. Return ONLY the corrected raw JSON object - no markdown "
    "fences, no commentary, no text before or after."
)


def build_repair_prompt(broken_text: str, error_message: str) -> str:
    """Build the user prompt that asks the LLM to fix malformed JSON."""
    return (
        "The JSON below failed to parse.\n\n"
        f"Parser error:\n{error_message}\n\n"
        f"Broken JSON:\n{broken_text}\n\n"
        "Fix the JSON and return ONLY the corrected raw JSON object. "
        "Do not add explanations, fences, or any text before or after the JSON."
    )


def extract_json(text: str) -> object:
    """Extract and parse the first JSON object/array found in model output."""
    cleaned = text.strip()

    # Strip markdown code fences.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # Locate the outermost JSON object or array.
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
    if not match:
        raise JSONExtractionError("No JSON object or array found in model output")

    candidate = match.group(1)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JSONExtractionError(f"Invalid JSON in model output: {exc}") from exc


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value.get("questions") or []
    return []
