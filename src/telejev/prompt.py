"""All model prompts in one place.

Keep every system prompt and prompt builder here so wording changes are local and
the serving path (local / SGLang / vLLM) shares exactly the same text.
"""

from __future__ import annotations

import json

# System prompt for Jev-style option readout: the model must reply with one letter.
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

# System prompt for the autoregressive baseline: emit the final decision object.
GENERATION_SYSTEM = (
    "You are a surveillance analyst. Decide from the attached image whether any "
    "person is present and which of the listed behaviours are present. Answer "
    "with only a JSON object, no explanation, no markdown."
)


def behavior_labels(criteria: list[dict]) -> list[str]:
    """Human labels of the violation criteria (everything except the person check)."""
    return [
        criterion.get("label") or criterion.get("id")
        for criterion in criteria
        if criterion.get("id") != "person"
    ]


def build_decision_payload(state, question: str, options: list[dict], letters: str) -> str:
    """User-turn JSON for a Jev-style decision: evidence + criterion + lettered options."""
    return json.dumps(
        {
            "evidence": state,
            "criterion": question,
            "options": [
                {"letter": letters[index], "description": option["description"]}
                for index, option in enumerate(options)
            ],
        },
        ensure_ascii=False,
    )


def build_generation_text(state, criteria: list[dict]) -> str:
    """User-turn text asking for ``{"has_person": 0|1, "violations": [...]}``."""
    lines = ["Evidence: " + json.dumps(state, ensure_ascii=False), "Decide from the image:"]
    for criterion in criteria:
        label = criterion.get("label") or criterion.get("id")
        field = "has_person (0 or 1)" if criterion.get("id") == "person" else f'violation "{label}"'
        lines.append(f"- {field}: {criterion['question']}")
    allowed = ", ".join(json.dumps(label, ensure_ascii=False) for label in behavior_labels(criteria))
    lines.append(f"Allowed violations: [{allowed}]")
    lines.append(
        'Respond with ONLY a JSON object exactly like {"has_person": 0, "violations": ["..."]}. '
        "has_person=1 if any person is present; list only the violation names that are present."
    )
    return "\n".join(lines)