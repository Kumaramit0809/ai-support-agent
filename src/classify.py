"""
Few-shot LLM intent classifier: given the taxonomy built by build_taxonomy.py,
classify a new customer message into exactly one intent id, with a confidence
score and short rationale.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import DEFAULT_MODEL, generate_json


@dataclass
class ClassificationResult:
    intent_id: str
    confidence: float
    rationale: str


def _format_taxonomy_for_prompt(taxonomy: dict) -> str:
    lines = []
    for it in taxonomy["intents"]:
        lines.append(f"- {it['id']}: {it['label']} — {it['description']}")
    return "\n".join(lines)


CLASSIFY_PROMPT_TEMPLATE = """You are classifying an incoming customer support message for {brand}
into exactly one of the following intents:

{taxonomy_block}

Message:
\"\"\"{message}\"\"\"

Respond with ONLY a JSON object:
{{"intent_id": "<one of the ids above, or \\"other\\" if none fit>", "confidence": <0.0-1.0>, "rationale": "<one short sentence>"}}
"""


def classify_message(message: str, taxonomy: dict, brand: str, model: str = DEFAULT_MODEL, client=None) -> ClassificationResult:
    prompt = CLASSIFY_PROMPT_TEMPLATE.format(
        brand=brand, taxonomy_block=_format_taxonomy_for_prompt(taxonomy), message=message
    )
    data = generate_json(prompt, model=model, client=client)
    return ClassificationResult(
        intent_id=data.get("intent_id", "other"),
        confidence=float(data.get("confidence", 0.0)),
        rationale=data.get("rationale", ""),
    )


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--message", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    with open(args.taxonomy) as f:
        taxonomy = json.load(f)
    result = classify_message(args.message, taxonomy, args.brand, args.model)
    print(result)
