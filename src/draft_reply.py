"""
Draft a reply to a customer message, grounded in retrieved historical
resolutions for similar past issues (see retrieve.py). The model is
instructed to follow the brand's tone/patterns from the examples and to
avoid inventing specifics (order numbers, refund amounts, policy claims)
that aren't supported by the examples or the incoming message itself.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.llm_client import DEFAULT_MODEL, generate_text
from src.retrieve import GroundingExample


@dataclass
class DraftedReply:
    reply_text: str
    grounding_used: list[str]


DRAFT_PROMPT_TEMPLATE = """You are drafting a customer support reply for {brand}'s Twitter support account,
in response to the message below. Match the brand's real historical tone and
structure shown in the examples (concise, empathetic opener, concrete next
step, sign-off style if the examples show one).

Rules:
- Do NOT invent specific facts (refund amounts, order status, ETAs) that
  aren't grounded in the examples or stated by the customer.
- If the issue needs account-specific lookup, direct the customer to DM/contact
  the appropriate channel, as the examples do.
- Keep it to 1-3 sentences, consistent with the brand's real reply length.

Classified intent: {intent_label} — {intent_description}

Similar past resolutions (for tone and pattern only, not facts to copy verbatim):
{examples_block}

Customer message:
\"\"\"{message}\"\"\"

Respond with ONLY the reply text, nothing else.
"""


def _format_examples(examples: list[GroundingExample]) -> str:
    if not examples:
        return "(no similar past resolutions found)"
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"{i}. Customer: {ex.customer_text}\n   Brand reply: {ex.brand_reply_text}")
    return "\n".join(lines)


def draft_reply(
    message: str,
    brand: str,
    intent_label: str,
    intent_description: str,
    examples: list[GroundingExample],
    model: str = DEFAULT_MODEL,
    client=None,
) -> DraftedReply:
    prompt = DRAFT_PROMPT_TEMPLATE.format(
        brand=brand,
        intent_label=intent_label,
        intent_description=intent_description,
        examples_block=_format_examples(examples),
        message=message,
    )
    text = generate_text(prompt, model=model, client=client)
    return DraftedReply(reply_text=text, grounding_used=[e.customer_text for e in examples])
