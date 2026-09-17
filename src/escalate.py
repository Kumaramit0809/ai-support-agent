"""
Decide whether a message should be auto-handled or escalated to a human,
with a stated reason. Two-layer policy:

  1. Hard rules (cheap, deterministic, auditable) that force escalation
     regardless of what the LLM thinks — e.g. account security/fraud,
     legal/regulatory language, self-harm mentions, or low classifier
     confidence. These exist because for a real support agent you do not
     want a probabilistic model to be the only thing standing between a
     compromised account and auto-handling.
  2. An LLM judgment call for the remaining cases, which also has to state
     a reason — used for judgment-heavy cases the rules don't cover
     (repeated complaints suggesting churn risk, ambiguous wording, etc.)

This mirrors how the report's "what's misleading about the headline number"
section should discuss precision/recall tradeoffs: hard rules trade recall
of "should have auto-handled" for higher precision on "never wrongly
auto-handle something dangerous."
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HARD_ESCALATION_PATTERNS = {
    "account_security": re.compile(r"\b(hacked|unauthorized|someone (used|accessed)|didn.t (make|authorize)|fraud(ulent)?)\b", re.I),
    "legal_threat": re.compile(r"\b(lawyer|sue|lawsuit|legal action|attorney|bbb complaint|fcc complaint)\b", re.I),
    "self_harm_or_crisis": re.compile(r"\b(kill myself|suicide|self.?harm)\b", re.I),
    "explicit_human_request": re.compile(r"\b(talk to a human|speak to a (real )?person|human agent|not a bot)\b", re.I),
}

LOW_CONFIDENCE_THRESHOLD = 0.55
REPEATED_COMPLAINT_TURN_THRESHOLD = 2  # prior_customer_turns from load_data.py


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str
    source: str  # "hard_rule" | "low_confidence" | "repeated_complaint" | "llm_judgment"


def rule_based_check(message: str, prior_customer_turns: int = 0) -> EscalationDecision | None:
    for name, pattern in HARD_ESCALATION_PATTERNS.items():
        if pattern.search(message):
            return EscalationDecision(escalate=True, reason=f"Matched hard escalation rule: {name}", source="hard_rule")
    if prior_customer_turns >= REPEATED_COMPLAINT_TURN_THRESHOLD:
        return EscalationDecision(
            escalate=True,
            reason=f"Thread already has {prior_customer_turns} prior turns — repeated contact suggests the automated reply path hasn't resolved it and churn risk is elevated.",
            source="repeated_complaint",
        )
    return None


LLM_ESCALATION_PROMPT = """You are deciding whether a customer support message for {brand} should be
auto-handled by an AI agent or escalated to a human agent.

Escalate if: the situation involves money/goodwill decisions beyond a standard
policy, strong customer frustration that a templated reply won't defuse,
ambiguity that could lead to a wrong automated action, or anything a
reasonable support lead would want a human to review before responding.
Otherwise, auto-handle.

Classified intent: {intent_label}
Classifier confidence: {confidence}

Message:
\"\"\"{message}\"\"\"

Respond with ONLY a JSON object:
{{"escalate": true/false, "reason": "<one short sentence stating why>"}}
"""


def llm_judgment_check(message: str, brand: str, intent_label: str, confidence: float, model: str, client=None) -> EscalationDecision:
    from src.llm_client import DEFAULT_MODEL, generate_json

    prompt = LLM_ESCALATION_PROMPT.format(brand=brand, intent_label=intent_label, confidence=confidence, message=message)
    data = generate_json(prompt, model=model or DEFAULT_MODEL, client=client)
    return EscalationDecision(escalate=bool(data.get("escalate", False)), reason=data.get("reason", ""), source="llm_judgment")


def decide_escalation(
    message: str,
    brand: str,
    intent_label: str,
    confidence: float,
    prior_customer_turns: int,
    model: str = None,
    client=None,
) -> EscalationDecision:
    hard = rule_based_check(message, prior_customer_turns)
    if hard is not None:
        return hard
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return EscalationDecision(
            escalate=True,
            reason=f"Intent classification confidence {confidence:.2f} is below the {LOW_CONFIDENCE_THRESHOLD} threshold — too uncertain to auto-handle safely.",
            source="low_confidence",
        )
    return llm_judgment_check(message, brand, intent_label, confidence, model, client)
