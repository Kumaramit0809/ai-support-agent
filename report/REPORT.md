# Report — AI Support Agent for AmazonHelp

## 1. Problem framing

**What "good" means for AmazonHelp:** The agent must never fabricate order-specific
facts (refund amounts, delivery dates, tracking status) not grounded in retrieved
historical context or the customer's own message. It must correctly route
account-security, fraud, legal-threat, or repeated-contact messages to a human every
time — zero tolerance for false negatives on that class, even at the cost of
over-escalating borderline cases. It should classify and draft a passable first-pass
reply for the ~11 real intent categories discovered in the brand's actual Twitter
support traffic.

**What I chose not to build:** Multi-turn conversation memory beyond the single most
recent customer message plus retrieved grounding examples. Actual refund/account
execution — the system only recommends escalation or drafts text, never performs an
account action. Native handling for every language present in the traffic: French and
Spanish got dedicated taxonomy clusters, but Portuguese and German (also present in
real AmazonHelp traffic) did not — a scope cut under time pressure, not an
architectural limitation.

## 2. Results vs. baselines

Evaluated on n=27 real, hand-labeled messages (see §4 for why n=27, not 150-250).

| System | Intent Accuracy | Intent Macro-F1 | Escalation Precision | Escalation Recall | Escalation F1 |
|---|---|---|---|---|---|
| Trivial baseline | 0.074 | 0.020 | 0.000 | 0.000 | 0.000 |
| Simple baseline (TF-IDF 1-NN) | 0.000 | 0.000 | 0.500 | 0.267 | 0.348 |
| **LLM agent** | **0.667** | **0.582** | **0.733** | **0.733** | **0.733** |

The agent clearly beats both baselines on every metric. The trivial baseline
(always guess the most common intent, never escalate) is near-useless by
construction. The simple baseline's 0.000 intent accuracy is worth unpacking: it
does 1-nearest-neighbor retrieval over historical messages but has no independent
way to assign one of our 11 taxonomy labels to the neighbor it finds — it was scored
against real intent ids it structurally cannot produce, making this a weak baseline
for intent specifically, though it's a fairer comparison for escalation (keyword
rules only), where it does reasonably (F1 0.348 vs. agent's 0.733).

**LLM-judge reply-quality scoring was not completed** — see §4.

## 3. Failure analysis — top 5 failure modes

1. **Order number mistaken for a Twitter handle in a drafted reply.** One reply
   opened with `@408-6673389-3841925 I'm sorry to hear about the delay...` — the
   model pattern-matched the retrieved grounding example's "@handle" opening
   convention and applied it to what was actually an order number embedded in the
   customer's message. Hypothesis: retrieval examples bias generated *format* too
   strongly without validating whether the leading token is a real handle.

2. **Escalation hard-rule matched for the wrong reason.** A customer wrote *"Why
   Fraud Offers?"* referring to a marketing complaint, not account compromise. Our
   keyword rule (`account_security` pattern includes "fraud") escalated it anyway —
   the right outcome by coincidence (it's a legitimate complaint deserving human
   review), but the *stated reason* ("matched hard rule: account_security") was
   factually wrong. An auditor reading escalation reasons would be misled about why.

3. **Escalation recall gap on subtly frustrated messages.** During manual
   relabeling, at least 2 of 27 messages were judged escalate=True by a human where
   the agent said False — e.g. a sarcastic Black Friday complaint, and a case where
   a customer's explicit delivery request was ignored. The LLM judgment layer
   appears to under-weight sarcasm/implicit frustration relative to overt anger or
   explicit keywords.

4. **Non-English leakage past the English-only filter.** Portuguese and German
   messages passed our ASCII-ratio-based English filter (they share the Latin
   alphabet with English) and were processed by English-tuned prompts, landing in
   a generic `general_inquiry_and_error` bucket rather than being flagged for
   translation/human routing.

5. **Small per-category support inflates noise.** Several real taxonomy intents
   (e.g. `missing_or_misdelivered_package`) had 0 examples in the n=27 eval sample
   despite being genuine clusters in the full 158,711-message taxonomy — a direct
   consequence of small forced sample size (§4), not a taxonomy flaw.

## 4. What is misleading about my headline number?

- **n=27, not 150-250.** I hand-labeled 150 real examples as required, but a
  terminal input-buffering bug corrupted ~87 of them (answers bled across CSV
  columns) partway through labeling. Rather than submit corrupted labels, I
  discarded the bad rows and relabeled — carefully, with strict input validation —
  only the 27 messages the agent had already successfully processed, so the golden
  set and agent predictions would actually overlap. 27 is small; confidence
  intervals on these metrics are wide, and category-level precision/recall for rare
  intents (§3.5) should not be trusted.
- **Why only 27 agent predictions exist at all:** across today's session I hit
  hard free-tier request/token quotas on **seven different LLM models across two
  providers** (Gemini: 3.8-flash, 3.5-flash-lite, 2.5-flash [deprecated mid-session],
  3.6-flash; Groq: gpt-oss-120b, gpt-oss-20b, compound-mini). This dominated the
  day's engineering time and directly capped how much real data I could generate,
  not a limitation of the pipeline logic itself, which runs cleanly whenever quota
  is available.
- **The simple baseline's 0.000 intent accuracy is an artifact of baseline design**,
  not evidence the agent is dramatically better at intent than a "real" simple
  system would be — see §2.
- **No LLM-judge reply-quality score is reported** (groundedness/helpfulness/tone/
  correctness). The harness is fully built and tested, but every attempt to run it
  today hit the same quota walls before completing even 30 rows. Omitted rather
  than reported on a broken partial run.
- **The English-only filter is a blunt ASCII-ratio heuristic** — it caught ~94% of
  non-English AmazonHelp traffic but missed Portuguese/German (§3.4), so "English
  agent" performance here is not purely English-only in practice.

## 5. What I'd do next with one more week

1. Move to a paid tier or self-hosted open model to remove the free-tier volume
   ceiling that consumed most of today's time, and re-run on the full 150-250
   golden set already labeled in spirit (the labeling *process* works, it needs a
   clean matching agent run).
2. Complete LLM-judge scoring + human-judge agreement validation (Spearman/kappa)
   — fully coded, never run to completion.
3. Fix the retrieval-format leakage causing order numbers to look like @handles
   (§3.1) with an explicit prompt instruction against opening a reply with an
   order/tracking-number-shaped token.
4. Make the escalation hard-rule reason-aware, not just keyword-aware (§3.2) —
   e.g. require "fraud" to co-occur with an account/security term before firing
   the `account_security` rule, so stated reasons stay accurate even when outcomes
   are already correct.
5. Add explicit non-English routing: either hard-escalate any non-ASCII-heavy
   message by default, or build dedicated taxonomies for the 3-4 languages
   (English, Spanish, French, Portuguese, German) actually present in real traffic.

---
See `report/decision_log.md` for the full list of engineering decisions behind this system.