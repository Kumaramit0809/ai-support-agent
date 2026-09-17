# Decision log

## Architecture decisions (made while building the scaffold)

1. **TF-IDF + KMeans for taxonomy bootstrapping, one LLM call to name all
   clusters** — cheaper and more data-grounded than asking an LLM to invent
   categories message-by-message.

2. **Capped taxonomy clustering to a 5,000-message random sample**, not the
   full 158,711. `silhouette_score` (used to auto-pick cluster count) is O(n²) —
   infeasible on the full corpus. Found this the hard way: the first attempt
   hung indefinitely trying to compute pairwise distances across all 158k
   messages.

3. **Held-out split between retrieval corpus and evaluation sample** so
   grounding retrieval can't trivially return the exact historical reply to the
   exact message being evaluated.

4. **Two-layer escalation policy** (hard regex rules, then LLM judgment) —
   account-security/fraud/legal-threat language is caught by cheap, auditable,
   always-on rules that override the LLM regardless of confidence, because a
   probabilistic model being wrong here is asymmetrically costly.

5. **Low intent-confidence forces escalation** rather than letting an uncertain
   classification silently flow into reply drafting.

## Data decisions (made while working with the real dataset)

6. **Filtered AmazonHelp to English-only** using a >85%-ASCII-character
   heuristic (94% of traffic passed). Cheap and fast, but a blunt instrument —
   it missed Portuguese and German, which share the Latin alphabet with
   English (see REPORT.md §3.4, §4).

7. **Golden set sampled from the agent's own successfully-processed messages**,
   not an independent random draw from the full corpus. This was forced by a
   real bug (see #10 below) — two independent random samples out of 158k
   messages essentially never overlap, which made the first golden set
   unusable for scoring against agent predictions.

## Bugs found and fixed during the session (all real, all costly)

8. **Silent failure bug in `pipeline.py`:** exceptions were caught and written
   to the output CSV but never printed to the terminal. A full 22-minute,
   300-message run completed with zero visible errors — every single row had
   actually failed (the model had been deprecated mid-session). Fixed by
   printing every caught exception to stderr immediately.

9. **Data-loss bug in `pipeline.py` and `llm_judge.py`:** both only wrote their
   output CSV once, at the very end of the full run. Any Ctrl+C or crash mid-run
   lost 100% of already-paid-for API work. Fixed by writing the CSV
   incrementally after every single row.

10. **Golden-set / agent-output ID mismatch:** the golden set (150 messages)
    and the agent's output (a separate random sample) shared zero message IDs,
    so `eval/metrics.py` had nothing to score. Fixed by writing
    `eval/eval_on_golden.py`, which runs the agent specifically against the
    golden set's exact message IDs instead of a fresh random sample.

11. **CSV encoding crash:** the labeling tool's file write used the system
    default encoding (cp1252 on Windows), which can't represent emoji or
    accented characters present in real customer messages. Crashed mid-session
    on a message containing both. Fixed by forcing UTF-8 on the file handle.

12. **Golden-set column corruption:** ~87 of 150 hand-labeled rows had
    answers bled across the wrong CSV columns (reason text ending up in the
    intent-id field), most likely from terminal input buffering during
    interactive labeling. Detected via `dtype=str` + manual inspection;
    recovered by discarding corrupted rows and building a stricter relabeling
    tool that validates every answer against the exact valid-intent list
    before accepting it, applied to the 24-25 messages that also had real
    agent predictions.

## Provider/model decisions (the day's biggest time sink)

13. **Started on Anthropic's API, switched to Google Gemini, then to Groq.**
    Across the session, seven different free-tier models were tried, and every
    one hit a hard request-per-day, token-per-day, or request-per-minute quota
    (Gemini 3.8-flash, 3.5-flash-lite, 2.5-flash [deprecated mid-session, a
    separate surprise], 3.6-flash; Groq gpt-oss-120b, gpt-oss-20b,
    compound-mini). This consumed most of the day's engineering time and
    directly determined the final n=27 sample size — not a limitation of the
    pipeline's logic, which runs cleanly whenever quota is available (proven
    repeatedly on 20-300 message batches across different models).

14. **Final model used for the reported results: `groq/compound-mini`.**
    Chosen last because it was the only model with untouched quota late in the
    session, not for any quality reason over the others tried.

15. **LLM-judge reply-quality scoring was scoped out of the final results**,
    not fabricated on a partial/broken run, once it became clear every attempt
    would hit the same quota wall before completing. The harness itself
    (`eval/llm_judge.py`, `eval/judge_agreement.py`) is fully built and unit-
    tested against synthetic data — it simply never got a clean full run
    against real quota headroom today.