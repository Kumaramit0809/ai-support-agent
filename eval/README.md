# Building the golden evaluation set

The assignment requires **150-250 hand-labeled examples you built yourself**, with a note
on sampling/labeling — this is intentionally not something this repo pre-fills, since that
would defeat the point of the deliverable. Here's the fastest honest path:

## 1. Generate agent predictions first (optional but recommended)

```bash
python3 -m src.pipeline --brand AmazonHelp --sample 300 --out data/processed/agent_outputs.csv
```

Having the agent's own predictions on hand while labeling lets the labeling tool show you
its guess as a *starting point to correct*, which is faster than labeling from a blank
slate — you're editing, not authoring from scratch. You still make the final call.

## 2. Run the labeling tool

```bash
python3 eval/labeling_tool.py \
    --pairs-csv data/processed/resolution_pairs.csv \
    --taxonomy data/processed/taxonomy.json \
    --agent-outputs data/processed/agent_outputs.csv \
    --target 200 \
    --out eval/golden_set.csv
```

This samples messages (stratified by message-length decile as a proxy for diversity —
see `SAMPLE_STRATEGY_NOTE` in the script, edit it if you use a different strategy), shows
you each one plus the agent's guess, and asks you to type the gold intent id, whether it
should escalate, why, and any reply-quality notes. Progress saves after every row.

Budget ~1-2 minutes per example → 200 examples is roughly 3-6 hours of focused labeling.
If you're short on time, 150 is the deliverable's floor — prioritize covering every intent
in the taxonomy at least a few times, plus the hard/ambiguous cases, over pure volume.

## 3. Score the agent (and baselines) against the golden set

```bash
python3 eval/metrics.py --golden-csv eval/golden_set.csv --pred-csv data/processed/agent_outputs.csv --label "LLM Agent"
python3 eval/metrics.py --golden-csv eval/golden_set.csv --pred-csv data/processed/baseline_trivial.csv --label "Trivial Baseline"
python3 eval/metrics.py --golden-csv eval/golden_set.csv --pred-csv data/processed/baseline_simple.csv --label "Simple Baseline"
```

## 4. Score reply quality with the LLM judge

```bash
python3 eval/llm_judge.py --outputs-csv data/processed/agent_outputs.csv --brand AmazonHelp --out data/processed/judge_scores.csv
```

## 5. Validate the judge against a human (required)

Pick ~30-50 rows from your golden set, fill in `eval/human_ratings_template.csv` by hand
(you rating groundedness/helpfulness/tone_fit/correctness 1-5 yourself), then:

```bash
python3 eval/judge_agreement.py --human-csv eval/my_human_ratings.csv --judge-csv data/processed/judge_scores.csv
```

Report the Spearman correlation and quadratic-weighted kappa in `report/REPORT.md`. If
agreement is weak (rho < ~0.4), that's a real finding to report, not a bug to hide — say
so, and note it as a limitation in the "what's misleading about my headline number" section.

`scripts/run_pipeline.sh` runs steps 1, 3, and 4 automatically once `eval/golden_set.csv`
exists; step 2 (labeling) and step 5 (human ratings) are inherently manual and stay outside
that script on purpose.
