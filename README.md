# Hiver SDE Intern Assignment — AI Support Agent

An AI customer-support agent built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, targeting a single brand end-to-end:

1. **Classify** each incoming customer message into a data-derived intent taxonomy.
2. **Draft a reply** grounded in how the brand has historically resolved similar issues (retrieval over real resolved threads).
3. **Decide** auto-handle vs. escalate, with a stated reason.

> **Status note (read me first):** This repo is fully wired and runnable end-to-end, but the
> real Kaggle CSV could not be fetched programmatically in the environment this repo was
> assembled in (it requires an authenticated Kaggle account). Everything below — pipeline,
> retrieval, classifier, escalation policy, baselines, eval harness, LLM-judge, report,
> decision log — is real, working code, smoke-tested against `data/sample_synthetic.csv`
> (a small fixture in the *exact* real schema). Drop the real `twcs.csv` into `data/raw/`
> and every command below produces genuine numbers instead of fixture numbers. The
> **golden evaluation set is intentionally left for you to hand-label** — see
> [`eval/README.md`](eval/README.md) — because the assignment requires real human judgment,
> which isn't something that can be legitimately pre-filled.

## Target brand

Default: `AmazonHelp` (large volume, clean multi-turn threads, resolved issues cover
account, orders, refunds, delivery — good intent diversity). Change with `--brand`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY=... (free key: https://aistudio.google.com/app/apikey)
```

## Get the real dataset

1. Download `twcs.csv` from Kaggle: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
   (free account required; "Download" button on the dataset page, or `kaggle datasets download -d thoughtvector/customer-support-on-twitter`
   if you have the Kaggle CLI configured).
2. Unzip and place `twcs.csv` at `data/raw/twcs.csv`.

Without this file, every command below automatically falls back to
`data/sample_synthetic.csv` (~60 rows, same schema) so the pipeline is still runnable and
reviewable — but headline numbers from the fixture are **not** meant to be reported as results.

## Reproduce headline results in under 15 minutes

```bash
bash scripts/run_pipeline.sh --brand AmazonHelp --sample 3000
```

This single command:
1. Loads and threads the raw tweets for the brand (`src/load_data.py`)
2. Builds the intent taxonomy from the data (`src/build_taxonomy.py`) → `data/processed/taxonomy.json`
3. Runs the full agent (classify → retrieve → draft → escalate) over a held-out sample
   (`src/pipeline.py`) → `data/processed/agent_outputs.csv`
4. Runs both baselines (`eval/baselines.py`)
5. Scores everything against `eval/golden_set.csv` once you've labeled it — automated
   metrics (`eval/metrics.py`) + LLM-as-judge (`eval/llm_judge.py`)
6. Prints a results table to stdout and writes `report/results.md`

Runtime is dominated by LLM calls; with `--sample 3000` and Gemini Flash it comfortably
finishes in under 15 minutes on a single machine with default rate limits. Use
`--model gemini-3.8-flash` (default) for the best quality/cost balance, or
`--model gemini-3.5-flash-lite` for a cheaper/faster run that fits comfortably in the
free tier's rate limits.

Offline / no API key: add `--offline` to run only the TF-IDF simple baseline (no LLM calls,
seconds to run) — useful as a smoke test.

## Repo layout

```
src/
  load_data.py       # ingest twcs.csv, filter to brand, reconstruct threads
  build_taxonomy.py  # cluster customer messages -> LLM-named intent taxonomy
  classify.py         # few-shot LLM intent classifier
  retrieve.py          # TF-IDF retrieval of grounding examples (historical resolutions)
  draft_reply.py         # LLM reply generation grounded in retrieved examples
  escalate.py              # escalation policy (rules + LLM) with stated reason
  pipeline.py               # orchestrates the full agent end-to-end
eval/
  labeling_tool.py    # interactive CLI to build the golden set quickly
  golden_set.csv        # YOU fill this in (150-250 rows) — see eval/README.md
  baselines.py            # trivial baseline + simple (non-LLM) baseline
  metrics.py                # intent accuracy/F1, escalation P/R/F1
  llm_judge.py                # rubric-based LLM-as-judge for reply quality
  judge_agreement.py            # judge-vs-human agreement (Spearman + quadratic kappa)
report/
  REPORT.md            # the mandated report, with TODOs for post-run numbers
  decision_log.md         # 10-15 non-obvious decisions
scripts/
  run_pipeline.sh          # one-command reproduction
```

## Citations / borrowed code

- Dataset: Kaggle `thoughtvector/customer-support-on-twitter` (see above).
- `sklearn` TF-IDF + KMeans used for retrieval and taxonomy bootstrapping (standard library
  usage, not borrowed from a specific source).
- No external code was copied from other repos or gists. If you extend this and borrow
  something, log it in `report/decision_log.md`.
