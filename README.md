# Hiver SDE Intern Assignment — AI Support Agent

An AI customer-support agent built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, targeting a single brand end-to-end:

1. **Classify** each incoming customer message into a data-derived intent taxonomy.
2. **Draft a reply** grounded in how the brand has historically resolved similar issues (retrieval over real resolved threads).
3. **Decide** auto-handle vs. escalate, with a stated reason.

> **Status note:** This is a fully working, end-to-end pipeline run against the real
> Kaggle dataset for the `AmazonHelp` brand. Due to free-tier LLM rate limits
> encountered across multiple providers/models during development (see
> `report/decision_log.md` for the full story), the final reported results are on a
> smaller real sample (n=27) than originally targeted, rather than the full 150-250.
> The golden evaluation set was hand-labeled by me — see `eval/golden_set_final.csv`
> and `report/REPORT.md` for real results, failure analysis, and honest limitations.
> LLM calls use the Groq API (`GROQ_API_KEY` in `.env`).

## Target brand

Default: `AmazonHelp` (large volume, clean multi-turn threads, resolved issues cover
account, orders, refunds, delivery — good intent diversity). Change with `--brand`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY=... (free key: https://console.groq.com/keys)
```

## Get the real dataset

1. Download `twcs.csv` from Kaggle: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
   (free account required; "Download" button on the dataset page).
2. Unzip and place `twcs.csv` at `data/raw/twcs.csv`.

Without this file, every command below automatically falls back to
`data/sample_synthetic.csv` (~60 rows, same schema) so the pipeline is still runnable and
reviewable — but headline numbers from the fixture are **not** meant to be reported as results.

## Reproducing results

```bash
python src/load_data.py --brand AmazonHelp --english-only --out data/processed/resolution_pairs.csv
python src/build_taxonomy.py --brand AmazonHelp --pairs-csv data/processed/resolution_pairs.csv --out data/processed/taxonomy.json
python -m src.pipeline --brand AmazonHelp --english-only --taxonomy data/processed/taxonomy.json --sample 50 --model groq/compound-mini --out data/processed/agent_outputs.csv
python eval/baselines.py --pairs-csv data/processed/resolution_pairs.csv --taxonomy data/processed/taxonomy.json --golden-csv eval/golden_set_final.csv
python eval/metrics.py --golden-csv eval/golden_set_final.csv --pred-csv data/processed/agent_outputs.csv --label "LLM Agent"
```

On Windows, use `scripts\run_pipeline.ps1` for a scripted version of the load+taxonomy+baselines
steps, or `run_agent.ps1` as a quick wrapper around the agent step alone.

**Runtime is dominated by LLM provider rate limits**, which vary significantly by provider,
model, and account tier. This project hit hard free-tier quota walls on seven different
models across two providers (Gemini and Groq) during development — see
`report/decision_log.md` for the full, honest account of what happened and why the final
sample size is n=27 rather than the originally planned 150-250.

Offline / no API key: `python src/build_taxonomy.py ... --offline` and
`python eval/baselines.py ...` run with zero LLM calls, for a quick smoke test.

## Repo layout