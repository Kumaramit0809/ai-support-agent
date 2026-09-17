#!/usr/bin/env bash
# One-command reproduction of headline results.
# Usage: bash scripts/run_pipeline.sh --brand AmazonHelp --sample 3000 [--offline] [--model MODEL]
set -euo pipefail

cd "$(dirname "$0")/.."

BRAND="AmazonHelp"
SAMPLE=500
MODEL="gemini-3.8-flash"
OFFLINE=""
DATA_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --brand) BRAND="$2"; shift 2 ;;
    --sample) SAMPLE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --offline) OFFLINE="1"; shift 1 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

DATA_ARG=()
if [[ -n "$DATA_PATH" ]]; then DATA_ARG=(--data-path "$DATA_PATH"); fi

echo "== 1/6 Loading + threading data for brand=$BRAND =="
python3 src/load_data.py --brand "$BRAND" "${DATA_ARG[@]}" --out data/processed/resolution_pairs.csv

echo "== 2/6 Building intent taxonomy =="
TAXONOMY_FLAGS=()
if [[ -n "$OFFLINE" || -z "${GEMINI_API_KEY:-}" ]]; then TAXONOMY_FLAGS+=(--offline); fi
python3 src/build_taxonomy.py --brand "$BRAND" --pairs-csv data/processed/resolution_pairs.csv \
  --out data/processed/taxonomy.json --model "$MODEL" "${TAXONOMY_FLAGS[@]}"

echo "== 3/6 Running baselines (trivial + simple) =="
python3 eval/baselines.py --pairs-csv data/processed/resolution_pairs.csv \
  --taxonomy data/processed/taxonomy.json --sample "$SAMPLE" \
  --out-trivial data/processed/baseline_trivial.csv --out-simple data/processed/baseline_simple.csv

if [[ -n "$OFFLINE" || -z "${GEMINI_API_KEY:-}" ]]; then
  echo "== 4-6/6 SKIPPED: GEMINI_API_KEY not set / --offline given =="
  echo "Baselines ran successfully. Get a free key at https://aistudio.google.com/app/apikey,"
  echo "set GEMINI_API_KEY in .env, and re-run without --offline to run the full LLM agent,"
  echo "judge, and golden-set scoring."
  exit 0
fi

echo "== 4/6 Running the LLM agent (classify -> retrieve -> draft -> escalate) =="
python3 -m src.pipeline --brand "$BRAND" "${DATA_ARG[@]}" --taxonomy data/processed/taxonomy.json \
  --sample "$SAMPLE" --model "$MODEL" --out data/processed/agent_outputs.csv

echo "== 5/6 LLM-as-judge reply quality scoring =="
python3 eval/llm_judge.py --outputs-csv data/processed/agent_outputs.csv --brand "$BRAND" \
  --model "$MODEL" --out data/processed/judge_scores.csv

if [[ -f eval/golden_set.csv ]]; then
  echo "== 6/6 Scoring against golden set =="
  {
    echo "# Results — brand=$BRAND, sample=$SAMPLE, model=$MODEL"
    echo
    echo '```'
    python3 eval/metrics.py --golden-csv eval/golden_set.csv --pred-csv data/processed/agent_outputs.csv --label "LLM Agent"
    echo
    python3 eval/metrics.py --golden-csv eval/golden_set.csv --pred-csv data/processed/baseline_trivial.csv --label "Trivial Baseline"
    echo
    python3 eval/metrics.py --golden-csv eval/golden_set.csv --pred-csv data/processed/baseline_simple.csv --label "Simple Baseline"
    echo '```'
    echo
    echo "Mean LLM-judge overall score (agent): $(python3 -c "import pandas as pd; print(round(pd.read_csv('data/processed/judge_scores.csv')['overall'].mean(),2))")"
  } | tee report/results.md
else
  echo "== 6/6 SKIPPED: eval/golden_set.csv not found =="
  echo "Run 'python3 eval/labeling_tool.py' first (see eval/README.md), then re-run this script"
  echo "to get the full comparison against the golden set."
fi

echo "Done."
