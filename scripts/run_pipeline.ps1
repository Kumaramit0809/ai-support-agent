# One-command reproduction of headline results (PowerShell version).
# Usage: .\scripts\run_pipeline.ps1 -Brand AmazonHelp -Sample 3000 [-Offline] [-Model gemini-3.8-flash]

param(
    [string]$Brand = "AmazonHelp",
    [int]$Sample = 500,
    [string]$Model = "gemini-3.8-flash",
    [switch]$Offline,
    [string]$DataPath = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# Load .env into the current process environment, if present
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)\s*=\s*(.*)\s*$') {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
        }
    }
}

$dataArgs = @()
if ($DataPath -ne "") { $dataArgs = @("--data-path", $DataPath) }

$hasKey = -not [string]::IsNullOrEmpty($env:GEMINI_API_KEY)

Write-Host "== 1/6 Loading + threading data for brand=$Brand =="
python src/load_data.py --brand $Brand @dataArgs --out data/processed/resolution_pairs.csv

Write-Host "== 2/6 Building intent taxonomy =="
$taxonomyFlags = @()
if ($Offline -or -not $hasKey) { $taxonomyFlags = @("--offline") }
python src/build_taxonomy.py --brand $Brand --pairs-csv data/processed/resolution_pairs.csv `
    --out data/processed/taxonomy.json --model $Model @taxonomyFlags

Write-Host "== 3/6 Running baselines (trivial + simple) =="
python eval/baselines.py --pairs-csv data/processed/resolution_pairs.csv `
    --taxonomy data/processed/taxonomy.json --sample $Sample `
    --out-trivial data/processed/baseline_trivial.csv --out-simple data/processed/baseline_simple.csv

if ($Offline -or -not $hasKey) {
    Write-Host "== 4-6/6 SKIPPED: GEMINI_API_KEY not set / -Offline given =="
    Write-Host "Baselines ran successfully. Get a free key at https://aistudio.google.com/app/apikey,"
    Write-Host "set GEMINI_API_KEY in .env, and re-run without -Offline to run the full LLM agent,"
    Write-Host "judge, and golden-set scoring."
    exit 0
}

Write-Host "== 4/6 Running the LLM agent (classify -> retrieve -> draft -> escalate) =="
python -m src.pipeline --brand $Brand @dataArgs --taxonomy data/processed/taxonomy.json `
    --sample $Sample --model $Model --out data/processed/agent_outputs.csv

Write-Host "== 5/6 LLM-as-judge reply quality scoring =="
python eval/llm_judge.py --outputs-csv data/processed/agent_outputs.csv --brand $Brand `
    --model $Model --out data/processed/judge_scores.csv

if (Test-Path "eval/golden_set.csv") {
    Write-Host "== 6/6 Scoring against golden set =="
    $results = @()
    $results += "# Results -- brand=$Brand, sample=$Sample, model=$Model"
    $results += ""
    $results += '```'
    $results += (python eval/metrics.py --golden-csv eval/golden_set.csv --pred-csv data/processed/agent_outputs.csv --label "LLM Agent" | Out-String)
    $results += (python eval/metrics.py --golden-csv eval/golden_set.csv --pred-csv data/processed/baseline_trivial.csv --label "Trivial Baseline" | Out-String)
    $results += (python eval/metrics.py --golden-csv eval/golden_set.csv --pred-csv data/processed/baseline_simple.csv --label "Simple Baseline" | Out-String)
    $results += '```'
    $meanScore = python -c "import pandas as pd; print(round(pd.read_csv('data/processed/judge_scores.csv')['overall'].mean(),2))"
    $results += ""
    $results += "Mean LLM-judge overall score (agent): $meanScore"
    $results -join "`n" | Tee-Object -FilePath "report/results.md"
} else {
    Write-Host "== 6/6 SKIPPED: eval/golden_set.csv not found =="
    Write-Host "Run 'python eval/labeling_tool.py' first (see eval/README.md), then re-run this script"
    Write-Host "to get the full comparison against the golden set."
}

Write-Host "Done."
