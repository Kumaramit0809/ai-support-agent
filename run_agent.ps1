param([int]$Sample = 150, [string]$Model = "openai/gpt-oss-20b")
python -m src.pipeline --brand AmazonHelp --english-only --taxonomy data/processed/taxonomy.json --sample $Sample --model $Model --out data/processed/agent_outputs.csv
