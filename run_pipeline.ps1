Write-Host "Running precompute phase..."
.venv\Scripts\python.exe precompute.py --input data/candidates.jsonl
if ($LASTEXITCODE -ne 0) { Write-Error "Precompute failed"; exit $LASTEXITCODE }

Write-Host "Running ranking phase..."
.venv\Scripts\python.exe rank.py --input data/candidates.jsonl --output output/submission.csv --top-k 100
if ($LASTEXITCODE -ne 0) { Write-Error "Ranking failed"; exit $LASTEXITCODE }

Write-Host "Done!"
