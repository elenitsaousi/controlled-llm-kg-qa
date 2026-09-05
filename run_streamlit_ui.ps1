$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

Write-Host "Starting True Demand KG QA Streamlit UI..."
Write-Host "This UI does not require Node.js or npm."
Write-Host "Open http://localhost:8501 if the browser does not open automatically."

python -m streamlit run app.py --server.port 8501
