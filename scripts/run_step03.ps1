$ErrorActionPreference = "Stop"

Write-Host "[V6 Step 3] Testing the cumulative project with the current V6 Python..."
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) {
    throw "The cumulative tests failed (exit code $LASTEXITCODE). Step 3 was not started."
}

Write-Host "[V6 Step 3] Running exact official-EnCOT source/runtime/bridge gate..."
python main_v6.py --stage step3
if ($LASTEXITCODE -ne 0) {
    throw "Step 3 failed (exit code $LASTEXITCODE). Do not continue to Step 4."
}

$ReturnZip = Join-Path (Get-Location) "v6_step03_results.zip"
if (-not (Test-Path -LiteralPath $ReturnZip -PathType Leaf)) {
    throw "Step 3 passed without producing the required return ZIP: $ReturnZip"
}
Write-Host "[V6 Step 3] COMPLETE. Return this file:"
Write-Host $ReturnZip
