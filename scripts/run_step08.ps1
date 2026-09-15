$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

Write-Host "[V6.1 Step 8] Testing the cumulative project with the current V6 Python..."
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) {
    throw "The cumulative tests failed (exit code $LASTEXITCODE). Step 8 was not started and the test remains sealed."
}

Write-Host "[V6.1 Step 8 R2] Running the audited 77-row frozen full-model and paired-ablation confirmation..."
python main_v6.py --stage step8
$StageExit = $LASTEXITCODE
$ReturnZip = Join-Path $ProjectRoot "v6_step08_results.zip"
if (-not (Test-Path $ReturnZip)) {
    throw "Step 8 produced no return ZIP: $ReturnZip"
}
if ($StageExit -ne 0) {
    throw "Step 8 R2 failed an integrity check (exit code $StageExit). Return the ZIP; preserve every output and do not retune."
}

Write-Host "[V6.1 Step 8 R2] COMPLETE. Return this file and the full console output:"
Write-Host $ReturnZip
