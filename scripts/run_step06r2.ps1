$ErrorActionPreference = "Stop"

Write-Host "[V6.1 Step 6R2] Testing the cumulative project with the current V6 Python..."
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) {
    throw "The cumulative tests failed (exit code $LASTEXITCODE). Step 6R2 was not started."
}

$ReturnZip = Join-Path $PSScriptRoot "..\v6_step06r2_results.zip"
Write-Host "[V6.1 Step 6R2] Running disclosed spent-validation recovery selection..."
python main_v6.py --stage step6r2
$StageExit = $LASTEXITCODE

if (-not (Test-Path $ReturnZip)) {
    throw "Step 6R2 produced no return ZIP: $ReturnZip"
}

Write-Host "[V6.1 Step 6R2] COMPLETE. Return this file and the full console output:"
Write-Host (Resolve-Path $ReturnZip)
if ($StageExit -ne 0) {
    throw "Step 6R2 failed or was blocked (exit code $StageExit). Return the ZIP and keep test/baselines/SOTA sealed."
}
