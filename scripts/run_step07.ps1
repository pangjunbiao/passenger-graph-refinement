$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

Write-Host "[V6.1 Step 7] Testing the cumulative project with the current V6 Python..."
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) {
    throw "The cumulative tests failed (exit code $LASTEXITCODE). Step 7 was not started."
}

Write-Host "[V6.1 Step 7] Running final 695-document, ten-seed pre-test refit and freeze..."
python main_v6.py --stage step7
$StageExit = $LASTEXITCODE
$ReturnZip = Join-Path $ProjectRoot "v6_step07_results.zip"
if (-not (Test-Path $ReturnZip)) {
    throw "Step 7 produced no return ZIP: $ReturnZip"
}
if ($StageExit -ne 0) {
    throw "Step 7 failed or was blocked (exit code $StageExit). Return the ZIP and keep test/comparators sealed."
}

Write-Host "[V6.1 Step 7] COMPLETE. Return this file and the full console output:"
Write-Host $ReturnZip
