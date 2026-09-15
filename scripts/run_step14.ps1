$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

Write-Host "[V6.1 Step 14 R3] Testing the cumulative project..."
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) {
    throw "Cumulative tests failed. Step 14 was not started."
}

Write-Host "[V6.1 Step 14 R3] Reporting frozen pre-test sensitivity for q, rho, and decoder calibration..."
python main_v6.py --stage step14
if ($LASTEXITCODE -ne 0) {
    throw "Step 14 failed."
}

$ReturnZip = Join-Path $ProjectRoot "v6_step14_sensitivity_results.zip"
if (-not (Test-Path $ReturnZip)) {
    throw "Step 14 produced no return ZIP: $ReturnZip"
}

Write-Host "[V6.1 Step 14 R3] PASS. Separate q/calibration and rho development studies were reported; no model fit, new evaluation, validation reopening, test access, or new selection occurred."
Write-Host "Return this file and the full console output:"
Write-Host $ReturnZip
