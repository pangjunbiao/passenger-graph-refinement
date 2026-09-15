$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

Write-Host "[V6.1 Step 11 R2] Testing the cumulative project..."
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) {
    throw "Cumulative tests failed. Step 11 was not started."
}

Write-Host "[V6.1 Step 11 R2] Synthesizing distinct frozen evidence (no fitting, tuning, or table replotting)..."
python main_v6.py --stage step11
$StageExit = $LASTEXITCODE
$ReturnZip = Join-Path $ProjectRoot "v6_step11_results.zip"
if (-not (Test-Path $ReturnZip)) {
    throw "Step 11 produced no return ZIP: $ReturnZip"
}
if ($StageExit -ne 0) {
    throw "Step 11 failed (exit code $StageExit). Return the ZIP and console output."
}

$Latest = Get-ChildItem (Join-Path $ProjectRoot "outputs\v6\step11_frozen_evidence") -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
$Contract = Get-Content (Join-Path $Latest.FullName "step11_contract.json") -Raw |
    ConvertFrom-Json
if ($Contract.status -ne "PASS") {
    throw "Step 11 contract did not pass."
}
Write-Host "[V6.1 Step 11 R2] COMPLETE. Return this file and the full console output:"
Write-Host $ReturnZip
