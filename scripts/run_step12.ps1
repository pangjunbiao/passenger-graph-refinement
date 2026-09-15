$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

Write-Host "[V6.1 Step 12 R3] Testing the cumulative project..."
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) {
    throw "Cumulative tests failed. Step 12 was not started."
}

Write-Host "[V6.1 Step 12 R3] Finalizing the three-annotator case study (no new human input and no model fitting)..."
python main_v6.py --stage step12
$StageExit = $LASTEXITCODE
$ReturnZip = Join-Path $ProjectRoot "v6_step12_results.zip"
if (-not (Test-Path $ReturnZip)) {
    throw "Step 12 produced no return ZIP: $ReturnZip"
}
if ($StageExit -ne 0) {
    throw "Step 12 failed (exit code $StageExit). Return the ZIP and console output."
}

$Latest = Get-ChildItem (Join-Path $ProjectRoot "outputs\v6\step12_human_case_finalization") -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
$Contract = Get-Content (Join-Path $Latest.FullName "step12_contract.json") -Raw |
    ConvertFrom-Json

if ($Contract.status -eq "PASS") {
    Write-Host "[V6.1 Step 12 R3] PASS. All three original response sets were retained; no fourth adjudicator or new human files are required."
    Write-Host "Complete the factual ethics/consent and recruitment disclosure before manuscript submission."
} else {
    throw "Unexpected Step 12 status: $($Contract.status)"
}
Write-Host "[V6.1 Step 12 R3] COMPLETE. Return this file and the full console output:"
Write-Host $ReturnZip
