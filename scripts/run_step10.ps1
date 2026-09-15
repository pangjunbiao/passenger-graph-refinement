$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

Write-Host "[V6.1 Step 10 R3] Testing the cumulative project..."
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) {
    throw "Cumulative tests failed. Step 10 was not started."
}

Write-Host "[V6.1 Step 10 R3] Auditing frozen evidence and preparing/finalizing validity outputs..."
python main_v6.py --stage step10
$StageExit = $LASTEXITCODE
$ReturnZip = Join-Path $ProjectRoot "v6_step10_results.zip"
if (-not (Test-Path $ReturnZip)) {
    throw "Step 10 produced no return ZIP: $ReturnZip"
}
if ($StageExit -ne 0) {
    throw "Step 10 failed (exit code $StageExit). Return the ZIP and console output."
}

$Latest = Get-ChildItem (Join-Path $ProjectRoot "outputs\v6\step10_reporting_validity") -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
$Contract = Get-Content (Join-Path $Latest.FullName "step10_contract.json") -Raw |
    ConvertFrom-Json

if ($Contract.status -eq "PASS") {
    Write-Host "[V6.1 Step 10 R3] FINAL PASS. Human and external evidence are complete."
} else {
    $HumanStatus = $Contract.human_evaluation.status
    $ExternalStatus = $Contract.external_evaluation.status
    Write-Host "[V6.1 Step 10 R3] PREPARATION PASS | human=$HumanStatus | external=$ExternalStatus"
    if (-not $Contract.human_evaluation.complete) {
        Write-Host "Complete the independent annotation packet under inputs\v6\step10\human."
    }
    if (-not $Contract.external_evaluation.complete) {
        Write-Host "Complete the dataset-specific external evaluation under inputs\v6\step10\external."
    }
}
Write-Host "[V6.1 Step 10 R3] COMPLETE. Return this file and the full console output:"
Write-Host $ReturnZip
