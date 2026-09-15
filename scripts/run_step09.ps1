$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

Write-Host "[V6.1 Step 9] Testing the cumulative project..."
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) {
    throw "Cumulative tests failed. Step 9 was not started."
}

Write-Host "[V6.1 Step 9] Running/resuming 100 frozen fair comparator fits..."
python main_v6.py --stage step9
$StageExit = $LASTEXITCODE
$ReturnZip = Join-Path $ProjectRoot "v6_step09_results.zip"
if (-not (Test-Path $ReturnZip)) {
    throw "Step 9 produced no diagnostic return ZIP: $ReturnZip"
}
if ($StageExit -ne 0) {
    Write-Host "[V6.1 Step 9] INCOMPLETE. Completed receipts were preserved. Return the ZIP and console output for a targeted fix, then rerun to resume."
    throw "Step 9 incomplete (exit code $StageExit); no V6.1 model change is permitted."
}
Write-Host "[V6.1 Step 9] COMPLETE. Return this file and the full console output:"
Write-Host $ReturnZip
