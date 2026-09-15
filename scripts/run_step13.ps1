$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

Write-Host "[V6.1 Step 13 R1] Testing the cumulative project..."
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) {
    throw "Cumulative tests failed. Step 13 was not started."
}

$Dataset = Join-Path $ProjectRoot "inputs\v6\step13\Transit-Tweet-Data-main.zip"
if (-not (Test-Path $Dataset)) {
    throw "Copy Transit-Tweet-Data-main.zip to $Dataset"
}

Write-Host "[V6.1 Step 13 R1] Running the preregistered external temporal replication..."
Write-Host "This is resumable. Completed method/seed receipts are hash-checked and reused."
python main_v6.py --stage step13
$StageExit = $LASTEXITCODE
$ReturnZip = Join-Path $ProjectRoot "v6_step13_results.zip"
if (-not (Test-Path $ReturnZip)) {
    throw "Step 13 produced no return ZIP: $ReturnZip"
}
if ($StageExit -ne 0) {
    throw "Step 13 is incomplete. Rerun to resume, or return the ZIP and console output."
}

Write-Host "[V6.1 Step 13 R1] PASS. Return this file and the full console output:"
Write-Host $ReturnZip
Write-Host "Then copy the contents of the generated Step-10 submission folder to inputs\v6\step10\external and rerun Step 10 R3."
