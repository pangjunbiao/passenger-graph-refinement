$ErrorActionPreference = "Stop"

Write-Host "[V6 Step 5R2] Testing the cumulative project with the current V6 Python..."
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) {
    throw "The cumulative tests failed (exit code $LASTEXITCODE). Step 5R2 was not started."
}

Write-Host "[V6 Step 5R2] Running target-aligned frozen-parent graph projection revision 5..."
$ReturnZip = Join-Path (Get-Location) "v6_step05r2_results.zip"
$PreviousHash = $null
if (Test-Path -LiteralPath $ReturnZip -PathType Leaf) {
    $PreviousHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ReturnZip).Hash
}
python main_v6.py --stage step5r2
$StageExit = $LASTEXITCODE
if (-not (Test-Path -LiteralPath $ReturnZip -PathType Leaf)) {
    throw "Step 5R2 produced no return ZIP: $ReturnZip"
}
$CurrentHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ReturnZip).Hash
if ($null -ne $PreviousHash -and $CurrentHash -eq $PreviousHash) {
    throw "Step 5R2 did not replace the previous return ZIP. Treat the file as stale."
}
Write-Host "[V6 Step 5R2] COMPLETE. Return this file and the full console output:"
Write-Host $ReturnZip
if ($StageExit -ne 0) {
    throw "Step 5R2 failed or was blocked (exit code $StageExit). Return the ZIP; do not continue to validation."
}
