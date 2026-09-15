$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

Write-Host "[V6.1 Step 11B R1] Testing the cumulative project..."
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) {
    throw "Cumulative tests failed. Step 11B was not started."
}

Write-Host "[V6.1 Step 11B R1] Rendering the manifest-verified English and Chinese graph figures..."
python main_v6.py --stage step11b
if ($LASTEXITCODE -ne 0) {
    throw "Step 11B failed."
}

$ReturnZip = Join-Path $ProjectRoot "v6_step11_bilingual_figures.zip"
if (-not (Test-Path $ReturnZip)) {
    throw "Step 11B produced no return ZIP: $ReturnZip"
}

Write-Host "[V6.1 Step 11B R1] PASS. English and Chinese figures were created without changing evidence."
Write-Host "Return this file and the full console output:"
Write-Host $ReturnZip

