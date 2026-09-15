[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ((Split-Path $ProjectRoot -Leaf) -notmatch "(?i)v6") {
    throw "Refusing to run: the isolated project directory name must contain V6."
}
Get-Command $Python -ErrorAction Stop | Out-Null

Set-Location -LiteralPath $ProjectRoot

& $Python -c "import sys, numpy, yaml, torch; assert sys.version_info[:2] == (3, 10), sys.version; print('python=', sys.executable); print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('numpy=', numpy.__version__); print('yaml=', yaml.__version__)"
if ($LASTEXITCODE -ne 0) { throw "V6 Step-1 environment preflight failed." }

& $Python -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw "V6 Step-1 unit tests failed." }

& $Python .\main_v6.py --stage step1 --project-root $ProjectRoot --config .\configs\v6\step01.yaml
if ($LASTEXITCODE -ne 0) { throw "V6 Step-1 contract failed closed." }

$Latest = Get-ChildItem -LiteralPath .\outputs\v6\step01_math_core -Directory |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1

Write-Host "V6 Step 1 completed. Return this directory:"
Write-Host $Latest.FullName
