$ErrorActionPreference = "Stop"

if ($PWD.Path -notmatch "(?i)V6") {
    throw "Run Step 2 from the isolated SC-HTM-V6 project root."
}

Write-Host "[preflight] Reusing the currently selected Python environment"
python -c "import sys, numpy, scipy, pandas, yaml; print('Python:',sys.executable); print('NumPy:',numpy.__version__); print('SciPy:',scipy.__version__); print('Pandas:',pandas.__version__); print('PyYAML:',yaml.__version__)"

Write-Host "[tests] Running the complete cumulative V6 test suite"
python -m pytest -q tests

Write-Host "[run] Building the Step-2 data/evaluator firewall"
python main_v6.py --stage step2
