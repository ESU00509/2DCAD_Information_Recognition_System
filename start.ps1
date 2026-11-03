$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Project environment missing. See README.md for setup.'
}
$env:PYTHONUTF8 = '1'
& $projectPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
