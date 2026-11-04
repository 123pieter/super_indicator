# Use the project's interpreter and configuration regardless of the caller's folder.
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$appFile = Join-Path $PSScriptRoot 'app.py'
Push-Location $PSScriptRoot
try {
    & $projectPython -m streamlit run $appFile @args
    $appExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $appExitCode
