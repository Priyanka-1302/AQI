param([switch]$Train)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Create .venv and install requirements first. See README.md.' }
if ($Train -or -not (Test-Path 'models/model_info.json')) {
    & $pythonExe -m ml.train
    if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }
}
if (-not (Test-Path 'frontend/dist/index.html')) {
    Push-Location frontend
    try {
        npm ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend install failed.' }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
}
Write-Host 'Aero is available at http://127.0.0.1:8000. Press Ctrl+C to stop.'
& $pythonExe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
