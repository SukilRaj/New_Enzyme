param([switch]$IncludeESM)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required. Install Python 3.12 and retry.' }
}
$requirementsFile = if ($IncludeESM) { 'requirements-local.txt' } else { 'requirements-core.txt' }
& ./.venv/Scripts/python.exe -m pip install -r $requirementsFile
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Host 'Ready. Double-click Start-Helix.cmd.'
