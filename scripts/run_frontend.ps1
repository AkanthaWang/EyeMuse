$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$FrontendSrc = Join-Path $Root 'frontend\src'
$Python = 'D:/py3.10/python.exe'

$env:PYTHONPATH = @(
    $Root,
    $FrontendSrc
) -join ';'

& $Python (Join-Path $Root 'frontend\main.py')