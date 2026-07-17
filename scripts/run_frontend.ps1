$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$FrontendSrc = Join-Path $Root 'frontend\src'
$Python = 'D:/Soft/Work/Anaconda/envs/pytorch/python.exe'

$env:PYTHONPATH = @(
    $Root,
    $FrontendSrc
) -join ';'

& $Python (Join-Path $Root 'frontend\main.py')