param(
    [string]$RuntimeRoot = 'G:\Auto-OpenMatte-Sdr2hdr-converter — kopia',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GuiArgs
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $RuntimeRoot '.venv\Scripts\python.exe'
$ffmpegBin = Join-Path $RuntimeRoot 'dev\ffmpeg-build\install\bin'
$bridgeBin = Join-Path $RuntimeRoot 'dev\v05-native'
$cudaBin = Join-Path $RuntimeRoot '.venv\Lib\site-packages\nvidia\cu13\bin\x86_64'
$runtimeTools = Join-Path $RuntimeRoot 'tools\openmatte_hdr'
$bridgePath = Join-Path $bridgeBin 'v5_gpu_bridge.dll'

foreach ($required in @($python, $ffmpegBin, $bridgeBin, $runtimeTools, $bridgePath)) {
    if (-not (Test-Path $required)) {
        throw "Required recovered V5 runtime path is missing: $required"
    }
}

$pathSeparator = [IO.Path]::PathSeparator
$env:OPENMATTE_V5_RUNTIME = $RuntimeRoot
$env:OPENMATTE_V5_TOOLS = $runtimeTools
$env:OPENMATTE_V5_BRIDGE_PATH = $bridgePath
$env:PYTHONPATH = @(
    $runtimeTools,
    $RuntimeRoot,
    (Join-Path $repoRoot 'src'),
    $repoRoot
) -join $pathSeparator
$env:PATH = @(
    (Join-Path $RuntimeRoot '.venv\Scripts'),
    $ffmpegBin,
    $bridgeBin,
    $cudaBin,
    $env:PATH
) -join $pathSeparator

& $python -m gui @GuiArgs
exit $LASTEXITCODE
