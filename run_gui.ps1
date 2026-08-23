param(
    [string]$RuntimeRoot = 'C:\Users\xroki\om-v5-runtime',
    # Directory holding the FFmpeg/ffprobe executables used for analysis and
    # rendering. Pass '' to keep the previous PATH-based resolution.
    [string]$MediaToolsBin = 'C:\ffmpeg\bin',
    # Run the EXTEND extension-strip transform on the GPU. Falls back to the CPU
    # automatically when CUDA is unavailable. Pass -GpuTransform:$false to force
    # the CPU path.
    [bool]$GpuTransform = $true,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GuiArgs
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
$python = Join-Path $RuntimeRoot '.venv\Scripts\python.exe'
$ffmpegBin = Join-Path $RuntimeRoot 'dev\ffmpeg-build\install\bin'
$bridgeBin = Join-Path $RuntimeRoot 'dev\v05-native'
$cudaBin = Join-Path $RuntimeRoot '.venv\Lib\site-packages\nvidia\cu13\bin\x86_64'
$runtimeTools = Join-Path $RuntimeRoot 'tools\openmatte_hdr'
$bridgePath = Join-Path $bridgeBin 'v5_gpu_bridge.dll'

foreach ($required in @($python, $ffmpegBin, $bridgeBin, $runtimeTools, $bridgePath)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required V5 runtime path is missing: $required"
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

# The runtime FFmpeg is built with --disable-x86asm, which decodes HEVC several
# times slower than an assembly-optimized build. Point only the subprocess tools
# at a faster static build; the runtime's shared FFmpeg libraries stay on PATH so
# the native GPU bridge keeps loading exactly what it expects.
if ($MediaToolsBin) {
    $toolFfmpeg = Join-Path $MediaToolsBin 'ffmpeg.exe'
    $toolFfprobe = Join-Path $MediaToolsBin 'ffprobe.exe'
    if ((Test-Path -LiteralPath $toolFfmpeg) -and (Test-Path -LiteralPath $toolFfprobe)) {
        $env:OPENMATTE_FFMPEG = $toolFfmpeg
        $env:OPENMATTE_FFPROBE = $toolFfprobe
    }
    else {
        Write-Warning "MediaToolsBin '$MediaToolsBin' has no ffmpeg.exe/ffprobe.exe; falling back to PATH resolution."
    }
}

if ($GpuTransform) {
    $env:OPENMATTE_GPU_TRANSFORM = '1'
}
else {
    Remove-Item Env:OPENMATTE_GPU_TRANSFORM -ErrorAction SilentlyContinue
}

& $python -m gui @GuiArgs
exit $LASTEXITCODE
