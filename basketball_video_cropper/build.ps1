$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv-yolo\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "YOLO Python environment not found: $python"
}

$ffmpegBinary = & $python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
if ($LASTEXITCODE -ne 0) {
    throw "Could not locate imageio-ffmpeg in the YOLO Python environment."
}

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --console `
    --name BasketballVideoCropper-YOLO `
    --paths $projectRoot `
    --add-binary "$ffmpegBinary;imageio_ffmpeg\binaries" `
    --add-data "$projectRoot\yolov8n.pt;models" `
    --collect-all ultralytics `
    --collect-all torch `
    --collect-all torchvision `
    --collect-all lap `
    --collect-submodules nvidia `
    --hidden-import backports.tarfile `
    --specpath (Join-Path $PSScriptRoot "build") `
    --distpath (Join-Path $PSScriptRoot "dist") `
    --workpath (Join-Path $PSScriptRoot "build") `
    (Join-Path $PSScriptRoot "crop_basketball_video.py")

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}
