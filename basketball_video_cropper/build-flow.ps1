$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$projectRoot = Split-Path -Parent $PSScriptRoot
$ffmpegBinary = py -3.10 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
if ($LASTEXITCODE -ne 0) {
    throw "Could not locate imageio-ffmpeg in Python 3.10."
}

py -3.10 -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --console `
    --name BasketballVideoCropper-Flow `
    --paths $projectRoot `
    --exclude-module ultralytics `
    --exclude-module torch `
    --exclude-module torchvision `
    --exclude-module imageio_ffmpeg `
    --add-binary "$ffmpegBinary;imageio_ffmpeg\binaries" `
    --specpath (Join-Path $PSScriptRoot "build-flow") `
    --distpath (Join-Path $PSScriptRoot "dist") `
    --workpath (Join-Path $PSScriptRoot "build-flow") `
    (Join-Path $PSScriptRoot "crop_basketball_video.py")

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}
