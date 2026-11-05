param(
    [string]$Python = 'D:\UIT\.venvs\ml-gpu\Scripts\python.exe',
    [string]$PipCache = 'D:\UIT\.cache\pip'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Khong tim thay Python venv: $Python"
}
New-Item -ItemType Directory -Force -Path $PipCache | Out-Null
& $Python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'Hay chon venv ml-gpu da cai PyTorch.' }
& $Python -m pip install --cache-dir $PipCache nbformat ipython lightgbm psutil
if ($LASTEXITCODE -ne 0) { throw 'Cai dependency chua thanh cong.' }
& $Python -c "import nbformat, IPython, lightgbm, psutil; print('ADVISOR_DEPENDENCIES_READY')"
if ($LASTEXITCODE -ne 0) { throw 'Kiem tra import chua thanh cong.' }
