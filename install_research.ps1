param(
    [string]$PythonExe = "D:\UIT\.venvs\ml-gpu\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$CacheRoot = "D:\UIT\.cache\baf-research"
New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null
$env:PIP_CACHE_DIR = Join-Path $CacheRoot "pip"
$env:TEMP = Join-Path $CacheRoot "temp"
$env:TMP = $env:TEMP
$env:TORCH_HOME = Join-Path $CacheRoot "torch"
$env:XDG_CACHE_HOME = $CacheRoot
$env:JUPYTER_DATA_DIR = "D:\UIT\.jupyter"
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Khong tim thay venv: $PythonExe. Truyen -PythonExe den venv dang dung."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Can Git for Windows trong PATH. Cai Git, mo lai terminal roi chay script."
}
& $PythonExe -c "import torch; print('Torch hien tai:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Venv hien tai chua import duoc Torch." }
& $PythonExe -m pip install --prefer-binary -r (Join-Path $ProjectRoot "requirements-research.txt")
if ($LASTEXITCODE -ne 0) { throw "Cai dependencies that bai; xem log pip." }
& $PythonExe -m ipykernel install --user --name uit-ml-gpu --display-name "Python (UIT ml-gpu)"
if ($LASTEXITCODE -ne 0) { throw "Dang ky kernel that bai." }
Write-Host "Xong. Chon kernel Python (UIT ml-gpu), mo notebook 11."
