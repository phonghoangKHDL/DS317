param(
    [string]$ZipPath,
    [string]$Project = 'D:\UIT\DS317\BAF_project'
)
$ErrorActionPreference = 'Stop'
if (-not $ZipPath) {
    $downloads = Join-Path $env:USERPROFILE 'Downloads'
    $item = Get-ChildItem -LiteralPath $downloads -Filter 'BAF_advisor_steps1-4_20261030*.zip' |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $item) { throw 'Khong tim thay ZIP. Truyen duong dan bang -ZipPath.' }
    $ZipPath = $item.FullName
}
if (-not (Test-Path -LiteralPath (Join-Path $Project 'src\gru.py'))) {
    throw "Thu muc dich chua co BAF_project goc: $Project"
}
Expand-Archive -LiteralPath $ZipPath -DestinationPath $Project -Force
Write-Host "Da giai nen vao $Project"
Write-Host 'Mo notebook 18, chon kernel ml-gpu. Xem README_ADVISOR.md de biet thu tu.'
