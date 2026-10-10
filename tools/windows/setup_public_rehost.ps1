[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$DownloadRoot,
    [Parameter(Mandatory = $true)][string]$ModelRoot,
    [Parameter(Mandatory = $true)][string]$LlamaSource,
    [string]$Distribution = 'Ubuntu',
    [string]$ModelId = 'llama-3.2-3b-instruct-q4-k-m',
    [string]$InstallRoot = ''
)

$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$metadataRoot = Join-Path $repository 'evaluation\publication\releases\evaluation1-release-v1'
$stageScript = Join-Path $repository 'evaluation\scripts\stage_public_evaluation_release.py'
$locationRaw = python $stageScript locate --download-root $DownloadRoot
if ($LASTEXITCODE -ne 0) { throw 'The downloaded Evaluation1 release could not be located.' }
$location = $locationRaw | ConvertFrom-Json
$releaseRoot = [string]$location.external_root
$modelRootResolved = (Resolve-Path $ModelRoot).Path
$llamaSourceResolved = (Resolve-Path $LlamaSource).Path
if (-not $InstallRoot) {
    $wslHome = (& wsl.exe -d $Distribution -- sh -lc 'printf %s "$HOME"').Trim()
    if (-not $wslHome -or -not $wslHome.StartsWith('/')) { throw 'WSL home resolution failed.' }
    $InstallRoot = "$wslHome/.local/share/laris/rehost"
}

python $stageScript initialize --metadata-root $metadataRoot --external-root $releaseRoot
if ($LASTEXITCODE -ne 0) { throw 'Public Evaluation1 metadata initialization failed.' }

python $stageScript verify `
    --metadata-root $metadataRoot --external-root $releaseRoot
if ($LASTEXITCODE -ne 0) { throw 'Public Evaluation1 release verification failed.' }

$repositoryWsl = (& wsl.exe -d $Distribution -- sh -lc 'wslpath -a -u "$1"' sh $repository).Trim()
$llamaWsl = (& wsl.exe -d $Distribution -- sh -lc 'wslpath -a -u "$1"' sh $llamaSourceResolved).Trim()
if (-not $repositoryWsl -or -not $llamaWsl) { throw 'WSL path conversion failed.' }

& wsl.exe -d $Distribution -- sh "$repositoryWsl/tools/wsl/setup_public_rehost_runtime.sh" `
    $repositoryWsl $llamaWsl $InstallRoot
if ($LASTEXITCODE -ne 0) { throw 'WSL rehosting runtime setup failed.' }

$receiptRaw = & wsl.exe -d $Distribution -- cat "$InstallRoot/setup-receipt.json"
if ($LASTEXITCODE -ne 0) { throw 'WSL setup receipt is unavailable.' }
$receipt = $receiptRaw | ConvertFrom-Json
$profile = Join-Path $env:APPDATA 'LARIS\rehost-profile.json'
$analysisRoot = Join-Path $env:APPDATA 'LARIS\analyses'

python (Join-Path $repository 'evaluation\scripts\create_public_rehost_profile.py') `
    --release-root $releaseRoot --model-id $ModelId --model-root $modelRootResolved `
    --profile $profile --analysis-root $analysisRoot --wsl-distribution $Distribution `
    --wsl-python $receipt.python --wsl-runtime-profile $receipt.runtime_profile `
    --wsl-helper $receipt.helper --wsl-helper-digest $receipt.helper_digest `
    --wsl-libllama $receipt.libllama --wsl-work-root $receipt.work_root --replace
if ($LASTEXITCODE -ne 0) { throw 'Investigator profile creation failed.' }

Write-Host "LARIS_PUBLIC_RELEASE_ROOT=$releaseRoot"
Write-Host "LARIS_PUBLIC_REHOST_READY=$profile"
