[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$VmHost,
    [Parameter(Mandatory = $true)][string]$VmUser,
    [Parameter(Mandatory = $true)][string]$IdentityFile,
    [Parameter(Mandatory = $true)][string]$BundleRoot,
    [Parameter(Mandatory = $true)][string]$NodeConfigRoot,
    [Parameter(Mandatory = $true)][string]$TenantId,
    [string]$WslDistribution = 'Ubuntu-24.04'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-Executable {
    param([Parameter(Mandatory = $true)][string]$Name)
    return (Get-Command $Name -ErrorAction Stop).Source
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $algorithm = [System.Security.Cryptography.SHA256]::Create()
        try { $digest = $algorithm.ComputeHash($stream) }
        finally { $algorithm.Dispose() }
    }
    finally { $stream.Dispose() }
    return ([System.BitConverter]::ToString($digest)).Replace('-', '').ToLowerInvariant()
}

function Quote-Remote {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value.Contains("'") -or $Value.Contains("`r") -or $Value.Contains("`n")) {
        throw 'Remote command value is unsafe.'
    }
    return "'$Value'"
}

function Invoke-ScpUpload {
    param([string]$Scp, [string]$Identity, [string]$Source, [string]$Target)
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        & $Scp '-q' '-i' $Identity '-o' 'IdentitiesOnly=yes' '--' $Source $Target
        if ($LASTEXITCODE -eq 0) { return }
        if ($LASTEXITCODE -ne 255) {
            throw "Rehost input upload failed with exit code $LASTEXITCODE."
        }
        if ($attempt -lt 3) { Start-Sleep -Seconds $attempt }
    }
    throw 'Rehost input upload failed after three transport attempts.'
}

function Invoke-ScpDownload {
    param([string]$Scp, [string]$Identity, [string]$Source, [string]$Destination)
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        & $Scp '-q' '-i' $Identity '-o' 'IdentitiesOnly=yes' '--' $Source $Destination
        if ($LASTEXITCODE -eq 0) { return }
        if ($LASTEXITCODE -ne 255) {
            throw "Rehost input download failed with exit code $LASTEXITCODE."
        }
        if ($attempt -lt 3) { Start-Sleep -Seconds $attempt }
    }
    throw 'Rehost input download failed after three transport attempts.'
}

function Install-ExactFile {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )
    $parent = Split-Path -Parent $Destination
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        if ((Get-Sha256 $Destination) -ne $ExpectedSha256) {
            throw "An installed rehost input differs: $Destination"
        }
        return
    }
    $pending = $Destination + '.pending-' + [Guid]::NewGuid().ToString('N')
    Copy-Item -LiteralPath $Source -Destination $pending
    if ((Get-Sha256 $pending) -ne $ExpectedSha256) {
        Remove-Item -LiteralPath $pending -Force
        throw "A staged rehost input differs: $Destination"
    }
    Move-Item -LiteralPath $pending -Destination $Destination
}

if ($VmHost -notmatch '^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$') {
    throw 'VmHost is invalid.'
}
if ($VmUser -notmatch '^[a-z_][a-z0-9_-]{0,31}$') {
    throw 'VmUser is invalid.'
}
if ($TenantId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
    throw 'TenantId is invalid.'
}
if ($WslDistribution -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
    throw 'WslDistribution is invalid.'
}

$identity = (Resolve-Path -LiteralPath $IdentityFile -ErrorAction Stop).Path
$bundle = (Resolve-Path -LiteralPath $BundleRoot -ErrorAction Stop).Path
$nodeConfig = (Resolve-Path -LiteralPath $NodeConfigRoot -ErrorAction Stop).Path
$receiptPath = Join-Path $nodeConfig 'installation-receipt.json'
if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
    throw 'The selected node installation receipt is unavailable.'
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$resolver = Join-Path $PSScriptRoot 'resolve_rehost_bundle_requirements.py'
$validator = Join-Path $PSScriptRoot 'validate_rehost_profile.py'
$exportShell = Join-Path $repoRoot 'tools\vm\export_rehost_profile_inputs.sh'
$exportPython = Join-Path $repoRoot 'tools\vm\export_rehost_profile_inputs.py'
$wslInstaller = Join-Path $repoRoot 'tools\wsl\install_rehost_runtime.sh'
$ssh = Resolve-Executable 'ssh.exe'
$scp = Resolve-Executable 'scp.exe'
$tar = Resolve-Executable 'tar.exe'
$python = Resolve-Executable 'python.exe'
$wsl = Resolve-Executable 'wsl.exe'

$requirementsText = (& $python $resolver $bundle $receiptPath) -join "`n"
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($requirementsText)) {
    throw 'The selected bundle requirements could not be resolved.'
}
$requirements = $requirementsText | ConvertFrom-Json
if ([string]$requirements.installation_receipt_sha256 -notmatch '^sha256:[0-9a-f]{64}$') {
    throw 'The installation receipt binding is invalid.'
}

$transferId = [Guid]::NewGuid().ToString('N')
$releaseId = $transferId.Substring(0, 16)
$target = "$VmUser@$VmHost"
$remoteShell = "/tmp/laris-rehost-export-$transferId.sh"
$remotePython = "/tmp/laris-rehost-export-$transferId.py"
$remoteRequirements = "/tmp/laris-rehost-requirements-$transferId.json"
$remoteArchive = "/tmp/laris-rehost-export-$transferId.tar"
$remoteDigest = "$remoteArchive.sha256"
$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) "laris-rehost-install-$transferId"
$downloadRoot = Join-Path $temporaryRoot 'download'
$extractRoot = Join-Path $temporaryRoot 'export'
$wslInput = Join-Path $temporaryRoot 'wsl-input'
[System.IO.Directory]::CreateDirectory($downloadRoot) | Out-Null
[System.IO.Directory]::CreateDirectory($extractRoot) | Out-Null
[System.IO.Directory]::CreateDirectory((Join-Path $wslInput 'lib')) | Out-Null
$requirementsPath = Join-Path $temporaryRoot 'bundle-requirements.json'
[System.IO.File]::WriteAllText(
    $requirementsPath,
    $requirementsText.TrimEnd() + "`n",
    [System.Text.UTF8Encoding]::new($false)
)

$remoteExportStarted = $false
try {
    Write-Host 'Step 1/5: exporting the receipt-bound rehost inputs from the selected node.'
    Invoke-ScpUpload $scp $identity $exportShell "${target}:$remoteShell"
    $remoteExportStarted = $true
    Invoke-ScpUpload $scp $identity $exportPython "${target}:$remotePython"
    Invoke-ScpUpload $scp $identity $requirementsPath "${target}:$remoteRequirements"
    $remoteChecks = @(
        "$(Get-Sha256 $exportShell)  $remoteShell",
        "$(Get-Sha256 $exportPython)  $remotePython",
        "$(Get-Sha256 $requirementsPath)  $remoteRequirements"
    ) -join '\n'
    $remoteCommand = @(
        "/usr/bin/printf '%b' $(Quote-Remote ($remoteChecks + '\n')) | /usr/bin/sha256sum -c -",
        "/usr/bin/sudo -- /bin/sh $(Quote-Remote $remoteShell) $(Quote-Remote $transferId) $(Quote-Remote ([string]$requirements.identity_key_id)) /var/lib/laris-install/installation-receipt.json $(Quote-Remote $remoteRequirements) $(Quote-Remote $remotePython)"
    ) -join ' && '
    & $ssh '-tt' '-i' $identity '-o' 'IdentitiesOnly=yes' '--' $target $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "The receipt-bound remote export failed with exit code $LASTEXITCODE."
    }
    $remoteExportStarted = $true

    Write-Host 'Step 2/5: downloading and validating the bounded export.'
    Invoke-ScpDownload $scp $identity "${target}:$remoteArchive" $downloadRoot
    Invoke-ScpDownload $scp $identity "${target}:$remoteDigest" $downloadRoot
    $archivePath = Join-Path $downloadRoot ([System.IO.Path]::GetFileName($remoteArchive))
    $digestPath = Join-Path $downloadRoot ([System.IO.Path]::GetFileName($remoteDigest))
    $digestLine = [System.IO.File]::ReadAllText($digestPath).Trim()
    if ($digestLine -notmatch '^([0-9a-f]{64})  laris-rehost-export-[0-9a-f]{32}\.tar$') {
        throw 'The remote export digest record is invalid.'
    }
    if ((Get-Sha256 $archivePath) -ne $Matches[1]) {
        throw 'The remote export archive digest differs.'
    }
    & $tar '-xf' $archivePath '-C' $extractRoot
    if ($LASTEXITCODE -ne 0) { throw 'The remote export extraction failed.' }
    $exportManifestPath = Join-Path $extractRoot 'export-manifest.json'
    $exportManifestText = [System.IO.File]::ReadAllText($exportManifestPath)
    $exportManifest = $exportManifestText | ConvertFrom-Json
    if ($exportManifest.schema_version -ne '1.0' -or
        $exportManifest.identity_key_id -ne $requirements.identity_key_id -or
        $exportManifest.installation_receipt_sha256 -ne $requirements.installation_receipt_sha256 -or
        $exportManifest.model_profile_id -ne $requirements.model_profile_id) {
        throw 'The remote export manifest binding differs.'
    }
    $entryMap = @{}
    foreach ($entry in $exportManifest.entries) {
        $entryPath = [string]$entry.path
        if ($entryPath -notmatch '^[A-Za-z0-9._/-]+$' -or $entryPath.Contains('..') -or $entryMap.ContainsKey($entryPath)) {
            throw 'The remote export inventory is invalid.'
        }
        $candidate = Join-Path $extractRoot ($entryPath.Replace('/', '\'))
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf) -or
            (Get-Item -LiteralPath $candidate).Length -ne [long]$entry.size_bytes -or
            ('sha256:' + (Get-Sha256 $candidate)) -ne [string]$entry.sha256) {
            throw "The exported input differs: $entryPath"
        }
        $entryMap[$entryPath] = $entry
    }

    Write-Host 'Step 3/5: installing the private Windows identity key and exact model.'
    $localRoot = Join-Path $env:APPDATA 'LARIS'
    $localKeyRoot = Join-Path $localRoot 'keys'
    $localAnalysisRoot = Join-Path $localRoot 'analyses'
    $localModelRoot = Join-Path $env:LOCALAPPDATA 'LARIS\models'
    $profilePath = Join-Path $localRoot 'rehost-profile.json'
    [System.IO.Directory]::CreateDirectory($localKeyRoot) | Out-Null
    [System.IO.Directory]::CreateDirectory($localAnalysisRoot) | Out-Null
    [System.IO.Directory]::CreateDirectory($localModelRoot) | Out-Null
    $keyEntry = $entryMap['identity.key']
    $modelRelative = 'model/' + [string]$requirements.model_filename
    $modelEntry = $entryMap[$modelRelative]
    if ($null -eq $keyEntry -or $null -eq $modelEntry) {
        throw 'The remote export is missing the key or model.'
    }
    $keyPath = Join-Path $localKeyRoot (([string]$requirements.identity_key_id) + '.bin')
    $modelPath = Join-Path $localModelRoot ([string]$requirements.model_filename)
    Install-ExactFile (Join-Path $extractRoot 'identity.key') $keyPath ([string]$keyEntry.sha256).Substring(7)
    Install-ExactFile (Join-Path $extractRoot ($modelRelative.Replace('/', '\'))) $modelPath ([string]$modelEntry.sha256).Substring(7)
    & icacls.exe $localKeyRoot /inheritance:r /grant:r ($env:USERNAME + ':(OI)(CI)F') | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'The private identity-key ACL could not be restricted.' }

    Write-Host 'Step 4/5: installing the private WSL rehost runtime.'
    $sourceArchive = Join-Path $wslInput 'laris_onpremise-src.tar'
    & $tar '-cf' $sourceArchive '--exclude=__pycache__' '--exclude=*.pyc' '-C' (Join-Path $repoRoot 'src') 'laris_onpremise'
    if ($LASTEXITCODE -ne 0) { throw 'The bounded LARIS source archive build failed.' }
    Copy-Item -LiteralPath (Join-Path $extractRoot 'native\llama-runtime-payload-rehost') -Destination (Join-Path $wslInput 'llama-runtime-payload-rehost')
    Copy-Item -LiteralPath (Join-Path $extractRoot 'runtime\runtime-profile.json') -Destination (Join-Path $wslInput 'runtime-profile.json')
    $libllamaName = $null
    foreach ($entry in $exportManifest.entries) {
        $entryPath = [string]$entry.path
        if ($entryPath.StartsWith('lib/')) {
            $libraryName = [System.IO.Path]::GetFileName($entryPath)
            Copy-Item -LiteralPath (Join-Path $extractRoot ($entryPath.Replace('/', '\'))) -Destination (Join-Path $wslInput ('lib\' + $libraryName))
            if ($libraryName -match '^libllama\.so') { $libllamaName = $libraryName }
        }
    }
    if ([string]::IsNullOrWhiteSpace($libllamaName)) { throw 'The exported libllama is unavailable.' }
    $manifestFiles = @('laris_onpremise-src.tar', 'llama-runtime-payload-rehost', 'runtime-profile.json')
    $manifestFiles += Get-ChildItem -LiteralPath (Join-Path $wslInput 'lib') -File | ForEach-Object { 'lib/' + $_.Name }
    $manifestLines = foreach ($name in $manifestFiles) {
        (Get-Sha256 (Join-Path $wslInput ($name.Replace('/', '\')))) + '  ' + $name
    }
    [System.IO.File]::WriteAllText(
        (Join-Path $wslInput 'manifest.sha256'),
        ($manifestLines -join "`n") + "`n",
        [System.Text.UTF8Encoding]::new($false)
    )
    $wslInputPath = (& $wsl '-d' $WslDistribution '--exec' '/usr/bin/wslpath' '-a' '-u' $wslInput).Trim()
    $wslInstallerPath = (& $wsl '-d' $WslDistribution '--exec' '/usr/bin/wslpath' '-a' '-u' $wslInstaller).Trim()
    $wslUserHome = (& $wsl '-d' $WslDistribution '--exec' '/bin/sh' '-c' 'getent passwd "$(id -un)" | cut -d: -f6').Trim()
    if (-not $wslInputPath.StartsWith('/') -or -not $wslInstallerPath.StartsWith('/') -or -not $wslUserHome.StartsWith('/')) {
        throw 'WSL path resolution failed.'
    }
    & $wsl '-d' $WslDistribution '--exec' '/bin/sh' $wslInstallerPath $wslInputPath $releaseId
    if ($LASTEXITCODE -ne 0) { throw 'The private WSL rehost runtime installation failed.' }

    Write-Host 'Step 5/5: generating and validating the active investigator profile.'
    $wslRelease = "$wslUserHome/.local/share/laris-rehost/releases/$releaseId"
    $helperComponent = $requirements.rehost_components | Where-Object { $_.role -eq 'helper' }
    if (@($helperComponent).Count -ne 1) { throw 'The helper receipt binding is ambiguous.' }
    $profileArguments = @(
        '-m', 'laris_onpremise.cli', 'rehost', 'profile', 'create',
        '--bundle-root', $bundle,
        '--profile', $profilePath,
        '--tenant-id', $TenantId,
        '--identity-key', (([string]$requirements.identity_key_id) + '=' + $keyPath),
        '--model-root', $localModelRoot,
        '--analysis-root', $localAnalysisRoot,
        '--wsl-distribution', $WslDistribution,
        '--wsl-python', "$wslRelease/venv/bin/python",
        '--wsl-runtime-profile', "$wslRelease/runtime-profile.json",
        '--wsl-helper', "$wslRelease/native/llama-runtime-payload-rehost",
        '--wsl-helper-digest', ([string]$helperComponent.sha256),
        '--wsl-libllama', "$wslRelease/lib/$libllamaName",
        '--wsl-work-root', "$wslUserHome/.local/share/laris-rehost/work",
        '--preserve-existing-identity-keys',
        '--replace'
    )
    & $python @profileArguments
    if ($LASTEXITCODE -ne 0) { throw 'The investigator profile installation failed.' }
    & $python $validator '--profile' $profilePath '--bundle-root' $bundle
    if ($LASTEXITCODE -ne 0) { throw 'The investigator profile validation failed.' }
    Write-Host 'LARIS_REHOST_PROFILE_INSTALL_COMPLETE'
    Write-Host "profile=$profilePath"
}
finally {
    if ($remoteExportStarted) {
        $cleanupTargets = @($remoteShell, $remotePython, $remoteRequirements, $remoteArchive, $remoteDigest)
        $cleanupCommand = '/bin/rm -f -- ' + (($cleanupTargets | ForEach-Object { Quote-Remote $_ }) -join ' ')
        & $ssh '-i' $identity '-o' 'IdentitiesOnly=yes' '--' $target $cleanupCommand 2>$null
    }
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
