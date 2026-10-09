[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$LarisNodeHost,
    [Parameter(Mandatory = $true)][string]$SshUser,
    [Parameter(Mandatory = $true)][string]$SshIdentityFile,
    [Parameter(Mandatory = $true)][string]$NodeConfigRoot,
    [string]$ProfilePath = (Join-Path $env:APPDATA 'LARIS\rehost-profile.json')
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
            throw "Identity-sync upload failed with exit code $LASTEXITCODE."
        }
        if ($attempt -lt 3) { Start-Sleep -Seconds $attempt }
    }
    throw 'Identity-sync upload failed after three transport attempts.'
}

function Invoke-ScpDownload {
    param([string]$Scp, [string]$Identity, [string]$Source, [string]$Destination)
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        & $Scp '-q' '-i' $Identity '-o' 'IdentitiesOnly=yes' '--' $Source $Destination
        if ($LASTEXITCODE -eq 0) { return }
        if ($LASTEXITCODE -ne 255) {
            throw "Identity-sync download failed with exit code $LASTEXITCODE."
        }
        if ($attempt -lt 3) { Start-Sleep -Seconds $attempt }
    }
    throw 'Identity-sync download failed after three transport attempts.'
}

if ($LarisNodeHost -notmatch '^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$') {
    throw 'LarisNodeHost is invalid.'
}
if ($SshUser -notmatch '^[a-z_][a-z0-9_-]{0,31}$') {
    throw 'SshUser is invalid.'
}

$identity = (Resolve-Path -LiteralPath $SshIdentityFile -ErrorAction Stop).Path
$nodeConfig = (Resolve-Path -LiteralPath $NodeConfigRoot -ErrorAction Stop).Path
$profile = (Resolve-Path -LiteralPath $ProfilePath -ErrorAction Stop).Path
$receiptPath = Join-Path $nodeConfig 'installation-receipt.json'
if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
    throw 'The selected LARIS node installation receipt is unavailable.'
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$resolver = Join-Path $PSScriptRoot 'resolve_rehost_identity_requirements.py'
$exportShell = Join-Path $repoRoot 'tools\vm\export_rehost_identity.sh'
$exportPython = Join-Path $repoRoot 'tools\vm\export_rehost_profile_inputs.py'
$ssh = Resolve-Executable 'ssh.exe'
$scp = Resolve-Executable 'scp.exe'
$tar = Resolve-Executable 'tar.exe'
$python = Resolve-Executable 'python.exe'

$requirementsText = (& $python $resolver $receiptPath) -join "`n"
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($requirementsText)) {
    throw 'The LARIS node identity requirements could not be resolved.'
}
$requirements = $requirementsText | ConvertFrom-Json
$identityKeyId = [string]$requirements.identity_key_id
if ($identityKeyId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
    throw 'The LARIS node identity key ID is invalid.'
}

$transferId = [Guid]::NewGuid().ToString('N')
$target = "$SshUser@$LarisNodeHost"
$remoteShell = "/tmp/laris-identity-sync-$transferId.sh"
$remotePython = "/tmp/laris-identity-sync-$transferId.py"
$remoteRequirements = "/tmp/laris-identity-sync-$transferId.json"
$remoteArchive = "/tmp/laris-rehost-export-$transferId.tar"
$remoteDigest = "$remoteArchive.sha256"
$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) "laris-identity-sync-$transferId"
$downloadRoot = Join-Path $temporaryRoot 'download'
$extractRoot = Join-Path $temporaryRoot 'export'
[System.IO.Directory]::CreateDirectory($downloadRoot) | Out-Null
[System.IO.Directory]::CreateDirectory($extractRoot) | Out-Null
$requirementsPath = Join-Path $temporaryRoot 'identity-requirements.json'
[System.IO.File]::WriteAllText(
    $requirementsPath,
    $requirementsText.TrimEnd() + "`n",
    [System.Text.UTF8Encoding]::new($false)
)

$remoteStarted = $false
try {
    Write-Host 'Step 1/3: exporting the current HMAC identity epoch from the LARIS node.'
    Write-Host 'Enter only the LARIS node sudo password at the remote prompt.'
    Invoke-ScpUpload $scp $identity $exportShell "${target}:$remoteShell"
    $remoteStarted = $true
    Invoke-ScpUpload $scp $identity $exportPython "${target}:$remotePython"
    Invoke-ScpUpload $scp $identity $requirementsPath "${target}:$remoteRequirements"
    $checks = @(
        "$(Get-Sha256 $exportShell)  $remoteShell",
        "$(Get-Sha256 $exportPython)  $remotePython",
        "$(Get-Sha256 $requirementsPath)  $remoteRequirements"
    ) -join '\n'
    $remoteCommand = @(
        "/usr/bin/printf '%b' $(Quote-Remote ($checks + '\n')) | /usr/bin/sha256sum -c -",
        "/usr/bin/sudo -- /bin/sh $(Quote-Remote $remoteShell) $(Quote-Remote $transferId) $(Quote-Remote $identityKeyId) /var/lib/laris-install/installation-receipt.json $(Quote-Remote $remoteRequirements) $(Quote-Remote $remotePython) --identity-only"
    ) -join ' && '
    & $ssh '-tt' '-i' $identity '-o' 'IdentitiesOnly=yes' '--' $target $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "The receipt-bound identity export failed with exit code $LASTEXITCODE."
    }

    Write-Host 'Step 2/3: downloading and validating the 32-byte identity key.'
    Invoke-ScpDownload $scp $identity "${target}:$remoteArchive" $downloadRoot
    Invoke-ScpDownload $scp $identity "${target}:$remoteDigest" $downloadRoot
    $archivePath = Join-Path $downloadRoot ([System.IO.Path]::GetFileName($remoteArchive))
    $digestPath = Join-Path $downloadRoot ([System.IO.Path]::GetFileName($remoteDigest))
    $digestLine = [System.IO.File]::ReadAllText($digestPath).Trim()
    if ($digestLine -notmatch '^([0-9a-f]{64})  laris-rehost-export-[0-9a-f]{32}\.tar$' -or
        (Get-Sha256 $archivePath) -ne $Matches[1]) {
        throw 'The identity export archive digest differs.'
    }
    $members = @(& $tar '-tf' $archivePath)
    if ($LASTEXITCODE -ne 0 -or $members.Count -ne 2 -or
        $members[0] -ne 'identity.key' -or $members[1] -ne 'export-manifest.json') {
        throw 'The identity export archive inventory is invalid.'
    }
    & $tar '-xf' $archivePath '-C' $extractRoot
    if ($LASTEXITCODE -ne 0) { throw 'The identity export extraction failed.' }
    $manifest = Get-Content -LiteralPath (Join-Path $extractRoot 'export-manifest.json') -Raw | ConvertFrom-Json
    $entries = @($manifest.entries)
    if ($manifest.schema_version -ne '1.0' -or
        $manifest.export_type -ne 'identity_only' -or
        $manifest.identity_key_id -ne $identityKeyId -or
        $manifest.installation_receipt_sha256 -ne $requirements.installation_receipt_sha256 -or
        $entries.Count -ne 1 -or $entries[0].path -ne 'identity.key' -or
        [long]$entries[0].size_bytes -ne 32) {
        throw 'The identity export manifest binding differs.'
    }
    $exportedKey = Join-Path $extractRoot 'identity.key'
    if ((Get-Item -LiteralPath $exportedKey).Length -ne 32 -or
        ('sha256:' + (Get-Sha256 $exportedKey)) -ne [string]$entries[0].sha256) {
        throw 'The exported identity key differs from its installation receipt.'
    }

    Write-Host 'Step 3/3: preserving existing epochs and updating the investigator profile.'
    $profileRoot = Split-Path -Parent $profile
    $keyRoot = Join-Path $profileRoot 'keys'
    [System.IO.Directory]::CreateDirectory($keyRoot) | Out-Null
    $keyPath = Join-Path $keyRoot ($identityKeyId + '.bin')
    if (Test-Path -LiteralPath $keyPath -PathType Leaf) {
        if ((Get-Sha256 $keyPath) -ne (Get-Sha256 $exportedKey)) {
            throw 'An installed identity key differs from the selected LARIS node.'
        }
    }
    else {
        $pendingKey = $keyPath + '.pending-' + [Guid]::NewGuid().ToString('N')
        Copy-Item -LiteralPath $exportedKey -Destination $pendingKey
        Move-Item -LiteralPath $pendingKey -Destination $keyPath
    }
    & icacls.exe $keyRoot /inheritance:r /grant:r ($env:USERNAME + ':(OI)(CI)F') | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'The private identity-key ACL could not be restricted.' }

    & $python '-m' 'laris_onpremise.cli' 'rehost' 'profile' 'add-identity' `
        '--profile' $profile '--identity-key' ($identityKeyId + '=' + $keyPath)
    if ($LASTEXITCODE -ne 0) { throw 'The investigator profile update failed.' }
    Write-Host "LARIS_REHOST_IDENTITY_SYNC_COMPLETE identity_key_id=$identityKeyId"
    Write-Host "profile=$profile"
}
finally {
    if ($remoteStarted) {
        $cleanup = '/bin/rm -f -- ' + ((@(
            $remoteShell, $remotePython, $remoteRequirements, $remoteArchive, $remoteDigest
        ) | ForEach-Object { Quote-Remote $_ }) -join ' ')
        & $ssh '-i' $identity '-o' 'IdentitiesOnly=yes' '--' $target $cleanup 2>$null
    }
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
