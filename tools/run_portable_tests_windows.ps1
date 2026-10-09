[CmdletBinding()]
param()

if ($PSVersionTable.PSVersion.Major -lt 7) {
    $pwsh = Get-Command pwsh.exe -ErrorAction Stop
    & $pwsh.Source -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath
    exit $LASTEXITCODE
}

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$sourceRoot = (Resolve-Path -LiteralPath (Join-Path $repo 'src')).Path
$tempParent = [System.IO.Path]::GetPathRoot($repo)
$testLeaf = 'laris-tests-{0}-{1}' -f $PID, ([guid]::NewGuid().ToString('N').Substring(0, 8))
$testRoot = Join-Path $tempParent $testLeaf
$originalLocation = (Get-Location).Path
$originalPythonPath = $env:PYTHONPATH
$exitCode = 1
$cleanupFailed = $false

function Invoke-PortableVerification {
    & python -m pytest tests -q -p no:cacheprovider --basetemp $testRoot
    if ($LASTEXITCODE -ne 0) {
        $script:exitCode = $LASTEXITCODE
        return
    }

    & python -m compileall -q src tests
    if ($LASTEXITCODE -ne 0) {
        $script:exitCode = $LASTEXITCODE
        return
    }

    $script:exitCode = 0
}

try {
    if (Test-Path -LiteralPath $testRoot) {
        throw "Refusing to reuse portable test root: $testRoot"
    }
    New-Item -ItemType Directory -Path $testRoot | Out-Null
    $created = Get-Item -LiteralPath $testRoot -Force
    if (-not $created.PSIsContainer) { throw 'Portable test root is not a directory' }
    if (($created.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw 'Portable test root is a reparse point'
    }
    if ((Split-Path -Parent $created.FullName) -cne $tempParent) {
        throw 'Portable test root parent differs'
    }

    Set-Location -LiteralPath $repo
    $env:PYTHONPATH = $sourceRoot
    Invoke-PortableVerification
}
catch {
    Write-Error $_
    $exitCode = 1
}
finally {
    Set-Location -LiteralPath $originalLocation
    $env:PYTHONPATH = $originalPythonPath
    if (Test-Path -LiteralPath $testRoot) {
        try {
            $candidate = Get-Item -LiteralPath $testRoot -Force
            if (-not $candidate.PSIsContainer) { throw 'Cleanup target is not a directory' }
            if (($candidate.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw 'Cleanup target is a reparse point'
            }
            if ((Split-Path -Parent $candidate.FullName) -cne $tempParent) {
                throw 'Cleanup target parent differs'
            }
            if ($candidate.Name -cne $testLeaf) { throw 'Cleanup target leaf differs' }
            if ($env:OS -eq 'Windows_NT') {
                & icacls.exe $candidate.FullName /reset /T /C /Q 2>&1 | Out-Null
                if ($LASTEXITCODE -ne 0) { throw 'Portable test ACL reset failed' }
            }
            Remove-Item -LiteralPath $candidate.FullName -Recurse -Force
        }
        catch {
            Write-Error $_
            $cleanupFailed = $true
        }
    }
}

if ($cleanupFailed -and $exitCode -eq 0) { $exitCode = 1 }
exit $exitCode
