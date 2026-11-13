param([switch]$WaitForExit)

$ErrorActionPreference = 'Stop'
$product = 'PyCharm2025.3'
$source = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "JetBrains/$product"))
$targetRoot = [IO.Path]::GetFullPath("E:/JetBrains/$product")
$destination = Join-Path $targetRoot 'system'
$settings = Join-Path $env:APPDATA "JetBrains/$product"
$statusPath = Join-Path $targetRoot 'migration-status.json'
New-Item -ItemType Directory -Path $targetRoot -Force | Out-Null

function Write-MigrationStatus($state, $message) {
    @{status=$state; message=$message; source=$source; destination=$destination; time=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

function Get-RunningPyCharm {
    Get-Process -Name pycharm64 -ErrorAction SilentlyContinue
}

try {
    # Copy only the explicitly identified IDE directory. Keep the original as a
    # recovery backup; never delete or move live Local History or chat databases.
    if (-not $source.EndsWith('\JetBrains\PyCharm2025.3', [StringComparison]::OrdinalIgnoreCase) -or
        -not $destination.StartsWith('E:\JetBrains\PyCharm2025.3\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Unexpected storage paths; refusing relocation.'
    }
    do {
        while (Get-RunningPyCharm) {
            if (-not $WaitForExit) { throw 'Close PyCharm first, or use -WaitForExit.' }
            Write-MigrationStatus 'waiting_for_pycharm_exit' 'Close PyCharm and allow the copy to finish before reopening.'
            Start-Sleep -Seconds 5
        }
        Write-MigrationStatus 'copying' 'Copying IDE storage to E:; the original remains available for recovery.'
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        & robocopy.exe $source $destination /E /COPY:DAT /DCOPY:DAT /XJ /R:2 /W:1 /NP /NFL /NDL "/LOG:$(Join-Path $targetRoot 'migration-copy.log')" | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "IDE storage copy failed (robocopy exit $LASTEXITCODE). Settings were not changed." }
        # If the IDE reopened while copying, leave its configuration untouched
        # and repeat the copy after it next exits.
    } while (Get-RunningPyCharm)

    $properties = Join-Path $settings 'idea.properties'
    $options = Join-Path $settings 'pycharm64.exe.vmoptions'
    New-Item -ItemType Directory -Path $settings -Force | Out-Null
    foreach ($path in @($properties, $options)) {
        if (Test-Path -LiteralPath $path) {
            Copy-Item -LiteralPath $path -Destination (Join-Path $targetRoot ((Split-Path $path -Leaf) + '.before-relocation')) -Force
        }
    }
    $lines = if (Test-Path -LiteralPath $properties) { @(Get-Content -LiteralPath $properties) } else { @() }
    $lines = @($lines | Where-Object { $_ -notmatch '^\s*idea\.(system|log)\.path\s*=' })
    $lines += 'idea.system.path=E:/JetBrains/PyCharm2025.3/system'
    $lines += 'idea.log.path=E:/JetBrains/PyCharm2025.3/log'
    $vm = if (Test-Path -LiteralPath $options) { @(Get-Content -LiteralPath $options) } else { @() }
    $vm = @($vm | Where-Object { $_ -notmatch '^\s*-Djava\.io\.tmpdir=' })
    $vm += '-Djava.io.tmpdir=E:/JetBrains/PyCharm2025.3/tmp'
    New-Item -ItemType Directory -Path (Join-Path $targetRoot 'tmp'),(Join-Path $targetRoot 'log') -Force | Out-Null
    [IO.File]::WriteAllLines($properties, [string[]]$lines, [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllLines($options, [string[]]$vm, [Text.UTF8Encoding]::new($false))
    Write-MigrationStatus 'complete' 'PyCharm may now be reopened. Caches, Local History, logs and Java temp use E:. The original C: directory is retained as a backup.'
} catch {
    Write-MigrationStatus 'failed' $_.Exception.Message
    throw
}
