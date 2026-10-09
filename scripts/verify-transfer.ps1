$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ChecksumFile = Join-Path $ProjectRoot "SHA256SUMS"
if (-not (Test-Path $ChecksumFile)) {
    throw "SHA256SUMS was not found. This is not a packaged transfer bundle."
}

$Failures = 0
Get-Content $ChecksumFile | ForEach-Object {
    if ($_ -match '^([a-f0-9]{64})  (.+)$') {
        $Expected = $Matches[1]
        $Relative = $Matches[2].Replace('/', [IO.Path]::DirectorySeparatorChar)
        $Path = Join-Path $ProjectRoot $Relative
        if (-not (Test-Path $Path)) {
            Write-Host "MISSING $Relative" -ForegroundColor Red
            $Failures++
        } else {
            $Actual = (Get-FileHash -Algorithm SHA256 $Path).Hash.ToLower()
            if ($Actual -ne $Expected) {
                Write-Host "MISMATCH $Relative" -ForegroundColor Red
                $Failures++
            }
        }
    }
}
if ($Failures -gt 0) { throw "$Failures transfer validation failure(s)." }
Write-Host "Transfer bundle checksums are valid." -ForegroundColor Green

