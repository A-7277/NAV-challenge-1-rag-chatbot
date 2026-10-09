param(
    [switch]$IngestIfMissing
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Checking Docker Desktop..."
docker version | Out-Null
docker compose version | Out-Null

$Drive = Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($ProjectRoot).TrimEnd(':\'))
if ($Drive.Free -lt 30GB) {
    Write-Warning "Less than 30 GB is free. Full corpus ingestion may fail."
}
$MemoryBytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
if ($MemoryBytes -lt 16GB) {
    Write-Warning "Less than 16 GB RAM detected. Keep ingestion workers at one."
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env. Add GROQ_API_KEY before generation." -ForegroundColor Yellow
}

foreach ($Directory in @("data", "artifacts", "models")) {
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
}

Write-Host "Building from the source code in $ProjectRoot..."
docker compose build app
docker compose up -d qdrant

$Snapshot = Get-ChildItem "artifacts/snapshots/*.snapshot" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($Snapshot) {
    docker compose run --rm app rag-snapshot restore "/workspace/$($Snapshot.FullName.Substring($ProjectRoot.Length + 1).Replace('\','/'))"
} elseif ($IngestIfMissing) {
    Write-Host "No snapshot found. Downloading models and ingesting the ten-report corpus..." -ForegroundColor Yellow
    docker compose run --rm app rag-models
    docker compose run --rm app rag-ingest --manifest data/corpus.yaml
} else {
    Write-Warning "No Qdrant snapshot found. Run '.\scripts\setup.ps1 -IngestIfMissing' or the documented rag-ingest command before the demo."
}

docker compose up -d app
Write-Host "Application: http://localhost:8501" -ForegroundColor Green
Write-Host "Source remains available locally at: $ProjectRoot"
