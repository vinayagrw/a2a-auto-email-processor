# Create necessary directories if they don't exist
if (-not (Test-Path -Path "./data")) {
    New-Item -ItemType Directory -Path "./data" | Out-Null
}
if (-not (Test-Path -Path "./output")) {
    New-Item -ItemType Directory -Path "./output" | Out-Null
}
if (-not (Test-Path -Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs" | Out-Null
}

# Build and start the services
Write-Host "Building and starting Docker services..." -ForegroundColor Cyan
docker-compose up --build -d

# Show the status of the services
Write-Host "`nService Status:" -ForegroundColor Green
docker-compose ps

Write-Host "`nServices are running!" -ForegroundColor Green
Write-Host "- Email Processor: http://localhost:8001"
Write-Host "- Response Agent:  http://localhost:8002"
Write-Host "- Summary Agent:   http://localhost:8003"
Write-Host "`nTo stop the services, run: docker-compose down" -ForegroundColor Yellow
