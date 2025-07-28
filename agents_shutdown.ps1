# Ports used by the agents
$ports = @(8001, 8002, 8003)

Write-Host "Searching for processes using ports $($ports -join ', ')" -ForegroundColor Cyan

foreach ($port in $ports) {
    # Find the process using the port
    $process = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | 
               Select-Object -ExpandProperty OwningProcess -ErrorAction SilentlyContinue |
               Get-Process -ErrorAction SilentlyContinue

    if ($process) {
        Write-Host "Found process using port $port" -ForegroundColor Yellow
        Write-Host "  PID: $($process.Id)"
        Write-Host "  Name: $($process.ProcessName)"
        Write-Host "  Path: $($process.Path)"
        
        # Kill the process
        try {
            Stop-Process -Id $process.Id -Force
            Write-Host "  Successfully terminated process" -ForegroundColor Green
        }
        catch {
            Write-Host "  Failed to terminate process: $_" -ForegroundColor Red
        }
    }
    else {
        Write-Host "No process found using port $port" -ForegroundColor Gray
    }
    Write-Host "----------------------------------------"
}

Write-Host "Done!" -ForegroundColor Green
