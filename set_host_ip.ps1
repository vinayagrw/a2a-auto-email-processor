# This script sets the HOST_IP environment variable for Docker Compose
# Get the default gateway IP address (host IP from container's perspective)
$hostIp = (Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -ne $null -and $_.NetAdapter.Status -eq "Up" }).IPv4Address.IPAddress

# Write to a temporary .env file
echo "HOST_IP=$hostIp" | Out-File -FilePath .env -Encoding ASCII

Write-Host "Set HOST_IP to $hostIp in .env file"
