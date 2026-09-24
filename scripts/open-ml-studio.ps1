param(
    [string]$JumpboxIp,
    [int]$SocksPort = 1080
)

if (-not $JumpboxIp) {
    $JumpboxIp = & terraform "-chdir=terraform" output -raw jumpbox_public_ip 2>$null
}

if (-not $JumpboxIp) {
    Write-Error "No jump box address. Pass -JumpboxIp <ip>, or run from the repository root with Terraform initialised and enable_jumpbox = true."
    exit 1
}

$tunnelUp = Test-NetConnection -ComputerName "127.0.0.1" -Port $SocksPort -InformationLevel Quiet -WarningAction SilentlyContinue

if (-not $tunnelUp) {
    Write-Host "Starting SSH SOCKS tunnel through the jump box..."
    Start-Process ssh -ArgumentList "-N", "-D", "$SocksPort", "azureuser@$JumpboxIp" -WindowStyle Hidden
    Start-Sleep -Seconds 3
} else {
    Write-Host "SOCKS tunnel already up on port $SocksPort."
}

Write-Host "Launching proxied Chrome..."
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
    --proxy-server="socks5://127.0.0.1:$SocksPort" `
    --user-data-dir="$env:TEMP\chrome-ml-proxy" `
    "https://ml.azure.com"
