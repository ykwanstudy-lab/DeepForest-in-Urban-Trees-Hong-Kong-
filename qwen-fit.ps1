# qwen-fit.ps1 - can this Windows machine run Qwen3.8-Flash-Next locally?
# Read-only: reports CPU/RAM/GPU/VRAM/free disk and which Strata quant pack fits.
#   usage: powershell -ExecutionPolicy Bypass -File .\qwen-fit.ps1
# Nothing is installed, downloaded or changed.

$ErrorActionPreference = 'SilentlyContinue'

function Line { Write-Host ('-' * 62) }

$os   = (Get-CimInstance Win32_OperatingSystem).Caption
$arch = $env:PROCESSOR_ARCHITECTURE
$cpu  = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
$cpu  = if ($cpu) { $cpu.Trim() } else { 'unknown' }
$cores = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum

$cs    = Get-CimInstance Win32_ComputerSystem
$ramGB = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
$sysDrive = if ($env:SystemDrive) { $env:SystemDrive.TrimEnd(':') } else { 'C' }
$freeGB = [math]::Round((Get-PSDrive $sysDrive).Free / 1GB, 0)

# --- GPU / VRAM -------------------------------------------------------------
$gpus = @()
$vramMB = 0

$smi = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $smi = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
}
if ($smi) {
    foreach ($row in $smi) {
        if ($row -match '^\s*(.+?),\s*(\d+)\s*MiB') {
            $name = $Matches[1].Trim()
            $mb   = [int]$Matches[2]
            $gpus += "  NVIDIA: $name - $([math]::Round($mb/1024,1)) GB VRAM"
            $vramMB += $mb
        }
    }
}

if ($vramMB -eq 0) {
    # AdapterRAM is 32-bit and caps at 4 GB, so read the 64-bit registry value instead
    $reg = Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0*" `
           -Name 'HardwareInformation.qwMemorySize' 2>$null
    if ($reg) {
        $best = ($reg | ForEach-Object { $_.'HardwareInformation.qwMemorySize' } | Sort-Object -Descending |
                 Select-Object -First 1)
        if ($best -gt 0) { $vramMB = [int]($best / 1MB) }
    }
    $names = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name)
    foreach ($n in $names) {
        if ($vramMB -gt 0) { $gpus += "  GPU: $n - $([math]::Round($vramMB/1024,1)) GB VRAM" } else { $gpus += "  GPU: $n - VRAM unknown" }
    }
}

$vramGB  = [math]::Round($vramMB / 1024, 1)
$hasGpu  = ($vramMB -gt 0)
$unified = $false   # no unified-memory x86 laptops in scope here

Write-Host ''
Write-Host '=== Qwen3.8-Flash-Next local fit check ==='
Write-Host "OS      : $os ($arch)"
Write-Host "CPU     : $cpu $(if ($cores) { "($cores cores)" })"
Write-Host "RAM     : $ramGB GB"
if ($gpus.Count -gt 0) { $gpus | ForEach-Object { Write-Host $_ } }
else { Write-Host 'GPU     : none detected (nvidia-smi / registry found no adapter)' }
Write-Host "Disk    : $($sysDrive):\ has $freeGB GB free (smallest pack ~80 GB, Q8_0 ~190 GB)"
Line

# --- verdict ----------------------------------------------------------------
# Strata's rule: RAM must hold the experts plus ~10 GB. VRAM buys speed, not a lower RAM need.
$usable = [math]::Round($ramGB - 10, 1)

# pack name, experts GB (discrete-GPU rule)
$packs = @(
    @{ Name = 'IQ3_S';   GB = 50.0 },
    @{ Name = 'IQ3_XXS'; GB = 43.0 },
    @{ Name = 'IQ2_XS';  GB = 35.5 },
    @{ Name = 'Q2_0';    GB = 34.0 },
    @{ Name = 'Coder';   GB = 23.0 }
)

$pack = 'none'
$need = 0
foreach ($p in $packs) { if ($usable -ge $p.GB) { $pack = $p.Name; $need = $p.GB; break } }

Write-Host "Budget  : $usable GB of system RAM minus 10 GB for Windows"
if ($pack -ne 'none') { Write-Host "Fits    : $pack (~$need GB of weights)" }

if (-not $hasGpu) {
    Write-Host 'Verdict : no supported GPU - a 6B-active MoE on CPU alone is impractical'
    Write-Host '          use the Qwen API, or Qwen3.8-27B (18 GB) via: ollama run qwen3.8'
}
else {
    switch ($pack) {
        'IQ3_S'   { Write-Host 'Verdict : GOOD   - strata launch --family qwen --model IQ3_S' }
        'IQ3_XXS' { Write-Host 'Verdict : GOOD   - strata launch --family qwen --model IQ3_XXS' }
        'IQ2_XS'  { Write-Host 'Verdict : GOOD   - strata launch --family qwen --model IQ2_XS' }
        'Q2_0'    { Write-Host 'Verdict : OK     - strata launch --family qwen --model Q2_0' }
        'Coder'   { Write-Host 'Verdict : CODE   - strata launch --family coder --model IQ1_M (weak outside code)' }
        default {
            if ($ramGB -ge 32 -and $vramGB -ge 20) {
                Write-Host 'Verdict : LOW-RAM - strata launch --family qwen --model Q2_0 (experts streamed from SSD)'
            }
            elseif ($ramGB -ge 32 -and $vramGB -ge 12) {
                Write-Host 'Verdict : LOW-RAM - strata launch --family coder --model IQ1_M (needs a 12-16 GB card)'
            }
            else {
                Write-Host 'Verdict : NO     - not enough memory for any pack'
                Write-Host '          use the Qwen API ($0.15/$0.47 per 1M tokens), or Qwen3.8-27B (18 GB)'
            }
        }
    }
    if ($vramGB -lt 24 -and $pack -ne 'none') {
        Write-Host 'Note    : under 24 GB VRAM the image reader runs on the CPU (slow, still works)'
    }
}

if ($freeGB -lt 80) { Write-Host 'Disk    : WARNING - free up space, the smallest packs need ~80 GB' }
Line
Write-Host 'Authoritative check after install (no download):'
Write-Host '  strata launch --plan --family qwen --model IQ2_XS'
Write-Host ''
