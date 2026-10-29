#!/bin/sh
# qwen-fit.sh - can this machine run Qwen3.8-Flash-Next locally?
# Read-only: detects CPU/RAM/GPU/VRAM/free disk and prints which quant packs fit.
# Works on Linux and macOS. Nothing is installed, downloaded or changed.
#   usage: sh qwen-fit.sh
# Notes: "fits" uses RAM+VRAM with 8 GB headroom; unified memory (Apple/Strix Halo) is counted once.
#        The authoritative check is:  strata launch --plan --family qwen --model IQ2_XS

hr() { printf '%.0s-' $(seq 1 62); echo; }
gb() { awk -v v="$1" 'BEGIN{printf "%.1f", v}'; }

case "$(uname -s)" in
  Darwin) os_name="macOS $(sw_vers -productVersion 2>/dev/null)" ;;
  Linux)  os_name="Linux $(uname -r)" ;;
  *)      os_name="$(uname -s) $(uname -r)" ;;
esac
arch=$(uname -m)

# ---- CPU ----
if [ "$(uname -s)" = "Darwin" ]; then
  cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model 2>/dev/null)
  cores=$(sysctl -n hw.ncpu 2>/dev/null)
else
  cpu=$(grep -m1 '^model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')
  [ -z "$cpu" ] && cpu=$(grep -m1 '^Hardware' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')
  cores=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null)
fi
[ -z "$cpu" ] && cpu="unknown"

# ---- RAM (GB) ----
if [ "$(uname -s)" = "Darwin" ]; then
  ram_bytes=$(sysctl -n hw.memsize 2>/dev/null)
else
  ram_kb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null)
  [ -n "$ram_kb" ] && ram_bytes=$((ram_kb * 1024))
fi
[ -n "$ram_bytes" ] && ram_gb=$(awk -v b="$ram_bytes" 'BEGIN{printf "%.1f", b/1073741824}')

# ---- GPU / VRAM ----
gpu_lines=""
vram_mib=0
unified=0

if command -v nvidia-smi >/dev/null 2>&1; then
  while read -r name mem; do
    [ -z "$name" ] && continue
    mem_mib=${mem% *}; mem_mib=${mem_mib%%MiB*}; mem_mib=$(echo "$mem_mib" | tr -dc '0-9')
    [ -z "$mem_mib" ] && mem_mib=0
    vram_mib=$((vram_mib + mem_mib))
    gpu_lines="$gpu_lines  NVIDIA: $name - $(gb "$(awk -v m="$mem_mib" 'BEGIN{print m/1024}')") GB VRAM
"
  done <<EOF
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)
EOF
fi

# AMD (Linux amdgpu sysfs)
for d in /sys/class/drm/card*/device; do
  [ -r "$d/mem_info_vram_total" ] || continue
  vbytes=$(cat "$d/mem_info_vram_total" 2>/dev/null)
  [ -z "$vbytes" ] && continue
  [ "$vbytes" = "0" ] && continue
  vg=$(awk -v b="$vbytes" 'BEGIN{printf "%.1f", b/1073741824}')
  gpu_lines="$gpu_lines  AMD: amdgpu card - $vg GB VRAM
"
  vram_mib=$((vram_mib + vbytes / 1048576))
done

# AMD (rocm-smi fallback, if sysfs was unreadable)
if [ "$vram_mib" -eq 0 ] && command -v rocm-smi >/dev/null 2>&1; then
  rtot=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Total Memory/{gsub(/[^0-9]/,"",$NF); s+=$NF} END{print s+0}')
  if [ "${rtot:-0}" -gt 0 ]; then
    vram_mib=$((rtot / 1048576))
    gpu_lines="$gpu_lines  AMD: ROCm GPU - $(gb "$(awk -v m="$vram_mib" 'BEGIN{print m/1024}')") GB VRAM
"
  fi
fi

# Apple Silicon: always unified memory, even when system_profiler finds the GPU
if [ "$(uname -s)" = "Darwin" ]; then
  gname=$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chipset Model/{print $2; exit}')
  [ -z "$gname" ] && [ "$arch" = "arm64" ] && gname="Apple Silicon GPU"
  [ -n "$gname" ] && gpu_lines="$gpu_lines  Apple: $gname (unified memory)
"
  [ "$arch" = "arm64" ] && unified=1
fi

# Models store copies in the user's home on most setups.
free_gb=$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2{printf "%.0f", $4/1048576}')
[ -z "$free_gb" ] && free_gb=$(df -Pk / 2>/dev/null | awk 'NR==2{printf "%.0f", $4/1048576}')

echo
echo "=== Qwen3.8-Flash-Next local fit check ==="
echo "OS      : $os_name ($arch)"
echo "CPU     : $cpu ${cores:+($cores cores)}"
echo "RAM     : ${ram_gb:-unknown} GB"
if [ -n "$gpu_lines" ]; then
  printf '%s' "$gpu_lines"
else
  echo "GPU     : none detected (nvidia-smi / amdgpu / ROCm not found)"
fi
[ "$unified" -eq 1 ] && echo "          unified memory: RAM and VRAM are the same pool, not added"
echo "Disk    : $HOME has ${free_gb:-?} GB free (smallest pack ~80 GB, Q8_0 ~190 GB)"
hr

# ---- verdict ----
has_gpu=0
[ "$vram_mib" -gt 0 ] && has_gpu=1
[ "$unified" -eq 1 ] && has_gpu=1
# Strata's documented rule: on a discrete GPU, RAM must hold the experts plus about 10 GB.
# A bigger card makes it faster; it does not lower the RAM needed (except in low-RAM mode).
# Unified memory (Apple Silicon, Strix Halo) is one pool and counts once.
if [ "$unified" -eq 1 ]; then
  usable=$(awk -v r="${ram_gb:-0}" 'BEGIN{printf "%.1f", r-6}')
  basis="unified memory minus 6 GB for the OS"
  best=$(awk -v u="$usable" 'BEGIN{
    n=split("IQ3_S:54.8 IQ3_XXS:47.0 IQ2_XS:39.2 Q2_0:37.6 Gyro-S:27.6", a, " ");
    for (i=1;i<=n;i++) { split(a[i], p, ":"); if (u+0 >= p[2]+0) { print p[1] " " p[2]; exit } }
    print "none 0"
  }')
else
  usable=$(awk -v r="${ram_gb:-0}" 'BEGIN{printf "%.1f", r-10}')
  basis="system RAM minus 10 GB for the OS"
  best=$(awk -v u="$usable" 'BEGIN{
    n=split("IQ3_S:50 IQ3_XXS:43 IQ2_XS:35.5 Q2_0:34 Coder:23", a, " ");
    for (i=1;i<=n;i++) { split(a[i], p, ":"); if (u+0 >= p[2]+0) { print p[1] " " p[2]; exit } }
    print "none 0"
  }')
fi

pack=$(echo "$best" | cut -d' ' -f1)
need=$(echo "$best" | cut -d' ' -f2)
vram_gb=$(awk -v m="$vram_mib" 'BEGIN{printf "%.1f", m/1024}')

echo "Budget  : $usable GB of $basis"
[ "$pack" != "none" ] && echo "Fits    : $pack (~$need GB of weights)"

if [ "$has_gpu" -eq 0 ]; then
  echo "Verdict : no supported GPU detected - a 6B-active MoE on CPU alone is impractical"
  echo "          use the Qwen API, or Qwen3.8-27B (18 GB) via ollama run qwen3.8"
else
  case "$pack" in
    IQ3_S)    echo "Verdict : GOOD   - strata launch --family qwen --model IQ3_S" ;;
    IQ3_XXS)  echo "Verdict : GOOD   - strata launch --family qwen --model IQ3_XXS" ;;
    IQ2_XS)   echo "Verdict : GOOD   - strata launch --family qwen --model IQ2_XS" ;;
    Q2_0)     echo "Verdict : OK     - strata launch --family qwen --model Q2_0" ;;
    Gyro-S)   echo "Verdict : TIGHT  - strata launch --family gyro --model Gyro-S (lowest quality)" ;;
    Coder)    echo "Verdict : CODE   - strata launch --family coder --model IQ1_M (weak outside code)" ;;
    *)
      # 32 GB tier: Strata's low-RAM mode reads experts from the SSD
      ram_ok=$(awk -v r="${ram_gb:-0}" 'BEGIN{print (r+0 >= 32)}')
      v24=$(awk -v v="$vram_gb" 'BEGIN{print (v+0 >= 20)}')
      v12=$(awk -v v="$vram_gb" 'BEGIN{print (v+0 >= 12)}')
      if [ "$ram_ok" = "1" ] && [ "$v24" = "1" ]; then
        echo "Verdict : LOW-RAM - strata launch --family qwen --model Q2_0 (experts streamed from SSD)"
      elif [ "$ram_ok" = "1" ] && [ "$v12" = "1" ]; then
        echo "Verdict : LOW-RAM - strata launch --family coder --model IQ1_M (needs a 12-16 GB card)"
      else
        echo "Verdict : NO     - not enough memory for any pack"
        echo "          use the Qwen API (\$0.15/\$0.47 per 1M tokens), or Qwen3.8-27B (18 GB)"
      fi ;;
  esac
fi
[ "$unified" -eq 0 ] && [ "$has_gpu" -eq 1 ] && [ "$pack" != "none" ] && \
  awk -v v="$vram_gb" 'BEGIN{ if (v+0 < 24) print "Note    : under 24 GB VRAM the image reader runs on the CPU (slow, still works)" }'

if [ -n "$free_gb" ] && [ "$free_gb" -lt 80 ]; then
  echo "Disk    : WARNING - free up space, the smallest packs need ~80 GB"
fi
hr
echo "Authoritative check after install (no download):"
echo "  strata launch --plan --family qwen --model IQ2_XS"
echo
