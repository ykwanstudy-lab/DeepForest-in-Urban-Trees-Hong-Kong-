#!/usr/bin/env python3
"""Static inspection of the SilvaLiDAR Air Windows installer (PE analysis).

No third-party deps: parses the DOS/PE headers, section table, imports and the
VERSIONINFO resource directly out of the raw bytes.
"""
import datetime
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = path.read_bytes()
print(f"file        : {path}")
print(f"size        : {len(data):,} bytes")

if data[:2] != b"MZ":
    sys.exit("not a PE file (no MZ magic)")
e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
print(f"e_lfanew    : 0x{e_lfanew:x}")
print(f"DOS stub    : {data[0x40:e_lfanew][:60]!r}")

assert data[e_lfanew:e_lfanew + 4] == b"PE\0\0", "no PE signature"
coff = e_lfanew + 4
machine, nsec, timestamp, _, _, optsize, chars = struct.unpack_from("<HHIIIHH", data, coff)
MACHINE = {0x14C: "i386 (32-bit)", 0x8664: "x86-64 (AMD64)", 0xAA64: "ARM64"}
print(f"machine     : 0x{machine:04x} {MACHINE.get(machine, '?')}")
print(f"sections    : {nsec}")
print(f"timestamp   : {datetime.datetime.utcfromtimestamp(timestamp)} UTC")
print(f"characterist: 0x{chars:04x}  dll={bool(chars & 0x2000)} exe={bool(chars & 0x2)}")

opt = coff + 20
magic = struct.unpack_from("<H", data, opt)[0]
pe32plus = magic == 0x20B
print(f"opt magic   : 0x{magic:04x} ({'PE32+' if pe32plus else 'PE32'})")
entry = struct.unpack_from("<I", data, opt + 16)[0]
if pe32plus:
    imagebase = struct.unpack_from("<Q", data, opt + 0x18)[0]
    subsysver = struct.unpack_from("<HH", data, opt + 0x30)
else:
    imagebase = struct.unpack_from("<I", data, opt + 0x1C)[0]
    subsysver = struct.unpack_from("<HH", data, opt + 0x30)
subsystem = struct.unpack_from("<H", data, opt + 0x44)[0]
SUBSYS = {1: "Native", 2: "Windows GUI", 3: "Windows CUI (console)"}
print(f"entrypoint  : 0x{entry:x}  imagebase=0x{imagebase:x}")
print(f"subsystem   : {subsystem} {SUBSYS.get(subsystem, '')} "
      f"(min OS {subsysver[0]}.{subsysver[1]})")
dllchars = struct.unpack_from("<H", data, opt + (0x46 if pe32plus else 0x46))[0]
print(f"dll chars   : 0x{dllchars:04x}")
numdir = struct.unpack_from("<I", data, opt + (0x6C if pe32plus else 0x5C))[0]
dd_off = opt + (0x70 if pe32plus else 0x60)
dirs = {}
for i in range(min(numdir, 16)):
    rva, size = struct.unpack_from("<II", data, dd_off + i * 8)
    if rva:
        dirs[i] = (rva, size)
DIRNAMES = {0: "Export", 1: "Import", 2: "Resource", 3: "Exception", 4: "Certificate",
            5: "BaseReloc", 6: "Debug", 9: "TLS", 10: "LoadConfig", 12: "IAT",
            14: "CLR/COM descriptor"}
for i, (rva, size) in sorted(dirs.items()):
    print(f"  data dir {i:2d} {DIRNAMES.get(i, ''):22s} rva=0x{rva:<8x} size={size:,}")

sections = []
sec_off = opt + optsize
for i in range(nsec):
    name, vsize, vaddr, rsize, raddr = struct.unpack_from("<8sIIII", data, sec_off + i * 40)
    raw_chars = struct.unpack_from("<I", data, sec_off + i * 40 + 36)[0]
    sections.append((name.rstrip(b"\0").decode(errors="replace"), vaddr, vsize, raddr, rsize))
    print(f"  section {name.rstrip(chr(0).encode()).decode():8s} vaddr=0x{vaddr:<7x} "
          f"vsize={vsize:<10,} raw=0x{raddr:<8x} rsize={rsize:,} flags=0x{raw_chars:08x}")


def rva_to_off(rva):
    for _, vaddr, vsize, raddr, rsize in sections:
        if vaddr <= rva < vaddr + max(vsize, rsize):
            return raddr + (rva - vaddr)
    return None


if 1 in dirs:
    off = rva_to_off(dirs[1][0])
    dlls = []
    while off:
        ilt, tstamp, fwd, name_rva, iat = struct.unpack_from("<IIIII", data, off)
        if not name_rva:
            break
        no = rva_to_off(name_rva)
        dlls.append(data[no:data.index(b"\0", no)].decode(errors="replace"))
        off += 20
    print(f"imports     : {len(dlls)} DLLs")
    for d in sorted(dlls):
        print(f"    {d}")

if 4 in dirs:
    rva, size = dirs[4]
    sig = data[rva:rva + 8]
    if sig[:4] == bytes([0x30, 0x82]):
        print("authenticode: PKCS#7 blob at 0x%x, %s bytes" % (rva, format(size, ",")))
    else:
        print("authenticode: unexpected magic %r" % sig)
else:
    print("authenticode: ABSENT - installer is NOT code-signed")

if 2 in dirs:
    print("resources   : present (16=VERSION, 24=MANIFEST, 3/14=ICON)")
    for key in (b"VS_VERSION_INFO", b"ProductName", b"FileDescription", b"CompanyName",
                b"ProductVersion", b"FileVersion", b"LegalCopyright", b"OriginalFilename"):
        idx = data.find(key)
        if idx == -1:
            continue
        raw = data[idx + len(key):idx + len(key) + 400]
        try:
            txt = raw[4:4 + struct.unpack_from("<H", raw, 2)[0] * 2].decode("utf-16-le")
        except Exception:
            txt = ""
        print(f"  {key.decode():18s}: {txt.strip()!r}")

fingerprints = {
    "Inno Setup": [b"Inno Setup", b"JR.Inno.Setup", b"InnoSetupVersion"],
    "NSIS": [b"Nullsoft.NSIS", b"NSIS Error", b"nsExec"],
    "InstallShield": [b"InstallShield"],
    "MSI/WiX bootstrapper": [b"WixBundle", b"Windows Installer"],
    "PyInstaller": [b"pyi-", b"_MEIPASS", b"PyInstaller", b"pyimod"],
    "Electron": [b"electron.asar", b"app.asar", b"ELECTRON_RUN_AS_NODE"],
    "Python runtime": [b"python3", b"python31", b"python_d.exe", b"Py_Initialize"],
    "Qt": [b"Qt5Core", b"Qt6Core", b"Qt5Widgets", b"Qt6Widgets"],
    "Tauri/Rust": [b"tauri", b"webview2"],
}
print("fingerprints:")
for label, needles in fingerprints.items():
    hits = [n.decode(errors="replace") for n in needles if n in data]
    if hits:
        print(f"  {label:24s}: {hits}")


def parse_version_info(data):
    """Pull the VERSIONINFO resource fields out of a PE image."""
    sig = data.find(bytes([0xBD, 0x04, 0xEF, 0xFE]))
    if sig < 0:
        print("version info: none")
        return
    ver = struct.unpack_from("<I", data, sig + 4)[0]
    fvms, fvls, pvms, pvls = struct.unpack_from("<IIII", data, sig + 8)
    flags_mask, flags = struct.unpack_from("<II", data, sig + 24)
    print(f"version info: struc={ver >> 16}.{ver & 0xFFFF}  "
          f"FileVersion={fvms >> 16}.{fvms & 0xFFFF}.{fvls >> 16}.{fvls & 0xFFFF}  "
          f"ProductVersion={pvms >> 16}.{pvms & 0xFFFF}.{pvls >> 16}.{pvls & 0xFFFF}")
    print(f"              flags=0x{flags:08x} (mask 0x{flags_mask:08x})"
          f"{' [DEBUG]' if flags & 1 else ''}{' [PRERELEASE]' if flags & 2 else ''}"
          f"{' [PATCHED]' if flags & 4 else ''}{' [PRIVATEBUILD]' if flags & 8 else ''}"
          f"{' [SPECIALBUILD]' if flags & 0x10 else ''}")
    key = "VS_VERSION_INFO".encode("utf-16-le")
    start = data.rfind(key, 0, sig + 4096)
    if start < 0:
        start = sig
    end = min(len(data), sig + 8192)
    blob = data[start:end]
    strings = []
    cur = bytearray()
    i = 0
    while i < len(blob) - 1:
        ch = blob[i] | (blob[i + 1] << 8)
        if 0x20 <= ch < 0x7F:
            cur.append(ch)
            i += 2
        else:
            if len(cur) >= 3:
                strings.append(cur.decode("ascii"))
            cur = bytearray()
            i += 1 if ch != 0 else 2
    if len(cur) >= 3:
        strings.append(cur.decode("ascii"))
    for s in strings[:40]:
        print(f"              {s!r}")


if 0 in dirs:
    off = rva_to_off(dirs[0][0])
    nfun, nname = struct.unpack_from("<II", data, off + 20)
    print(f"exports     : {nfun} functions, {nname} names")
    if nname:
        name_rvas = off + 32
        names = []
        for i in range(min(nname, 60)):
            rva = struct.unpack_from("<I", data, name_rvas + i * 4)[0]
            o = rva_to_off(rva)
            names.append(data[o:data.index(b"\0", o)].decode(errors="replace"))
        for n in names:
            print(f"    {n}")

parse_version_info(data)
