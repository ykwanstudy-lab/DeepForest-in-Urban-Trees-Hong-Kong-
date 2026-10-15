# SilvaLiDAR Air 下載檔研究報告

- 研究對象：`https://carlos-alberto-silva.github.io/silvalab/downloads/SilvaLiDAR_Air_Setup_latest_x64.exe`
- 下載頁：`https://carlos-alberto-silva.github.io/silvalab/silvalidar-air.html`
- 研究日期：2026-10-15
- 方法：直接下載 → PE 靜態分析（自寫 parser）→ 用自行編譯嘅 `innoextract 1.13.0` 解包 → 檢查 payload、內附文件、字串、匯出/匯入表

---

## 0. 一句總結

呢個「下載檔」係一個 **Inno Setup 打包嘅 Windows x64 安裝程式**（32-bit stub + x64 payload），
內含一個 **Qt 6.11.1 + C++ (MSVC 2022) 寫嘅桌面 GIS／LiDAR 處理軟件**，
**完全無 Python／Electron／.NET，亦無任何機器學習模型檔**；
安裝包**無數位簽章、無密碼**，所以任何人都可以完整解包檢查（本報告就係咁做）。

---

## 1. 檔案身份

| 項目 | 值 |
|---|---|
| HTTP 狀態 | `200`（GitHub Pages / Fastly，`content-type: application/octet-stream`） |
| Content-Length | **60,495,940 bytes**（57.7 MiB，即頁面寫嘅「60.5 MB」） |
| Last-Modified | `Tue, 13 Oct 2026 14:45:04 GMT` |
| ETag | `"6ace43f0-39b1844"` |
| SHA-256 | `dd9dfd183bdd45de3824b32567a01f49e3c09cea50d97736b719c85a805e3c8b` |
| MD5 | `2d7d44bb0bd70590ee1e53b9dee9249d` |
| PE 類型 | PE32，`machine 0x014c i386`（**32-bit stub**），subsystem = Windows GUI |
| 數位簽章 | **無**（PE certificate directory 不存在） |
| VERSIONINFO | ProductName `SilvaLiDAR Air`；FileVersion `0.1.8.11.0`；ProductVersion `0.1.8.11`；CompanyName `SilvaLab`；Comments `This installation was built with Inno Setup.` |
| 打包器 | **Inno Setup**：`Inno Setup Setup Data (6.7.0)`、`Inno Setup Messages (6.5.0)`、`InnoSetupLdrWindow`、imports 只有 `advapi32/comctl32/kernel32/oleaut32/user32` |
| 密碼 | 無（`innoextract --info`：`Setup is not passworded!`） |
| 語言 | english（單一語言） |
| 安裝模式 | stub manifest 為 `requestedExecutionLevel level="asInvoker"`（Inno 6.5+ 支援非管理員安裝模式） |
| 內部時間戳 | stub 2026-05-22（Inno Setup 自身編譯時間）；**payload 檔案 2026-10-12 / 10-13**（即真正 build 時間） |

> 注意：呢個檔案係「安裝程式」，唔係 portable 版。內附 User Guide 提到官方另有 portable package（unzip 即用），但下載頁只提供 installer。

---

## 2. 解包結果：安裝包內有咩

`innoextract --list` 顯示 **499 個檔案項目（493 個唯一路徑）**，全部放喺 `app/` 之下，
解包後約 **187.7 MiB（196,807,534 bytes）**，即安裝後約 188 MB。

### 目錄結構

```
app/
├── SilvaLiDARAir.exe          21.9 MiB  ← 主程式（Qt6 GUI）
├── SilvaLiDARAirCLI.exe        3.0 MiB  ← 命令行版
├── SilvaLaserAirCore.dll       1.06 MiB ← 核心運算引擎（506 exports）
├── BuildingDetection.dll       673 KiB  ← 工具 plugin
├── CanopyGapDetection.dll      673 KiB  ← 工具 plugin
├── LowestPercentileGround.dll  676 KiB  ← 工具 plugin（地面濾波法之一）
├── PowerLineDetection.dll      707 KiB  ← 工具 plugin
├── aec.dll / legacy.dll        32 KiB / 108 KiB（OpenSSL 3.6.4 相關）
├── 121 個 DLL（Qt6、GDAL、PDAL、PROJ、GEOS、ICU78、HDF5、netCDF、spatialite、boost 1.92 …）
├── assets/  report_template*.html、silvaflow/default_template.sfw、home_app_logo.png
├── Libraries/3D Models/pine_tree.glb
├── Documentation/  User Guide、CLI Guide、Plugin Developer Guide、About、Citations、
│                   Open_Source_Components.txt、Third_Party_Notices/（大量 licence 檔）
├── share/gdal/、share/proj/（proj.db 9.7 MiB）← GDAL/PROJ 資料
└── platforms/ styles/ tls/ imageformats/ generic/ networkinformation/ ← Qt plugins
```

### 最大檔案（佔體積者）

| 檔案 | 大小 | 用途 |
|---|---|---|
| `icudt78.dll` | 31.6 MiB | ICU 78 Unicode 資料 |
| `gdal.dll` | 22.1 MiB | GDAL 光柵/向量 IO |
| `SilvaLiDARAir.exe` | 21.9 MiB | 主程式 |
| `share/proj/proj.db` | 9.73 MiB | PROJ 坐標轉換資料庫 |
| `spatialite.dll` | 7.21 MiB | SpatiaLite（SQLite 空間擴展） |
| `Qt6Gui/Widgets/Core.dll` | 6.6 / 5.9 / 5.1 MiB | Qt 6.11.1 |
| `pdalcpp.dll` | 5.49 MiB | **PDAL** 點雲處理 |
| `assimp-vc143-mt.dll` | 5.25 MiB | 3D 模型載入（虛擬森林 glb） |
| `libcrypto-3-x64.dll` | 5.09 MiB | OpenSSL 3.x |
| `pine_tree.glb` | 4.71 MiB | 唯一 3D 樹模型（其餘樹種可能靠程序生成） |

### 值得注意嘅封裝細節

- **無 uninstaller 檔**（`unins000.exe` 係安裝時由 Inno 現場生成，屬正常）。
- **無 VC++ redistributable**：`msvcp140.dll` / `vcruntime140.dll` 唔在包內，但兩個 EXE 都 import 佢 → **目標電腦必須已裝 Microsoft Visual C++ 2015–2022 x64 Redistributable**，否則會開唔到。
- **無 Python**（0 個 `.py/.pyd`）、**無 Electron/asar**、**無 .NET/CLR**。
- **無任何 ML 模型檔**（`.onnx/.pt/.tflite/.h5` 全部 0 個）→ 「AI」係 LLM 助手，唔係內建深度學習偵測器。
- **封裝瑕疵**：`Documentation/` 6 個檔案在清單出現兩次（第二次標示 `- overwritten`），即 installer 重複安裝同一份文件，白白多佔約 0.3 MB。

---

## 3. 技術棧指紋（由 PE import/export、字串、licence 檔交叉核對）

| 層 | 用咩 |
|---|---|
| 語言／編譯器 | C++，MSVC 143（VS 2022 x64），`/openmp`（`VCOMP140.DLL`）、`MSVCP140_ATOMIC_WAIT` |
| GUI | **Qt 6.11.1**（Core/Gui/Widgets/Network/OpenGL/OpenGLWidgets/PrintSupport/Sql/Xml/Concurrent/Test/DBus）+ Qt plugins（platforms、styles `qmodernwindowsstyle.dll`、tls、imageformats） |
| 點雲 | **PDAL**（`pdalcpp.dll`、`libpdal_plugin_kernel_fauxplugin.dll`）+ LASzip、Eigen、nanoflann、Qhull |
| GIS | GDAL、PROJ 9、GEOS、SpatiaLite、libxml2、pugixml、netCDF、HDF5、SQLite、libcurl、OpenSSL 3.6.4、TIFF/JPEG/PNG/WebP/JP2/LERC |
| 其他 | Boost 1.92、ICU 78、FreeType、HarfBuzz、assimp、double-conversion、brotli、zstd、lz4、bz2 |
| 演算法（編入程式） | **LASzip**、**CSF (Cloth Simulation Filter)**、**PTD (Progressive TIN Densification)** + 增量 Delaunay（Shewchuk predicates）、nanoflann、Eigen |
| 授權模式 | 專有但免費（見第 5 節）；Qt 以 LGPL-3 動態連結；**無 GPL 元件** |

主程式 manifest：`requestedExecutionLevel level='asInvoker'`（app 本身唔需要管理員權限），min OS 6.0。

---

## 4. 網絡連線、AI 與其他「隱藏」功能（由字串挖出）

| 功能 | 相關字串／端點 | 備註 |
|---|---|---|
| USGS 3DEP LiDAR 下載 | `--3dep-check`、`--3dep-ui-check` | 內建工具，可直接抓美國公開 LiDAR（TNM / National Map） |
| NAIP 正射影像 | `https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}` | 另有 Mapbox（自備 token）、自訂 XYZ/TMS（只限 HTTPS）、本地 GeoTIFF/COG/JP2 |
| **SilvaAI（LLM 助手）** | `http://127.0.0.1:11434/api/tags`、`/api/pull`（**Ollama**）、`https://ollama.com/download/OllamaSetup.exe`、`https://api.openai.com/v1/audio/speech`（`gpt-4o-mini-tts`）、`/v1/audio/transcriptions`、模型名 `gpt-5-mini`、`claude-sonnet-4-20250514` | 支援 Ollama 本地模型／OpenAI／Anthropic／Gemini／自訂 OpenAI-compatible endpoint；API key 存本機 |
| 語音對話 | SRGS XML 語法 `root="silva"`、TTS/STT 端點 | Windows 語音輸入 + 雲端 TTS/轉錄 |
| **MCP server** | `SilvaLiDARAir.exe --mcp`（加 `--show`） | 可被 Claude Desktop / Claude Code 驅動；提供 `get_status`、`add_point_clouds`、`run_workflow`、`screenshot`、`read_log` 等 |
| FFmpeg（非內附） | `https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip` | 「Flight Through」匯出 MP4 時才按你需要下載，或指向自備 ffmpeg.exe |
| **C2PA 內容憑證** | `c2pa.claim.v2`、`c2pa.hash.data`、`c2pa.watermarked.unbound`、`urn:c2pa:24e74b85-…`、`org.contentauth.c2pa_rs` **0.79.2**、spec **2.2.0**、`SSL.com C2PA ICA R1 2025`、`OpenAI TSA Issuing CA`、`digitalSourceType … trainedAlgorithmicMedia`、`gpt-image`、`the synthetic imagery source is valid` | 主程式**內嵌 c2pa-rs（Rust C2PA 實作）同 OpenAI／SSL.com 憑證鏈**，加上 `gpt-image`、`trainedAlgorithmicMedia`、synthetic imagery 等字串 → 極可能係用嚟**驗證／標示 OpenAI 影像服務產生嘅 AI 影像來源憑證**；注意：用戶文件完全冇提及此功能（屬未公開／內部功能，屬推斷） |
| 設定儲存 | `QSettings`、`LOCALAPPDATA` | 無 registry 依賴 |
| User-Agent | `SilvaLiDARAir/0.1.8.11 (+https://silvalab.ca)` | app 自報官網係 **silvalab.ca** |
| 內建 CLI 選項 | `--tool chm|ground|dtm|trees|canopy-gap|colorize|tiles|merge|homogenize|catalog-pipeline|plot-products|area-metrics|earth-imagery`、`--chm-resolution`、`--csf-threshold`、`--aba-report-job`、`--all-dialogs-smoke-test` … | 大量 `*-check` 係開發者自測入口，亦可作功能探測 |

**遙測／追蹤**：用 `telemetry|analytics|sentry|mixpanel|posthog|segment|crashpad` 等關鍵字掃主程式，**未發現**常見分析或崩潰回報 SDK 字串（只撞到產品文案 “forest analytics”）。此為**關鍵字掃描結論，非完整行為分析**。app 會主動連網（3DEP、NAIP、AI provider、FFmpeg／Ollama 下載），要監控就用防火牆／代理。

---

## 5. 授權（內附 `Documentation/Third_Party_Notices/SilvaLiDAR_Air_LICENSE.txt`）

- 專有軟件，**免費**使用；可裝在任意數量電腦，任何用途（研究、教學、商業、作業）。
- **你用軟件產生嘅成果（raster、vector、點雲、metrics、模型、報告）完全屬於你**，授權無附加條件。
- 限制：不得轉售／轉授權／當成其他產品一部分分發（需書面同意）；**不得反編譯／逆向**（法律強制允許者除外）；不得移除版權／授權標示。**可以免費分享未經修改嘅 installer 或 portable 包**（連 Documentation 一齊）。
- 第三方元件各有自己授權（Qt = LGPL-3，可用自己 build 嘅同名 Qt 6.11.1 替換 DLL）；**無 GPL 元件**。
- 無保養、責任限制、違反條款即終止。

---

## 6. 安全評估

| 項目 | 結論 |
|---|---|
| 數位簽章 | **完全無簽**（installer、兩個 EXE、所有 DLL 都無 Authenticode）→ SmartScreen 會警告；官方文件亦承認 “Windows may warn about an unsigned program”，教你 More info → Run anyway |
| 完整性驗證 | 無官方公佈 hash；本報告已記錄 SHA-256／MD5，可作日後比對 |
| 解包難度 | 無密碼、無加密 → 任何人都可完整檢視內容（對研究／審計有利，亦代表保護只靠授權條款） |
| 權限 | app 為 `asInvoker`（不需 admin）；installer 用 Inno 預設（一般裝到 Program Files，會提權） |
| 執行風險 | 未見可疑行為（無遙測、無常駐服務、無隱藏 exe）；主要風險係**第三方雲端 AI**：用雲端 provider 時，對話內容＋**專案摘要＋視窗截圖**會傳去供應商（官方文件亦有警告） |
| 建議 | 只從官方頁下載並核對 SHA-256；敏感資料用本地 Ollama；先在 VM／受控機試 |


---

## 7. 功能總覽（由內附 User Guide / CLI Guide 整理）

處理鏈（林業標準流程，工具可單獨或串連執行）：

`Noise Removal → Ground Filtering（CSF / PTD / Lowest percentile 等）→ DTM → Height Normalization
→ CHM（多種方法，預設 point-to-raster subcircle 0.15 m；另有 pit-free）→ Individual Tree Detection
& Extraction → Individual Tree Metrics（樹高、冠幅、冠面積、AlphaShape3D 冠體積）
→ Canopy Gap / Cloud Metrics / Grid Metrics / Voxel Metrics → Hillshade/Slope/Aspect/Contours/TRI`

- **SilvaFlow**：圖形化 workflow 編輯器（每個 box 一個工具，可存 `.sfw`），會重用未變更嘅步驟。
- **Catalog**：把 SilvaFlow 分塊（tile，預設 500 m、可 adaptive）平行跑，處理超大範圍，最後合併成品；可設定同時 tile 數、核心數、buffer。
- **資料 IO**：LAS/LAZ（可選保留 RGB／intensity／classification／returns／GPS time／extra bytes）、「Load」或「Streamed collection」兩種載入模式（後者適合超大檔案）；匯出 GeoTIFF / GeoPackage / GeoJSON / Shapefile / LAS/LAZ、截圖。
- **影像**：USGS NAIP、Mapbox、自訂 XYZ/TMS、本地 GeoTIFF/COG/JP2；可 clip 到專案範圍並存成 GeoTIFF；亦可**用影像為點雲上色**（Colorize，含 CRS 自動重投影）。
- **3DEP**：內建下載美國 USGS 3DEP LiDAR（可依範圍／來源／tile 下載）。
- **可視化**：2D 地圖 + 3D Scene（OpenGL、EDL）、動畫（Spin、Flight Through、Walk with the forester、Birds、虛擬森林 3D 模型）。
- **Area-based modeling (ABA)**：載入 plot CSV → 相關性／PCA 探索 → 建模比較 → 驗證 → 製圖與不確定性，輸出 HTML/PDF 報告。
- **報告**：內建 HTML 模板（`report_template*.html`），可出 PDF。
- **SilvaAI**：LLM 助手，知道工具清單、參數同專案狀態，可用自然語言建構並執行 SilvaFlow（有 system prompt 限制佢唔可以亂講能力）；支援語音對話。
- **MCP server**：`--mcp` 俾 Claude Desktop / Claude Code 等直接操作 app（開專案、加點雲、跑 workflow、截圖、讀 log）。
- **CLI**：`SilvaLiDARAirCLI.exe --tool <name>` 覆蓋大部分運算（適合 HPC／伺服器／批次）。

---

## 8. 對本 repo（DeepForest in Urban Trees Hong Kong）嘅意義

- SilvaLiDAR Air **唔係**同 DeepForest 競爭嘅偵測器：佢**無內建深度學習模型**，單株樹偵測係傳統演算法（CHM local maxima / 冠幅分割類），靠 LiDAR 點雲。
- 但佢係一部好用嘅**「地面真值 / 林冠指標生產機」**：
  - 可以喺 Windows 上用 GUI 快速出 CHM、樹高、冠幅、冠體積、grid metrics；
  - 有完整 CLI 可以批次化，方便做對照組（traditional vs deep learning）；
  - 可以為 DeepForest 偵測結果提供 LiDAR 基準（例如樹冠高度 vs 影像 bbox 推估高度）。
- 限制／注意：
  - **只支援 Windows x64**（Qt6 GUI，無 macOS/Linux 版），Linux 上只能解包睇內容（如本報告）；
  - 3DEP 只有美國資料，香港要用自己嘅 LiDAR（如地政總署／土木工程拓展署資料）；
  - 授權禁止反編譯，但允許免費使用同分享未修改 installer。
- 想同 DeepForest 互補嘅做法：用 SilvaLiDAR Air（或 CLI）產生 HK 試區嘅 CHM／樹冠多邊形 → 轉 GeoJSON/GeoPackage → 同 DeepForest 預測嘅 bbox 做 IoU／高度一致性比較。

---

## 9. 重現方法（本次研究所用步驟）

```bash
# 1) 下載
curl -sSL -o SilvaLiDAR_Air_Setup_latest_x64.exe \
  https://carlos-alberto-silva.github.io/silvalab/downloads/SilvaLiDAR_Air_Setup_latest_x64.exe
sha256sum SilvaLiDAR_Air_Setup_latest_x64.exe

# 2) PE 靜態分析（自寫，無第三方依賴）
python3 analyze_pe.py SilvaLiDAR_Air_Setup_latest_x64.exe        # installer
python3 analyze_pe.py extracted/app/SilvaLiDARAir.exe            # 主程式

# 3) 解包（Inno Setup 6.7 data 需要 innoextract >= 1.11）
#    Debian/GitHub 官方 binary 需要 glibc 2.38 / boost 1.83+，本機 glibc 2.36 → 自行編譯 1.13.0
cd silvalidar_research/tools
./fetch_deps.sh      # 抓 boost 1.74 dev+runtime、zlib/lzma headers、innoextract 源碼，套用 patches/
./build.sh           # 用 GCC 12 編出 ./innoextract

cd ..
tools/innoextract --info  SilvaLiDAR_Air_Setup_latest_x64.exe
tools/innoextract --list  SilvaLiDAR_Air_Setup_latest_x64.exe
LD_LIBRARY_PATH=tools/root/usr/lib/x86_64-linux-gnu \
  tools/innoextract --extract --output-dir extracted SilvaLiDAR_Air_Setup_latest_x64.exe

# 4) 檢視內容
ls extracted/app | head
python3 - <<'EOF'
import re, html
t = open('extracted/app/Documentation/SilvaLiDAR_Air_User_Guide.html', encoding='utf-8').read()
for lvl, txt in re.findall(r'<h([1-3])[^>]*>(.*?)</h\1>', t, re.S):
    print('  ' * (int(lvl) - 1) + html.unescape(re.sub('<[^>]+>', '', txt)).strip())
EOF
```

> 註：`tools/` 內嘅 `innoextract 1.13.0` 係本機用 GCC 12 + boost 1.74 自行編譯（因為官方 Linux binary 要求 glibc 2.38，而容器只有 2.36；innoextract 1.9 又唔識 Inno Setup 6.7 格式）。build 步驟同手寫嘅 `configure.hpp` / `release.cpp` 都留咗喺 `tools/`。

---

## 附錄 A：關鍵檔案雜湊（解包後）

| 檔案 | SHA-256 |
|---|---|
| `SilvaLiDARAir.exe` | `75c4f13dcf39f4a15a4d478600967c60908f4b30812325e6da55a56f3a60478a` |
| `SilvaLiDARAirCLI.exe` | `3bfeefb1a0706f0e4d1386c439ddbf4ef6be55b31f19081e0a9ae2010f563362` |
| `SilvaLaserAirCore.dll` | `0bb0d91d439ad63010257738967fa204a22d3dd7bff0678ada079a54cd2810a1` |
| `BuildingDetection.dll` | `3e50e3140b41f46db79e55d7a52c6d13c218a41d0641c0085a5ea9a589cecc26` |
| installer | `dd9dfd183bdd45de3824b32567a01f49e3c09cea50d97736b719c85a805e3c8b` |

## 附錄 B：本目錄檔案

| 檔案 | 內容 |
|---|---|
| `REPORT.md` | 本報告 |
| `analyze_pe.py` | 自寫 PE 分析器（headers／sections／imports／exports／VERSIONINFO／簽章／打包器指紋） |
| `filelist.txt` | `innoextract --list` 完整檔案清單（499 項） |
| `tools/fetch_deps.sh` | 抓取並解開 build/執行依賴（boost 1.74 dev+runtime、zlib/lzma headers、innoextract 源碼），套用 `patches/` |
| `tools/build.sh` | 用 GCC 12 編譯 innoextract 1.13.0 |
| `tools/patches/configure.hpp`、`tools/patches/release.cpp` | 手寫代替 CMake 生成檔（本機無 cmake） |
| `tools/innoextract`、`tools/debs/`、`tools/root*/`、`extracted/` | 建置產物同依賴（**未入 git**，跑 `fetch_deps.sh` + `build.sh` 即可重建） |
| `tools/`、`deps/` 其餘檔案 | 依賴 headers／libs（未入 git） |
| `SilvaLiDAR_Air_Setup_latest_x64.exe`、`extracted/` | 原始 installer 同解包結果（未入 git，可重下／重解） |

