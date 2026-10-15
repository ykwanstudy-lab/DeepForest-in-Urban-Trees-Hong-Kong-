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


---

# 第二輪深入挖掘（追加 findings）

## 10. 引擎內部：AI 系統提示＋CLI `--help` 揭示嘅完整能力清單

主程式內嵌一段俾 SilvaAI 用嘅 **AUTHORITATIVE TOOL REGISTRY** 系統提示（明文列出所有工具、方法、預設參數、以及「唔准講有」嘅功能），加埋 CLI Guide 內每個 `--tool` 嘅完整 `--help`，等於一份**官方級 capability 清單**：

### 地面濾波（8 種，全部有預設值）
| 方法 | 關鍵預設 |
|---|---|
| PTD（Progressive TIN Densification） | seed 20 m、max angle 30°、max dist 2.0 m、min ground spacing 0.25 m、TIN buffer 30 m、50 iterations |
| MCC（Multiscale Curvature） | scale 1.50 m、curvature 0.30、100 passes |
| CSF（Cloth Simulation Filter） | class threshold 0.50 m、cloth res 0.50 m、time step 0.65、rigidness 3、500 iterations |
| PMF（Progressive Morphological） | cell 1.0 m、max window 33 m、slope 1.0、exp=true |
| SMRF（Simple Morphological） | cell 1.0 m、slope 0.15、max window 18 m、scalar 1.25 |
| WLS/IRI（Kraus–Pfeifer robust interpolation） | cell 7 m、g=−2、w=2.5、iterations 5 |
| MV-HCS（multi-view hierarchical cloth） | cell 2 m、adaptive window 6–20 m、3 hierarchies |
| AMT v2（Adaptive Multi-Scale Terrain） | scales 20/10/5/2.5 m、max slope 50°、threads auto |

> 一般 GIS／林業軟件通常只有 2–4 種；8 種＋完整參數係少見。

### 其餘核心能力（節錄）
- **DTM**：TIN/Delaunay（推薦）、KNN-IDW、Kriging（可外插 k/p）；resolution 預設 0.5 m
- **CHM**：`p2r`、`p2r-subcircle`、`p2r-tin-fill`、`p2r-knn-idw-fill`、`pitfree`（Khosravipour 2014）、`spikefree`（2016）、`tin`、`idw`、`natural-neighbor`、`hybrid`、`streaming-p2r`、**`gpu-p2r`**；`--chm-plugin-dir` 可載入**外部 `.dll`／`.so` CHM plugin**（＝CHM 演算法可擴充，且提到 `.so` 暗示有跨平台打算）
- **單株樹偵測（8 個家族）**：Silva et al. 2016（CHM 冠幅分割，含 Popescu 可變窗）、Li 2012/TreeSeg（點雲分割）、Dalponte & Coomes 2016、Ferraz AMS3D（3D 自適應 mean-shift）、Marker-controlled Watershed、region-growing、Falkowski wavelet、Yang et al. 2020（watershed + 3D profiles + k-means）
- **單株樹指標**：HMAX、高度百分位／分佈、crown base height（+percentile）、crown length、crown radius、crown area、**crown diameter major/minor**、crown volume（AlphaShape3D，alpha 0.75 m，可 Balanced/Detailed）；內部參數有 vertical bins 32、min points 30、**stem search fraction 0.62、minimum stem height 1.30 m**、crown expansion 2.75、**leaning threshold 12°、fallen threshold 62°、snag crown ratio 0.10**、soft time limit 30 s
- **Metrics**：Cloud Metrics（HAG／Elevation、1–99 百分位、intensity／return／class／strata、**LAI/LAD + light profiles 可選**）、Grid Metrics（20 m 格、可出每指標 GeoTIFF、stable grid alignment）、Voxel Metrics、**Custom Metric Designer**
- **地形衍生**：只有 Slope／Aspect／Hillshade／TRI／Contours；**Curvature、Profile/Plan Curvature、Roughness、TPI、Landforms、Relative Elevation 明確「未實作」並從選單隱藏**
- **大檔策略**：需要「point records + display index ≤ Windows 可用記憶體 65%，並留 1.5 GiB」；約 **180 bytes/point**；`.lax` 標準空間索引 + 自家 `.slx` cache；Catalog tile 預設 500 m（可 adaptive）、buffer 自動／手動
- **桌面獨有（CLI 未有）**：SilvaAI、SilvaFlow 圖編輯器、Noise Removal、TRI／Voxel／ABA、手繪 Clip、重投影（建議用 GDAL/OGR）；其餘幾乎全部有 CLI（`catalog-pipeline`、`plot-products`、`area-metrics`）
- **報告模板章節**（`report_template.html`）：Dataset Overview、Processing Pipeline、Noise Removal、Point Density、Ground Classification、DTM、Hillshade、Slope、Contours、Height Normalization、Classification、CHM、Clip、CloudMetrics、Grid Metrics、Voxel Metrics、Individual Tree Detection、Individual Tree Metrics、Catalog Mode、Canopy Gaps、Final Products & QC
- **`.sfw` workflow 檔**係自家文字格式（`SILVAFLOW,13` + `NODE,…` 行），內含每個 box 嘅全部參數
- **Plugin 系統**：每個 plugin DLL 內嵌 **JSON manifest（ABI v2）**，例如
  - `PowerLineDetection.dll` → `org.silvalab.test.power_lines`（v1.1.0）：*"Experimental unsupervised wire-candidate classification using local height above ground and horizontal neighborhood linearity"*，參數 minimum_height 12 m、ground_cell 15 m、neighborhood_radius 4 m、vertical_tolerance 1.25 m、min_neighbors 5、**min linearity 0.88**，required_buffer 30 m，輸出 LAS class 14 + `power_line_candidates.csv`
  - `BuildingDetection.dll` → `org.silvalab.test.buildings`（v1.0.0）：*"Experimental elevated low-roughness grid screening"*，cell 5 m，輸出 `building_candidates.csv`
  - id 用 `org.silvalab.test.*` 前綴 → **屬實驗性工具**；plugin 可掛入 standalone／SilvaFlow／Catalog，並自報 buffer 需求
- **AI 明文護欄（即「做唔到」清單）**：無內建 calibrated biomass 模型、無物種分類、無 stem taper profile；crown envelope volume ≠ 木材／樹幹體積或生物量；P2R+Subcircle 只屬 CHM、Li 2012 只做分割唔量樹幹體積

## 11. 香港坐標系：可以直接用

`app/share/proj/proj.db`（PROJ 資料庫，13,790 個 CRS）已包含香港坐標系：
**EPSG:2326 Hong Kong 1980 Grid System**、EPSG:4611 Hong Kong 1980、EPSG:4738/4739 HK 1963、EPSG:8425–8427 Hong Kong Geodetic CS、**EPSG:5738 HKPD height**、EPSG:5739/7976 HKCD/HKPD depth。
→ 即係可以直接以 HK1980 Grid 出 CHM／metrics／GeoPackage，唔需要自己 reproject。


## 12. SilvaLiDAR Air vs QGIS vs ArcGIS（定位同能力比較）

| 面向 | **SilvaLiDAR Air** | **QGIS** | **ArcGIS Pro** |
|---|---|---|---|
| 定位 | 垂直領域工具：**機載 LiDAR → 林業／地形產品** | 通用開源 GIS 平台 | 商業企業級 GIS 平台 |
| 價格／授權 | 免費（專有，禁反編譯） | 免費（GPL） | 訂閱制（各 extension 另計） |
| 平台 | **只有 Windows x64**（Qt6 GUI） | Windows／macOS／Linux | **只有 Windows x64** |
| LiDAR 點雲 | 內建完整管線（LAS/LAZ 直讀、串流、.lax、Catalog 分塊、8 種地面濾波、多種 CHM、8 種樹偵測） | QGIS 3.3x+ 原生點雲支援（PDAL/Untwine 索引）；樹木分析要靠 plugin（LAStools toolbox、lidR via R、FUSION、PDAL/`pdal_wrench`）自行砌 | LAS dataset／LAS 點雲圖層、3D Analyst（Classify LAS Ground、LAS Dataset to Raster、Tile LAS）；**有 pretrained deep learning 模型：Tree Detection／Tree Segmentation（影像）同 Tree Point Classification（點雲）**，需 Image Analyst／Deep Learning 授權 |
| 林業專用演算法 | **內建＋有文獻依據**（Khosravipour pit-free、Silva 2016、Dalponte 2016、Ferraz AMS3D、Li 2012、Yang 2020、CSF、PTD、MCC、SMRF、Kraus-Pfeifer…）＋單株樹指標（冠幅／冠基高／冠體積 alpha shape） | 要逐個裝／寫 script（R `lidR` 最接近） | 靠工具箱組合＋Esri 預訓練模型；林業專用單株樹指標要自己砌 model builder／Python |
| 報告／可追溯性 | **一鍵 HTML/PDF 報告（20+ 章節）、每個產品記錄參數同 provenance、可重跑** | 要自己用 Atlas／Layout 砌 | 有 Layout／Task，但要自己設計 |
| 批次／HPC | CLI + Catalog 分塊平行（單檔多核、多 tile 並行） | 靠 GDAL/PDAL CLI、`qgis_process`、Python | ArcPy／Jupyter，企業排程 |
| 通用 GIS 能力 | **冇**（無向量編輯、無 raster calculator、無 layout 出圖、無 WMS/WFS 編輯、無空間統計工具集） | 極強（向量／柵格編輯、地理處理工具箱、Layout、WMS/WMTS/WFS、Python/Processing） | 極強＋企業級（地理資料庫、版本控制、分享、協作） |
| 影像底圖 | NAIP／Mapbox／自訂 XYZ/TMS／本地 GeoTIFF/COG/JP2 | 極多 WMS/WMTS/XYZ（含 HK 圖層） | 極多（含 Living Atlas、HK 圖層） |
| 生態／擴展 | 自家 plugin ABI v2（C++ DLL，內嵌 JSON manifest）＋ CHM 插件（`.dll`/`.so`）；**無 Python 擴展** | 巨大 plugin 生態＋ Python/R | Python (ArcPy)、SDK、Marketplace |
| 多用戶／企業 | 無（單機） | 靠 PostGIS/QGIS Server | 企業級（Portal／AGOL／版本管理） |
| AI 助手 | **內建 SilvaAI（Ollama 本地／OpenAI／Anthropic／Gemini）＋ MCP server** | 無內建（可外接） | 有 AI assistants／Copilot 類功能（雲端） |

**一句總結**：
SilvaLiDAR Air **唔係「另一個 GIS」**，而係一部**專門做「點雲 → 森林／地形產品」嘅垂直工具**，同 QGIS／ArcGIS 係互補關係。它嘅底層其實都係同一批開源庫（PDAL、GDAL、LASzip、PROJ）——QGIS 要自己砌、ArcGIS 要自己砌＋付費；SilvaLiDAR Air 嘅價值就係**預設參數、文獻化方法、單株樹指標、報告同 provenance**。反過來，任何製圖、向量編輯、資料庫、企業協作、物種／深度學習偵測，就一定要用 QGIS／ArcGIS。

## 13. 對 arboriculture（樹木管理／風險評估）嘅 insight

### 香港現況（研究時查證）
- **CEDD GEO 開放 LiDAR**：2010 同 2020 年**全港航空 LiDAR 測量**，經 CEDD Spatial Data Portal（`sdportal.cedd.gov.hk`）取得 → 香港**有免費、全港、兩期**點雲可做變化偵測。
- **HyD（路政署）**：多光譜航空照片 → Tree Health Index (THI) + Tree Health Monitoring System (THMS)；用 **UAV LiDAR（HawkScan HS300）** 抽樹高、冠幅，並**用 3D 樹形參數回歸估算 DBH**；用地面 LiDAR 做樹枝分割；高光譜 204 波段建 19 種香港城市樹種光譜庫。
- **Smart Lab S-0812（AT-Vibe）**：手持／無人機 LiDAR 掃描，自動計 **DBH、冠幅、樹高、光透射率、傾斜角**，傾斜角 >10° 判為高風險，效率提升 5–10 倍。

### SilvaLiDAR Air 喺 arboriculture 做得到
- **樹冠層指標**：樹高（HMAX）、冠幅（major/minor diameter）、冠面積、冠基高、冠長、冠體積（alpha shape，可檢 `crown_volume_valid` / `is_alpha`）
- **傾斜／倒伏／枯立相關參數**：`leaning threshold 12°`、`fallen threshold 62°`、`snag crown ratio 0.10`、`stem search fraction 0.62`、`minimum stem height 1.30 m`（**注意：User Guide 未文件化，屬引擎內部能力，需要實測驗證**）
- **林分／格網指標**：canopy cover、LAI/LAD 光剖面、height percentiles、strata、voxel metrics → 用於**樹冠覆蓋率、綠化目標、棲地／碳匯指標**
- **樹冠空隙（Canopy Gap）** → 城市林空隙／補植機會、樹冠連通性分析
- **多期變化偵測**：同一範圍 2010 vs 2020（或颱風前後）LiDAR → 樹冠高度／體積變化、消失（移除／倒樹）、新增
- **基建衝突篩查**：`PowerLineDetection`（高於地面 12 m + 水平線性度 ≥0.88 → LAS class 14 + CSV）→ **樹線衝突（tree–wire conflict）** 候選點，直接對應兩電嘅修剪計劃；`BuildingDetection`（elevated low-roughness）→ 城市環境遮擋／建築物背景
- **全港／區級自動化**：Catalog 分塊平行 + CLI + MCP（AI 客戶端可叫佢跑）→ 可以每晚／每季自動出區級樹冠報告
- **報告同審計**：一鍵 HTML/PDF + 參數 provenance → 適合納入樹木管理系統嘅記錄（可追溯、可重現）

### 做唔到（對 arboriculture 係關鍵限制）
| 需要嘅 | SilvaLiDAR Air | 原因 |
|---|---|---|
| **DBH／樹幹直徑** | ✗（無模型，AI 亦被禁止亂估） | 機載 LiDAR 睇唔到樹幹；HyD 做法係用 3D 樹形參數**回歸**估 DBH → 你可自行用 SilvaLiDAR Air 出嘅冠幅／樹高做 predictors，但**模型要自己 calibrate**（同 ABA 建模概念一致） |
| **物種識別** | ✗（明文唔支援） | 需要高光譜／影像＋ML |
| **缺陷／腐朽／樹洞／根系** | ✗ | 要 VTA／TLS／GPR／聲波斷層 |
| **單枝結構、光透射率（逐棵）** | ✗（LAI 只係格網／雲級，唔係逐棵） | 需要地面／背負式 LiDAR 高密度點雲 |
| **樹木風險等級（TRA Form 2/3）** | ✗ | 屬專業判斷，工具只提供指標 |
| **颱風即時／雨後倒樹清單** | 部分（要新點雲 + 人工核對） | 要有覆蓋當區嘅新 LiDAR／航拍 |

### 建議嘅互補工作流（HK 情境）
1. 由 CEDD Spatial Data Portal 取 2010 / 2020（或最新）LiDAR → 轉 LAS/LAZ
2. SilvaLiDAR Air（CRS 用 **EPSG:2326 HK1980 Grid**）：`tiles` 分塊 → `ground(CSF/PTD)` → `dtm` → `normalize` → `chm(pitfree)` → `trees(silva2016 / dalponte2016)` → `plot-products` / `area-metrics`
3. 輸出：逐棵樹表（tree_id、x、y、height、crown_radius、crown_base_height + `_full.csv` 全欄位）、樹冠覆蓋／空隙 raster、Grid/Voxel metrics CSV、HTML/PDF 報告
4. 同現有系統結合：用座標同 GLTM／TRAM 嘅個別樹記錄（DBH、物種、健康狀況、風險等級）join → 建立「LiDAR 指標 ↔ 風險」篩選模型（例如：樹高 × 冠幅 × 傾斜 × 鄰近目標）
5. 用 `PowerLineDetection` 做樹線走廊候選點 → 交電纜／修剪團隊核實
6. 兩期比較 → 生長率、移除／倒樹、樹冠覆蓋變化 → 匯報綠化政策 KPI
7. 敏感／未公開資料：用 **Ollama 本地模型**跑 SilvaAI，唔好交去雲端 provider

