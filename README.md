# 图析 · CAD Insight

基于 **PaddleOCR + DeepSeek API + LangChain / LangGraph + Streamlit** 的 2D CAD 机械图纸信息识别与结构化提取系统。

上传一张机械图纸，系统完成本地 OCR、区域定位、规则解析与大模型协同提取，输出零件图号、材料、表面处理、总体尺寸、关键特征及技术要求，并支持逐项查看原图证据、人工复核、历史任务与 JSON / CSV 导出。

- 本地 OCR 模式无需 API Key，完全离线运行。
- 大小模型协同模式将图纸局部区域发送给 DeepSeek，由多模态大模型完成语义提取与冲突复核。
- 所有自动结果与人工修订分开保存，原始证据（文字、坐标、分数、区域）完整保留。

---

## 目录

- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [环境要求](#环境要求)
- [快速开始（Windows）](#快速开始windows)
- [快速开始（Linux / macOS）](#快速开始linux--macos)
- [配置 DeepSeek API](#配置-deepseek-api)
- [启动与访问](#启动与访问)
- [使用流程](#使用流程)
- [识别模式](#识别模式)
- [目录结构](#目录结构)
- [命令行工具](#命令行工具)
- [数据与评测](#数据与评测)
- [隐私与安全](#隐私与安全)
- [常见问题](#常见问题)
- [已知限制](#已知限制)

---

## 功能特性

| 模块 | 说明 |
| --- | --- |
| 图纸输入 | 内置 100 张 JPG 样例图纸；支持上传 JPG / PNG（≤25MB、≤4000 万像素） |
| 图像处理 | EXIF 方向校正、分块 OCR、坐标还原、区域裁剪、缩放预览与证据框叠加 |
| OCR | PaddleOCR PP-OCRv5 mobile 中英文识别，可选 90° 旋转文字识别 |
| 区域定位 | 默认按标题栏、技术要求、主体左/右生成 4 个区域，支持手动调整百分比范围 |
| 规则解析 | 图号、材料、表面处理、单位等明确字段规则，缺失值保持为空，不臆造 |
| 大模型协同 | LangChain Runnable 调用 DeepSeek 图像 / 文本接口，按区域理解并做冲突复核 |
| 证据追溯 | 每个字段保存 OCR 原文、坐标、分数或区域裁剪图，可跳转查看来源 |
| 人工复核 | 表格逐字段改值 / 状态 / 证据 / 备注，或直接编辑完整 JSON，保存独立修订稿 |
| 任务档案 | 历史任务、原图、自动结果、人工修订、运行日志与 API 用量 |
| 导出 | JSON（完整记录）、CSV（字段表），已处理常见电子表格公式前缀 |
| 实验评测 | 按图号家族划分数据集、空白真值模板、基础准确率 / 覆盖率 / 特征 P-R-F1 评测 |

## 技术栈

- Python 3.11
- Streamlit 1.66
- PaddlePaddle 3.2.2 (CPU) / PaddleOCR 3.3.2
- LangChain Core / LangGraph
- OpenAI Python SDK（调用 DeepSeek 兼容接口）
- Pydantic 2、Pillow、OpenCV（headless）、python-dotenv
- pytest / Playwright（测试与浏览器验证）

工作流：

```text
图纸验证 → OCR 分块/去重 → 区域定位与规则解析
                              ├─ 本地模式 → 证据校验
                              └─ API 模式 → 分区域理解 → 冲突复核 → 证据校验
                                                              ↓
                                              本地结果 / 人工复核 / 导出
```

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Windows 10/11（推荐，已在 Windows 验证）；Linux 可运行；macOS 取决于 PaddlePaddle 是否提供对应 wheel |
| Python | **3.11**（建议，项目在该版本验证）；3.10–3.12 通常可用 |
| 内存 | 建议 ≥ 8GB（OCR 与图像处理为 CPU 推理） |
| 磁盘 | 代码与数据集约 200MB；首次还需下载 OCR 模型与依赖，预留 ≥ 3GB |
| 网络 | 首次安装依赖、下载 OCR 模型需要联网；使用 DeepSeek 模式需要联网与有效 API Key |
| 显卡 | 非必需；默认 CPU 运行，无需 CUDA |

> 本项目数据集为 100 张机械图纸 JPG（约 125MB），单文件均小于 100MB，已直接包含在仓库中，克隆后即可选择本地样例。

## 快速开始（Windows）

### 1. 克隆仓库

```powershell
git clone https://github.com/ESU00509/2DCAD_Information_Recognition_System.git
cd 2DCAD_Information_Recognition_System
```

### 2. 创建独立 Python 3.11 环境

使用官方 Python 安装包或 Conda 均可。以下两种方式二选一：

**方式 A：venv（推荐，需已安装 Python 3.11）**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**方式 B：Conda**

```powershell
conda create -p .venv python=3.11 -y
conda run -p .venv python -m pip install --upgrade pip
conda run -p .venv python -m pip install -r requirements.txt
```

> 项目脚本默认使用 `.venv\Scripts\python.exe`（Windows）或 `.venv/python.exe`，请保持环境目录名为 `.venv`。

### 3. 下载 OCR 模型（首次必做）

```powershell
.\.venv\Scripts\python.exe -m cad_insight.cli setup-ocr
```

模型（PP-OCRv5 mobile 检测 / 识别）会下载到项目内 `.cache/paddlex`，之后可离线使用。首次下载需联网，体积约数百 MB。

### 4. 配置 DeepSeek（可选）

```powershell
Copy-Item .env.example .env
notepad .env
```

在 `.env` 中填入 `DEEPSEEK_API_KEY`。不配置也可以使用“本地 OCR”模式。

### 5. 启动

双击 `start.bat`，或在 PowerShell 执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

保持窗口开启，浏览器访问 **http://127.0.0.1:8501**。

### 6. 自检

```powershell
.\.venv\Scripts\python.exe -m cad_insight.cli doctor
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.cache\pytest-local
```

`doctor` 会打印 Python 版本、关键依赖版本、API Key 是否已配置（不会打印密钥内容）与数据集数量。

## 快速开始（Linux / macOS）

```bash
git clone https://github.com/ESU00509/2DCAD_Information_Recognition_System.git
cd 2DCAD_Information_Recognition_System

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m cad_insight.cli setup-ocr
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY（可选）

python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

> `start.bat` / `start.ps1` 仅适用于 Windows；Linux/macOS 请直接使用上面的 `streamlit` 命令。
> 若 macOS / ARM 环境安装 PaddlePaddle 失败，建议改用 Linux 或 Windows，或参考 [PaddlePaddle 官方安装文档](https://www.paddlepaddle.org.cn/install/quick) 选择可用版本。

## 配置 DeepSeek API

1. 复制 `.env.example` 为 `.env`，只在本地编辑，**不要提交到仓库**。
2. 填入 `DEEPSEEK_API_KEY`。
3. 按需调整以下变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 空 | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | 兼容 OpenAI Chat Completions 的服务地址 |
| `DEEPSEEK_MODEL` | `deepseek-flash` | 支持图像输入的模型名，可按账户可用能力调整 |
| `DEEPSEEK_TIMEOUT` | `90` | 单次请求超时（秒） |
| `DEEPSEEK_MAX_REQUESTS` | `8` | 单张图纸最大实际请求数（含重试与冲突复核，1–30） |
| `DEEPSEEK_MAX_OUTPUT_TOKENS` | `10000` | 单次输出上限（1024–32768） |
| `DEEPSEEK_THINKING` | `disabled` | 设为 `enabled` 开启思考模式（会占用输出额度） |
| `DEEPSEEK_INPUT_PRICE` | 空 | 每百万输入 token 价格（人民币），仅用于界面估算 |
| `DEEPSEEK_OUTPUT_PRICE` | 空 | 每百万输出 token 价格（人民币），仅用于界面估算 |

配置后在网页“连接设置”点击“重新读取本地配置”，再回工作台选择“大小模型协同”。

**重要：** `deepseek-flash` 的图像能力与可用性取决于账户与官方文档，请以 [DeepSeek Vision 文档](https://api-docs.deepseek.com/guides/vision/) 为准。未确认模型权限时，先用“本地 OCR”模式。

## 启动与访问

- 默认监听 `127.0.0.1:8501`，仅本机可访问，适合个人使用。
- 如需局域网共享，可在 `.streamlit/config.toml` 或启动命令中改为 `--server.address 0.0.0.0`，并自行配置防火墙与反向代理；本项目未实现多用户登录与权限控制，不建议直接暴露到公网。
- 上传大小上限在 [.streamlit/config.toml](.streamlit/config.toml) 中设置（当前 25MB）。

## 使用流程

1. 从内置 100 张图纸中选择，或上传 JPG / PNG。
2. 选择识别模式（见下节）。
3. 默认自动生成 4 个区域；位置不合适可在“高级选项”中手动调整百分比范围，预览立即显示区域框。
4. 点击开始，查看原图、结果与证据。图像支持缩放、缩小和滚动。
5. 在“人工复核”中改值、状态、证据与备注，保存独立修订稿；新增特征可使用完整 JSON 编辑器。
6. 下载 JSON / CSV；历史任务与修订稿可从“任务档案”再次打开或下载。

## 识别模式

| 模式 | 是否调用 API | 说明 |
| --- | --- | --- |
| 本地 OCR | 否 | 提取文字、图号、材料与部分特征，展示证据，可完全离线 |
| 大小模型协同 | 是 | OCR + 区域裁剪 + DeepSeek 分区理解 + 冲突复核（推荐） |
| 整图 API | 是 | 整图直接发送，作为对比基线，不做 OCR |
| OCR + 文本 API | 是 | 只发送 OCR 文本与坐标，不发送图片，作为降级路径；**不等于视觉识别** |

约束：

- 单图最多 8 次实际 API 请求（含网络重试、JSON 修复与冲突复核）；网络 / 429 / 5xx 最多重试两次；鉴权、余额或格式错误立即停止。
- 每个区域最多一次冲突复核；两种路径矛盾时保留候选并标记待确认。
- 成功区域与 OCR 结果会缓存，重复运行可复用；缓存命中会单独标记。
- API 未完成时保留部分结果与警告，状态标记为 `partial`。

## 目录结构

```text
.
├── app.py                      Streamlit 网页入口
├── assets/style.css            工作台视觉样式
├── cad_insight/
│   ├── settings.py             路径、模型缓存目录与 .env 配置
│   ├── schema.py               字段、证据、特征与关系结构
│   ├── imaging.py              图片验证、分块、区域定位与证据标注
│   ├── ocr.py                  PaddleOCR 推理、坐标还原、缓存
│   ├── rules.py                明确字段规则、冲突合并
│   ├── deepseek.py             LangChain Runnable、请求限额 / 重试 / 缓存
│   ├── pipeline.py             LangGraph 工作流
│   ├── storage.py              任务原稿、人工修订与导出
│   ├── evaluation.py           数据集划分与基础评测
│   ├── ui.py                   界面组件
│   └── cli.py                  命令行工具
├── tests/test_core.py          核心逻辑测试
├── scripts/
│   ├── check_api.py            只读检查账户模型列表（不打印密钥）
│   ├── run_api_samples.py      三张样例的 API 联调（会产生费用）
│   └── browser_check.py        浏览器端到端验证
├── data/
│   ├── dataset-split.json      按图号家族划分：开发 53 / 验证 34 / 测试 13
│   └── gold-template.json      空白人工真值模板（不是已完成标注）
├── 2D图纸数据集/                100 张 JPG 机械图纸样例
├── .streamlit/config.toml      端口、上传大小与主题
├── .env.example                环境变量模板
├── start.bat / start.ps1       Windows 启动脚本
├── requirements.txt            直接依赖
└── requirements.lock.txt       开发机完整依赖快照（可选）
```

运行时会自动创建 `data/jobs/`（原图、区域图、结果、修订、日志）与 `data/cache/`（OCR / API 缓存）。

## 命令行工具

```powershell
# Windows 示例；Linux/macOS 将 .\.venv\Scripts\python.exe 换成 ./.venv/bin/python

# 环境与配置自检（不打印 API Key）
.\.venv\Scripts\python.exe -m cad_insight.cli doctor

# 首次下载 OCR 模型
.\.venv\Scripts\python.exe -m cad_insight.cli setup-ocr

# 本地识别
.\.venv\Scripts\python.exe -m cad_insight.cli run "2D图纸数据集\M2773808G006.jpg" --mode ocr

# 真实 API 识别（需配置密钥）
.\.venv\Scripts\python.exe -m cad_insight.cli run "2D图纸数据集\M2773808G006.jpg" --mode hybrid

# 数据集划分与空白真值模板
.\.venv\Scripts\python.exe -m cad_insight.cli split
.\.venv\Scripts\python.exe -m cad_insight.cli gold-template

# 评测：自动结果对人工真值
.\.venv\Scripts\python.exe -m cad_insight.cli evaluate --prediction "data\jobs\任务编号\result.json" --gold "人工真值.json"

# 运行测试（临时文件写入项目内）
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.cache\pytest-local
```

## 数据与评测

- 数据集已按图号家族保守分组：**开发 53 / 验证 34 / 测试 13**，优先避免同家族泄漏，未强凑 60/20/20；正式实验仍建议排查跨家族近重复。
- `data/gold-template.json` 只是空白模板，**不是已标注真值**。需要人工逐图填写基础字段、单位、名义尺寸与偏差、特征类型 / 数量 / 参数，并引用对应证据；无法判断的字段置 `null` 或写入 `unjudgeable_paths`。
- 评测统计基础字段准确率、覆盖率与严格特征 P/R/F1，缺失回答计为漏检；总体尺寸同时检查单位。
- 当前**没有**生成或声称任何模型准确率成绩。人工修订稿与自动识别原稿必须分开报告。

## 隐私与安全

- `.env` 已被 `.gitignore` 忽略，请勿把 API Key 写入源码或提交到仓库。
- “大小模型协同 / 整图 API”会把图纸图像发送到 DeepSeek；“OCR + 文本 API”只发送识别文字与坐标；“本地 OCR”不联网。
- 任务原图、识别结果与修订稿默认保存在本机 `data/jobs/`，不会上传到其他服务。
- 本项目默认单人本地使用，未实现用户认证、权限隔离与审计；如需多人部署，请自行增加鉴权与网络防护。
- 请勿在公开环境使用含有敏感或受版权保护的图纸数据。

## 常见问题

**1. 启动提示找不到 `.venv` 或 `Project environment missing`**

按“快速开始”重新创建 `.venv` 并安装依赖；Windows 脚本固定使用 `.venv/Scripts/python.exe`，请确认环境名与路径。

**2. 安装 PaddlePaddle 失败**

- 确认 Python 版本为 3.11，且为 64 位。
- 国内网络可配置 pip 镜像后重试。
- 若系统 / 架构无可用 wheel，请改用 Windows 或 Linux；GPU 用户参考 PaddlePaddle 官方安装命令。

**3. OCR 模型下载失败 / 识别时报模型不存在**

- 重新执行 `python -m cad_insight.cli setup-ocr`。
- 模型缓存位于 `.cache/paddlex`，确保该目录可写且未被清理。
- 首次下载需联网，代理环境下请确认 `HTTPS_PROXY` 等变量。

**4. 页面能打开但识别报错**

先运行 `python -m cad_insight.cli doctor` 检查依赖与配置；再运行 `pytest` 确认核心逻辑环境正常。

**5. 8501 端口被占用**

修改 `.streamlit/config.toml` 的 `port`，或启动时追加 `--server.port 8502`。

**6. API 返回 401 / 402 / 429**

- 401：Key 无效或未正确写入 `.env`。
- 402：账户余额不足。
- 429 / 5xx：触发限流或服务异常，系统会有限重试；请降低并发或稍后再试。

**7. 上传图片失败**

检查格式（JPG / JPEG / PNG）、大小（≤25MB）与像素（≤4000 万），并确认 `.streamlit/config.toml` 的 `maxUploadSize` 设置。

**8. `requirements.lock.txt` 与本机不兼容**

`requirements.lock.txt` 是开发机的完整依赖快照，包含平台相关包；跨平台部署请优先使用 `requirements.txt`。

## 已知限制

- 当前关联为**标注级关系**；复杂跨视图尺寸链、引线追踪、3D 重建、DWG/DXF 解析尚不支持。
- OCR mobile 模型对 Ø、沉孔符号、垂直尺寸等可能漏读；未证实前不会补写符号，可开启旋转识别或用局部图让 API 复核。
- “流程完成”不等于“字段齐全或全部正确”；总体长宽高在无明确依据时保持为空。
- 不把通用规则中的“如需喷砂”当成该零件明确喷砂；设备名称不直接当零件名称。
- 模型别名可能更新；正式实验建议关闭 API 缓存并记录响应模型与日期。
- 未配置 token 单价时，界面只显示用量，不显示费用估算；估算值不替代供应商账单。

---

## 参考资料

- [DeepSeek Vision 文档](https://api-docs.deepseek.com/guides/vision/)
- [PaddleOCR 文档](https://www.paddleocr.ai/)
- [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/graph-api)

> 识别结果用于辅助图纸判读，具体数值、公差与工艺要求请结合原图人工复核。
