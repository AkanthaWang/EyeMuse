# EyeMuse
> EyeMuse: Multimodal Desktop Pet with Real-time Fatigue and Emotion Detection via Camera &amp; LLM Interaction

![](logo.png)

## 🌟 项目概述
EyeMuse 是一款多模态情绪感知桌面宠物，旨在探索人与人工智能之间更自然、更具共情能力的交互方式。

与传统 AI 助手仅响应用户的显式输入不同，EyeMuse 通过以下方式持续感知用户的实时状态：

+ 👁️ 基于摄像头的面部与疲劳分析
+ 🎤 语音交互
+ 💬 文本输入
+ 🧠 大语言模型（LLM）推理

通过融合多模态感知与基于 LLM 的情感理解，EyeMuse 能够生成个性化的回应，并通过动画、语音和交互行为来表达其反应。

EyeMuse 的目标是打造一个不仅 理解用户说了什么，更能 感知用户感受如何 的 AI 伙伴。


## 技术栈

- 🐍 Python
- 🖼️ GUI：PySide6 或 PyQt6
- 🧩 架构：Qt 桌面界面 + Python 业务模块
- 📦 数据存储：本地文件（JSON / SQLite 等）
- 🎞️ 像素动画：帧数据驱动

## 项目框架

当前仓库已经整理为“桌面前端 + 本地 Python 能力模块”的结构，前端界面直接调用后端模块，不依赖额外的 Web 服务。

```text
EyeMuse/
├─ frontend/
│  ├─ main.py
│  └─ src/
│     ├─ app.py
│     └─ __init__.py
├─ backend/
│  └─ app/
│     ├─ __init__.py
│     ├─ modules/
│     │  ├─ __init__.py
│     │  ├─ llm/
│     │  │  ├─ __init__.py
│     │  │  └─ client.py
│     │  └─ realtime_face_detection/
│     │     ├─ __init__.py
│     │     ├─ common.py
│     │     ├─ mediapipe_analyzer.py
│     │     ├─ preview.py
│     │     ├─ service.py
│     │     └─ yolo_detector.py
│     └─ weights/
│        ├─ face_landmarker_v2_with_blendshapes.task
│        └─ yolov11n-face.pt
├─ scripts/
│  └─ run_frontend.ps1
├─ README.md
├─ INIT.md
└─ PLAN.md
```

### 当前模块说明

+ `frontend/main.py`
  桌面程序入口。

+ `frontend/src/app.py`
  当前主界面与交互核心，负责：
  1. 文本对话区与状态卡片展示。
  2. 摄像头开关、画面预览与检测结果渲染。
  3. 启动 `LLMStreamWorker` 实现流式回复。
  4. 在 LLM 失败时回退到本地规则回复。

+ `backend/app/modules/realtime_face_detection/yolo_detector.py`
  封装 YOLO 人脸检测，优先输出脸框，供前端展示与后续分析裁剪使用。

+ `backend/app/modules/realtime_face_detection/mediapipe_analyzer.py`
  封装 MediaPipe 面部关键点、blendshape 提取，以及压力/疲劳分数估计。
  当前逻辑为优先使用 YOLO 提供的人脸框进行裁剪，再交给 MediaPipe 做分析。

+ `backend/app/modules/realtime_face_detection/common.py`
  放置检测通用数据结构、脸框区域对象、压力计算、疲劳估计等公共逻辑。

+ `backend/app/modules/realtime_face_detection/service.py`
  提供本地摄像头预览调试入口，用于单独验证检测链路。

+ `backend/app/modules/llm/client.py`
  封装 LLM 调用，读取 `.env` 中的模型配置，按 OpenAI 兼容的 `chat/completions` 接口进行流式请求。

### 新增功能补充

+ `backend/app/modules/rppg/processor.py`
  新增基于 MediaPipe ROI + POS 算法的 rPPG 处理模块，用于从面部时序信号中估计 `HEART RATE`、`RESPIRATION`、`HRV`、`SNR` 与 `rPPG progress`。

+ `backend/app/modules/dashboard_data/repository.py`
  新增本地历史数据聚合模块，用于支撑可视化分析页、日报、周报与自定义时间区间报告。

+ `frontend/src/app.py`
  新增并整合了以下前端能力：
  1. 人脸关键点轮廓展示。
  2. `HEART RATE`、`RESPIRATION`、`HRV` 指标卡片展示。
  3. 可视化分析大屏与健康报告页面。
  4. 基于一段时间均值与阈值的宠物状态切换逻辑。

+ `frontend/assets/vendor/echarts.min.js`
  新增本地图表资源，保证可视化大屏在无外网环境下也能正常加载。

### 当前能力状态

+ 文本对话：已接入大模型调用。
+ 流式输出：已支持逐 chunk 刷新到前端对话区。
+ 摄像头预览：已支持本地摄像头打开、关闭与画面显示。
+ 人脸检测：已切换为 YOLO 预训练模型做人脸检测。
+ 面部分析：已使用 MediaPipe 做关键点与 blendshape 分析。
+ 状态评估：在检测到稳定人脸并完成基线校准后，展示压力估计与疲劳状态分数。
+ 关键点可视化：前端已叠加显示人脸关键点轮廓，便于观察关键点跟踪稳定性。
+ 生理指标分析：前端已展示 `HEART RATE`、`RESPIRATION`、`HRV`，并由后端 rPPG 模块持续更新。
+ 宠物状态机：`idle / listening / thinking / responding / alert / offline` 不再依赖单帧数值，而是基于最近一段时间的均值与阈值平滑切换。
+ 历史可视化：已支持可视化分析大屏、日报、周报与自定义时间区间报告。

### 当前交互链路

+ 文本链路
  用户输入文本 -> 前端启动 `LLMStreamWorker` -> `backend.app.modules.llm.client.LLMClient` 流式请求模型 -> chunk 实时追加到界面。

+ 视觉链路
  摄像头取帧 -> YOLO 输出人脸框 -> 使用脸框裁剪/辅助 MediaPipe 分析 -> 生成 `face_count`、`stress_score`、`fatigue_score` -> 前端卡片与提示文案同步刷新。

+ rPPG 链路
  摄像头取帧 -> MediaPipe 提供稳定关键点与 ROI -> POS 算法提取脉搏波形 -> 输出 `heart_rate`、`respiration_rate`、`hrv` -> 前端卡片、宠物状态与提示文案同步更新。

+ 状态决策链路
  压力估计、疲劳状态、`HEART RATE`、`RESPIRATION`、`HRV` -> 进入滑动时间窗口求均值 -> 与预设阈值比较 -> 切换 `idle / listening / thinking / responding / alert / offline`。

+ 历史分析链路
  实时状态结果 -> `dashboard_data` 写入本地 SQLite -> 聚合日 / 周 / 月 / 自定义区间数据 -> 驱动可视化分析页与 Markdown 健康报告。

### 运行方式

当前项目通过 PowerShell 脚本直接启动桌面端：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_frontend.ps1
```

当前脚本会：

+ 设置项目根目录与 `frontend/src` 到 `PYTHONPATH`
+ 使用脚本内指定的 Python 解释器启动 `frontend/main.py`

### .env 配置

LLM 相关配置从 `.env` 中读取，至少需要以下字段：

```env
MODEL_NAME=GLM-4.6V
MODEL_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions
MODEL_APIKEY=your_api_key
```

说明：

+ `MODEL_NAME`：模型名称。
+ `MODEL_URL`：完整接口地址，当前应填写到 `/chat/completions`，而不是只写到 `/v4`。
+ `MODEL_APIKEY`：模型平台分配的密钥。

### 权重文件

当前仓库已经内置两类关键权重文件：

+ `backend/app/weights/yolov11n-face.pt`
  YOLO 预训练人脸检测模型。

+ `backend/app/weights/face_landmarker_v2_with_blendshapes.task`
  MediaPipe Face Landmarker 模型文件，用于关键点与 blendshape 分析。

+ rPPG 说明
  当前 rPPG POS 实现不额外依赖新的深度学习权重，直接复用现有 MediaPipe 人脸关键点结果进行 ROI 采样与时序计算。

### 环境安装

项目根目录已补充 `requirements.txt`，推荐直接安装：

```powershell
pip install -r .\requirements.txt
```


### 当前限制与说明

+ 压力估计和疲劳状态不是单独模型推理结果，而是基于 MediaPipe blendshape 信号做的规则估计。
+ `HEART RATE`、`RESPIRATION`、`HRV` 来自 rPPG 时序估计，受光照、头动、遮挡、帧率稳定性影响较大，仅用于交互反馈与趋势观察，不应视作医疗结果。
+ 当 YOLO 能检测到脸，但 MediaPipe 尚未完成稳定关键点分析时，前端可能先显示“检测到面部”，但压力/疲劳仍处于等待或校准中。
+ 宠物状态切换已改为“时间窗口均值 + 阈值”方式，因此状态变化会比单帧判断更平滑，但也会有一定缓冲时间。
+ 当 LLM 流式调用失败时，界面会显示“LLM 回退到本地回复”，这是当前的兜底机制，不代表程序崩溃。



## 相关项目

+ https://github.com/MelanTech/Dororo - 宠物形象gif
+ https://github.com/HanLoney/OpenDesktop-Pet
+ https://github.com/CanFlyhang/Desktop-Pixel-Pet/tree/main - 宠物形象像素风
+ https://github.com/cjz-wr/DesktopPetByAi
+ https://blog.csdn.net/guyuealian/article/details/131718648 - 疲劳检测数据集
