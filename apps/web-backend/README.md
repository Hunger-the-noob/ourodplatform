# ODPlatform Web

Django 5 + Django REST Framework + Vue 3 单页控制台，用于生活垃圾数据理解、
YOLO 模型训练、测试集评估、模型推理和图片/视频识别。

## 页面模块

1. 项目介绍：平台能力、工作流与技术架构。
2. 数据理解：选择数据集、构建流水线、实时进度、16 项质检明细、历史报告对比和下载。
3. 模型训练：网页启动训练、实时进度/日志、指标图表、数据验证和测试集评估。
4. 模型推理：单图推理和目标框坐标检查。
5. 仪表盘：检测记录、识别次数、目标数量和最近活动。
6. 图片与视频识别：上传生活垃圾图片或视频，显示标注结果和识别历史。
7. **智能助手**：对话式问答，注入台账 / 模型指标 / 历史视频分析上下文；接任意 OpenAI 兼容接口，
   未配置 Key 时退回本地规则模式。
8. **视频智能分析（LLM 解读）**：上传视频 → 本地 YOLO 抽样推理 → 结构化统计与关键帧 →
   大模型中文报告 → 可一键归档为检测记录。

浏览器标题固定为 `ODPlatform`。Vue 3.5.44 和 Axios 1.20.0 已放入
`frontend/static/frontend/vendor/`，运行网页不需要 Node 或外部 CDN。

## 启动

```powershell
conda activate ODP
cd C:\Users\DELL\ODPlatform
python -m pip install -r ".\apps\web-backend\requirements.txt"
cd apps\web-backend
Copy-Item .env.example .env    # 填 LLM_API_KEY；YOLO 解释器按需填 ODP_YOLO_PYTHON
cd ..\..
powershell -ExecutionPolicy Bypass -File .\scripts\run_web.ps1
```

`run_web.ps1` 会自动挑出「装了 Django」和「装了 ultralytics」的两个解释器，并执行数据库迁移。
智能助手 / 视频分析的完整说明见 [`docs/assistant-llm-video.md`](../../docs/assistant-llm-video.md)。

访问：

- 控制台：<http://127.0.0.1:8000/>
- 可指定模块：<http://127.0.0.1:8000/?module=training>
- Django 后台：<http://127.0.0.1:8000/admin/>

创建后台管理员：

```powershell
cd apps\web-backend
python manage.py createsuperuser
```

## 网页训练流程

“模型训练”模块支持：

- 选择 epoch、输入尺寸、batch 和 CPU/GPU
- 后台异步启动 `odp_platform.cli.train_model`
- 实时显示任务阶段、进度和最近训练日志
- 在网页停止训练
- 数据重新验证
- 使用独立 test split 执行模型评估
- 展示 Precision、Recall、mAP50、mAP50-95
- 展示训练指标曲线、损失曲线、类别 mAP、混淆矩阵和 PR 曲线

训练结束后，最新完整运行记录会自动显示；取消的任务不会覆盖已完成模型指标。
正式训练建议使用 GPU 和 50–100 个 epoch。

## 数据工程网页操作

本次功能在原有文件基础上增量接入：数据概览、样本和质检接口仍定义在
`defect/api_views.py`；原有 YAML 读取函数、路径常量和数据统计仍保留在
`defect/data_understanding.py`。新增文件负责后台任务、进度、历史报告等新增能力。
仓库根目录 `D3_D4_新增功能与修改代码.txt` 是源码和修改对照的阅读副本，
不参与程序运行，也不替代源文件。

打开 `http://127.0.0.1:8000/?module=understanding`，在“数据理解”的数据工程工作台操作：

1. 选择本机数据集；没有真实数据时页面显示缺失提示，图片数量按当前磁盘统计。
2. 展开“流水线与质检配置”，设置格式、检测/分割、划分策略、种子、比例和类别顺序。
3. 点击“构建数据集”并确认，后台调用 D3。原始数据保留，成品数据按参数重建。
4. 点击“运行 16 项质检”，后台调用 D4；可自动对比上次、指定基线或关闭对比。
5. 查看实际阶段进度、处理数量、速度和预计剩余时间；刷新页面后会继续查询当前任务。
6. 在“质检明细与历史对比”按级别筛选，展开检查项查看坏图、标签行或 SHA256 证据，
   切换历史报告，下载 JSON、结果 CSV 和对比 CSV。

进度百分比表示**当前阶段**的完成量，进入新阶段会重新计数。质检发现 ERROR / WARNING
时任务仍会正常完成并保存报告，页面显示检查结论与证据；只有执行异常才显示任务失败。
Web 数据任务日志及状态位于 `runs/web_data/`，质检报告仍位于 `runs/data_validation/`。
同一开发服务同时只执行一个数据任务；构建数据和模型训练的按钮会互相检查运行状态。

仓库自带 YAML 可能保留其他电脑的绝对路径。请在本机准备 `data/raw/<名称>/images`
及对应标注目录后，从页面重建成品及 YAML；仅有配置时不会显示原电脑的历史图片数量。

开发验证（全部使用隔离临时数据，不改动真实数据集）：

```powershell
python -m pip install -r apps/web-backend/requirements.txt -r requirements-data-dev.txt
cd apps/web-backend
python manage.py test defect
```

可选浏览器验证，在仓库根执行：

```powershell
python -m pip install playwright
# 使用本机 Edge / Chrome；无本机浏览器时先执行 python -m playwright install chromium
python scripts/check_data_web.py --output-dir runs/web_data_smoke
```

浏览器测试对数据工程 API 使用真实后端和子进程，只为无关的模型、识别和训练查询提供
测试响应；验证进度、16 项结果、坏图修复、基线、下载与手机布局，并保存截图。

## 核心 API

| 方法 | 地址 | 用途 |
|---|---|---|
| GET | `/api/data/?dataset=<名称>` | 当前数据集实际统计、类别、样本和质检 |
| GET | `/api/data/catalog/` | 本机数据集、原始数据/成品可用性及默认参数 |
| POST | `/api/data/transform/` | 后台构建数据集，立即返回 202 |
| POST | `/api/data/validate/` | 后台执行 16 项质检，立即返回 202 |
| GET | `/api/data/jobs/` | 当前后台数据任务、阶段实际进度、速度、剩余时间及日志 |
| GET | `/api/data/reports/?dataset=<名称>` | 当前配置与任务的最近 30 份有效质检报告 |
| GET | `/api/data/reports/<run_id>/?dataset=<名称>` | 逐项结果、错误证据、历史对比和下载链接 |
| GET | `/api/data/reports/<run_id>/download/<文件名>/?dataset=<名称>` | 下载 report.json、results.csv 或 comparison.csv |
| GET/POST | `/api/training/` | 查询状态或启动训练 |
| POST | `/api/training/cancel/` | 停止训练 |
| GET | `/api/model/` | 最新模型、历史指标和评估结果 |
| GET/POST | `/api/model/evaluate/` | 查询或运行测试集评估 |
| GET | `/api/model/evaluation-artifact/{name}/` | 获取混淆矩阵、PR 曲线 |
| GET/POST | `/api/recognition/` | 图片识别及历史 |
| GET/POST | `/api/video-recognition/` | 视频识别及历史 |
| GET | `/api/dashboard/` | 仪表盘聚合数据 |

## 代码位置

```text
apps/web-backend/
├── manage.py
├── .env.example                      大模型 / YOLO 推理配置样例
├── config/
│   ├── settings.py                   数据库、静态文件、CORS、DRF、ASSISTANT 配置
│   ├── urls.py
│   └── views.py
├── defect/
│   ├── models.py                     检测、图片识别和视频识别模型
│   ├── api_views.py                  REST 接口
│   ├── data_understanding.py         数据统计与样本接口
│   ├── training_jobs.py              网页训练进程管理
│   ├── evaluation.py                 test split 评估
│   ├── recognition.py                图片推理
│   ├── video_recognition.py          视频推理
│   ├── model_info.py                 模型、历史和图表数据
│   ├── serializers.py
│   ├── admin.py
│   └── migrations/
├── assistant/
│   ├── models.py                     VideoAnalysisJob 视频分析任务
│   ├── api_views.py                  meta / chat / video / record 接口
│   ├── llm.py                        大模型客户端（OpenAI 兼容）
│   ├── context.py                    台账 / 模型 / 历史任务上下文
│   ├── stats.py                      检测统计聚合
│   ├── explain.py                    本地规则解释（无 Key 兜底）
│   ├── video.py                      任务调度 + 解释器/权重探测
│   ├── yolo_worker.py                独立进程 YOLO 视频推理 worker
│   └── tests.py                      单元测试
└── frontend/
    ├── index.html                    七模块页面结构（新增“智能助手”模块）
    └── static/frontend/
        ├── app.js                    Vue 状态、训练轮询、识别和图表
        ├── assistant.js              智能助手 + 视频分析前端逻辑
        ├── styles.css
        └── vendor/
```

数据库位于 `apps/web-backend/db.sqlite3`，上传结果位于
`apps/web-backend/media/`。训练任务日志位于 `runs/web_training/`。

## 边界

- 当前模型是 CPU 上 1 epoch 的演示结果，精度只用于验证功能。
- 原数据类别名为 `class_0` 到 `class_9`；有真实类别名称后应重新生成数据 YAML。
- 视频识别最多处理 300 个采样帧，单文件最大 100 MB。
- 当前是本地开发配置，API 没有登录鉴权。生产环境必须增加认证、关闭 DEBUG、
  限制 CORS，并使用正式 WSGI/ASGI 服务和任务队列。

## 智能助手 / 视频 LLM 解读 API

| 方法 | 地址 | 用途 |
|---|---|---|
| GET | `/api/assistant/meta/?probe=1` | 大模型配置状态 + 本地 YOLO 环境探测 |
| POST | `/api/assistant/chat/` | 智能助手对话（注入台账上下文） |
| GET/POST | `/api/assistant/video/` | 查询最近分析任务 / 上传视频发起分析 |
| GET | `/api/assistant/video/{id}/` | 分析进度、关键帧、大模型解读 |
| POST | `/api/assistant/video/{id}/record/` | 把分析结果归档为检测记录 |

## Rubbish 数据集训练

```powershell
cd C:\Users\DELL\ODPlatform
conda activate ODP
powershell -ExecutionPolicy Bypass -File .\scripts\train_rubbish.ps1
```

默认执行 1 epoch、416 图像尺寸的 CPU 演示训练。正式训练示例：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_rubbish.ps1 `
  -Epochs 50 -ImageSize 640 -Batch 16 -Device 0
```

训练完成后刷新 Web 页面，“模型训练”区域会自动读取最新运行现场。
智能助手 / 视频分析的完整说明见 [`docs/assistant-llm-video.md`](../../docs/assistant-llm-video.md)。
