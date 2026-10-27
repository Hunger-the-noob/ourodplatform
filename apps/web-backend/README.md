# ODPlatform Web

Django 5 + Django REST Framework + Vue 3 单页控制台，用于生活垃圾数据理解、
YOLO 模型训练、测试集评估、模型推理和图片/视频识别。

## 页面模块

1. 项目介绍：平台能力、工作流与技术架构。
2. 数据理解：数据划分、类别分布、样本图片和质检查询。
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
cd C:\Users\NAYOUdUO\ourodplatform
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

## 核心 API

| 方法 | 地址 | 用途 |
|---|---|---|
| GET | `/api/data/` | 数据统计、类别、样本和质检 |
| POST | `/api/data/validate/` | 重新执行数据验证 |
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
cd C:\Users\NAYOUdUO\ourodplatform
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
