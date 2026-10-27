# ODPlatform 新增代码阅读副本

这个文件夹是为了答辩、代码审查和讲解而制作的**阅读副本**，不参与网站运行。
真正生效的代码仍在原项目路径中，修改本文件夹里的内容不会改变网页行为。

## 对应任务

| 图中任务 | 实际实现 | 阅读副本 |
|---|---|---|
| D3 大数据集执行进度 | 转换、扫描、分层划分和物化阶段均有数量、速度、剩余时间 | `01_数据工程/进度条_progress.py`、`D3_数据流水线_orchestrator.py` |
| pytest 单元测试 | D3/D4 回归测试不依赖预置 checkpoint | `01_数据工程/test_*.py` |
| pre-commit | Ruff 检查、导入排序和格式化 | `02_算法工程/.pre-commit-config.yaml` |
| D4 图片完整性 | Pillow 容器校验与实际像素解码 | `01_数据工程/D4_图片完整性_image_integrity.py` |
| 标签格式/越界/类别 ID | 列数、NaN/Inf、归一化、框边界和 `class_id < nc` | `01_数据工程/D4_标签格式_label_format.py` |
| manifest 逐样本 SHA256 | 流式读取，显示进度，区分 changed/moved/missing/unreadable | `01_数据工程/D4_样本哈希_manifest_lineage.py` |
| 历史报告对比 | 对比新增、修复、变化、持续问题，并生成 CSV | `01_数据工程/D4_历史对比_history.py` |
| D5 配置代码去重 | train/val/infer 共用 `_build_config()` | `02_算法工程/D5_统一配置构建_runtime_config.py` |
| 生成器测试 | 正常生成、跳过、覆盖备份、空字段和特殊字符 | `02_算法工程/D5_配置生成测试_test_generator.py` |
| runs 自动清理 | 默认 dry-run，保护 lineage，拒绝符号链接 | `02_算法工程/D5_runs清理_clean_runs.py` |
| Gradio 演示 | 图片、视频检测和本地 YOLO + 智能解读 | `02_算法工程/Gradio演示_app.py` |
| Django + Vue 网页 | 项目介绍、数据理解、训练、推理、仪表盘、识别 | `03_Web网页/` |
| 智能助手 | 对话、真实台账上下文、OpenAI 兼容接口、离线规则兜底 | `04_智能助手/` |
| 视频本地推理 + 大模型解释 | 本地 YOLO 抽样推理，结构化统计交给 LLM | `04_智能助手/video.py`、`yolo_worker.py`、`explain.py` |

## 本次网页调整

- 删除旧的悬浮助手抽屉样式和未使用的导航状态，避免与新“智能助手”模块重复。
- 明确“模型推理”是单图技术验证，“图片与视频识别”是业务识别入口。
- 数据概览只保留页面上方的质检入口，避免出现两次“运行质检/重新验证”。
- 检测指标趋势和训练损失改为上下全宽布局，增加 Y 轴、起止 epoch、曲线数据点和当前指标条。
- 增加逐轮指标明细表，完整展示 Precision、Recall、mAP@50、mAP@50-95 和三类损失。

## 验证结果

```text
apps/platform/tests                 215 passed
apps/web-backend defect + assistant  43 passed
Django manage.py check               no issues
```

## 启动项目

```powershell
conda activate ODP
cd C:\Users\DELL\ODPlatform
powershell -ExecutionPolicy Bypass -File .\scripts\run_web.ps1
```

浏览器访问 <http://127.0.0.1:8000/>。

## 注意

- 数据库 `apps/web-backend/db.sqlite3`、训练权重、原始数据和媒体文件都没有复制到这里。
- 智能助手不配置大模型 Key 时仍能工作，但会使用本地规则解释。
- Gradio 是可选依赖：`python -m pip install -e ".\apps\platform[demo]"`。
- 代码副本保留了原文件中的中文说明；本 README 及各子目录 README 标出了阅读顺序。
