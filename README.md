# ODPlatform

通用目标检测开发平台（Monorepo）。

## 布局

- `apps/platform/` — 核心引擎与 `od-platform` Python 包
- `apps/desktop/` — PySide6 桌面推理演示
- `apps/web-backend/` — Django + DRF + Vue 3 缺陷检测工作台（含智能助手、视频智能分析）
- `data/` — 原始数据与处理产物
- `models/` — 预训练权重与训练产物
- `runs/` — 数据、训练与推理运行现场
- `tests/` — 跨端 E2E 测试占位
- `docs/architecture/` — 架构决策记录
- `docs/assistant-llm-video.md` — 智能助手 + 视频 YOLO → 大模型解读说明

## 安装

```powershell
conda activate ODP
cd C:\Users\NAYOUdUO\ourodplatform
python -m pip install -e "./apps/platform[dev,desktop]"
odp-init
```

只需要命令行引擎时，可以省略 `desktop`：

```powershell
python -m pip install -e "./apps/platform[dev]"
```

## 常用命令

```powershell
odp-transform --dataset <数据集名> --format pascal_voc --task detect
odp-validate "apps\platform\configs\datasets\<数据集名>.yaml"
odp-train --data <数据集名> --epochs 10
odp-infer --source "D:\images\test.jpg" --model "models\pretrained\yolov8n.pt" --no-show
powershell -ExecutionPolicy Bypass -File .\scripts\run_desktop.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_web.ps1
python .\scripts\make_demo_video.py --output .\runs\demo\demo.mp4
```

完整运行说明见各子项目 README。
