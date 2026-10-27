# D3 / D4 数据工程新增功能

本次在原目录和接口基础上扩展，不移动源文件。原分支已经具备 pytest 配置和 15 项质检，因此复用原检查并加强边界情况，新增 `image_integrity` 后共 16 项检查。

## 使用

在仓库根目录安装轻量开发依赖并启用提交前检查：

```powershell
python -m pip install -r requirements-dev.txt
python -m pre_commit install
powershell -ExecutionPolicy Bypass -File scripts/run_web.ps1
```

打开 `http://127.0.0.1:8000/?module=understanding`，在“数据工程 · D3 / D4”中选择数据集。将原始数据放在项目已有的 `data/raw/<dataset>/images` 和 `labels`（YOLO）或 `annotations`（VOC/COCO）目录；YOLO 必须填写与原标签一致的类别顺序。点击“运行 D3 转换与划分”生成成品，再点击“运行 D4 质量检查”。训练期间页面及 API 拒绝重新转换数据。

进度条显示当前阶段的实际完成数和总数；阶段切换时计数重新开始。无法预先计数的准备、划分和报告阶段显示处理中。刷新页面后会重新连接当前后台任务。质检发现 ERROR/WARNING 会完成并展示问题报告；运行异常才显示执行失败。

命令行也默认显示进度条，自动化调用可用 `--no-progress` 关闭：

```powershell
$env:PYTHONPATH = 'apps/platform/src'
python -m od_platform.cli.transform_data --dataset demoyolo --format yolo --classes head 反光衣
python -m od_platform.cli.validate_data apps/platform/configs/datasets/demoyolo.yaml
```

## 质检与历史

- 图片：先校验文件容器，再实际解码像素，空图、损坏图及截断图片均报 ERROR。
- 标签：复用原有结构、`0 <= class_id < nc`、坐标归一化与边界检查；新增 NaN/Infinity、非正框尺寸、编码/读取错误处理。bbox 边界仍按原有容差及比例分级。
- 契约：逐样本 SHA256 独立重算的语义保持一致，加入进度反馈。
- 报告：保存到 `runs/data_validation/<run_id>/report.json`、`results.csv`，JSON 采用原子替换；运行 ID 包含微秒和随机后缀，显式重复 ID 会被拒绝。
- 历史：只比较同一 YAML 路径、同一任务的上一次报告。首次运行建立基线。图片、标签格式、框越界、契约按文件/行/问题类型精确比较，其他原有检查按检查项跟踪。完整问题保存在 JSON，页面每类预览最多 100 条、历史记录最多 50 次，总数不受预览限制。跳过、崩溃和无法读取的文件不冒充已修复；旧报告粒度不同时重新建立该检查的基线。

新增 API：`GET /api/data/engineering/` 查看可用数据集及检查，`GET/POST /api/data/jobs/` 查询或启动任务，`GET /api/data/history/?dataset=<name>` 查询历史。原 `/api/data/validate/` 现为后台接口，POST 返回 HTTP 202，GET 查询状态；调用者需轮询任务状态，报告严重度位于 `result.overall_severity`。

Web 任务管理沿用项目现有训练管理器的单进程方式，适配 `scripts/run_web.ps1`；服务重启会中断内存中的任务，磁盘质检历史保留。多 worker 部署时需先迁移到共享任务队列。

## 验证与协作

```powershell
python -m pytest --ignore=apps/platform/tests/test_inference_config.py
python apps/web-backend/manage.py test defect --noinput
node --check apps/web-backend/frontend/static/frontend/app.js
```

数据工程测试生成真实可解码图片、临时源数据、成品和流转章，无需提前运行 checkpoint 或下载训练模型。轻量环境没有安装 torch/ultralytics，所以上述命令仅排除依赖该推理运行时的原有 `test_inference_config.py`；安装完整 platform 依赖后可运行不带排除参数的全量 pytest。

`.pre-commit-config.yaml` 对本次提交的 Python 文件执行 Ruff lint 和格式检查。为避免本次功能扩大为全仓格式重写，GitHub Actions 同样只检查变更文件；测试覆盖轻量环境中的核心套件和全部 Django 测试。每位协作者克隆后需自行执行 `python -m pre_commit install`。

可选浏览器验收（本机 Edge 可直接使用，无需下载 Chromium）：

```powershell
python -m pip install playwright
python scripts/check_data_engineering_ui.py --browser-channel msedge --screenshot tmp/data-engineering.png
```

该脚本启动临时数据库和真实 Django API，验证 D3、D4、进度、损坏/修复对比与移动端布局，退出后清理临时数据。使用默认 Chromium 时需先运行 `python -m playwright install chromium`。

## 原讲义副本

`D3_new_function.txt`、`D4_new_function.txt` 的前半部分逐字节复制自用户提供的 D3/D4 原讲义，后半部分为新增功能说明和代码补丁。附录使用 `+` 标明新增行、`-` 标明替换的旧行，并对每个文件说明用途。原讲义中的旧检查数量、空图片演示及测试指令保留原样，当前实现以附录和本指南为准。
