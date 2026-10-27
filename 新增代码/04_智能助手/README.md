# 智能助手与视频解读说明

执行链路：

```text
上传视频
  → Django 创建后台任务
  → 独立 YOLO 进程抽样推理
  → 聚合类别、置信度、关键帧和时间分布
  → 大模型读取结构化统计并生成中文解读
  → 可归档为检测记录
```

主要文件：

- `assistant.js`：对话、上传、进度、关键帧、解读和归档界面。
- `api_views.py`：meta、chat、video、detail、record 接口。
- `video.py`：任务调度、运行环境探测、权重选择和解释兜底。
- `yolo_worker.py`：独立进程执行本地 YOLO，避免阻塞 Django。
- `context.py`：注入真实台账、模型指标和历史视频任务。
- `llm.py`：OpenAI 兼容 HTTP 客户端。
- `explain.py`：未配置 Key 时的本地规则解释。
- `tests.py`：统计、接口、上传、归档与运行环境测试。

大模型配置写在 `apps/web-backend/.env`，不要把真实 API Key 提交到 GitHub。
