"""视频分析 + 大模型解读应用。

把「本地 YOLO 推理」与「大模型解释」串成一条链路：

1. 前端上传视频 → 保存到 media/videos/
2. 后台线程调用独立进程里的本地 YOLO（yolo_worker.py）逐帧推理
3. 推理结果汇总成结构化统计（stats.summarize）
4. 把统计交给大模型生成中文解读（llm.chat），无 Key 时退化为本地规则解释
5. 结果落库（VideoAnalysisJob），并可一键归档为 defect.DefectRecord

另外提供不带视频的通用智能助手：把台账 / 模型指标作为上下文注入对话。
"""
