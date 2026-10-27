# 数据工程代码说明

建议按以下顺序阅读：

1. `进度条_progress.py`：统一进度输出，异常退出后自动恢复安静模式。
2. `D3_数据流水线_orchestrator.py`：转换、扫描、平衡报告、划分、物化和 YAML 输出。
3. `D4_图片完整性_image_integrity.py`：`verify()` 后再 `load()`，发现截断或伪造图片。
4. `D4_标签格式_label_format.py`：检测/分割标签格式、坐标、NaN/Inf 和类别 ID 校验。
5. `D4_样本哈希_manifest_lineage.py`：逐样本 SHA256，支持进度并区分移动、改动和缺失。
6. `D4_历史对比_history.py`：同数据身份报告之间的新增、修复、变化与持续问题。
7. `test_*.py`：用临时演示数据从零验证 D3/D4，不修改真实数据集。
8. `web_*.py`：网页端的异步任务、状态轮询、历史报告和下载接口。

其中原有中文 docstring 已保留。测试命令：

```powershell
cd C:\Users\DELL\ODPlatform
python -m pytest -q -c pyproject.toml apps/platform/tests
```
