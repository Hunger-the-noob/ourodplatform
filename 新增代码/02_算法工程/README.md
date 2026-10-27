# 算法工程代码说明

- `D5_统一配置构建_runtime_config.py`：`build_train_config()`、
  `build_val_config()`、`build_infer_config()` 共用同一个 `_build_config()`，
  合并优先级为 YAML < 额外来源 < CLI。
- `D5_配置生成测试_test_generator.py`：覆盖要求中的五类生成场景，并验证三种
  build 函数共用实现。
- `D5_runs清理_clean_runs.py`：`odp-clean` 默认只预览，必须显式 `--yes` 才删除；
  只处理 `runs/<子系统>/<run_id>` 一层目录，并保护 `lineage`。
- `D7_可视化_visualization.py`：类别配色、中文标签映射、置信度和图例开关。
- `Gradio演示_app.py`：图片、视频和智能解读三个演示页签。
- `.pre-commit-config.yaml`：提交前自动运行 Ruff 检查、导入排序和格式化。
- `platform_pyproject.toml`：测试、可执行命令、`dev`/`desktop`/`demo` 可选依赖。
- `project_pyproject.toml`：全仓库 Ruff、Mypy 和 pytest 共享配置。

示例：

```powershell
odp-clean --keep 5 --dry-run
python -m pre_commit run
python -m pip install -e ".\apps\platform[demo]"
python apps\demo\app.py
```
