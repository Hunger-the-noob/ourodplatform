/* 数据工程：复用后台真实阶段进度，异步执行任务并浏览历史质检报告。 */
function useDataEngineering({ dataset, refreshOverview, showToast, getError }) {
    const { ref, reactive, computed } = Vue;
    const selectedDataset = ref("Rubbish_data_set");
    const datasetOptions = ref([]);
    const currentDataset = computed(() => datasetOptions.value.find((item) => item.name === selectedDataset.value));
    const dataJob = reactive({ status: "idle", progress: 0, progress_event: null, logs: [], result: null });
    const submitting = ref(false);
    const dataError = ref("");
    const pollWarning = ref("");
    const dataBusy = computed(() => submitting.value || dataJob.status === "running");
    const validatingData = computed(() => dataBusy.value && dataJob.action === "validate");
    const transformForm = reactive({ format: "yolo", task: "detect", split_strategy: "random", seed: 42, train_rate: 0.8, val_rate: 0.1, classes: "" });
    const compareMode = ref("auto");
    const baselineRunId = ref("");
    const reportHistory = ref([]);
    const selectedReportId = ref("");
    const validationReport = ref(null);
    const reportLoading = ref(false);
    const severityFilter = ref("all");
    const checks = computed(() => (validationReport.value?.results || []).filter((item) => severityFilter.value === "all" || item.severity === severityFilter.value));
    const checkCounts = computed(() => {
        const counts = { PASS: 0, INFO: 0, WARNING: 0, ERROR: 0 };
        for (const item of validationReport.value?.results || []) counts[item.severity] = (counts[item.severity] || 0) + 1;
        return counts;
    });
    const comparisonGroups = computed(() => [
        { key: "new_issues", label: "新增问题", tone: "ERROR" },
        { key: "resolved_issues", label: "已修复", tone: "PASS" },
        { key: "changed_issues", label: "问题变化", tone: "WARNING" },
        { key: "unchanged_issues", label: "持续问题", tone: "INFO" },
    ].map((item) => ({ ...item, rows: validationReport.value?.comparison?.[item.key] || [] })));
    const labels = {
        image_integrity: "图片完整性", label_format: "标签格式与类别范围", bbox_within_image: "目标框边界",
        manifest_lineage: "样本来源与 SHA256", yaml_schema: "数据配置结构", pair_existence: "图片与标签配对",
        fingerprint_match: "契约指纹", box_geometry: "目标框几何", duplicate_annotations: "重复标注",
        annotation_coverage: "标注覆盖率", orphan_labels: "孤立标签", stem_collision: "同名图片冲突",
        split_presence: "划分完整性与规模", split_uniqueness: "跨划分重复", class_presence: "类别出现性", class_balance: "类别均衡",
    };
    const checkLabel = (name) => labels[name] || name;
    const severityClass = (severity) => `quality-${String(severity || "unknown").toLowerCase()}`;
    const prettyEvidence = (details) => JSON.stringify(details, null, 2);
    const dataStageLabel = computed(() => {
        if (dataJob.status === "failed") return "任务执行失败";
        if (dataJob.status === "completed") return dataJob.action === "transform" ? "数据集构建完成" : "质检完成";
        const stage = dataJob.stage || "starting";
        const split = stage.includes("train") ? "训练集" : stage.includes("val") ? "验证集" : stage.includes("test") ? "测试集" : "";
        const stages = [
            ["D3 discover", "发现标注类别"], ["D3 index", "建立标注索引"], ["D3 convert", "转换标注格式"],
            ["D3 scan", "扫描转换标签"], ["D3 stratify", "按类别分层划分"], ["D3 hash", "记录标签指纹"],
            ["D3 copy", "复制图片和标签"], ["D4 scan", "扫描标签与统计"], ["D4 checks", "执行质量检查"],
            ["D4 decode", "检查图片完整性"], ["D4 label format", "检查标签格式与类别"], ["D4 manifest", "核验样本 SHA256"],
        ];
        return (stages.find(([prefix]) => stage.startsWith(prefix))?.[1] || "准备数据任务") + (split ? ` · ${split}` : "");
    });
    let pollTimer = null;
    let disposed = false;
    let handledJob = "";
    const params = () => ({ dataset: selectedDataset.value });
    const errorMessage = (error) => {
        const body = error.response?.data;
        if (body?.errors) return Object.values(body.errors).flat().join("；");
        return getError(error);
    };

    const applyDefaults = () => {
        const item = currentDataset.value;
        if (!item) return;
        Object.assign(transformForm, { format: item.format, task: item.task, split_strategy: item.strategy,
            seed: item.seed, train_rate: item.rates?.train ?? 0.8, val_rate: item.rates?.val ?? 0.1,
            classes: (item.classes || []).join(", ") });
    };
    const loadCatalog = async () => {
        const response = await axios.get("/api/data/catalog/");
        datasetOptions.value = response.data.datasets;
        if (!datasetOptions.value.some((item) => item.name === selectedDataset.value) && datasetOptions.value.length) {
            selectedDataset.value = datasetOptions.value[0].name;
        }
        applyDefaults();
    };
    const loadReport = async () => {
        if (!selectedReportId.value) { validationReport.value = null; return; }
        const name = selectedDataset.value;
        const runId = selectedReportId.value;
        reportLoading.value = true;
        try {
            const response = await axios.get(`/api/data/reports/${encodeURIComponent(runId)}/`, { params: { dataset: name } });
            if (name === selectedDataset.value && runId === selectedReportId.value) validationReport.value = response.data;
        } catch (error) {
            if (name === selectedDataset.value) { validationReport.value = null; dataError.value = errorMessage(error); }
        } finally { reportLoading.value = false; }
    };
    const loadReports = async (preferred = "") => {
        const name = selectedDataset.value;
        const response = await axios.get("/api/data/reports/", { params: { dataset: name } });
        if (name !== selectedDataset.value) return;
        reportHistory.value = response.data.reports;
        selectedReportId.value = reportHistory.value.some((item) => item.run_id === preferred) ? preferred : (reportHistory.value[0]?.run_id || "");
        await loadReport();
    };
    const schedule = (delay = 850) => {
        if (pollTimer) window.clearTimeout(pollTimer);
        if (!disposed) pollTimer = window.setTimeout(pollDataJob, delay);
    };
    const pollDataJob = async () => {
        try {
            const response = await axios.get("/api/data/jobs/");
            if (disposed) return;
            Object.assign(dataJob, response.data);
            pollWarning.value = "";
            if (dataJob.status === "running") { schedule(); return; }
            if (!dataJob.id || handledJob === dataJob.id) return;
            handledJob = dataJob.id;
            if (dataJob.status === "failed") {
                dataError.value = dataJob.error || "数据任务失败，请查看任务日志。";
                showToast(dataError.value, "error");
                return;
            }
            if (dataJob.dataset === selectedDataset.value) {
                await loadCatalog();
                await Promise.all([refreshOverview(), loadReports(dataJob.result?.run_id)]);
            }
            const severity = dataJob.result?.severity;
            showToast(dataJob.action === "transform" ? "数据集构建完成，可继续运行质检" : `质检完成：${severity || "已生成报告"}`, severity === "ERROR" ? "error" : "success");
        } catch (error) {
            pollWarning.value = "暂时无法读取任务进度，正在重试。";
            schedule(2500);
        }
    };
    const startJob = async (action) => {
        if (dataBusy.value) return;
        if (action === "transform" && !window.confirm(`将按当前参数重新构建 ${selectedDataset.value} 的成品数据。原始数据保留，继续执行？`)) return;
        submitting.value = true;
        dataError.value = "";
        try {
            let payload = { dataset: selectedDataset.value };
            if (action === "transform") {
                payload = { ...payload, ...transformForm, seed: Number(transformForm.seed), train_rate: Number(transformForm.train_rate), val_rate: Number(transformForm.val_rate),
                    classes: transformForm.classes.trim() ? transformForm.classes.split(/[,，\n]/).map((item) => item.trim()).filter(Boolean) : null };
            } else {
                payload.compare_previous = compareMode.value !== "off";
                if (compareMode.value === "specific") {
                    if (!baselineRunId.value) throw new Error("请选择历史对比基线。");
                    payload.baseline_run_id = baselineRunId.value;
                }
            }
            const response = await axios.post(`/api/data/${action}/`, payload);
            Object.assign(dataJob, response.data);
            schedule(200);
        } catch (error) {
            dataError.value = error.response ? errorMessage(error) : (error.message || "数据任务启动失败");
            showToast(dataError.value, "error");
        } finally { submitting.value = false; }
    };
    const selectDataset = async () => {
        applyDefaults();
        dataError.value = "";
        baselineRunId.value = "";
        validationReport.value = null;
        selectedReportId.value = "";
        reportHistory.value = [];
        try { await Promise.all([refreshOverview(), loadReports()]); }
        catch (error) { dataError.value = errorMessage(error); }
    };
    const initialize = async () => {
        try {
            await loadCatalog();
            const response = await axios.get("/api/data/jobs/");
            Object.assign(dataJob, response.data);
            if (dataJob.status === "running") schedule();
            else handledJob = dataJob.id || "";
            await loadReports();
        } catch (error) { dataError.value = errorMessage(error); }
    };
    const refreshDataEngineering = async () => {
        try { await Promise.all([loadReports(selectedReportId.value), pollDataJob()]); }
        catch (error) { dataError.value = errorMessage(error); }
    };
    const dispose = () => { disposed = true; if (pollTimer) window.clearTimeout(pollTimer); };
    return { selectedDataset, datasetOptions, currentDataset, dataJob, dataBusy, dataError, pollWarning,
        validatingData, transformForm, compareMode, baselineRunId, reportHistory, selectedReportId,
        validationReport, reportLoading, severityFilter, checks, checkCounts, comparisonGroups,
        dataStageLabel, checkLabel, severityClass, prettyEvidence, selectDataset, loadReport,
        initialize, refreshDataEngineering, dispose,
        validateDataset: () => startJob("validate"), transformDataset: () => startJob("transform") };
}
