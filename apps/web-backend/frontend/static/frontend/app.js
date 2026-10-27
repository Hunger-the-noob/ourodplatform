const {
    computed,
    createApp,
    onBeforeUnmount,
    onMounted,
    reactive,
    ref,
} = Vue;

const API = "/api";
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_VIDEO_BYTES = 100 * 1024 * 1024;

createApp({
    setup() {
        const modules = [
            {
                id: "intro",
                icon: "⌂",
                label: "项目介绍",
                caption: "PROJECT OVERVIEW",
                description: "了解 ODPlatform 的定位、能力边界和完整视觉工作流。",
            },
            {
                id: "understanding",
                icon: "▦",
                label: "数据理解",
                caption: "DATA UNDERSTANDING",
                description: "查看生活垃圾数据集的划分、类别分布、样本与质检结果。",
            },
            {
                id: "training",
                icon: "◎",
                label: "模型训练",
                caption: "TRAIN · VALIDATE · TEST",
                description: "在网页启动训练，实时查看日志，并使用测试集评估模型。",
            },
            {
                id: "inference",
                icon: "⌖",
                label: "模型推理",
                caption: "MODEL INFERENCE",
                description: "使用最新权重执行单图推理，并检查每个目标框与置信度。",
            },
            {
                id: "dashboard",
                icon: "◫",
                label: "仪表盘",
                caption: "DASHBOARD",
                description: "汇总检测记录、图片识别、视频任务和模型指标。",
            },
            {
                id: "media",
                icon: "▷",
                label: "图片与视频识别",
                caption: "IMAGE & VIDEO RECOGNITION",
                description: "上传生活垃圾图片或视频，查看标注结果与识别历史。",
            },
        ];

        const requestedModule = new URLSearchParams(window.location.search).get("module");
        const activeModule = ref(
            modules.some((item) => item.id === requestedModule)
                ? requestedModule
                : "intro",
        );
        const pageLoading = ref(true);
        const apiOnline = ref(false);
        const toast = reactive({ message: "", type: "success", timer: null });

        const dataset = reactive({
            name: "Rubbish_data_set",
            classes: [],
            class_count: 0,
            splits: {},
            class_counts: {},
            total_images: 0,
            total_boxes: 0,
            strategy: "",
            seed: null,
            validation: {},
            samples: [],
        });
        const dashboard = reactive({
            defects: {},
            recognition: {},
            model: {},
            dataset: {},
            recent_defects: [],
        });
        const modelInfo = reactive({
            status: "loading",
            run_id: "",
            experiment_name: "",
            dataset: { name: "—", classes: [], class_count: 0, splits: {}, validation: {} },
            training: {},
            metrics: {},
            metrics_history: [],
            class_map_50_95: {},
            weights: {},
            evaluation: null,
            curve_url: null,
        });

        const trainingState = reactive({
            id: "",
            status: "idle",
            stage: "idle",
            progress: 0,
            current_epoch: 0,
            total_epochs: 0,
            imgsz: 0,
            batch: "",
            device: "",
            logs: [],
            error: null,
        });
        const trainingForm = reactive({ epochs: 1, imgsz: 416, batch: 16, device: "cpu" });
        const trainingError = ref("");
        const validatingData = ref(false);
        const evaluatingModel = ref(false);
        let trainingPollTimer = null;
        // 新增：数据工程后台任务、真实阶段进度与历史报告。
        const engineering = reactive({ datasets: [], checks: [], formats: [], strategies: [] });
        const dataForm = reactive({ dataset: "Rubbish_data_set", format: "yolo", task: "detect",
            classes: "", train_rate: 0.8, val_rate: 0.1, seed: 42, split_strategy: "random" });
        const dataJob = reactive({ status: "idle", progress: {}, result: null, error: null });
        const dataHistory = reactive({ latest: null, reports: [] });
        const dataSubmitting = ref(false);
        const dataRunning = computed(() => dataSubmitting.value || dataJob.status === "running");
        const selectedDataset = computed(() => engineering.datasets.find((d) => d.name === dataForm.dataset));
        const comparison = computed(() => dataHistory.latest?.comparison || {});
        const severityClass = (severity) => ({ ERROR: "severity-error", WARNING: "severity-warning",
            INFO: "severity-info", PASS: "severity-pass" }[severity] || "");
        const dataStageLabel = computed(() => {
            const stage = dataJob.progress?.stage || "starting";
            const [name, detail] = stage.split(":");
            const labels = { starting: "准备任务", prepare: "检查原始数据", convert: "转换标注",
                discover_classes: "发现类别", index: "建立标注索引", scan_labels: "扫描转换结果",
                split: "固定种子划分", manifest: "生成划分契约", materialize: "复制图片与标签",
                write_yaml: "写入数据集配置", snapshot: "扫描数据集", checks: "执行质量检查",
                image_integrity: "图片完整性与解码", label_format: "标签格式、类别与归一化",
                bbox_within_image: "检查目标框边界", manifest_lineage: "逐样本 SHA256 核验",
                report: "生成报告和历史对比", complete: "全部阶段完成" };
            return `${labels[name] || name}${detail ? ` · ${detail}` : ""}`;
        });
        let dataPollTimer = null;

        const inferenceFile = ref(null);
        const inferencePreview = ref("");
        const inferenceResult = ref(null);
        const inferenceConfidence = ref(0.25);
        const inferenceLoading = ref(false);
        const inferenceError = ref("");

        const requestedMediaTab = new URLSearchParams(window.location.search).get("tab");
        const mediaTab = ref(requestedMediaTab === "video" ? "video" : "image");
        const mediaFile = ref(null);
        const mediaVideoFile = ref(null);
        const mediaImagePreview = ref("");
        const mediaVideoPreview = ref("");
        const mediaResult = ref(null);
        const videoResult = ref(null);
        const mediaConfidence = ref(0.25);
        const videoFrameStride = ref(2);
        const mediaLoading = ref(false);
        const videoLoading = ref(false);
        const mediaError = ref("");
        const videoError = ref("");
        const recognitionHistory = ref([]);
        const videoHistory = ref([]);

        const currentModule = computed(
            () => modules.find((item) => item.id === activeModule.value) || modules[0],
        );
        const trainingRunning = computed(() => trainingState.status === "running");
        const modelStatusLabel = computed(() => {
            const labels = {
                completed: "训练完成",
                training: "训练中",
                not_found: "未训练",
                loading: "读取中",
            };
            return labels[modelInfo.status] || "未知";
        });
        const trainingStageLabel = computed(() => {
            const labels = {
                idle: "等待启动",
                starting: "正在启动训练进程",
                preparing: "正在加载模型与数据",
                training: "模型正在训练",
                evaluating: "正在验证与计算指标",
                finalizing: "正在归档模型",
                completed: "训练完成",
                failed: "训练失败",
                cancelled: "训练已停止",
            };
            return labels[trainingState.stage] || trainingState.stage;
        });

        const classRows = computed(() =>
            Object.entries(dataset.class_counts || {})
                .map(([label, count]) => ({ label, count: Number(count) }))
                .sort((a, b) => a.label.localeCompare(b.label)),
        );
        const classMapRows = computed(() =>
            Object.entries(modelInfo.class_map_50_95 || {})
                .map(([name, value]) => ({ name, value: Number(value) }))
                .sort((a, b) => b.value - a.value),
        );
        const recognitionClassRows = computed(() =>
            Object.entries(dashboard.recognition?.class_counts || {})
                .map(([label, count]) => ({ label, count: Number(count) }))
                .sort((a, b) => b.count - a.count),
        );
        const splitGradient = computed(() => {
            const train = Number(dataset.splits?.train?.images || 0);
            const val = Number(dataset.splits?.val?.images || 0);
            const test = Number(dataset.splits?.test?.images || 0);
            const total = Math.max(train + val + test, 1);
            const a = (train / total) * 100;
            const b = a + (val / total) * 100;
            return `conic-gradient(#198d80 0 ${a}%, #e0a24a ${a}% ${b}%, #7e61c0 ${b}% 100%)`;
        });

        const formatNumber = (value) => Number(value || 0).toLocaleString("zh-CN");
        const confidencePercent = (value) => {
            const number = Number(value);
            return Number.isFinite(number) ? `${Math.round(number * 100)}%` : "—";
        };
        const metricPercent = (value) => {
            if (value === null || value === undefined || value === "") return "—";
            const number = Number(value);
            return Number.isFinite(number) ? `${Math.round(Math.max(0, Math.min(1, number)) * 100)}%` : "—";
        };
        const formatDate = (value, withTime) => {
            const date = new Date(value);
            if (Number.isNaN(date.getTime())) return value || "—";
            const options = withTime
                ? { hour: "2-digit", minute: "2-digit", hour12: false }
                : { year: "numeric", month: "2-digit", day: "2-digit" };
            return new Intl.DateTimeFormat("zh-CN", options).format(date);
        };
        const barPercent = (value, rows, key = "count") => {
            if (Number(value || 0) <= 0) return "0%";
            const max = Math.max(...rows.map((item) => Number(item[key] || 0)), 1);
            return `${Math.max(4, Math.round((Number(value || 0) / max) * 100))}%`;
        };

        const chartDots = (key) => {
            const history = modelInfo.metrics_history || [];
            const width = 508;
            const left = 34;
            return history.map((row, index) => {
                const x = history.length === 1
                    ? left + width / 2
                    : left + (index * width) / (history.length - 1);
                const value = Number(row[key]);
                const clamped = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
                return { x, y: 190 - clamped * 160 };
            });
        };
        const chartPoints = (key) => chartDots(key).map((point) => `${point.x},${point.y}`).join(" ");
        const lossPoints = (key) => {
            const history = modelInfo.metrics_history || [];
            const keys = ["train_box_loss", "train_cls_loss", "train_dfl_loss"];
            const max = Math.max(
                ...history.flatMap((row) => keys.map((item) => Number(row[item]) || 0)),
                1,
            );
            const width = 508;
            const left = 34;
            return history.map((row, index) => {
                const x = history.length === 1
                    ? left + width / 2
                    : left + (index * width) / (history.length - 1);
                const value = Number(row[key]) || 0;
                return `${x},${190 - (value / max) * 160}`;
            }).join(" ");
        };

        const showToast = (message, type = "success") => {
            if (toast.timer) window.clearTimeout(toast.timer);
            toast.message = message;
            toast.type = type;
            toast.timer = window.setTimeout(() => { toast.message = ""; }, 3500);
        };
        const getError = (error) => {
            const data = error.response?.data;
            if (!data) return "无法连接 Django 服务。";
            if (typeof data === "string") return data;
            if (data.detail) return data.detail;
            return Object.entries(data).map(([key, value]) => `${key}: ${value}`).join("；");
        };

        const scheduleTrainingPoll = (delay = 1200) => {
            if (trainingPollTimer) window.clearTimeout(trainingPollTimer);
            trainingPollTimer = window.setTimeout(pollTraining, delay);
        };
        const pollTraining = async () => {
            try {
                const response = await axios.get(`${API}/training/`);
                Object.assign(trainingState, response.data);
                if (response.data.status === "running") {
                    scheduleTrainingPoll();
                } else {
                    await loadAll(false);
                }
            } catch (error) {
                trainingError.value = getError(error);
                scheduleTrainingPoll(3000);
            }
        };

        const loadAll = async (showSpinner = true) => {
            if (showSpinner) pageLoading.value = true;
            try {
                const [
                    dashboardResponse,
                    datasetResponse,
                    modelResponse,
                    recognitionResponse,
                    videoResponse,
                    trainingResponse,
                ] = await Promise.all([
                    axios.get(`${API}/dashboard/`),
                    axios.get(`${API}/data/`),
                    axios.get(`${API}/model/`),
                    axios.get(`${API}/recognition/`),
                    axios.get(`${API}/video-recognition/`),
                    axios.get(`${API}/training/`),
                ]);
                Object.assign(dashboard, dashboardResponse.data);
                Object.assign(dataset, datasetResponse.data);
                Object.assign(modelInfo, modelResponse.data);
                recognitionHistory.value = recognitionResponse.data;
                videoHistory.value = videoResponse.data;
                Object.assign(trainingState, trainingResponse.data);
                await loadDataEngineering();
                apiOnline.value = true;
                if (trainingState.status === "running") scheduleTrainingPoll();
                if (!mediaResult.value && recognitionResponse.data.length) {
                    mediaResult.value = recognitionResponse.data[0];
                    mediaImagePreview.value = recognitionResponse.data[0].original_img_url;
                }
                if (!videoResult.value && videoResponse.data.length) {
                    videoResult.value = videoResponse.data[0];
                }
            } catch (error) {
                apiOnline.value = false;
                showToast(getError(error), "error");
            } finally {
                pageLoading.value = false;
            }
        };
        const refreshAll = async () => {
            await loadAll();
            showToast("数据已刷新");
        };

        const startTraining = async () => {
            trainingError.value = "";
            try {
                const response = await axios.post(`${API}/training/`, {
                    epochs: Number(trainingForm.epochs),
                    imgsz: Number(trainingForm.imgsz),
                    batch: String(trainingForm.batch),
                    device: trainingForm.device,
                });
                Object.assign(trainingState, response.data);
                showToast("训练任务已启动");
                scheduleTrainingPoll(800);
            } catch (error) {
                trainingError.value = getError(error);
                showToast(trainingError.value, "error");
            }
        };
        const cancelTraining = async () => {
            try {
                const response = await axios.post(`${API}/training/cancel/`);
                Object.assign(trainingState, response.data);
                showToast("训练任务已停止");
                await loadAll(false);
            } catch (error) {
                showToast(getError(error), "error");
            }
        };
        const validateDataset = async () => {
            await startDataJob("validate");
        };
        const loadDataHistory = async () => {
            const selected = dataForm.dataset;
            const response = await axios.get(`${API}/data/history/`, { params: { dataset: selected } });
            if (dataForm.dataset === selected) Object.assign(dataHistory, response.data);
        };
        const selectEngineeringDataset = async () => {
            const selected = selectedDataset.value;
            if (selected) {
                dataForm.classes = (selected.classes || []).join(", ");
                dataForm.format = engineering.formats.includes(selected.format) ? selected.format : "yolo";
                dataForm.task = selected.task || "detect";
            }
            try { await loadDataHistory(); } catch (error) { showToast(getError(error), "error"); }
        };
        const scheduleDataPoll = () => {
            if (dataPollTimer) window.clearTimeout(dataPollTimer);
            dataPollTimer = window.setTimeout(pollDataJob, 800);
        };
        const pollDataJob = async () => {
            try {
                const response = await axios.get(`${API}/data/jobs/`);
                Object.assign(dataJob, response.data);
                validatingData.value = dataJob.status === "running";
                if (dataJob.status === "running") scheduleDataPoll();
                else {
                    await loadAll(false);
                    showToast(dataJob.status === "failed" ? dataJob.error : "数据任务已完成，请查看报告",
                        dataJob.status === "failed" ? "error" : "success");
                }
            } catch (error) {
                showToast(getError(error), "error");
                scheduleDataPoll();
            }
        };
        const loadDataEngineering = async () => {
            const [catalog, job] = await Promise.all([
                axios.get(`${API}/data/engineering/`), axios.get(`${API}/data/jobs/`),
            ]);
            const firstLoad = engineering.datasets.length === 0;
            Object.assign(engineering, catalog.data);
            Object.assign(dataJob, job.data);
            validatingData.value = dataJob.status === "running";
            if (!selectedDataset.value && engineering.datasets.length) dataForm.dataset = engineering.datasets[0].name;
            if (firstLoad) await selectEngineeringDataset();
            else await loadDataHistory();
            if (dataJob.status === "running") scheduleDataPoll();
        };
        const startDataJob = async (operation) => {
            dataSubmitting.value = true;
            try {
                const payload = { ...dataForm, operation,
                    classes: dataForm.classes.split(/[,，\n]/).map((c) => c.trim()).filter(Boolean) };
                if (payload.format !== "yolo" && !payload.classes.length) delete payload.classes;
                const response = await axios.post(`${API}/data/jobs/`, payload);
                Object.assign(dataJob, response.data);
                validatingData.value = true;
                scheduleDataPoll();
            } catch (error) { showToast(getError(error), "error"); }
            finally { dataSubmitting.value = false; }
        };
        const evaluateModel = async () => {
            evaluatingModel.value = true;
            try {
                const response = await axios.post(`${API}/model/evaluate/`, {
                    imgsz: Number(trainingForm.imgsz),
                    device: trainingForm.device,
                }, { timeout: 180000 });
                modelInfo.evaluation = response.data;
                showToast("测试集评估完成");
                await loadAll(false);
            } catch (error) {
                showToast(getError(error), "error");
            } finally {
                evaluatingModel.value = false;
            }
        };

        const releaseUrl = (url) => {
            if (url?.startsWith("blob:")) URL.revokeObjectURL(url);
        };
        const validateImageFile = (file, maxBytes = MAX_IMAGE_BYTES) => {
            if (!file.type.startsWith("image/")) throw new Error("请选择图片文件。");
            if (file.size > maxBytes) throw new Error("图片不能超过 10 MB。");
        };
        const setInferenceFile = (file) => {
            validateImageFile(file);
            releaseUrl(inferencePreview.value);
            inferenceFile.value = file;
            inferencePreview.value = URL.createObjectURL(file);
            inferenceResult.value = null;
            inferenceError.value = "";
        };
        const onInferenceSelect = (event) => {
            const file = event.target.files?.[0];
            if (file) {
                try { setInferenceFile(file); } catch (error) { inferenceError.value = error.message; }
            }
        };
        const onInferenceDrop = (event) => {
            const file = event.dataTransfer?.files?.[0];
            if (file) {
                try { setInferenceFile(file); } catch (error) { inferenceError.value = error.message; }
            }
        };
        const clearInference = () => {
            releaseUrl(inferencePreview.value);
            inferenceFile.value = null;
            inferencePreview.value = "";
            inferenceResult.value = null;
            inferenceError.value = "";
            const input = document.querySelector('input[type="file"][accept="image/*"]');
            if (input) input.value = "";
        };
        const runInference = async () => {
            if (!inferenceFile.value) return;
            inferenceLoading.value = true;
            inferenceError.value = "";
            const data = new FormData();
            data.append("image", inferenceFile.value);
            data.append("confidence", inferenceConfidence.value);
            data.append("iou", "0.7");
            try {
                const response = await axios.post(`${API}/recognition/`, data, { timeout: 120000 });
                inferenceResult.value = response.data;
                recognitionHistory.value = [response.data, ...recognitionHistory.value];
                showToast(`推理完成：${response.data.object_count} 个目标`);
            } catch (error) {
                inferenceError.value = getError(error);
            } finally {
                inferenceLoading.value = false;
            }
        };

        const setMediaFile = (file) => {
            validateImageFile(file);
            releaseUrl(mediaImagePreview.value);
            mediaFile.value = file;
            mediaImagePreview.value = URL.createObjectURL(file);
            mediaResult.value = null;
            mediaError.value = "";
        };
        const onMediaImageSelect = (event) => {
            const file = event.target.files?.[0];
            if (file) {
                try { setMediaFile(file); } catch (error) { mediaError.value = error.message; }
            }
        };
        const onMediaImageDrop = (event) => {
            const file = event.dataTransfer?.files?.[0];
            if (file) {
                try { setMediaFile(file); } catch (error) { mediaError.value = error.message; }
            }
        };
        const runMediaRecognition = async () => {
            if (!mediaFile.value) return;
            mediaLoading.value = true;
            mediaError.value = "";
            const data = new FormData();
            data.append("image", mediaFile.value);
            data.append("confidence", mediaConfidence.value);
            data.append("iou", "0.7");
            try {
                const response = await axios.post(`${API}/recognition/`, data, { timeout: 120000 });
                mediaResult.value = response.data;
                recognitionHistory.value = [response.data, ...recognitionHistory.value];
                showToast(`图片识别完成：${response.data.object_count} 个目标`);
                await loadAll(false);
            } catch (error) {
                mediaError.value = getError(error);
            } finally {
                mediaLoading.value = false;
            }
        };
        const clearMedia = () => {
            releaseUrl(mediaImagePreview.value);
            mediaFile.value = null;
            mediaImagePreview.value = "";
            mediaResult.value = null;
            mediaError.value = "";
        };
        const selectMediaRecord = (record) => {
            releaseUrl(mediaImagePreview.value);
            mediaImagePreview.value = record.original_img_url || "";
            mediaResult.value = record;
            mediaFile.value = null;
        };
        const deleteRecognition = async (record) => {
            if (!window.confirm(`删除图片识别记录 #${record.id}？`)) return;
            try {
                await axios.delete(`${API}/recognition/${record.id}/`);
                recognitionHistory.value = recognitionHistory.value.filter((item) => item.id !== record.id);
                if (mediaResult.value?.id === record.id) clearMedia();
                showToast("图片识别记录已删除");
                await loadAll(false);
            } catch (error) {
                showToast(getError(error), "error");
            }
        };

        const setVideoFile = (file) => {
            if (!file.type.startsWith("video/")) throw new Error("请选择视频文件。");
            if (file.size > MAX_VIDEO_BYTES) throw new Error("视频不能超过 100 MB。");
            releaseUrl(mediaVideoPreview.value);
            mediaVideoFile.value = file;
            mediaVideoPreview.value = URL.createObjectURL(file);
            videoResult.value = null;
            videoError.value = "";
        };
        const onMediaVideoSelect = (event) => {
            const file = event.target.files?.[0];
            if (file) {
                try { setVideoFile(file); } catch (error) { videoError.value = error.message; }
            }
        };
        const onMediaVideoDrop = (event) => {
            const file = event.dataTransfer?.files?.[0];
            if (file) {
                try { setVideoFile(file); } catch (error) { videoError.value = error.message; }
            }
        };
        const runVideoRecognition = async () => {
            if (!mediaVideoFile.value) return;
            videoLoading.value = true;
            videoError.value = "";
            const data = new FormData();
            data.append("video", mediaVideoFile.value);
            data.append("confidence", mediaConfidence.value);
            data.append("iou", "0.7");
            data.append("frame_stride", videoFrameStride.value);
            try {
                const response = await axios.post(`${API}/video-recognition/`, data, { timeout: 300000 });
                videoResult.value = response.data;
                videoHistory.value = [response.data, ...videoHistory.value];
                showToast(`视频识别完成：${response.data.object_count} 个目标`);
                await loadAll(false);
            } catch (error) {
                videoError.value = getError(error);
            } finally {
                videoLoading.value = false;
            }
        };
        const clearVideo = () => {
            releaseUrl(mediaVideoPreview.value);
            mediaVideoFile.value = null;
            mediaVideoPreview.value = "";
            videoResult.value = null;
            videoError.value = "";
        };
        const selectVideoRecord = (record) => {
            releaseUrl(mediaVideoPreview.value);
            mediaVideoPreview.value = "";
            videoResult.value = record;
            mediaVideoFile.value = null;
        };
        const deleteVideo = async (record) => {
            if (!window.confirm(`删除视频识别记录 #${record.id}？`)) return;
            try {
                await axios.delete(`${API}/video-recognition/${record.id}/`);
                videoHistory.value = videoHistory.value.filter((item) => item.id !== record.id);
                if (videoResult.value?.id === record.id) clearVideo();
                showToast("视频识别记录已删除");
                await loadAll(false);
            } catch (error) {
                showToast(getError(error), "error");
            }
        };

        onMounted(loadAll);
        onBeforeUnmount(() => {
            if (dataPollTimer) window.clearTimeout(dataPollTimer);
            if (trainingPollTimer) window.clearTimeout(trainingPollTimer);
            if (toast.timer) window.clearTimeout(toast.timer);
            [inferencePreview, mediaImagePreview, mediaVideoPreview].forEach((item) => releaseUrl(item.value));
        });

        return {
            modules,
            activeModule,
            currentModule,
            pageLoading,
            apiOnline,
            toast,
            dataset,
            engineering, dataForm, dataJob, dataHistory, dataRunning, selectedDataset,
            comparison, severityClass, dataStageLabel, selectEngineeringDataset, startDataJob,
            dashboard,
            modelInfo,
            modelStatusLabel,
            trainingState,
            trainingForm,
            trainingError,
            trainingRunning,
            trainingStageLabel,
            validatingData,
            evaluatingModel,
            classRows,
            classMapRows,
            recognitionClassRows,
            recognitionHistory,
            videoHistory,
            splitGradient,
            inferenceFile,
            inferencePreview,
            inferenceResult,
            inferenceConfidence,
            inferenceLoading,
            inferenceError,
            mediaTab,
            mediaFile,
            mediaVideoFile,
            mediaImagePreview,
            mediaVideoPreview,
            mediaResult,
            videoResult,
            mediaConfidence,
            videoFrameStride,
            mediaLoading,
            videoLoading,
            mediaError,
            videoError,
            formatNumber,
            confidencePercent,
            metricPercent,
            formatDate,
            barPercent,
            chartPoints,
            chartDots,
            lossPoints,
            loadAll,
            refreshAll,
            startTraining,
            cancelTraining,
            validateDataset,
            evaluateModel,
            onInferenceSelect,
            onInferenceDrop,
            clearInference,
            runInference,
            onMediaImageSelect,
            onMediaImageDrop,
            runMediaRecognition,
            clearMedia,
            selectMediaRecord,
            deleteRecognition,
            onMediaVideoSelect,
            onMediaVideoDrop,
            runVideoRecognition,
            clearVideo,
            selectVideoRecord,
            deleteVideo,
        };
    },
}).mount("#app");
