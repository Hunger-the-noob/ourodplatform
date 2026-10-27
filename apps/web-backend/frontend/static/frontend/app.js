const { computed, createApp, onMounted, onUnmounted, reactive, ref } = Vue;

const API_BASE = "/api";
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;

function emptyForm() {
    return {
        defect_type: "",
        detect_person: "",
        device_id: "",
        confidence: 0.85,
        remark: "",
    };
}

createApp({
    setup() {
        const records = ref([]);
        const typeList = ref([]);
        const stats = reactive({
            total_records: 0,
            today_records: 0,
            high_confidence: 0,
            total_types: 0,
            average_confidence: 0,
            type_counts: [],
        });
        const modelLoading = ref(true);
        const modelInfo = reactive({
            status: "loading",
            run_id: "",
            experiment_name: "",
            dataset: {
                name: "—",
                classes: [],
                class_count: 0,
                splits: {},
                validation: {},
            },
            training: {},
            metrics: {},
            weights: {},
            class_map_50_95: {},
            curve_url: null,
        });

        const loading = ref(true);
        const saving = ref(false);
        const apiOnline = ref(false);
        const searchText = ref("");
        const filterType = ref("");

        const formOpen = ref(false);
        const editingId = ref(null);
        const form = reactive(emptyForm());
        const originalFile = ref(null);
        const defectFile = ref(null);
        const previewOriginal = ref("");
        const previewDefect = ref("");
        const formError = ref("");

        const detailOpen = ref(false);
        const detailRecord = ref(null);
        const toast = reactive({ message: "", type: "success", timer: null });

        const filteredRecords = computed(() => {
            const keyword = searchText.value.toLowerCase();
            return records.value.filter((record) => {
                const matchesType =
                    !filterType.value ||
                    String(record.defect_type) === filterType.value;
                if (!matchesType) return false;
                if (!keyword) return true;
                return [
                    record.id,
                    record.defect_type_name,
                    record.detect_person,
                    record.device_id,
                    record.remark,
                ]
                    .join(" ")
                    .toLowerCase()
                    .includes(keyword);
            });
        });

        const typeStats = computed(() => {
            const items = stats.type_counts || [];
            return items.filter((item) => item.count > 0).slice(0, 6);
        });

        const modelStatusLabel = computed(() => {
            const labels = {
                completed: "训练完成",
                training: "训练进行中",
                not_found: "暂无训练记录",
                loading: "读取中",
            };
            return labels[modelInfo.status] || "未知状态";
        });

        const classMetricRows = computed(() => {
            return Object.entries(modelInfo.class_map_50_95 || {})
                .filter(([, value]) => Number.isFinite(Number(value)))
                .map(([name, value]) => ({ name, value: Number(value) }))
                .sort((left, right) => right.value - left.value);
        });

        const typePercent = (count) => {
            const maxCount = Math.max(
                ...typeStats.value.map((item) => Number(item.count) || 0),
                1,
            );
            return Math.max(8, Math.round((Number(count) / maxCount) * 100));
        };

        const confidencePercent = (value) => {
            const numericValue = Number(value);
            if (!Number.isFinite(numericValue)) return "0%";
            return `${Math.round(numericValue * 100)}%`;
        };

        const metricPercent = (value) => {
            if (value === null || value === undefined) return "—";
            const numericValue = Number(value);
            if (!Number.isFinite(numericValue)) return "—";
            return `${Math.round(Math.max(0, Math.min(1, numericValue)) * 100)}%`;
        };

        const formatDate = (value, withTime) => {
            const date = new Date(value);
            if (Number.isNaN(date.getTime())) return value || "—";
            const options = withTime
                ? { hour: "2-digit", minute: "2-digit", hour12: false }
                : { year: "numeric", month: "2-digit", day: "2-digit" };
            return new Intl.DateTimeFormat("zh-CN", options).format(date);
        };

        const thumbStyle = (record) => {
            if (!record.original_img_url) {
                return {
                    backgroundImage:
                        "linear-gradient(135deg, #d6ebe7, #b7d8d3)",
                };
            }
            return {
                backgroundImage: `url("${record.original_img_url}")`,
            };
        };

        const showToast = (message, type = "success") => {
            if (toast.timer) window.clearTimeout(toast.timer);
            toast.message = message;
            toast.type = type;
            toast.timer = window.setTimeout(() => {
                toast.message = "";
            }, 3200);
        };

        const getErrorMessage = (error) => {
            const data = error.response?.data;
            if (!data) return "无法连接 Django 服务，请确认后端已启动。";
            if (typeof data === "string") return data;
            if (data.detail) return data.detail;
            if (typeof data === "object") {
                const messages = Object.entries(data).map(([field, value]) => {
                    const text = Array.isArray(value) ? value.join("；") : value;
                    return `${field}: ${text}`;
                });
                if (messages.length) return messages.join("\n");
            }
            return "操作失败，请检查输入后重试。";
        };

        const loadAll = async () => {
            loading.value = true;
            modelLoading.value = true;
            try {
                const [recordResponse, typeResponse, statsResponse, modelResponse] =
                    await Promise.all([
                        axios.get(`${API_BASE}/defect-record/`),
                        axios.get(`${API_BASE}/defect-type/`),
                        axios.get(`${API_BASE}/stats/`),
                        axios.get(`${API_BASE}/model/`),
                    ]);
                records.value = recordResponse.data;
                typeList.value = typeResponse.data;
                Object.assign(stats, statsResponse.data);
                Object.assign(modelInfo, modelResponse.data);
                apiOnline.value = true;
            } catch (error) {
                apiOnline.value = false;
                modelInfo.status = "not_found";
                showToast(getErrorMessage(error), "error");
            } finally {
                loading.value = false;
                modelLoading.value = false;
            }
        };

        const scrollToModel = () => {
            document.querySelector("#model-panel")?.scrollIntoView({
                behavior: "smooth",
                block: "start",
            });
        };

        const releaseObjectUrl = (url) => {
            if (url?.startsWith("blob:")) URL.revokeObjectURL(url);
        };

        const resetForm = () => {
            Object.assign(form, emptyForm());
            originalFile.value = null;
            defectFile.value = null;
            releaseObjectUrl(previewOriginal.value);
            releaseObjectUrl(previewDefect.value);
            previewOriginal.value = "";
            previewDefect.value = "";
            formError.value = "";
        };

        const openCreate = () => {
            if (!typeList.value.length) {
                showToast("请先在 Django 后台创建至少一种缺陷类型。", "error");
                return;
            }
            resetForm();
            editingId.value = null;
            form.defect_type = typeList.value[0].id;
            formOpen.value = true;
        };

        const openEdit = (record) => {
            resetForm();
            editingId.value = record.id;
            Object.assign(form, {
                defect_type: record.defect_type,
                detect_person: record.detect_person,
                device_id: record.device_id,
                confidence: Number(record.confidence),
                remark: record.remark || "",
            });
            previewOriginal.value = record.original_img_url || "";
            previewDefect.value = record.defect_img_url || "";
            formOpen.value = true;
        };

        const closeForm = () => {
            if (saving.value) return;
            formOpen.value = false;
            releaseObjectUrl(previewOriginal.value);
            releaseObjectUrl(previewDefect.value);
        };

        const validateImage = (file) => {
            if (!file.type.startsWith("image/")) {
                throw new Error("请选择有效的图片文件。");
            }
            if (file.size > MAX_IMAGE_BYTES) {
                throw new Error("图片不能超过 10 MB。");
            }
        };

        const onOriginalChange = (event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            try {
                validateImage(file);
                releaseObjectUrl(previewOriginal.value);
                originalFile.value = file;
                previewOriginal.value = URL.createObjectURL(file);
                formError.value = "";
            } catch (error) {
                event.target.value = "";
                formError.value = error.message;
            }
        };

        const onDefectChange = (event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            try {
                validateImage(file);
                releaseObjectUrl(previewDefect.value);
                defectFile.value = file;
                previewDefect.value = URL.createObjectURL(file);
                formError.value = "";
            } catch (error) {
                event.target.value = "";
                formError.value = error.message;
            }
        };

        const submitForm = async () => {
            formError.value = "";
            const confidence = Number(form.confidence);
            if (!form.defect_type || !form.detect_person || !form.device_id) {
                formError.value = "请填写缺陷类型、检测人员和设备编号。";
                return;
            }
            if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
                formError.value = "置信度必须是 0 到 1 之间的数字。";
                return;
            }
            if (!editingId.value && !originalFile.value) {
                formError.value = "新增记录必须上传原始图片。";
                return;
            }

            const payload = new FormData();
            payload.append("defect_type", form.defect_type);
            payload.append("detect_person", form.detect_person);
            payload.append("device_id", form.device_id);
            payload.append("confidence", confidence);
            payload.append("remark", form.remark);
            if (originalFile.value) payload.append("original_img", originalFile.value);
            if (defectFile.value) payload.append("defect_img", defectFile.value);

            saving.value = true;
            try {
                if (editingId.value) {
                    await axios.patch(
                        `${API_BASE}/defect-record/${editingId.value}/`,
                        payload,
                    );
                    showToast("检测记录已更新");
                } else {
                    await axios.post(`${API_BASE}/defect-record/`, payload);
                    showToast("检测记录已创建");
                }
                formOpen.value = false;
                await loadAll();
            } catch (error) {
                formError.value = getErrorMessage(error);
            } finally {
                saving.value = false;
            }
        };

        const openDetail = (record) => {
            detailRecord.value = record;
            detailOpen.value = true;
        };

        const editFromDetail = () => {
            const record = detailRecord.value;
            detailOpen.value = false;
            if (record) openEdit(record);
        };

        const removeRecord = async (record) => {
            const confirmed = window.confirm(
                `确认删除记录 #${String(record.id).padStart(4, "0")}？此操作不可撤销。`,
            );
            if (!confirmed) return;
            try {
                await axios.delete(`${API_BASE}/defect-record/${record.id}/`);
                showToast("检测记录已删除");
                if (detailRecord.value?.id === record.id) detailOpen.value = false;
                await loadAll();
            } catch (error) {
                showToast(getErrorMessage(error), "error");
            }
        };

        // 智能助手 + 视频分析：状态与方法由 assistant.js 提供，这里只负责组装
        const assistant = window.AssistantModule.create({
            notify: showToast,
            getErrorMessage,
            onArchived: loadAll,
        });

        onMounted(() => {
            loadAll();
            assistant.init();
        });
        onUnmounted(() => assistant.dispose());

        return {
            ...assistant,
            records,
            typeList,
            stats,
            modelLoading,
            modelInfo,
            modelStatusLabel,
            classMetricRows,
            loading,
            saving,
            apiOnline,
            searchText,
            filterType,
            filteredRecords,
            typeStats,
            typePercent,
            confidencePercent,
            metricPercent,
            formatDate,
            thumbStyle,
            formOpen,
            editingId,
            form,
            previewOriginal,
            previewDefect,
            formError,
            detailOpen,
            detailRecord,
            toast,
            loadAll,
            openCreate,
            openEdit,
            closeForm,
            onOriginalChange,
            onDefectChange,
            submitForm,
            openDetail,
            editFromDetail,
            removeRecord,
            scrollToModel,
        };
    },
}).mount("#app");
