/**
 * 智能助手 + 视频智能分析前端模块。
 *
 * 与 app.js 共用同一个 Vue 应用：app.js 在 setup() 里调用
 * window.AssistantModule.create({ notify, getErrorMessage, onArchived })
 * 把返回的对象展开进 setup 的返回值即可，避免主文件继续膨胀。
 */
(function () {
    "use strict";

    const { computed, nextTick, reactive, ref } = Vue;
    const API_BASE = "/api/assistant";
    const MAX_VIDEO_BYTES = 200 * 1024 * 1024;

    const VIDEO_STATUS_LABEL = {
        pending: "排队中",
        running: "分析中",
        completed: "已完成",
        failed: "失败",
    };

    const MODEL_SOURCE_LABEL = {
        trained: "自训练权重",
        pretrained: "预训练权重",
        "pretrained-download": "预训练权重（自动下载）",
        configured: "指定权重",
        "configured-autoload": "指定权重（自动下载）",
    };

    // ---------------------------------------------------------------- markdown
    function escapeHtml(text) {
        return String(text).replace(/[&<>"']/g, (char) => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;",
        }[char]));
    }

    function renderInline(text) {
        return text
            .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
            .replace(/`([^`]+)`/g, "<code>$1</code>")
            .replace(
                /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
                '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
            );
    }

    function renderMarkdown(raw) {
        if (!raw) return "";
        let text = escapeHtml(String(raw)).replace(/\r\n/g, "\n");
        const blocks = [];
        text = text.replace(/```[a-zA-Z0-9_-]*\n([\s\S]*?)```/g, (match, code) => {
            blocks.push(`<pre class="md-code"><code>${code.replace(/\n$/, "")}</code></pre>`);
            return `\u0000CODE${blocks.length - 1}\u0000`;
        });

        const lines = text.split("\n");
        const html = [];
        let listType = null;
        let paragraph = [];

        const closeList = () => {
            if (listType) {
                html.push(`</${listType}>`);
                listType = null;
            }
        };
        const flushParagraph = () => {
            if (paragraph.length) {
                html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
                paragraph = [];
            }
        };
        const openList = (type) => {
            if (listType !== type) {
                closeList();
                html.push(`<${type}>`);
                listType = type;
            }
        };

        for (let index = 0; index < lines.length; index += 1) {
            const line = lines[index];
            const trimmed = line.trim();

            if (!trimmed) {
                flushParagraph();
                closeList();
                continue;
            }
            if (/^\u0000CODE\d+\u0000$/.test(trimmed)) {
                flushParagraph();
                closeList();
                html.push(trimmed);
                continue;
            }

            const heading = trimmed.match(/^(#{1,4})\s+(.*)$/);
            if (heading) {
                flushParagraph();
                closeList();
                const level = Math.min(heading[1].length + 2, 6);
                html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
                continue;
            }

            if (/^(---+|\*\*\*+)$/.test(trimmed)) {
                flushParagraph();
                closeList();
                html.push("<hr>");
                continue;
            }

            const quote = trimmed.match(/^&gt;\s?(.*)$/);
            if (quote) {
                flushParagraph();
                closeList();
                html.push(`<blockquote>${renderInline(quote[1])}</blockquote>`);
                continue;
            }

            if (/^\|.*\|$/.test(trimmed)) {
                flushParagraph();
                closeList();
                const rows = [];
                while (index < lines.length && /^\|.*\|$/.test(lines[index].trim())) {
                    rows.push(lines[index].trim());
                    index += 1;
                }
                index -= 1;
                const cells = (row) => row.replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
                const isDivider = (row) => /^\|[\s:|-]+\|$/.test(row) && row.includes("-");
                const header = cells(rows[0]).map((cell) => `<th>${renderInline(cell)}</th>`).join("");
                const bodyRows = rows.slice(1).filter((row) => !isDivider(row));
                const body = bodyRows
                    .map((row) => `<tr>${cells(row).map((cell) => `<td>${renderInline(cell)}</td>`).join("")}</tr>`)
                    .join("");
                html.push(
                    `<div class="md-table"><table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table></div>`,
                );
                continue;
            }

            const bullet = trimmed.match(/^[-*+]\s+(.*)$/);
            if (bullet) {
                flushParagraph();
                openList("ul");
                html.push(`<li>${renderInline(bullet[1])}</li>`);
                continue;
            }

            const ordered = trimmed.match(/^\d+[.)]\s+(.*)$/);
            if (ordered) {
                flushParagraph();
                openList("ol");
                html.push(`<li>${renderInline(ordered[1])}</li>`);
                continue;
            }

            closeList();
            paragraph.push(trimmed);
        }

        flushParagraph();
        closeList();
        return html
            .join("\n")
            .replace(/\u0000CODE(\d+)\u0000/g, (match, order) => blocks[Number(order)] || "");
    }

    // ---------------------------------------------------------------- helpers
    function nowLabel() {
        return new Date().toLocaleTimeString("zh-CN", {
            hour: "2-digit",
            minute: "2-digit",
            hour12: false,
        });
    }

    function formatSize(bytes) {
        const value = Number(bytes) || 0;
        if (value >= 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
        if (value >= 1024) return `${(value / 1024).toFixed(0)} KB`;
        return `${value} B`;
    }

    function formatSeconds(seconds) {
        const value = Number(seconds);
        if (!Number.isFinite(value)) return "—";
        if (value >= 60) {
            const minutes = Math.floor(value / 60);
            return `${minutes}分${(value % 60).toFixed(1)}秒`;
        }
        return `${value.toFixed(1)} 秒`;
    }

    function percentText(value) {
        const number = Number(value);
        if (!Number.isFinite(number)) return "—";
        return `${Math.round(Math.max(0, Math.min(1, number)) * 100)}%`;
    }

    // ---------------------------------------------------------------- module
    function create({ notify, getErrorMessage, onArchived }) {
        const assistantOpen = ref(false);
        const assistantTab = ref("chat");
        const activeNav = ref("ledger");
        const chatMessages = ref([]);
        const chatInput = ref("");
        const chatSending = ref(false);
        const chatEngineLabel = ref("未配置（本地规则解释）");
        const assistantScroll = ref(null);

        const llmStatus = reactive({
            configured: false,
            provider: "",
            model: "",
            label: "未配置（本地规则解释）",
        });
        const videoRuntime = reactive({
            python: { available: null, python: "", detail: "尚未探测" },
            model: { path: "", source: "", exists: false },
            limits: { max_upload_mb: 200, max_frames: 24, max_keyframes: 6, conf: 0.25 },
            accept: [".mp4", ".avi", ".mov", ".mkv"],
        });
        const probingRuntime = ref(false);

        const videoFile = ref(null);
        const videoLocalUrl = ref("");
        const metaLoaded = ref(false);
        const videoUploading = ref(false);
        const videoUploadPercent = ref(0);
        const videoJob = ref(null);
        const videoJobs = ref([]);
        const archiving = ref(false);
        const videoPollTimer = ref(null);
        const videoAcceptText = ref(".mp4、.avi、.mov、.mkv");
        const videoForm = reactive({
            conf: 0.25,
            maxFrames: 24,
            person: "",
            device: "",
        });

        const chatChips = [
            "现在台账里有多少条记录？",
            "当前模型的 mAP@50 是多少？",
            "最近一次视频分析发现了什么？",
            "如何把分析结果归档到台账？",
        ];

        const llmNote = computed(() =>
            llmStatus.configured
                ? `引擎：${llmStatus.label}。回答基于台账实时快照，不会编造数据。`
                : "未配置 LLM_API_KEY：助手与视频解读会退化为本地规则模式，但读取的仍是真实台账数据。",
        );

        const videoBusy = computed(
            () =>
                videoUploading.value ||
                (videoJob.value && ["pending", "running"].includes(videoJob.value.status)),
        );
        const videoBadgeLabel = computed(() => {
            if (videoUploading.value) return "上传中";
            if (!videoJob.value) return "待分析";
            return VIDEO_STATUS_LABEL[videoJob.value.status] || videoJob.value.status;
        });
        const videoBadgeClass = computed(() => {
            if (videoUploading.value) return "running";
            if (!videoJob.value) return "idle";
            return videoJob.value.status;
        });
        const videoClasses = computed(() => videoJob.value?.result?.summary?.classes || []);
        const videoClassMax = computed(() =>
            Math.max(1, ...videoClasses.value.map((item) => Number(item.detections) || 0)),
        );
        const videoKeyframes = computed(() => videoJob.value?.result?.keyframes || []);
        const videoSummary = computed(() => videoJob.value?.result?.summary || {});
        const videoModelSource = computed(() => {
            const source = videoJob.value?.result?.model?.source || "";
            return MODEL_SOURCE_LABEL[source] || source || "—";
        });
        const videoLimits = computed(() => videoRuntime.limits || {});
        const runtimeLabel = computed(() => {
            const info = videoRuntime.python || {};
            if (info.available === true) {
                const name = info.label || (info.python || "").split(/[\\/]/).pop();
                return `已就绪 · ${name}`;
            }
            if (info.available === false) return "缺少 ultralytics / opencv";
            return "尚未检测";
        });
        const videoFileName = computed(() => videoFile.value?.name || "");
        // accept 属性必须是英文逗号分隔；界面上用中文顿号更好读
        const videoAcceptAttr = computed(() => (videoRuntime.accept || []).join(","));

        function showAvatar(message) {
            return message.role === "user" ? "我" : "AI";
        }

        async function loadMeta(probe) {
            try {
                const { data } = await axios.get(`${API_BASE}/meta/${probe ? "?probe=1" : ""}`);
                Object.assign(llmStatus, data.llm || {});
                Object.assign(videoRuntime.python, (data.video || {}).python || {});
                Object.assign(videoRuntime.model, (data.video || {}).model || {});
                Object.assign(videoRuntime.limits, (data.video || {}).limits || {});
                if (Array.isArray(data.video_accept)) videoRuntime.accept = data.video_accept;
                videoAcceptText.value = (videoRuntime.accept || []).join("、");
                // 只在首次加载时套用默认参数，避免「重新检测环境」把用户改过的值冲掉
                if (!metaLoaded.value) {
                    videoForm.conf = Number(videoRuntime.limits.conf) || 0.25;
                    videoForm.maxFrames = Number(videoRuntime.limits.max_frames) || 24;
                    metaLoaded.value = true;
                }
            } catch (error) {
                notify?.(getErrorMessage?.(error) || "读取助手配置失败", "error");
            }
        }

        async function probeRuntime(silent) {
            probingRuntime.value = true;
            try {
                await loadMeta(true);
                if (!silent) notify?.("已刷新推理环境检测结果");
            } finally {
                probingRuntime.value = false;
            }
        }

        /**
         * 首屏初始化：读配置、恢复进行中的任务、支持 ?job=<id> 与 #assistant 深链。
         * 推理环境探测较慢，放到最后异步执行，不阻塞首屏。
         */
        async function init() {
            await loadMeta(false);
            await loadVideoJobs();

            const params = new URLSearchParams(window.location.search);
            const deepLinkJobId = params.get("job");
            let initialJob = null;
            if (deepLinkJobId) {
                initialJob = videoJobs.value.find((item) => String(item.id) === deepLinkJobId) || null;
            }
            if (!initialJob) {
                initialJob = videoJobs.value.find((item) => ["pending", "running"].includes(item.status)) || null;
            }
            if (initialJob) openVideoJob(initialJob);

            if (window.location.hash === "#assistant") openAssistant();

            if (videoRuntime.python.available === null) {
                window.setTimeout(() => probeRuntime(true), 1200);
            }
        }

        function toggleAssistant() {
            assistantOpen.value = !assistantOpen.value;
            if (assistantOpen.value) {
                nextTick(() => scrollChatToBottom());
            }
        }

        function openAssistant() {
            assistantOpen.value = true;
            nextTick(() => scrollChatToBottom());
        }

        function scrollChatToBottom() {
            const element = assistantScroll.value;
            if (element) element.scrollTop = element.scrollHeight;
        }

        async function sendChat(preset) {
            const text = String(preset ?? chatInput.value ?? "").trim();
            if (!text || chatSending.value) return;
            chatInput.value = "";
            chatMessages.value.push({ role: "user", content: text, time: nowLabel() });
            chatSending.value = true;
            nextTick(() => scrollChatToBottom());
            try {
                const history = chatMessages.value
                    .slice(-10)
                    .map((item) => ({ role: item.role, content: item.content }));
                const { data } = await axios.post(`${API_BASE}/chat/`, { messages: history });
                chatMessages.value.push({
                    role: "assistant",
                    content: data.reply,
                    time: nowLabel(),
                });
                chatEngineLabel.value = data.engine || "本地规则解释";
                llmStatus.configured = Boolean(data.llm_configured);
                if (data.error && data.llm_configured) {
                    notify?.(`大模型调用失败：${data.error}`, "error");
                }
            } catch (error) {
                chatMessages.value.push({
                    role: "assistant",
                    content: `请求失败：${getErrorMessage?.(error) || "未知错误"}`,
                    time: nowLabel(),
                });
            } finally {
                chatSending.value = false;
                nextTick(() => scrollChatToBottom());
            }
        }

        function onChatKeydown(event) {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendChat();
            }
        }

        function clearChat() {
            chatMessages.value = [];
        }

        // ------------------------------------------------------------ video
        function releaseVideoUrl() {
            if (videoLocalUrl.value.startsWith("blob:")) URL.revokeObjectURL(videoLocalUrl.value);
            videoLocalUrl.value = "";
        }

        function stopPolling() {
            if (videoPollTimer.value) {
                window.clearInterval(videoPollTimer.value);
                videoPollTimer.value = null;
            }
        }

        async function loadVideoJobs() {
            try {
                const { data } = await axios.get(`${API_BASE}/video/`);
                videoJobs.value = data.jobs || [];
            } catch (error) {
                /* 历史列表失败不打断主流程 */
            }
        }

        function onVideoChange(event) {
            const file = event.target.files?.[0];
            if (!file) return;
            const suffix = (file.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
            const allowed = videoRuntime.accept || [];
            if (allowed.length && !allowed.includes(suffix)) {
                event.target.value = "";
                notify?.(`不支持 ${suffix || "该"} 格式，请上传 ${allowed.join("、")}`, "error");
                return;
            }
            if (file.size > MAX_VIDEO_BYTES) {
                event.target.value = "";
                notify?.("视频不能超过 200 MB", "error");
                return;
            }
            releaseVideoUrl();
            stopPolling();
            videoFile.value = file;
            videoLocalUrl.value = URL.createObjectURL(file);
            videoJob.value = null;
            videoUploadPercent.value = 0;
        }

        function startPolling(jobId) {
            stopPolling();
            const tick = async () => {
                try {
                    const { data } = await axios.get(`${API_BASE}/video/${jobId}/`);
                    videoJob.value = data.job;
                    if (["completed", "failed"].includes(data.job.status)) {
                        stopPolling();
                        loadVideoJobs();
                        if (data.job.status === "completed") {
                            const total = data.job.result?.summary?.total_detections ?? 0;
                            notify?.(`视频分析完成：检出 ${total} 个目标`);
                        } else {
                            notify?.(data.job.error || "视频分析失败", "error");
                        }
                    }
                } catch (error) {
                    stopPolling();
                    notify?.(getErrorMessage?.(error) || "查询分析进度失败", "error");
                }
            };
            tick();
            videoPollTimer.value = window.setInterval(tick, 1500);
        }

        async function startVideoAnalysis() {
            if (!videoFile.value || videoBusy.value) return;
            videoUploading.value = true;
            videoUploadPercent.value = 0;
            const payload = new FormData();
            payload.append("video", videoFile.value);
            payload.append("conf", String(videoForm.conf));
            payload.append("max_frames", String(videoForm.maxFrames));
            payload.append("person", videoForm.person);
            payload.append("device", videoForm.device);
            try {
                const { data } = await axios.post(`${API_BASE}/video/`, payload, {
                    onUploadProgress: (event) => {
                        if (event.total) {
                            videoUploadPercent.value = Math.min(
                                99,
                                Math.round((event.loaded / event.total) * 100),
                            );
                        }
                    },
                });
                videoJob.value = data.job;
                notify?.("视频已上传，正在本地推理...");
                startPolling(data.job.id);
            } catch (error) {
                notify?.(getErrorMessage?.(error) || "上传失败", "error");
            } finally {
                videoUploading.value = false;
            }
        }

        function openVideoJob(job) {
            videoJob.value = job;
            releaseVideoUrl();
            if (job && ["pending", "running"].includes(job.status)) startPolling(job.id);
            else stopPolling();
        }

        function resetVideoPanel() {
            stopPolling();
            releaseVideoUrl();
            videoFile.value = null;
            videoJob.value = null;
            videoUploadPercent.value = 0;
        }

        async function archiveVideoJob() {
            if (!videoJob.value || videoJob.value.record_id || archiving.value) return;
            archiving.value = true;
            try {
                const { data } = await axios.post(`${API_BASE}/video/${videoJob.value.id}/record/`);
                videoJob.value = data.job;
                loadVideoJobs();
                notify?.(`已归档为检测记录 #${String(data.record.id).padStart(4, "0")}`);
                onArchived?.();
            } catch (error) {
                notify?.(getErrorMessage?.(error) || "归档失败", "error");
            } finally {
                archiving.value = false;
            }
        }

        function scrollToVideo() {
            activeNav.value = "video";
            document.querySelector("#video-panel")?.scrollIntoView({
                behavior: "smooth",
                block: "start",
            });
        }

        function scrollToLedger() {
            activeNav.value = "ledger";
            window.scrollTo({ top: 0, behavior: "smooth" });
        }

        function updateActiveNav() {
            const position = window.scrollY + 160;
            const recordsPanel = document.querySelector("#records-panel");
            if (recordsPanel && position >= recordsPanel.offsetTop) {
                activeNav.value = "ledger";
                return;
            }
            const modelPanel = document.querySelector("#model-panel");
            if (modelPanel && position >= modelPanel.offsetTop) {
                activeNav.value = "model";
                return;
            }
            const videoPanel = document.querySelector("#video-panel");
            if (videoPanel && position >= videoPanel.offsetTop) {
                activeNav.value = "video";
                return;
            }
            activeNav.value = "ledger";
        }

        function scrollSpyHandler() {
            updateActiveNav();
        }

        function dispose() {
            stopPolling();
            releaseVideoUrl();
        }

        return {
            // 助手
            assistantOpen,
            assistantTab,
            toggleAssistant,
            openAssistant,
            chatMessages,
            chatInput,
            chatSending,
            chatEngineLabel,
            chatChips,
            assistantScroll,
            llmStatus,
            llmNote,
            sendChat,
            onChatKeydown,
            clearChat,
            showAvatar,
            renderMarkdown,
            // 导航
            activeNav,
            scrollToVideo,
            scrollToLedger,
            // 运行时信息
            videoRuntime,
            videoLimits,
            videoAcceptText,
            videoAcceptAttr,
            probingRuntime,
            runtimeLabel,
            probeRuntime,
            loadMeta,
            init,
            // 视频分析
            videoFile,
            videoFileName,
            videoLocalUrl,
            videoUploading,
            videoUploadPercent,
            videoJob,
            videoJobs,
            videoBusy,
            videoBadgeLabel,
            videoBadgeClass,
            videoClasses,
            videoClassMax,
            videoKeyframes,
            videoSummary,
            videoModelSource,
            videoForm,
            archiving,
            onVideoChange,
            startVideoAnalysis,
            openVideoJob,
            resetVideoPanel,
            archiveVideoJob,
            loadVideoJobs,
            stopPolling,
            // 通用
            formatSize,
            formatSeconds,
            percentText,
            dispose,
        };
    }

    window.AssistantModule = { create, renderMarkdown };
})();
