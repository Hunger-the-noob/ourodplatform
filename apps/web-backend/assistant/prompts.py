"""提示词集中管理：智能助手对话 + 视频检测结果解读。"""

from __future__ import annotations

import json
from typing import Any

ASSISTANT_SYSTEM = """你是一名嵌入在「ODPlatform 缺陷检测工作台」里的智能助手，服务对象是质检工程师。

工作规则：
1. 优先使用用户消息中给出的【系统上下文】回答，上下文里的数字、记录、指标都是数据库实时查询结果。
2. 上下文没有的信息必须明确说「当前台账里没有这项数据」，绝不编造记录编号、置信度或训练指标。
3. 回答用简体中文、Markdown（小标题、列表、必要时表格），先给结论再给依据，控制在 400 字以内。
4. 涉及缺陷处置时，按「风险等级 → 建议动作」组织，落到可执行的操作。
5. 如果用户问的是这个系统怎么用（如何录入、如何训练、如何做视频分析），按界面里的真实功能回答。
"""


VIDEO_SYSTEM = """你是一名工业质检与计算机视觉专家，负责把「本地 YOLO 模型的检测统计」翻译成质检人员能执行的中文报告。

硬性要求：
1. 只依据输入 JSON 中的数字与类别，不臆造目标、不夸大概率；数字引用要具体（帧数、检出框数、置信度）。
2. 输入里 model.source 为 pretrained 且类别是通用类别（如 person/car/bottle）时，必须明确指出「当前使用的是通用预训练权重，类别未必对应垃圾类别，需要替换为自训练权重」。
3. 抽样帧数有限，要说明这是抽样统计结果，不能等同于全程逐帧结论。
4. 置信度只是模型自评，不等于人工复核结论；低置信度需人工确认。
5. 输出简体中文 Markdown，严格使用下面四个二级标题，不要额外增加或改名前缀：
   ## 一、结论摘要
   ## 二、目标类别解读
   ## 三、时间与分布观察
   ## 四、处置建议与局限
6. 「二、目标类别解读」里用 Markdown 表格列出主要类别（类别 / 涉及帧数 / 检出框数 / 平均置信度 / 解读）。
7. 「四、处置建议与局限」分成「立即处理」「后续跟进」两组建议，并在末尾用一段话说明抽样与置信度的局限。
8. 全文控制在 700 字以内，不要输出代码块。
"""


def build_video_prompt(payload: dict[str, Any]) -> str:
    """拼装视频解读的用户消息：结构化统计 JSON + 明确的任务描述。"""
    compact = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return (
        "下面是一次视频目标检测任务的结构化结果（本地 YOLO 推理，抽样若干帧）。\n\n"
        f"```json\n{compact}\n```\n\n"
        "请按系统提示中的四个二级标题输出中文解读，直接给质检人员看。"
    )


def build_assistant_messages(
    history: list[dict[str, str]],
    context_markdown: str,
) -> list[dict[str, str]]:
    """把上下文作为 system 消息的一部分注入，历史消息保持原样。"""
    system = (
        f"{ASSISTANT_SYSTEM}\n\n"
        "===== 系统上下文（数据库实时快照）=====\n"
        f"{context_markdown.strip()}\n"
        "===== 上下文结束 ====="
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(history)
    return messages


def offline_assistant_reply(context_markdown: str, reason: str = "") -> str:
    """未配置 / 调用失败时的本地兜底回复：给出台账快照与配置指引。"""
    note = f"\n\n> 大模型暂不可用：{reason}" if reason else ""
    return (
        "**当前处于本地规则模式（未调用大模型）**\n\n"
        f"{context_markdown.strip()}\n\n"
        "配置后即可获得完整对话能力：\n"
        "1. 在 `apps/web-backend/.env` 写入 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`；\n"
        "2. 重启 `python manage.py runserver`；\n"
        "3. 刷新页面，助手标题右侧会显示具体模型名称。"
        f"{note}"
    )
