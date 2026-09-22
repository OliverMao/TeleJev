"""所有模型提示词集中在这里。

设计要点（APC / 前缀缓存友好）：

    [system]  固定的通用规则（全局唯一）        <- 所有请求共用
    [user]    image(s)                          <- 帧
              任务清单文本                      <- 尾部，可变化

把「每路不同的任务清单」放在图像之后，不同摄像头仍能共享 system + 帧序列的 KV
前缀；编辑任务只影响尾部文本，不会让整段前缀失效。

输出契约：``violations`` 中的每一项必须是任务名（taskName），且只能反映
**最后一帧**的状态，以便前端按名字点亮对应的监测卡片。
"""

from __future__ import annotations

import json

# Jev 式读 logits 的 system 提示：只回一个大写字母。
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

# 自回归基线的 system 提示：全局通用规则，所有请求共用（APC 前缀）。
GENERATION_SYSTEM = """你是一名实时视频流助手，逐帧观察连续的摄像头画面，并按要求判定当前是否存在指定的监测行为。

【最重要的判定原则】
你输出的 violations 必须且只能反映【最后一帧】的状态。例如前 10 帧出现过某行为、后 10 帧已恢复正常，则不得报告该行为。
- 历史帧（最后一帧之前的所有帧）仅作为连续运动趋势的辅助参考，用于判断动作是否仍在持续、是否刚刚发生、是否已经结束。
- 如果某个行为在历史帧出现过，但在最后一帧已经结束或恢复常态，则该行为【不得】出现在 violations 中。
- 只有当某个行为在【最后一帧当下仍在发生或仍然保持该姿态】时，才计入 violations。
- 不要累计、沿用、记忆历史帧中已结束的行为；不要做“曾经发生过就报”的统计。

【输出格式】
严格输出如下 JSON，不要添加任何多余文本或解释，不要开启思考模式：
{"has_person": 0 或 1, "violations": ["行为名称", ...]}

- has_person：最后一帧画面中是否有人（1 有 / 0 无 / -1 无法判断）。
- violations：最后一帧当下仍在发生的监测行为的【名称】，必须【逐字】使用用户在任务清单中给出的名称。
- 如果最后一帧没有任何清单中的行为，violations 必须为空数组 []。
"""

# 内置默认任务的判定标准（当清单里出现这些名称且未提供优化描述时附加，提高准确率）。
BUILTIN_RULES: dict[str, str] = {
    "摔倒": "包含真实的失去平衡意外跌倒，以及躯干大面积接触地面的情形。只要【最后一帧】人体出现趴卧在地、双膝跪伏、平躺、侧卧，"
            "甚至主动且有意识直接坐在地板上（无正常座椅支撑），无论人是否朝向摄像头，都要判定为“摔倒”。"
            "若历史帧曾摔倒但最后一帧已重新站起或坐回正常座椅，则【不报】。",
    "挥手": "判定标准为【最后一帧】人物手部有大幅度的挥动动作（手腕超过头顶），都要判定为“挥手”，"
            "注意不要把打电话误识别为挥手。若历史帧曾挥手但最后一帧手已放下，则【不报】。",
    "弯腰": "指【最后一帧】人物上半身出现明显的伏身状态，看起来像是为了减轻腰部压力、缓解疼痛或身体不适而被迫弯曲扭身，"
            "都要判定为“弯腰”；不要把正常的身体晃动识别为弯腰。若历史帧曾弯腰但最后一帧已直起身，则【不报】。",
    "打架": "指【最后一帧】存在明显的肢体冲突、推搡、挥拳、踢打等对抗动作。若历史帧曾冲突但最后一帧已停止、分开或恢复常态，则【不报】。",
    "捂胸口": "指【最后一帧】人物手部按在胸口/胸前区域，可能伴随身体前倾或不适姿态。不要把手持物品、整理衣物误判为捂胸口；"
              "若历史帧曾捂胸口但最后一帧手已放下，则【不报】。",
}

# 任务清单为空时使用的兜底指令。
DEFAULT_USER_TEXT = (
    "请只根据最后一帧（当前时刻）的画面状态判定当前是否仍存在监测行为；"
    "历史帧仅作为动作趋势参考，已结束的行为不要计入当前结果。"
)


def behavior_labels(criteria: list[dict]) -> list[str]:
    """违规任务名（除“是否有人”外的每一项）。"""
    return [
        criterion.get("label") or criterion.get("id")
        for criterion in criteria
        if criterion.get("id") != "person"
    ]


def build_decision_payload(state, question: str, options: list[dict], letters: str) -> str:
    """Jev 式决策的 user 轮 JSON：evidence + criterion + 带字母的 options。"""
    return json.dumps(
        {
            "evidence": state,
            "criterion": question,
            "options": [
                {"letter": letters[index], "description": option["description"]}
                for index, option in enumerate(options)
            ],
        },
        ensure_ascii=False,
    )


def build_task_block(criteria: list[dict]) -> str:
    """把任务清单渲染成注入 user content 尾部的文本块。"""
    tasks = [criterion for criterion in criteria if criterion.get("id") != "person"]
    if not tasks:
        return DEFAULT_USER_TEXT

    lines: list[str] = [
        "【本摄像头的监测任务清单】",
        "请只判定下列行为；violations 中必须逐字使用下列【任务名称】：",
        "",
    ]
    for index, task in enumerate(tasks, 1):
        name = str(task.get("label") or task.get("id") or "").strip()
        if not name:
            continue
        # 判定标准的注入优先级：description_opt > 内置规则 > 原始描述
        desc = str(task.get("description") or task.get("question") or "").strip()
        desc_opt = str(task.get("description_opt") or "").strip()
        exclusions = task.get("exclusions_opt") or []
        if not isinstance(exclusions, list):
            exclusions = []
        exclusions = [str(item).strip() for item in exclusions if str(item).strip()]
        period = str(task.get("carePeriod") or "").strip()
        level = str(task.get("alertLevel") or "").strip()
        meta = " / ".join(part for part in (period, level) if part)

        lines.append(f"{index}. 任务名称：{name}" + (f"（{meta}）" if meta else ""))
        if desc_opt:
            lines.append(f"   判定标准：{desc_opt}")
        elif name in BUILTIN_RULES:
            lines.append(f"   判定标准：{BUILTIN_RULES[name]}")
        elif desc:
            lines.append(f"   判定标准：{desc}")
        if exclusions:
            lines.append("   排除项（以下情形不应计入本任务）：" + "；".join(exclusions))

    lines += [
        "",
        "【判定要求】",
        "- 只依据最后一帧的当下状态判定，历史帧仅作趋势参考。",
        "- violations 中的名称必须与上面【任务名称】完全一致，不要改写、不要翻译、不要添加清单外的项目。",
        "  例如任务清单中写的是“捂胸口”，就必须输出“捂胸口”，不要输出“捂住胸口”“胸口不适”等变体。",
        "- 没有命中的任务不要出现在 violations 中；全部未命中时 violations 为 []。",
    ]
    return "\n".join(lines)


def build_generation_text(state, criteria: list[dict]) -> str:
    """自回归基线的 user 轮文本：图像在前，任务清单在尾部。

    返回值直接作为 user content 的文本部分（图像作为 image 部分放在它前面），
    因此不同任务清单共享的 system + 图像前缀可以命中 APC。
    """
    parts: list[str] = []
    if isinstance(state, str) and state.strip():
        parts.append("画面说明：" + state.strip())
    parts.append(build_task_block(criteria))
    return "\n".join(parts)