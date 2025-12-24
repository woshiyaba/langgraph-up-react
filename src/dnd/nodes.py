from typing import Any, Dict, Literal, cast  # noqa: D100
import json
import re

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel

from src.common import Context
from src.common.utils import load_chat_model
from src.dnd import prompt
from src.dnd.dnd_state import GameState, Player


class IntentRouteResult(BaseModel):
    """结构化的意图路由结果，只允许一个 action 字段。"""

    action: Literal[
        "explore",
        "talk",
        "skill_check",
        "attack",
        "cast_spell",
        "start_combat",
        "story",
    ]


_VALID_ACTIONS = {
    "explore",
    "talk",
    "skill_check",
    "attack",
    "cast_spell",
    "start_combat",
    "story",
}


def _extract_action_from_text(text: str) -> str:
    """从模型的原始文本中尽力解析出合法的 action。"""
    # 1. 先尝试从文本中的 JSON 片段里解析
    for match in re.findall(r"\{.*?\}", text, flags=re.S):
        try:
            obj = json.loads(match)
        except Exception:
            continue
        if isinstance(obj, dict):
            action = obj.get("action")
            if isinstance(action, str) and action in _VALID_ACTIONS:
                return action

    # 2. 关键字兜底：按优先级从文本中猜测
    lowered = text.lower()
    for act in [
        "start_combat",
        "attack",
        "cast_spell",
        "skill_check",
        "talk",
        "explore",
        "story",
    ]:
        if act in lowered:
            return act

    # 3. 实在解析不到就当作闲聊
    return "story"


async def intent_route_node(state: GameState, runtime: Runtime[Context]):
    """意图路由节点：只输出 action，不生成故事."""
    llm = load_chat_model(runtime.context.model)
    print("intent_route_node param", *state.messages)

    action: str

    # 优先尝试结构化输出，让 LLM 只填充 IntentRouteResult
    try:
        structured_llm = llm.with_structured_output(IntentRouteResult)
        result = await structured_llm.ainvoke(
            [{"role": "system", "content": prompt.intent_route}, *state.messages]
        )
        action = result.action
    except Exception:
        # 兜底：退回普通文本调用，再从文本中解析 action
        response = cast(
            AIMessage,
            await llm.ainvoke(
                [{"role": "system", "content": prompt.intent_route}, *state.messages]
            ),
        )
        print("intent_route_node raw", response.content)
        action = _extract_action_from_text(response.content)

    clean_content = json.dumps({"action": action}, ensure_ascii=False)
    clean_message = AIMessage(content=clean_content)
    print("intent_route_node action", clean_content)
    return {"messages": [clean_message]}


# === 默认玩家属性配置 ===
DEFAULT_PLAYER_STATS = {
    "Warrior": {
        "hp": 25, "ac": 14, "damage_dice": "1d10+2",
        "stats": {"STR": 16, "DEX": 12, "CON": 14, "INT": 8, "WIS": 10, "CHA": 10},
    },
    "Mage": {
        "hp": 15, "ac": 10, "damage_dice": "1d6",
        "stats": {"STR": 8, "DEX": 12, "CON": 10, "INT": 16, "WIS": 14, "CHA": 12},
    },
    "Rogue": {
        "hp": 18, "ac": 12, "damage_dice": "1d8+3",
        "stats": {"STR": 10, "DEX": 16, "CON": 12, "INT": 12, "WIS": 10, "CHA": 14},
    },
}


async def init_player_node(
    state: GameState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """初始化玩家节点：通过线程ID初始化玩家信息.

    如果玩家已存在，则跳过初始化。
    """
    # 获取线程ID作为玩家唯一标识
    thread_id = getattr(runtime, "thread_id", None) or "default_player"

    # 检查是否已经初始化过
    if thread_id in state.players:
        print(f"[init_player_node] 玩家 {thread_id} 已存在，跳过初始化")
        return {"current_user_id": thread_id}

    # 使用默认职业 Warrior 的属性
    player_class = "Warrior"
    defaults = DEFAULT_PLAYER_STATS[player_class]

    # 创建新玩家
    player = Player(
        id=thread_id,
        name=f"冒险者_{thread_id[:8]}",
        hp=defaults["hp"],
        max_hp=defaults["hp"],
        ac=defaults["ac"],
        stats=defaults["stats"].copy(),
        damage_dice=defaults["damage_dice"],
        description="一位勇敢的冒险者，踏上了未知的旅程。",
        player_class=player_class,
        level=1,
    )

    print(f"[init_player_node] 初始化玩家: {player.name} (id={thread_id})")

    # 返回更新后的状态
    return {
        "players": {thread_id: player},
        "current_user_id": thread_id,
    }

