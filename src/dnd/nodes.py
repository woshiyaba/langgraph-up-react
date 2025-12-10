from typing import Literal, cast  # noqa: D100
import json
import re

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel

from src.common import Context
from src.common.utils import load_chat_model
from src.dnd import prompt
from src.dnd.dnd_state import GameState


class IntentRouteResult(BaseModel):
    """结构化的意图路由结果，只允许一个 action 字段。"""

    action: Literal[
        "explore",
        "talk",
        "skill_check",
        "attack",
        "cast_spell",
        "start_combat",
        "store",
    ]


_VALID_ACTIONS = {
    "explore",
    "talk",
    "skill_check",
    "attack",
    "cast_spell",
    "start_combat",
    "store",
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
        "store",
    ]:
        if act in lowered:
            return act

    # 3. 实在解析不到就当作闲聊
    return "store"


async def intent_route_node(state: GameState, runtime: Runtime[Context]):
    """意图路由节点：只输出 action，不生成故事。"""
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


async def store_engine_node(state: GameState, runtime: Runtime[Context]):
    llm = load_chat_model(runtime.context.model)
    response = cast(
        AIMessage,
        await llm.ainvoke(
            [{"role": "system", "content": prompt.store_engine}, *state.messages]
        ),
    )

    print("store_engine_node", response.content)
    return {"messages": [response]}
