from typing import cast  # noqa: D100

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.common import Context
from src.common.utils import load_chat_model
from src.dnd import prompt
from src.dnd.dnd_state import GameState


async def intent_route_node(state: GameState, runtime: Runtime[Context]):
    llm = load_chat_model(runtime.context.model)
    print("intent_route_node param", *state.messages)
    response = cast(
        AIMessage,
        await llm.ainvoke(
            [{"role": "system", "content": prompt.intent_route}, *state.messages]
        ),
    )
    print("intent_route_node", response.content)
    return {"messages": [response]}


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
