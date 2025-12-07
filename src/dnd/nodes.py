from langchain.chains.question_answering.map_reduce_prompt import messages
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from typing import cast
from src.common import Context
from src.dnd.dnd_state import GameState
from src.dnd import prompt
from src.common.utils import load_chat_model


async def intent_route_node(state: GameState, runtime: Runtime[Context]):
    llm = load_chat_model(runtime.context.model)
    response = cast(
        AIMessage,
        await llm.ainvoke(
            [{"role": "system", "content": prompt.intent_route}, *state.messages]
        ),
    )
    return {"messages": [response]}

async def store_engine_node(state: GameState, runtime: Runtime[Context]):
    llm = load_chat_model(runtime.context.model)
    response = cast(
        AIMessage,
        await llm.ainvoke(
            [{"role": "system", "content": prompt.store_engine}, *state.messages]
        ),
    )
    return {"messages": [response]}
