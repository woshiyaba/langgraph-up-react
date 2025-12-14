from typing import cast

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.common import Context, load_chat_model
from src.dnd import prompt
from src.dnd.dnd_state import GameState
from src.dnd.story.tools import story_create, get_story_tools


async def store_engine_node(state: GameState, runtime: Runtime[Context]):
    llm = load_chat_model(runtime.context.model).bind_tools(await get_story_tools())

    response = cast(
        AIMessage,
        await llm.ainvoke(
            [{"role": "system", "content": prompt.store_engine}, *state.messages]
        ),
    )

    print("store_engine_node", response.content)
    return {"messages": [response]}