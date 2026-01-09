from langgraph.runtime import Runtime

from src.common import Context, load_chat_model
from src.dnd.dnd_state import GameState


async def dm_assistant(state: GameState, runtime: Runtime[Context]):
    """DM助手节点."""
    llm = load_chat_model(runtime.context.model)
    response = await llm.ainvoke(
        [{"role": "system", "content": prompt.dm_assistant}, *state.messages]
    )
    return {"messages": [response]}