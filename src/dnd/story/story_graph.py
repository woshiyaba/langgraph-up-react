from typing import List, Dict, cast, Literal

from langchain_core.messages import ToolMessage, AIMessage
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime

from src.common import Context
from src.dnd.dnd_state import GameState
from src.dnd.story.story_node import store_engine_node
from src.dnd.story.tools import get_story_tools

workflow = StateGraph(GameState, context_schema=Context)


async def dynamic_tools_node(
        state: GameState, runtime: Runtime[Context]
) -> Dict[str, List[ToolMessage]]:
    """Execute tools dynamically based on configuration.

    This function gets the available tools based on the current configuration
    and executes the requested tool calls from the last message.
    """
    # Get available tools based on configuration
    available_tools = await get_story_tools()

    # Create a ToolNode with the available tools
    tool_node = ToolNode(available_tools)

    # Execute the tool node
    result = await tool_node.ainvoke(state)

    return cast(Dict[str, List[ToolMessage]], result)


workflow.add_node("store_engine_node", store_engine_node)
workflow.add_node("tools", dynamic_tools_node)
workflow.set_entry_point("store_engine_node")


def route_model_output(state: GameState) -> Literal["__end__", "tools"]:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"

workflow.add_conditional_edges("store_engine_node",
                               route_model_output)
workflow.add_edge("tools", "store_engine_node")

story_graph = workflow.compile()
