from langgraph.graph import StateGraph, END
from src.common import Context
from src.dnd.dnd_state import GameState
from src.dnd.nodes import intent_route_node,store_engine_node

workflow = StateGraph(GameState, context_schema=Context)

# 添加节点
workflow.add_node("intent_route_node", intent_route_node)
workflow.add_node("store_engine_node", store_engine_node)

workflow.add_edge("__start__", "intent_route_node")
workflow.add_edge("intent_route_node", "store_engine_node")

app = workflow.compile()