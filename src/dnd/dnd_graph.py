from langgraph.graph import END, StateGraph

from src.common import Context
from src.dnd.attack import attack_graph
from src.dnd.dnd_state import GameState
from src.dnd.nodes import intent_route_node
from src.dnd.story.story_graph import story_graph

workflow = StateGraph(GameState, context_schema=Context)

# 添加节点
workflow.add_node("intent_route_node", intent_route_node)
workflow.add_node("store_engine_node", story_graph)
workflow.add_node("attack_graph", attack_graph)

workflow.add_edge("__start__", "intent_route_node")
workflow.add_edge("intent_route_node", "store_engine_node")
workflow.add_edge("store_engine_node", "attack_graph")
workflow.add_edge("attack_graph", END)

app = workflow.compile()