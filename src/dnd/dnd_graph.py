import json
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime

from src.common import Context
from src.dnd.attack import attack_graph
from src.dnd.dnd_state import GameState
from src.dnd.nodes import init_player_node, intent_route_node
from src.dnd.story.story_graph import story_graph

workflow = StateGraph(GameState, context_schema=Context)

# 添加节点
workflow.add_node("init_player_node", init_player_node)
workflow.add_node("intent_route_node", intent_route_node)
workflow.add_node("store_engine_node", story_graph)
workflow.add_node("attack_graph", attack_graph)


def start_route_fun(state: GameState, runtime: Runtime[Context]):
    """起始路由：判断玩家是否已初始化，决定是否跳过初始化节点."""
    thread_id = getattr(runtime, "thread_id", None) or "default_player"
    
    if thread_id in state.players:
        print(f"[start_route] 玩家 {thread_id} 已初始化，跳过 init_player_node")
        return "intent_route_node"
    
    print(f"[start_route] 玩家 {thread_id} 未初始化，进入 init_player_node")
    return "init_player_node"


def intent_route_fun(state: GameState, runtime: Runtime[Context]):
    """路由节点，通过意图识别路由到不同的子图."""
    # intent_route_node 会把 action 输出为 JSON 字符串：
    #   {"action": "..."}
    # 这里需要解析（json.loads），而不是 dump（写文件）。
    content = state.messages[-1].content
    try:
        intent = json.loads(content) if isinstance(content, str) else {}
    except Exception:
        intent = {}

    action = intent.get("action") if isinstance(intent, dict) else None
    if action in {"attack", "start_combat"}:
        return "attack_graph"

    # 其它动作（含 explore/talk/skill_check/cast_spell/story/store/未知/解析失败）
    # 都交给故事引擎，保证 conditional_edges 的返回值合法。
    return "store_engine_node"


# 起始条件边：已初始化则跳过 init_player_node
workflow.add_conditional_edges(
    "__start__", start_route_fun, ["init_player_node", "intent_route_node"]
)
workflow.add_edge("init_player_node", "intent_route_node")
workflow.add_conditional_edges(
    "intent_route_node", intent_route_fun, ["store_engine_node", "attack_graph"]
)

app = workflow.compile()
