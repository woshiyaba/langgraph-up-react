"""
战斗系统 LangGraph 图定义

流程图:
    ┌─────────────────────────────────────────────────────────────┐
    │                      START                                  │
    │                        │                                    │
    │                        ▼                                    │
    │              ┌─────────────────┐                            │
    │              │ init_combat_node │  检查并初始化战斗列表     │
    │              └────────┬────────┘                            │
    │                       │                                     │
    │                       ▼                                     │
    │              ┌─────────────────┐                            │
    │         ┌───►│ process_turn    │  处理当前角色回合          │
    │         │    └────────┬────────┘                            │
    │         │             │                                     │
    │         │             ▼                                     │
    │         │    ┌─────────────────┐                            │
    │         │    │ check_death     │  检查死亡，移除HP<=0的     │
    │         │    └────────┬────────┘                            │
    │         │             │                                     │
    │         │             ▼                                     │
    │         │    ┌─────────────────┐                            │
    │         │    │ should_continue │  判断战斗是否继续          │
    │         │    └────────┬────────┘                            │
    │         │             │                                     │
    │         │     continue│      end                            │
    │         │             ▼         ▼                           │
    │         │    ┌─────────────────┐    ┌──────┐                │
    │         └────┤ rotate_turn     │    │ END  │                │
    │              └─────────────────┘    └──────┘                │
    │              将当前角色移到队尾                              │
    └─────────────────────────────────────────────────────────────┘
"""  # noqa: D212, D415
from typing import Literal

from langgraph.graph import END, StateGraph

from src.common import Context
from src.dnd.attack.attack_node import (
    check_death_node,
    init_combat_node,
    process_turn_node,
    rotate_turn_node,
    should_continue_combat,
)
from src.dnd.dnd_state import Faction, GameState


def build_attack_graph() -> StateGraph:
    """构建战斗系统子图"""
    
    workflow = StateGraph(GameState, context_schema=Context)
    
    # ============================================================
    # 添加节点
    # ============================================================
    workflow.add_node("init_combat", init_combat_node)
    workflow.add_node("process_turn", process_turn_node)
    workflow.add_node("check_death", check_death_node)
    workflow.add_node("rotate_turn", rotate_turn_node)
    
    # ============================================================
    # 设置入口
    # ============================================================
    workflow.set_entry_point("init_combat")
    
    # ============================================================
    # 添加边
    # ============================================================
    
    # init_combat -> process_turn
    workflow.add_edge("init_combat", "process_turn")
    
    # process_turn -> check_death
    workflow.add_edge("process_turn", "check_death")
    
    # check_death -> 条件路由 (continue/end)
    workflow.add_conditional_edges(
        "check_death",
        should_continue_combat,
        {
            "continue": "rotate_turn",
            "end": END
        }
    )
    
    # rotate_turn -> process_turn (循环)
    workflow.add_edge("rotate_turn", "process_turn")
    
    return workflow


# 编译后的战斗图（可直接作为子图使用）
attack_graph = build_attack_graph().compile()


# ============================================================
# 提供一个入口函数，方便主图调用
# ============================================================
async def run_attack_graph(state: GameState, runtime) -> dict:
    """运行战斗图并返回结果"""
    result = await attack_graph.ainvoke(state, {"configurable": {"context": runtime.context}})
    return result

