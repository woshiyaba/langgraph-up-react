"""
战斗系统 LangGraph 图定义

流程图:
    ┌────────────────────────────────────────────────────────────────────────┐
    │                           START                                        │
    │                             │                                          │
    │                             ▼                                          │
    │                   ┌───────────────────┐                                │
    │                   │ is_combat_active? │                                │
    │                   └─────────┬─────────┘                                │
    │                    false    │    true                                  │
    │                      ▼      │      ▼                                   │
    │           ┌─────────────────┐   ┌───────────────────┐                  │
    │           │ init_combat     │   │ check_turn_type   │◄───────┐         │
    │           └────────┬────────┘   └─────────┬─────────┘        │         │
    │                    │         player_turn  │ player_action    │         │
    │                    │              │       │ _ready   npc     │         │
    │                    │              ▼       │    │      │      │         │
    │                    │    ┌───────────────┐ │    │      ▼      │         │
    │                    │    │ await_player  │ │    │ ┌─────────┐ │         │
    │                    │    │ (INTERRUPT)   │ │    │ │npc_skill│ │         │
    │                    │    └───────┬───────┘ │    │ └────┬────┘ │         │
    │                    │            │         │    │      │      │         │
    │                    │            ▼         │    ▼      ▼      │         │
    │                    │          [END]       │ ┌─────────────┐  │         │
    │                    │      (用户再次输入)   │ │combat_intent│◄─┘         │
    │                    │            │         │ └──────┬──────┘            │
    │                    │            │         │        │                   │
    │                    │            └─────────┼────────┤                   │
    │                    │                      │        ▼                   │
    │                    │                      │ ┌─────────────┐            │
    │                    └──────────────────────┼►│process_turn │            │
    │                                           │ └──────┬──────┘            │
    │                                           │        │                   │
    │                                           │        ▼                   │
    │                                           │ ┌─────────────┐            │
    │                                           │ │ check_death │            │
    │                                           │ └──────┬──────┘            │
    │                                           │        │                   │
    │                                           │        ▼                   │
    │                                           │ ┌─────────────┐            │
    │                                           │ │should_cont? │            │
    │                                           │ └──────┬──────┘            │
    │                                           │   cont │   end             │
    │                                           │        ▼    ▼              │
    │                                           │ ┌─────────────┐            │
    │                                           │ │ rotate_turn │────────────┘
    │                                           │ └─────────────┘
    │                                           │        │
    │                                           │        ▼
    │                                           │     ┌─────┐
    │                                           └────►│ END │
    │                                                 └─────┘
    └────────────────────────────────────────────────────────────────────────┘

玩家回合流程:
  1. check_turn_type 检测到玩家角色 + awaiting_player_input=False -> player_turn
  2. await_player 设置 awaiting_player_input=True -> END (中断等待输入)
  3. 用户输入后 graph 恢复，再次进入 check_turn
  4. check_turn_type 检测到玩家角色 + awaiting_player_input=True -> player_action_ready
  5. combat_intent 解析用户输入并清除 awaiting_player_input -> process_turn
"""  # noqa: D212, D415

from typing import Literal

from langgraph.graph import END, StateGraph

from src.common import Context
from src.dnd.attack.attack_node import (
    await_player_input_node,
    check_death_node,
    check_turn_type,
    combat_intent,
    init_combat_node,
    npc_skill_node,
    process_turn_node,
    rotate_turn_node,
    should_continue_combat,
)
from src.dnd.dnd_state import ControllerType, Faction, GameState


def build_attack_graph() -> StateGraph:
    """构建战斗系统子图."""
    workflow = StateGraph(GameState, context_schema=Context)

    # ============================================================
    # 添加节点
    # ============================================================
    workflow.add_node("init_combat", init_combat_node)
    workflow.add_node("check_turn", _check_turn_node)  # 路由判断节点
    workflow.add_node("await_player", await_player_input_node)
    workflow.add_node("npc_skill", npc_skill_node)
    workflow.add_node("combat_intent", combat_intent)
    workflow.add_node("process_turn", process_turn_node)
    workflow.add_node("check_death", check_death_node)
    workflow.add_node("rotate_turn", rotate_turn_node)

    # ============================================================
    # 设置入口：根据 is_combat_active 路由
    # ============================================================
    def conditional_entry_point(
        state: GameState,
    ) -> Literal["init_combat", "check_turn"]:
        if not state.is_combat_active:
            return "init_combat"
        return "check_turn"

    workflow.set_conditional_entry_point(
        conditional_entry_point, ["init_combat", "check_turn"]
    )

    # ============================================================
    # 添加边
    # ============================================================

    # init_combat -> check_turn
    workflow.add_edge("init_combat", "check_turn")

    # check_turn -> 条件路由 (player_turn / npc_turn / player_action_ready)
    workflow.add_conditional_edges(
        "check_turn",
        check_turn_type,
        {
            "player_turn": "await_player",
            "npc_batch": "npc_skill",
            "player_action_ready": "combat_intent"  # 玩家已输入，直接处理战斗
        }
    )

    # await_player -> END (中断等待玩家输入，下次从 check_turn 继续)
    workflow.add_edge("await_player", END)

    # npc_skill -> combat_intent
    workflow.add_edge("npc_skill", "combat_intent")

    # combat_intent -> process_turn
    workflow.add_edge("combat_intent", "process_turn")

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

    # rotate_turn -> check_turn (循环回去判断下一个行动者)
    workflow.add_edge("rotate_turn", "check_turn")

    return workflow


async def _check_turn_node(state: GameState, runtime) -> dict:
    """检查当前回合类型的空节点，仅用于路由判断."""
    return {}


# 编译后的战斗图（可直接作为子图使用）
attack_graph = build_attack_graph().compile()


# ============================================================
# 提供一个入口函数，方便主图调用
# ============================================================
async def run_attack_graph(state: GameState, runtime) -> dict:
    """运行战斗图并返回结果."""
    result = await attack_graph.ainvoke(
        state, {"configurable": {"context": runtime.context}}
    )
    return result
