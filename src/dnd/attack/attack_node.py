"""战斗系统节点实现."""
from typing import Any, Dict, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from src.common import Context, load_chat_model
from src.dnd import prompt
from src.dnd.attack.attack_tools import (
    ExtractedCharacters,
    attack_roll,
    create_combatant_from_extracted,
    damage_roll,
    get_attack_tools,
    sort_combatants_by_initiative,
)
from src.dnd.dnd_state import Combatant, Faction, GameState


# ============================================================
# 提取角色的 Prompt
# ============================================================
EXTRACT_CHARACTERS_PROMPT = """
你是一个DnD战斗角色提取器。请从最近的对话中提取出所有参与战斗的角色。

请识别：
1. 队友 (ally): 玩家角色、友方NPC
2. 敌人 (enemy): 怪物、敌对NPC

对于每个角色，请估算其属性：
- name: 角色名称
- faction: "ally" 或 "enemy"
- hp/max_hp: 根据角色类型估算生命值 (普通人类20, 战士30-50, 怪物根据描述)
- ac: 护甲等级 (无甲10-12, 轻甲13-15, 重甲16-18)
- dex: 敏捷值 (普通10, 敏捷类角色14-18, 笨重类6-8)
- damage_dice: 伤害骰 (拳头1d4, 匕首1d4, 剑1d8, 大剑2d6)
- description: 简短描述

请仔细阅读对话，找出所有明确或暗示参与战斗的角色。
"""


async def init_combat_node(state: GameState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """初始化战斗节点：检查列表，提取角色，按敏捷排序."""
    # 如果战斗列表不为空，直接跳过初始化
    if state.combat_order and len(state.combat_order) > 0:
        return {}
    
    # 获取最近5条消息
    recent_messages = list(state.messages[-5:]) if len(state.messages) >= 5 else list(state.messages)
    
    if not recent_messages:
        return {
            "combat_order": [],
            "is_combat_active": False,
            "combat_log": ["[系统] 无法从对话中识别战斗参与者"]
        }
    
    # 使用 LLM 提取角色
    llm = load_chat_model(runtime.context.model)
    
    try:
        structured_llm = llm.with_structured_output(ExtractedCharacters)
        result = await structured_llm.ainvoke([
            {"role": "system", "content": EXTRACT_CHARACTERS_PROMPT},
            *[{"role": "user" if isinstance(m, HumanMessage) else "assistant", 
               "content": m.content if hasattr(m, 'content') else str(m)} 
              for m in recent_messages]
        ])
        
        if not result or not result.characters:
            return {
                "combat_log": ["[系统] 未能从对话中识别出战斗参与者"]
            }
        
        # 转换为 Combatant 对象
        combatants = [
            create_combatant_from_extracted(char, i) 
            for i, char in enumerate(result.characters)
        ]
        
        # 按先攻排序
        sorted_combatants = sort_combatants_by_initiative(combatants)
        
        # 生成战斗日志
        combat_log = ["[系统] ===== 战斗开始 ====="]
        combat_log.append("[系统] 先攻顺序:")
        for i, c in enumerate(sorted_combatants):
            faction_str = "【队友】" if c.faction == Faction.ALLY else "【敌人】"
            combat_log.append(f"  {i+1}. {faction_str} {c.name} (DEX: {c.dexterity}, HP: {c.hp}/{c.max_hp})")
        
        return {
            "combat_order": sorted_combatants,
            "is_combat_active": True,
            "current_round": 1,
            "combat_log": combat_log
        }
        
    except Exception as e:
        return {
            "combat_log": [f"[系统] 初始化战斗失败: {str(e)}"]
        }


async def process_turn_node(state: GameState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """处理当前角色的战斗回合：取第一个角色执行攻击判定."""
    if not state.combat_order:
        return {
            "combat_log": ["[系统] 战斗列表为空，无法处理回合"],
            "is_combat_active": False
        }
    
    # 获取当前行动者
    current_actor = state.combat_order[0]
    combat_log = list(state.combat_log) if state.combat_log else []
    combat_log.append(f"\n[回合 {state.current_round}] {current_actor.name} 的回合")
    
    # 获取可攻击的目标（敌对阵营的存活角色）
    target_faction = Faction.ALLY if current_actor.faction == Faction.ENEMY else Faction.ENEMY
    available_targets = [c for c in state.combat_order if c.faction == target_faction and c.is_alive]
    
    if not available_targets:
        combat_log.append(f"  {current_actor.name} 没有可攻击的目标")
        return {"combat_log": combat_log}
    
    # 选择目标（简单策略：攻击第一个可用目标）
    target = available_targets[0]
    
    # 计算攻击加值（简化：使用力量调整值）
    str_mod = (current_actor.stats.get("STR", 10) - 10) // 2
    
    # 执行攻击
    attack_result = attack_roll.invoke({
        "attacker_name": current_actor.name,
        "target_name": target.name,
        "attack_bonus": str_mod,
        "target_ac": target.ac
    })
    
    combat_log.append(f"  {attack_result['details']}")
    
    # 如果命中，计算伤害
    updated_combatants = list(state.combat_order)
    if attack_result["hit"]:
        damage_result = damage_roll.invoke({
            "damage_dice": current_actor.damage_dice,
            "is_critical": attack_result["is_critical"]
        })
        
        combat_log.append(f"  {damage_result['details']}")
        
        # 更新目标生命值
        target_index = next(i for i, c in enumerate(updated_combatants) if c.id == target.id)
        updated_target = updated_combatants[target_index]
        new_hp = max(0, updated_target.hp - damage_result["damage"])
        
        # 创建更新后的 Combatant
        updated_combatants[target_index] = Combatant(
            id=updated_target.id,
            name=updated_target.name,
            faction=updated_target.faction,
            hp=new_hp,
            max_hp=updated_target.max_hp,
            ac=updated_target.ac,
            stats=updated_target.stats,
            damage_dice=updated_target.damage_dice,
            description=updated_target.description
        )
        
        combat_log.append(f"  {target.name} 受到 {damage_result['damage']} 点伤害! (HP: {updated_target.hp} -> {new_hp})")
        
        if new_hp <= 0:
            combat_log.append(f"  💀 {target.name} 被击败了!")
    
    return {
        "combat_order": updated_combatants,
        "combat_log": combat_log
    }


async def check_death_node(state: GameState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """检查死亡并移除：HP<=0的角色从列表移除，判断战斗是否结束."""
    if not state.combat_order:
        return {"is_combat_active": False}
    
    combat_log = list(state.combat_log) if state.combat_log else []
    
    # 过滤存活的角色
    alive_combatants = [c for c in state.combat_order if c.is_alive]
    
    # 检查战斗是否结束
    allies_alive = [c for c in alive_combatants if c.faction == Faction.ALLY]
    enemies_alive = [c for c in alive_combatants if c.faction == Faction.ENEMY]
    
    combat_ended = False
    if not enemies_alive:
        combat_log.append("\n[系统] ===== 战斗胜利！所有敌人被击败 =====")
        combat_ended = True
    elif not allies_alive:
        combat_log.append("\n[系统] ===== 战斗失败...所有队友倒下 =====")
        combat_ended = True
    
    return {
        "combat_order": alive_combatants,
        "is_combat_active": not combat_ended,
        "combat_log": combat_log
    }


async def rotate_turn_node(state: GameState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """轮转回合：将当前行动者移到队列尾部."""
    if not state.combat_order or len(state.combat_order) < 2:
        return {}
    
    # 将第一个移到最后
    rotated_order = state.combat_order[1:] + [state.combat_order[0]]
    
    # 简单的回合计数（这里可以根据需要优化）
    new_round = state.current_round
    # 假设当原第一人回到第一位时算一轮结束（这里简化处理）
    
    combat_log = list(state.combat_log) if state.combat_log else []
    combat_log.append(f"  -> 下一位: {rotated_order[0].name}")
    
    return {
        "combat_order": rotated_order,
        "current_round": new_round,
        "combat_log": combat_log
    }


async def combat_engine_node(state: GameState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """战斗引擎节点：使用LLM生成战斗叙述."""
    llm = load_chat_model(runtime.context.model).bind_tools(get_attack_tools())
    
    # 构建战斗状态摘要
    combat_summary = _build_combat_summary(state)
    
    response = cast(
        AIMessage,
        await llm.ainvoke([
            {"role": "system", "content": prompt.combat_engine},
            {"role": "user", "content": combat_summary},
            *state.messages[-3:]  # 最近的对话上下文
        ])
    )
    
    return {"messages": [response]}


def _build_combat_summary(state: GameState) -> str:
    """构建战斗状态摘要供LLM使用."""
    lines = ["当前战斗状态:"]
    lines.append(f"回合: {state.current_round}")
    lines.append("\n战斗顺序:")
    
    for i, c in enumerate(state.combat_order):
        marker = ">>>" if i == 0 else "   "
        faction = "队友" if c.faction == Faction.ALLY else "敌人"
        lines.append(f"{marker} {i+1}. [{faction}] {c.name} - HP: {c.hp}/{c.max_hp}, AC: {c.ac}")
    
    if state.combat_log:
        lines.append("\n最近战斗日志:")
        for log in state.combat_log[-5:]:
            lines.append(log)
    
    return "\n".join(lines)


def should_continue_combat(state: GameState) -> Literal["continue", "end"]:
    """判断战斗是否应该继续的路由函数."""
    if not state.is_combat_active:
        return "end"
    if not state.combat_order:
        return "end"
    
    # 检查是否还有两个阵营的角色存活
    allies = [c for c in state.combat_order if c.faction == Faction.ALLY and c.is_alive]
    enemies = [c for c in state.combat_order if c.faction == Faction.ENEMY and c.is_alive]
    
    if not allies or not enemies:
        return "end"
    
    return "continue"

