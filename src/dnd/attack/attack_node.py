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
from src.dnd.attack.prompt import COMBAT_INTENT, NPC_SKILL_PROMPT
from src.dnd.dnd_state import (
    Combatant,
    CombatCommand,
    ControllerType,
    Faction,
    GameState,
)

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
- is_player: 是否为玩家控制的角色（true/false）
  * 玩家角色：对话中的"我"、用户扮演的角色、明确说是玩家的角色
  * NPC：怪物、敌人、友方NPC、队友NPC等
- hp/max_hp: 根据角色类型估算生命值 (普通人类20, 战士30-50, 怪物根据描述)
- ac: 护甲等级 (无甲10-12, 轻甲13-15, 重甲16-18)
- dex: 敏捷值 (普通10, 敏捷类角色14-18, 笨重类6-8)
- damage_dice: 伤害骰 (拳头1d4, 匕首1d4, 剑1d8, 大剑2d6)
- description: 简短描述

请仔细阅读对话，找出所有明确或暗示参与战斗的角色。
注意区分玩家控制的角色和NPC，这很重要！

如果没有npc则模拟创建一个怪物进行战斗，方便测试
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
        
        # 添加当前玩家到战斗列表
        current_player = state.players.get(state.current_user_id)
        if current_player:
            # 检查是否已存在同名或同ID的角色，避免重复
            existing_ids = {c.id for c in combatants}
            existing_names = {c.name.lower() for c in combatants}
            
            if current_player.id not in existing_ids and current_player.name.lower() not in existing_names:
                player_combatant = current_player.to_combatant()
                combatants.append(player_combatant)
                print(f"[init_combat] 添加玩家 {current_player.name} 到战斗列表")
            else:
                print(f"[init_combat] 玩家 {current_player.name} 已在战斗列表中，跳过添加")
        
        # 按先攻排序
        sorted_combatants = sort_combatants_by_initiative(combatants)
        
        # 生成战斗日志
        combat_log = ["[系统] ===== 战斗开始 ====="]
        combat_log.append("[系统] 先攻顺序:")
        for i, c in enumerate(sorted_combatants):
            faction_str = "【队友】" if c.faction == Faction.ALLY else "【敌人】"
            ctrl_str = "🎮" if c.controller == ControllerType.PLAYER else "🤖"
            combat_log.append(f"  {i+1}. {ctrl_str}{faction_str} {c.name} (DEX: {c.dexterity}, HP: {c.hp}/{c.max_hp})")
        
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

async def combat_intent(state: GameState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """理解战斗意图，解析玩家输入或NPC生成的行动指令."""
    llm = load_chat_model(runtime.context.model)
    structured_llm = llm.with_structured_output(CombatCommand)
    
    # 优先使用 NPC 生成的行动指令，否则使用玩家输入
    if state.npc_action_text:
        action_text = state.npc_action_text
    else:
        action_text = state.messages[-1].content if state.messages else ""
    
    res = await structured_llm.ainvoke([
            {"role": "system", "content": COMBAT_INTENT},
            {"role": "user", "content": action_text}
        ]
    )
    return {
        "combat_command": res,
        "npc_action_text": None,  # 清空 NPC 行动指令
        "awaiting_player_input": False  # 清除玩家输入等待标志
    }


async def npc_skill_node(state: GameState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """NPC技能选择节点：使用LLM为当前NPC选择最优行动."""
    if not state.combat_order:
        return {"npc_action_text": None}
    
    current_actor = state.combat_order[0]
    combat_log = list(state.combat_log) if state.combat_log else []
    
    # 获取可攻击的目标（敌对阵营的存活角色）
    target_faction = Faction.ALLY if current_actor.faction == Faction.ENEMY else Faction.ENEMY
    available_targets = [c for c in state.combat_order if c.faction == target_faction and c.is_alive]
    
    if not available_targets:
        combat_log.append(f"  {current_actor.name} 没有可攻击的目标")
        return {
            "npc_action_text": None,
            "combat_log": combat_log
        }
    
    # 构建战斗上下文
    combat_context = _build_combat_summary(state)
    
    # 构建目标信息
    targets_info = "\n".join([
        f"- {t.name} (HP: {t.hp}/{t.max_hp}, AC: {t.ac})"
        for t in available_targets
    ])
    
    # NPC可用技能（简化：基于角色类型）
    available_skills = _get_npc_skills(current_actor)
    
    # 构建提示词
    prompt_text = NPC_SKILL_PROMPT.format(
        combat_context=combat_context,
        actor_name=current_actor.name,
        actor_faction="敌人" if current_actor.faction == Faction.ENEMY else "队友",
        actor_hp=current_actor.hp,
        actor_max_hp=current_actor.max_hp,
        available_skills=", ".join(available_skills),
        targets_info=targets_info
    )
    
    # 调用 LLM 生成行动指令
    llm = load_chat_model(runtime.context.model)
    response = await llm.ainvoke([
        {"role": "system", "content": prompt_text}
    ])
    
    npc_action = response.content.strip()
    combat_log.append(f"\n[回合 {state.current_round}] {current_actor.name} (NPC) 的回合")
    combat_log.append(f"  [AI决策] {npc_action}")
    
    return {
        "npc_action_text": npc_action,
        "combat_log": combat_log
    }


def _get_npc_skills(combatant: Combatant) -> list[str]:
    """根据NPC类型返回可用技能列表."""
    # 简化实现：根据名称关键词匹配技能
    name_lower = combatant.name.lower()
    
    # 基础技能
    skills = ["普通攻击"]
    
    # 根据角色类型添加特殊技能
    if "哥布林" in name_lower or "goblin" in name_lower:
        skills.extend(["利爪", "偷袭"])
    elif "骷髅" in name_lower or "skeleton" in name_lower:
        skills.extend(["骨剑", "死亡凝视"])
    elif "狼" in name_lower or "wolf" in name_lower:
        skills.extend(["撕咬", "扑击"])
    elif "兽人" in name_lower or "orc" in name_lower:
        skills.extend(["重击", "狂暴"])
    elif "法师" in name_lower or "mage" in name_lower:
        skills.extend(["火球术", "冰霜箭"])
    else:
        skills.extend(["猛击", "冲锋"])
    
    return skills

async def process_turn_node(state: GameState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """处理当前角色的战斗回合：根据combat_command执行攻击判定."""
    if not state.combat_order:
        return {
            "combat_log": ["[系统] 战斗列表为空，无法处理回合"],
            "is_combat_active": False
        }
    
    combat_log = list(state.combat_log) if state.combat_log else []
    updated_combatants = list(state.combat_order)
    
    # 从 combat_command 获取攻击信息
    cmd = state.combat_command
    if not cmd or not cmd.attacker or not cmd.defender:
        combat_log.append("[系统] 无法解析战斗指令")
        return {"combat_log": combat_log, "combat_command": None}
    
    # 查找攻击者
    attacker = _find_combatant_by_name(updated_combatants, cmd.attacker)
    if not attacker:
        combat_log.append(f"[系统] 找不到攻击者: {cmd.attacker}")
        return {"combat_log": combat_log, "combat_command": None}
    
    # 查找目标
    target = _find_combatant_by_name(updated_combatants, cmd.defender)
    if not target:
        combat_log.append(f"[系统] 找不到目标: {cmd.defender}")
        return {"combat_log": combat_log, "combat_command": None}
    
    skill_name = cmd.skill or "普通攻击"
    combat_log.append(f"  {attacker.name} 使用 [{skill_name}] 攻击 {target.name}!")
    
    # 计算攻击加值（简化：使用力量调整值）
    str_mod = (attacker.stats.get("STR", 10) - 10) // 2
    
    # 执行攻击
    attack_result = attack_roll.invoke({
        "attacker_name": attacker.name,
        "target_name": target.name,
        "attack_bonus": str_mod,
        "target_ac": target.ac
    })
    
    combat_log.append(f"  {attack_result['details']}")
    
    # 如果命中，计算伤害
    if attack_result["hit"]:
        # 技能加成
        damage_bonus = _get_skill_damage_bonus(skill_name)
        damage_result = damage_roll.invoke({
            "damage_dice": attacker.damage_dice,
            "is_critical": attack_result["is_critical"]
        })
        
        total_damage = damage_result["damage"] + damage_bonus
        bonus_text = f" +{damage_bonus}技能加成" if damage_bonus > 0 else ""
        combat_log.append(f"  {damage_result['details']}{bonus_text}")
        
        # 更新目标生命值
        target_index = next(i for i, c in enumerate(updated_combatants) if c.id == target.id)
        updated_target = updated_combatants[target_index]
        new_hp = max(0, updated_target.hp - total_damage)
        
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
            description=updated_target.description,
            controller=updated_target.controller
        )
        
        combat_log.append(f"  {target.name} 受到 {total_damage} 点伤害! (HP: {updated_target.hp} -> {new_hp})")
        
        if new_hp <= 0:
            combat_log.append(f"  💀 {target.name} 被击败了!")
    
    return {
        "combat_order": updated_combatants,
        "combat_log": combat_log,
        "combat_command": None  # 清空已处理的命令
    }


def _find_combatant_by_name(combatants: list[Combatant], name: str) -> Combatant | None:
    """根据名称模糊匹配查找战斗者."""
    name_lower = name.lower()
    for c in combatants:
        if name_lower in c.name.lower() or c.name.lower() in name_lower:
            return c
    return None


def _get_skill_damage_bonus(skill_name: str) -> int:
    """获取技能的额外伤害加成."""
    skill_bonuses = {
        "普通攻击": 0,
        "至圣斩": 10,
        "重击": 5,
        "猛击": 3,
        "火球术": 8,
        "冰霜箭": 6,
        "利爪": 2,
        "偷袭": 6,
        "骨剑": 3,
        "死亡凝视": 4,
        "撕咬": 3,
        "扑击": 4,
        "狂暴": 5,
        "冲锋": 3,
    }
    return skill_bonuses.get(skill_name, 0)


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


def check_turn_type(state: GameState) -> Literal["player_turn", "npc_batch", "player_action_ready"]:
    """判断当前是玩家回合还是NPC批量处理的路由函数."""
    if not state.combat_order:
        return "npc_batch"
    
    current_actor = state.combat_order[0]
    if current_actor.controller == ControllerType.PLAYER and current_actor.is_alive:
        # 如果 awaiting_player_input 为 True，说明之前已等待玩家输入
        # 此时用户再次输入后应直接路由到 combat_intent 处理战斗逻辑
        if state.awaiting_player_input:
            return "player_action_ready"  # 玩家已输入，处理战斗
        return "player_turn"  # 玩家回合，等待输入
    else:
        return "npc_batch"    # NPC回合，批量处理


async def await_player_input_node(state: GameState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """等待玩家输入节点：标记状态为等待输入，返回给前端."""
    if not state.combat_order:
        return {}
    
    current_actor = state.combat_order[0]
    combat_log = list(state.combat_log) if state.combat_log else []
    combat_log.append(f"\n[回合 {state.current_round}] 轮到 {current_actor.name} (玩家) 行动")
    combat_log.append("请输入你的行动，例如: '使用普通攻击攻击哥布林' 或 '使用至圣斩攻击史莱姆'")
    
    return {
        "awaiting_player_input": True,
        "combat_log": combat_log
    }


async def process_player_action_node(state: GameState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """处理玩家输入的动作节点：解析玩家指令并执行."""
    if not state.combat_order or not state.pending_player_action:
        return {"awaiting_player_input": False, "pending_player_action": None}
    
    current_actor = state.combat_order[0]
    player_input = state.pending_player_action
    combat_log = list(state.combat_log) if state.combat_log else []
    
    # 解析玩家输入
    action_info = _parse_player_action(player_input, state)
    
    if not action_info["valid"]:
        combat_log.append(f"  [错误] {action_info['error']}")
        return {
            "combat_log": combat_log,
            "awaiting_player_input": True,  # 继续等待有效输入
            "pending_player_action": None
        }
    
    target = action_info["target"]
    skill_name = action_info["skill_name"]
    damage_bonus = action_info.get("damage_bonus", 0)
    
    combat_log.append(f"  {current_actor.name} 使用 [{skill_name}] 攻击 {target.name}!")
    
    # 计算攻击加值
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
        # 技能可以有额外伤害加成
        base_damage_dice = current_actor.damage_dice
        damage_result = damage_roll.invoke({
            "damage_dice": base_damage_dice,
            "is_critical": attack_result["is_critical"]
        })
        
        total_damage = damage_result["damage"] + damage_bonus
        combat_log.append(f"  {damage_result['details']}" + (f" +{damage_bonus}技能加成" if damage_bonus > 0 else ""))
        
        # 更新目标生命值
        target_index = next(i for i, c in enumerate(updated_combatants) if c.id == target.id)
        updated_target = updated_combatants[target_index]
        new_hp = max(0, updated_target.hp - total_damage)
        
        updated_combatants[target_index] = Combatant(
            id=updated_target.id,
            name=updated_target.name,
            faction=updated_target.faction,
            hp=new_hp,
            max_hp=updated_target.max_hp,
            ac=updated_target.ac,
            stats=updated_target.stats,
            damage_dice=updated_target.damage_dice,
            description=updated_target.description,
            controller=updated_target.controller
        )
        
        combat_log.append(f"  {target.name} 受到 {total_damage} 点伤害! (HP: {updated_target.hp} -> {new_hp})")
        
        if new_hp <= 0:
            combat_log.append(f"  💀 {target.name} 被击败了!")
    
    return {
        "combat_order": updated_combatants,
        "combat_log": combat_log,
        "awaiting_player_input": False,
        "pending_player_action": None
    }


def _parse_player_action(player_input: str, state: GameState) -> Dict[str, Any]:
    """解析玩家的动作指令.
    
    支持格式：
    - "使用普通攻击攻击哥布林"
    - "使用至圣斩攻击史莱姆"
    - "攻击哥布林"
    """
    import re
    
    current_actor = state.combat_order[0]
    target_faction = Faction.ALLY if current_actor.faction == Faction.ENEMY else Faction.ENEMY
    available_targets = [c for c in state.combat_order if c.faction == target_faction and c.is_alive]
    
    if not available_targets:
        return {"valid": False, "error": "没有可攻击的目标"}
    
    # 技能映射表（可以扩展）
    skill_bonuses = {
        "普通攻击": 0,
        "至圣斩": 10,
        "重击": 5,
        "猛击": 3,
        "火球术": 8,
        "冰霜箭": 6,
    }
    
    # 尝试匹配 "使用XXX攻击YYY" 格式
    pattern1 = r"使用(.+?)攻击(.+)"
    match1 = re.search(pattern1, player_input)
    
    if match1:
        skill_name = match1.group(1).strip()
        target_name = match1.group(2).strip()
    else:
        # 尝试匹配 "攻击XXX" 格式
        pattern2 = r"攻击(.+)"
        match2 = re.search(pattern2, player_input)
        if match2:
            skill_name = "普通攻击"
            target_name = match2.group(1).strip()
        else:
            return {"valid": False, "error": f"无法理解指令: {player_input}。请使用格式: '使用XXX攻击YYY' 或 '攻击YYY'"}
    
    # 查找目标
    target = None
    for t in available_targets:
        if target_name in t.name or t.name in target_name:
            target = t
            break
    
    if not target:
        target_names = [t.name for t in available_targets]
        return {"valid": False, "error": f"找不到目标 '{target_name}'。可用目标: {', '.join(target_names)}"}
    
    # 获取技能加成
    damage_bonus = skill_bonuses.get(skill_name, 0)
    
    return {
        "valid": True,
        "skill_name": skill_name,
        "target": target,
        "damage_bonus": damage_bonus
    }


async def process_npc_batch_node(state: GameState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """批量处理所有NPC回合，直到轮到玩家或战斗结束."""
    if not state.combat_order:
        return {"is_combat_active": False}
    
    combat_log = list(state.combat_log) if state.combat_log else []
    updated_combatants = list(state.combat_order)
    current_round = state.current_round
    
    # 循环处理NPC回合
    max_iterations = 100  # 防止无限循环
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        
        # 检查战斗是否结束
        allies = [c for c in updated_combatants if c.faction == Faction.ALLY and c.is_alive]
        enemies = [c for c in updated_combatants if c.faction == Faction.ENEMY and c.is_alive]
        
        if not allies:
            combat_log.append("\n[系统] ===== 战斗失败...所有队友倒下 =====")
            return {
                "combat_order": updated_combatants,
                "is_combat_active": False,
                "combat_log": combat_log,
                "current_round": current_round
            }
        
        if not enemies:
            combat_log.append("\n[系统] ===== 战斗胜利！所有敌人被击败 =====")
            return {
                "combat_order": updated_combatants,
                "is_combat_active": False,
                "combat_log": combat_log,
                "current_round": current_round
            }
        
        # 过滤掉死亡的角色
        updated_combatants = [c for c in updated_combatants if c.is_alive]
        
        if not updated_combatants:
            break
        
        current_actor = updated_combatants[0]
        
        # 如果当前是玩家，停止批量处理
        if current_actor.controller == ControllerType.PLAYER:
            break
        
        # 处理NPC回合
        combat_log.append(f"\n[回合 {current_round}] {current_actor.name} (NPC) 的回合")
        
        # 获取可攻击的目标
        target_faction = Faction.ALLY if current_actor.faction == Faction.ENEMY else Faction.ENEMY
        available_targets = [c for c in updated_combatants if c.faction == target_faction and c.is_alive]
        
        if not available_targets:
            combat_log.append(f"  {current_actor.name} 没有可攻击的目标")
        else:
            # 选择目标（简单策略：攻击第一个可用目标）
            target = available_targets[0]
            
            # 计算攻击加值
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
                
                updated_combatants[target_index] = Combatant(
                    id=updated_target.id,
                    name=updated_target.name,
                    faction=updated_target.faction,
                    hp=new_hp,
                    max_hp=updated_target.max_hp,
                    ac=updated_target.ac,
                    stats=updated_target.stats,
                    damage_dice=updated_target.damage_dice,
                    description=updated_target.description,
                    controller=updated_target.controller
                )
                
                combat_log.append(f"  {target.name} 受到 {damage_result['damage']} 点伤害! (HP: {updated_target.hp} -> {new_hp})")
                
                if new_hp <= 0:
                    combat_log.append(f"  💀 {target.name} 被击败了!")
        
        # 轮转：将当前角色移到队列尾部
        if len(updated_combatants) >= 2:
            updated_combatants = updated_combatants[1:] + [updated_combatants[0]]
            combat_log.append(f"  -> 下一位: {updated_combatants[0].name}")
    
    return {
        "combat_order": updated_combatants,
        "combat_log": combat_log,
        "current_round": current_round,
        "awaiting_player_input": False
    }

