"""战斗系统工具集合"""
import random
import re
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.dnd.dnd_state import Combatant, ControllerType, Faction


class ExtractedCharacter(BaseModel):
    """从对话中提取的角色信息."""
    name: str = Field(description="角色名称")
    faction: str = Field(description="阵营: ally 或 enemy")
    is_player: bool = Field(default=False, description="是否为玩家控制的角色")
    hp: int = Field(default=20, description="生命值")
    max_hp: int = Field(default=20, description="最大生命值")
    ac: int = Field(default=12, description="护甲等级")
    dex: int = Field(default=10, description="敏捷属性")
    damage_dice: str = Field(default="1d6", description="伤害骰")
    description: Optional[str] = Field(default=None, description="角色描述")


class ExtractedCharacters(BaseModel):
    """提取的角色列表"""
    characters: List[ExtractedCharacter] = Field(description="角色列表")


@tool
def roll_initiative(modifier: int = 0) -> Dict[str, Any]:
    """
    投掷先攻骰 (1d20 + 敏捷调整值)
    
    Args:
        modifier: 敏捷调整值
    
    Returns:
        先攻结果
    """
    roll = random.randint(1, 20)
    total = roll + modifier
    return {
        "roll": roll,
        "modifier": modifier,
        "total": total,
        "details": f"先攻: {roll} + {modifier} = {total}"
    }


@tool
def attack_roll(attacker_name: str, target_name: str, attack_bonus: int, target_ac: int) -> Dict[str, Any]:
    """
    执行攻击骰判定
    
    Args:
        attacker_name: 攻击者名称
        target_name: 目标名称
        attack_bonus: 攻击加值
        target_ac: 目标护甲等级
    
    Returns:
        攻击结果
    """
    roll = random.randint(1, 20)
    is_critical = roll == 20
    is_fumble = roll == 1
    total = roll + attack_bonus
    
    # 大成功必中，大失败必失
    if is_critical:
        hit = True
    elif is_fumble:
        hit = False
    else:
        hit = total >= target_ac
    
    return {
        "attacker": attacker_name,
        "target": target_name,
        "roll": roll,
        "attack_bonus": attack_bonus,
        "total": total,
        "target_ac": target_ac,
        "hit": hit,
        "is_critical": is_critical,
        "is_fumble": is_fumble,
        "details": f"{attacker_name} 攻击 {target_name}: {roll}+{attack_bonus}={total} vs AC{target_ac} -> {'暴击!' if is_critical else '命中!' if hit else '大失败!' if is_fumble else '未命中'}"
    }


@tool
def damage_roll(damage_dice: str, is_critical: bool = False) -> Dict[str, Any]:
    """
    投掷伤害骰
    
    Args:
        damage_dice: 伤害骰表达式，如 "1d8+3"
        is_critical: 是否暴击（伤害骰翻倍）
    
    Returns:
        伤害结果
    """
    # 解析骰子表达式
    pattern = r'(\d*)d(\d+)([+-]\d+)?'
    match = re.match(pattern, damage_dice)
    
    if not match:
        return {"damage": 0, "details": "无效的伤害骰表达式"}
    
    num_dice = int(match.group(1)) if match.group(1) else 1
    dice_sides = int(match.group(2))
    modifier = int(match.group(3)) if match.group(3) else 0
    
    # 暴击时骰子数量翻倍
    actual_dice = num_dice * 2 if is_critical else num_dice
    
    rolls = [random.randint(1, dice_sides) for _ in range(actual_dice)]
    total = sum(rolls) + modifier
    
    return {
        "damage": total,
        "rolls": rolls,
        "modifier": modifier,
        "is_critical": is_critical,
        "expression": damage_dice,
        "details": f"伤害: {rolls} + {modifier} = {total}" + (" (暴击!)" if is_critical else "")
    }


def calculate_dex_modifier(dex: int) -> int:
    """计算敏捷调整值"""
    return (dex - 10) // 2


def sort_combatants_by_initiative(combatants: List[Combatant]) -> List[Combatant]:
    """
    按先攻顺序排序战斗者列表
    
    先攻 = 1d20 + 敏捷调整值
    敏捷高的在前
    """
    # 为每个角色投掷先攻
    initiative_rolls = []
    for c in combatants:
        dex_mod = calculate_dex_modifier(c.dexterity)
        roll_result = roll_initiative.invoke({"modifier": dex_mod})
        initiative_rolls.append((c, roll_result["total"]))
    
    # 按先攻值降序排序
    initiative_rolls.sort(key=lambda x: x[1], reverse=True)
    
    return [c for c, _ in initiative_rolls]


def create_combatant_from_extracted(extracted: ExtractedCharacter, index: int) -> Combatant:
    """从提取的角色信息创建 Combatant 对象"""
    faction = Faction.ALLY if extracted.faction.lower() == "ally" else Faction.ENEMY
    controller = ControllerType.PLAYER if extracted.is_player else ControllerType.NPC
    
    return Combatant(
        id=f"{extracted.faction}_{index}_{extracted.name}",
        name=extracted.name,
        faction=faction,
        hp=extracted.hp,
        max_hp=extracted.max_hp,
        ac=extracted.ac,
        stats={
            "STR": 10,
            "DEX": extracted.dex,
            "CON": 10,
            "INT": 10,
            "WIS": 10,
            "CHA": 10
        },
        damage_dice=extracted.damage_dice,
        description=extracted.description,
        controller=controller
    )


def get_attack_tools():
    """获取战斗工具列表"""
    return [roll_initiative, attack_roll, damage_roll]

