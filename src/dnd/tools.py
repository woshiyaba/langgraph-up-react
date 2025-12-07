# 定义骰子工具
import random
from typing import Dict, Any
from langchain_core.tools import tool

class DiceTools:
    """骰子工具集合"""

    @staticmethod
    @tool
    def roll_dice(dice_expression: str) -> Dict[str, Any]:
        """
        解析骰子表达式并投掷
        支持格式：1d20, 2d6+3, d100等
        """
        import re

        # 解析骰子表达式
        pattern = r'(\d*)d(\d+)([+-]\d+)?'
        match = re.match(pattern, dice_expression)

        if not match:
            return {"result": 0, "details": "Invalid dice expression"}

        num_dice = int(match.group(1)) if match.group(1) else 1
        dice_sides = int(match.group(2))
        modifier = int(match.group(3)) if match.group(3) else 0

        # 投掷骰子
        rolls = [random.randint(1, dice_sides) for _ in range(num_dice)]
        total = sum(rolls) + modifier

        return {
            "result": total,
            "rolls": rolls,
            "modifier": modifier,
            "expression": dice_expression,
            "details": f"Rolled {dice_expression}: {rolls} + {modifier} = {total}"
        }

    @staticmethod
    def skill_check(skill: str, difficulty: int = 15) -> Dict[str, Any]:
        """技能检定"""
        roll_result = DiceTools.roll_dice("1d20")
        success = roll_result["result"] >= difficulty

        return {
            **roll_result,
            "skill": skill,
            "difficulty": difficulty,
            "success": success,
            "message": f"{skill} check: {roll_result['result']} vs DC {difficulty}. {'Success!' if success else 'Failure!'}"
        }

    @staticmethod
    def attack_roll(attacker: str, target_ac: int) -> Dict[str, Any]:
        """攻击骰"""
        roll_result = DiceTools.roll_dice("1d20")
        hit = roll_result["result"] >= target_ac

        return {
            **roll_result,
            "attacker": attacker,
            "target_ac": target_ac,
            "hit": hit,
            "message": f"{attacker} attacks: {roll_result['result']} vs AC {target_ac}. {'Hit!' if hit else 'Miss!'}"
        }

    @staticmethod
    def damage_roll(damage_dice: str) -> Dict[str, Any]:
        """伤害骰"""
        return DiceTools.roll_dice(damage_dice)