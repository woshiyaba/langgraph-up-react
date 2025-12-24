from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


# 1. 物品结构
@dataclass
class Item:
    name: str
    description: str
    type: str  # "quest_item", "weapon", "consumable"
    effect_id: Optional[str]  # 关联到具体的代码逻辑或伏笔标记


# 2. 技能结构
@dataclass
class Skill:
    name: str
    class_requirement: str  # 职业限制，例如 "Mage"
    level_requirement: int
    damage_formula: str  # 例如 "2d6 + int_mod"


# 3. 战斗角色阵营
class Faction(str, Enum):
    ALLY = "ally"  # 队友
    ENEMY = "enemy"  # 敌人


# 3.1 控制类型（玩家/NPC）
class ControllerType(str, Enum):
    PLAYER = "player"  # 玩家控制，需要等待输入
    NPC = "npc"  # AI控制，自动处理


# 4. 战斗角色（统一队友和敌人）
@dataclass
class Combatant:
    """战斗参与者，可以是玩家、队友或敌人"""

    id: str  # 唯一标识
    name: str  # 显示名称
    faction: Faction  # 阵营：ally/enemy
    hp: int  # 当前生命值
    max_hp: int  # 最大生命值
    ac: int  # 护甲等级
    stats: Dict[str, int]  # 属性 {"STR": 10, "DEX": 14, "CON": 12...}
    damage_dice: str  # 伤害骰，如 "1d8+2"
    description: Optional[str] = None  # 角色描述
    controller: ControllerType = ControllerType.NPC  # 控制类型：玩家/NPC
    skills: List[Skill] = field(default_factory=list)

    @property
    def dexterity(self) -> int:
        """获取敏捷值，用于先攻排序."""
        return self.stats.get("DEX", 10)

    @property
    def is_alive(self) -> bool:
        """是否存活."""
        return self.hp > 0


@dataclass
class Player:
    """玩家角色，属性与 Combatant 对齐，方便战斗时转换."""

    # === 与 Combatant 对齐的核心属性 ===
    id: str  # 唯一标识（使用 thread_id）
    name: str  # 显示名称
    hp: int  # 当前生命值
    max_hp: int  # 最大生命值
    ac: int  # 护甲等级
    stats: Dict[str, int]  # 属性 {"STR": 10, "DEX": 14, "CON": 12...}
    damage_dice: str  # 伤害骰，如 "1d8+2"
    description: Optional[str] = None  # 角色描述
    skills: List[Skill] = field(default_factory=list)

    # === Player 特有属性 ===
    player_class: str = "Warrior"  # 职业：Warrior, Mage, Rogue
    level: int = 1  # 等级
    inventory: List[Item] = field(default_factory=list)  # 背包
    plot_flags: Dict[str, bool] = field(default_factory=dict)  # 剧情伏笔

    # === 兼容 Combatant 的属性方法 ===
    @property
    def dexterity(self) -> int:
        """获取敏捷值，用于先攻排序."""
        return self.stats.get("DEX", 10)

    @property
    def is_alive(self) -> bool:
        """是否存活."""
        return self.hp > 0

    def to_combatant(self) -> "Combatant":
        """转换为 Combatant 用于战斗."""
        return Combatant(
            id=self.id,
            name=self.name,
            faction=Faction.ALLY,
            hp=self.hp,
            max_hp=self.max_hp,
            ac=self.ac,
            stats=self.stats,
            damage_dice=self.damage_dice,
            description=self.description,
            controller=ControllerType.PLAYER,
            skills=self.skills,
        )


"""攻击者 被攻击者 和技能."""


class CombatCommand(BaseModel):
    # 攻击者
    attacker: Optional[str] = None
    # 被攻击者
    defender: Optional[str] = None
    # 技能
    skill: Optional[str] = None


# 5. 全局状态 (传入所有节点的上下文)
@dataclass
class GameState:
    # --- 基础聊天 ---
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    players: Dict[str, Player] = field(default_factory=dict)  # 玩家列表
    current_user_id: str = ""  # 当前用户ID

    # --- 战斗系统 ---
    combat_order: List[Combatant] = field(default_factory=list)  # 战斗顺序列表
    is_combat_active: bool = False  # 是否正在战斗中
    current_round: int = 0  # 当前回合数
    combat_log: List[str] = field(default_factory=list)  # 战斗日志
    awaiting_player_input: bool = False  # 是否等待玩家输入
    pending_player_action: Optional[str] = None  # 玩家待处理的动作指令
    combat_command: Optional[CombatCommand] = None  # 战斗命令
    npc_action_text: Optional[str] = None  # NPC生成的行动指令文本
