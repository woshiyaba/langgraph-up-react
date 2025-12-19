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
    type: str # "quest_item", "weapon", "consumable"
    effect_id: Optional[str] # 关联到具体的代码逻辑或伏笔标记

# 2. 技能结构
@dataclass
class Skill:
    name: str
    class_requirement: str # 职业限制，例如 "Mage"
    level_requirement: int
    damage_formula: str # 例如 "2d6 + int_mod"

# 3. 战斗角色阵营
class Faction(str, Enum):
    ALLY = "ally"      # 队友
    ENEMY = "enemy"    # 敌人

# 3.1 控制类型（玩家/NPC）
class ControllerType(str, Enum):
    PLAYER = "player"  # 玩家控制，需要等待输入
    NPC = "npc"        # AI控制，自动处理

# 4. 战斗角色（统一队友和敌人）
@dataclass
class Combatant:
    """战斗参与者，可以是玩家、队友或敌人"""
    id: str                          # 唯一标识
    name: str                        # 显示名称
    faction: Faction                 # 阵营：ally/enemy
    hp: int                          # 当前生命值
    max_hp: int                      # 最大生命值
    ac: int                          # 护甲等级
    stats: Dict[str, int]            # 属性 {"STR": 10, "DEX": 14, "CON": 12...}
    damage_dice: str                 # 伤害骰，如 "1d8+2"
    description: Optional[str] = None  # 角色描述
    controller: ControllerType = ControllerType.NPC  # 控制类型：玩家/NPC

    @property
    def dexterity(self) -> int:
        """获取敏捷值，用于先攻排序"""
        return self.stats.get("DEX", 10)
    
    @property
    def is_alive(self) -> bool:
        """是否存活"""
        return self.hp > 0

@dataclass
class Player:
        # --- 角色数据 (RPG核心) ---
    player_name: str
    player_class: str # Warrior, Mage, Rogue
    level: int
    hp: int
    max_hp: int
    stats: Dict[str, int] # {"STR": 10, "INT": 16...}
    
    # --- 物品与技能 ---
    inventory: List[Item]
    skills: List[Skill]
    
    # --- 剧情伏笔 (关键！) ---
    # 这里存储键值对，例如 {"found_ancient_key": True, "met_beggar_king": True}
    plot_flags: Dict[str, bool] 

    # --- 战斗状态 ---
    is_in_combat: bool
    enemy_state: Optional[Dict] # 当前敌人的数据
    phase: str # 当前阶段

    
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
    players: Dict[str, Player] = field(default_factory=dict) # 玩家列表
    current_user_id: str = "" # 当前用户ID
    
    # --- 战斗系统 ---
    combat_order: List[Combatant] = field(default_factory=list)  # 战斗顺序列表
    is_combat_active: bool = False  # 是否正在战斗中
    current_round: int = 0  # 当前回合数
    combat_log: List[str] = field(default_factory=list)  # 战斗日志
    awaiting_player_input: bool = False  # 是否等待玩家输入
    pending_player_action: Optional[str] = None  # 玩家待处理的动作指令
    combat_command: Optional[CombatCommand] = None  # 战斗命令
    npc_action_text: Optional[str] = None  # NPC生成的行动指令文本
