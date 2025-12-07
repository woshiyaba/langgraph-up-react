from dataclasses import field, dataclass
from typing import TypedDict, List, Dict, Optional, Annotated, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

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

# 3. 全局状态 (传入所有节点的上下文)
@dataclass
class GameState:
    # --- 基础聊天 ---
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    players: Dict[str, Player] = field(default_factory=dict) # 玩家列表
    current_user_id: str = "" # 当前用户ID
