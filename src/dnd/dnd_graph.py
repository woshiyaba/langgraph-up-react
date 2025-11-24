import uuid
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.runtime import Runtime
from common import Context
from common.utils import load_chat_model
from src.dnd.dnd_state import GameState, Character
import json


def _extract_message_content(message) -> str:
    """从不同类型的 message 中安全地取出文本内容."""
    if isinstance(message, BaseMessage):
        return message.content
    if isinstance(message, dict):
        # 兼容 LangGraph / HTTP JSON 形式: {"role": "...", "content": "..."}
        return str(message.get("content", ""))
    # 兜底: 直接转成字符串
    return str(message)

# === 节点 A: 公会接待员 (负责注册) ===
def registration_node(state: GameState, runtime: Runtime[Context]):
    user_id = state.get("current_user_id", "")

    last_raw_message = state["messages"][-10:]
    last_message = _extract_message_content(last_raw_message)
    print(f"消息总结之后 {last_message}")
    # 定义接待员的人设
    system_prompt = """
    你是一个豪爽、直接的 D&D 地下城主。有新人加入了，别废话，直接给他发一张角色卡。
    
    请随机生成一个 D&D 5E 的 1级角色 (包含姓名、种族、职业、HP、简单的随身物品、一句话背景)。
    职业请在 [战士, 法师, 盗贼, 牧师, 游侠, 圣骑士] 中随机选。
    
    **必须** 返回 JSON 格式，不要包含 Markdown 代码块标记，格式如下：
    {
        "name": "角色名",
        "race": "种族",
        "char_class": "职业",
        "hp": 10, 
        "inventory": ["长剑", "冒险家背包"],
        "backstory": "为了寻找失踪的父亲而踏上旅途。",
        "intro": "一段简短的开场白，描述该角色正处于什么境地（比如酒馆、地牢入口、森林营地）。"
    }
    """
    
    # 调用 LLM
    response = load_chat_model(runtime.context.model).invoke([
        SystemMessage(content=system_prompt),
    ])
    
    content = response.content
    # 清理可能存在的 markdown 标记 (以防万一)
    clean_content = content.replace("```json", "").replace("```", "").strip()
    try:
        char_data = json.loads(clean_content)
        
        # 1. 构建角色对象
        new_char: Character = {
            "name": char_data["name"],
            "char_class": char_data["char_class"],
            "race": char_data["race"],
            "hp": char_data["hp"],
            "max_hp": char_data["hp"],
            "inventory": char_data.get("inventory", ["基本衣物"]),
            "backstory": char_data.get("backstory", "")
        }
        
        # 2. 更新玩家数据库
        new_players = state.get("players", {}).copy()
        new_players[user_id] = new_char
        
        # 3. 生成给用户的欢迎语 (结合了角色介绍和开场剧情)
        welcome_msg = (
            f"🎲 **欢迎来到被遗忘的国度！**\n\n"
            f"老DM打量了你一番，说道：\n"
            f"“哈！看你这身板，一定是 **{new_char['race']} {new_char['char_class']}** 吧！"
            f"我也听过你的大名，**{new_char['name']}**，听说你 {new_char['backstory']}”\n\n"
            f"--- 属性 ---\n"
            f"❤️ HP: {new_char['hp']} | 🎒 装备: {', '.join(new_char['inventory'])}\n"
            f"-------------\n\n"
            f"📜 **当前状况**: {char_data['intro']}\n\n"
            f"*(你现在已经自动准备就绪。请告诉我，你接下来要做什么？)*"
        )
        
        return {
            "players": new_players,
            "messages": [HumanMessage(content=welcome_msg, name="Old_DM")]
        }
        
    except Exception as e:
        print(f"JSON 解析错误: {e}")
        # 容错处理：万一 JSON 解析失败，给个保底角色
        fallback_char = {"name": "冒险者", "char_class": "战士", "hp": 12, "race": "人类", "inventory": ["剑"], "backstory": "新人", "max_hp": 12}
        new_players = state["players"].copy()
        new_players[user_id] = fallback_char
        return {
            "players": new_players,
            "messages": [HumanMessage(content=f"出了点小差错，不过没关系，给你个新手套装（战士）。你现在在村口，要做什么？ ({str(e)})")]
        }


# --- 节点 1: Dungeon Master (DM) ---
# DM 负责分析局势，进行判定，并输出剧情
def dungeon_master_node(state: GameState, runtime: Runtime[Context]):
    llm = load_chat_model(runtime.context.model)
    players_info = json.dumps(state['players'], ensure_ascii=False)
    # 当前阶段: {state['phase']}
    # 构建 Prompt，告诉 LLM 它是 DM
    system_prompt = f"""
    你是一个严谨公正的龙与地下城(D&D 5E) 地下城主(DM)。
    当前玩家状态: {players_info}

    请根据玩家的上一条行动，判断结果。
    1. 如果涉及战斗，请计算伤害并明确指出谁扣了多少 HP。
    2. 如果玩家试图做某事，请决定是否需要投骰子。
    3. 简短生动地描述发生了什么。
    4. 总是以“现在轮到 [玩家名]”结尾，或者询问玩家接下来做什么。
    
    请以 JSON 格式输出你的决定，格式如下：
    {{
      "narrative": "剧情描述...",
      "update_hp": {{"player_id": -5}},  // 可选，更新血量
      "next_phase": "combat" // 可选，切换阶段
    }}
    """
    
    # 获取最后一条消息（玩家的输入）
    last_raw_message = state['messages'][-10:] if state.get('messages') else "游戏开始"
    last_message = _extract_message_content(last_raw_message)

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_message)])
    
    # 解析 LLM 的 JSON 输出 (这里简化处理，实际需要更强的解析逻辑)
    try:
        decision = json.loads(response.content)
        narrative = decision.get("narrative", "")
        
        # 模拟状态更新逻辑
        new_players = state['players'].copy()
        if "update_hp" in decision:
            for pid, change in decision["update_hp"].items():
                if pid in new_players:
                    new_players[pid]['hp'] += change
        
        return {
            "messages": [AIMessage(content=f"DM: {narrative}")],
            "players": new_players,
            "phase": decision.get("next_phase", state['phase'])
        }
    except:
        # 如果解析失败，直接作为文本返回
        return {"messages": [AIMessage(content=f"DM: {response.content}")]}

# --- 节点 2: 路由逻辑 ---
# 判断游戏是否结束，或者是否需要特定处理
def should_continue(state: GameState):
    # 这里可以添加逻辑，比如所有人生存则继续，全灭则结束
    return "wait_for_input"

def route_user(state: GameState):
    user_id = state.get("current_user_id")
    players = state.get("players", {})
    print(f"玩家列表 {players}")
    # 核心判断：用户是否存在于玩家列表中？
    if user_id in players:
        return "dungeon_master"  # 去玩游戏
    else:
        return "registration_node" # 去注册
# --- 构建图 ---
workflow = StateGraph(GameState, context_schema=Context)

# 添加节点
workflow.add_node("dungeon_master", dungeon_master_node)
workflow.add_node("registration_node", registration_node)

# 设置入口
workflow.set_conditional_entry_point(
    route_user,
    {
        "registration_node": "registration_node",
        "dungeon_master": "dungeon_master"
    }
)

# 添加边 (DM -> 等待输入 -> DM 的循环)
# 注意：在真实的服务器代码中，“等待输入”通常意味着中断图的执行，等待 API 调用
# 这里我们用一个逻辑上的“结束”来模拟一回合的结束，等待下一次 invoke

workflow.add_edge("dungeon_master", END) 
workflow.add_edge("registration_node", END) 

app = workflow.compile()