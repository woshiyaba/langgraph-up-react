from typing import TypedDict
from langchain_core.tools import tool
import random

from src.common.utils import search_with_score


class RuleReference(TypedDict):
    """单条规则参考."""
    content: str
    score: float


class DNDRulesResult(TypedDict):
    """DND规则检索结果."""
    query: str
    found: bool
    rules: list[RuleReference]
    summary: str


@tool
def story_create() -> int:
    """生成一个1-6的随机数，用于决定故事类型。
    
    返回值含义：
    - 1/2/3: 普通故事剧情
    - 4/5: 附带重要NPC的故事
    - 6: 战斗铺垫剧情
    """
    print("story_create called")
    # return random.randint(1, 6)
    return 6


@tool
def search_dnd_rules(query: str) -> DNDRulesResult:
    """从DND规则库中检索相关规则和知识，作为故事生成的依据。
    
    在生成故事之前调用此工具，获取与玩家行动相关的DND规则参考。
    这能确保故事内容符合DND规则体系。
    
    Args:
        query: 查询内容，描述玩家的行动或需要查询的规则
        
    Returns:
        包含检索结果的结构化字典：
        - query: 原始查询
        - found: 是否找到相关规则
        - rules: 规则列表，每条包含content和score
        - summary: 规则摘要，便于快速理解
    """
    print(f"search_dnd_rules called with query: {query}")
    
    # 调用RAG检索
    results = search_with_score(query, k=3)
    
    if not results:
        return DNDRulesResult(
            query=query,
            found=False,
            rules=[],
            summary="未找到相关DND规则，请根据通用DND知识生成故事。"
        )
    
    # 构建规则列表
    rules: list[RuleReference] = [
        RuleReference(content=content, score=round(score, 4))
        for content, score in results
    ]
    
    # 生成摘要（取前两条的开头）
    summary_parts = []
    for i, (content, _) in enumerate(results[:2], 1):
        # 截取前100个字符作为摘要
        snippet = content[:100].replace('\n', ' ').strip()
        if len(content) > 100:
            snippet += "..."
        summary_parts.append(f"[{i}] {snippet}")
    
    return DNDRulesResult(
        query=query,
        found=True,
        rules=rules,
        summary="\n".join(summary_parts)
    )


async def get_story_tools():
    """获取故事生成相关的所有工具."""
    print("get_story_tools called")
    return [story_create, search_dnd_rules]
