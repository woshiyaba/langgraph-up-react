from langchain_core.tools import tool
import random


@tool
def story_create() -> int:
    """生成一个1-20的随机数"""
    print("story_create called")
    # return random.randint(1, 6)
    return 6

async def get_story_tools():
    print("get_story_tools called")
    return [story_create]
