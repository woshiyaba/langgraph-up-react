"""DND 规则检索器.

提供简单易用的检索接口，供游戏节点调用。
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from src.rag.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
)
from src.rag.indexer import get_embeddings

logger = logging.getLogger(__name__)


class DNDRuleRetriever:
    """DND 规则检索器.
    
    使用单例模式，避免重复加载向量数据库。
    
    Example:
        >>> retriever = DNDRuleRetriever()
        >>> results = retriever.search("火球术的伤害")
        >>> print(results[0])
    """
    
    _instance: Optional["DNDRuleRetriever"] = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, persist_directory: Optional[Path] = None):
        """初始化检索器.
        
        Args:
            persist_directory: 向量数据库目录
        """
        if self._initialized:
            return
        
        from langchain_chroma import Chroma
        
        self.persist_directory = persist_directory or CHROMA_PERSIST_DIR
        
        if not self.persist_directory.exists():
            logger.warning(f"⚠️ 向量数据库不存在: {self.persist_directory}")
            logger.warning("请先运行: python -m src.rag.indexer")
            self._vectordb = None
        else:
            self._vectordb = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=get_embeddings(),
                collection_name=COLLECTION_NAME
            )
            logger.info(f"✅ 加载向量数据库: {self.persist_directory}")
        
        self._initialized = True
    
    @property
    def is_available(self) -> bool:
        """检查检索器是否可用."""
        return self._vectordb is not None
    
    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[str]:
        """搜索相关规则.
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相关文档内容列表
        """
        if not self.is_available:
            logger.warning("检索器不可用，返回空结果")
            return []
        
        try:
            docs = self._vectordb.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
    
    def search_with_metadata(
        self, 
        query: str, 
        k: int = DEFAULT_TOP_K
    ) -> list[Document]:
        """搜索相关规则，返回完整 Document（含元数据）.
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            Document 列表
        """
        if not self.is_available:
            return []
        
        try:
            return self._vectordb.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
    
    def search_with_score(
        self, 
        query: str, 
        k: int = DEFAULT_TOP_K
    ) -> list[tuple[str, float]]:
        """搜索相关规则，返回内容和相关度分数.
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            (内容, 分数) 元组列表，分数越低越相关
        """
        if not self.is_available:
            return []
        
        try:
            results = self._vectordb.similarity_search_with_score(query, k=k)
            return [(doc.page_content, score) for doc, score in results]
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
    
    def search_by_filter(
        self,
        query: str,
        filter_dict: dict,
        k: int = DEFAULT_TOP_K
    ) -> list[str]:
        """带过滤条件的搜索.
        
        Args:
            query: 查询文本
            filter_dict: 过滤条件，如 {"page": 10}
            k: 返回结果数量
            
        Returns:
            相关文档内容列表
        """
        if not self.is_available:
            return []
        
        try:
            docs = self._vectordb.similarity_search(
                query, 
                k=k,
                filter=filter_dict
            )
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []


# ============================================================
# 便捷函数
# ============================================================

@lru_cache(maxsize=1)
def get_retriever() -> DNDRuleRetriever:
    """获取检索器实例（带缓存）."""
    return DNDRuleRetriever()


def search_rule(query: str, k: int = DEFAULT_TOP_K) -> list[str]:
    """快捷搜索函数.
    
    Args:
        query: 查询文本
        k: 返回结果数量
        
    Returns:
        相关文档内容列表
        
    Example:
        >>> from src.rag.retriever import search_rule
        >>> results = search_rule("法术位的恢复规则")
    """
    return get_retriever().search(query, k)


def format_context(results: list[str], max_chars: int = 2000) -> str:
    """格式化检索结果，用于拼接到 Prompt.
    
    Args:
        results: 检索结果列表
        max_chars: 最大字符数
        
    Returns:
        格式化后的上下文字符串
    """
    if not results:
        return ""
    
    formatted_parts = []
    total_chars = 0
    
    for i, content in enumerate(results, 1):
        part = f"[参考{i}] {content}"
        if total_chars + len(part) > max_chars:
            break
        formatted_parts.append(part)
        total_chars += len(part)
    
    return "\n\n".join(formatted_parts)


# ============================================================
# 调试工具
# ============================================================

def interactive_search():
    """交互式搜索（调试用）."""
    print("🔍 DND 规则检索器 - 交互模式")
    print("输入 'quit' 退出\n")
    
    retriever = get_retriever()
    
    if not retriever.is_available:
        print("❌ 检索器不可用，请先构建索引")
        return
    
    while True:
        query = input("查询> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        
        if not query:
            continue
        
        results = retriever.search_with_score(query, k=3)
        
        if not results:
            print("未找到相关结果\n")
            continue
        
        print(f"\n找到 {len(results)} 条结果:\n")
        for i, (content, score) in enumerate(results, 1):
            print(f"--- 结果 {i} (相关度: {score:.4f}) ---")
            print(content[:300] + "..." if len(content) > 300 else content)
            print()


if __name__ == "__main__":
    interactive_search()

