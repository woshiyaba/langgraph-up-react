"""RAG (检索增强生成) 模块.

用于将 DND 规则书 PDF 建立向量索引，并提供检索接口。
"""

from src.rag.retriever import DNDRuleRetriever

__all__ = ["DNDRuleRetriever"]

