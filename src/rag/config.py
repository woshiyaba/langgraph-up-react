"""RAG 模块配置."""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROJECT_NAME = Path(__file__).parent.parent.parent
# RAG 相关路径
RAG_DIR = PROJECT_ROOT / PROJECT_NAME / "rag"
PDF_PATH = RAG_DIR / PROJECT_NAME / "rag/5eDnD_玩家手册PHB_中译v1.72版.pdf"
CHROMA_PERSIST_DIR = RAG_DIR / "chroma_db"
print("RAG_DIR ", RAG_DIR)
print("CHROMA_PERSIST_DIR ", CHROMA_PERSIST_DIR)
# 分块配置
CHUNK_SIZE = 256  # 每个文档块的字符数
CHUNK_OVERLAP = 50  # 块之间的重叠字符数

# 检索配置
DEFAULT_TOP_K = 3  # 默认检索的文档数量

# Embedding 模型配置（支持多种后端）
EMBEDDING_PROVIDER = os.getenv("RAG_EMBEDDING_PROVIDER", "openai")

# OpenAI Embedding
OPENAI_EMBEDDING_MODEL = "text-embedding-v4"
# Embedding 向量维度（仅 text-embedding-v4 支持，可选值：256, 512, 1024）
EMBEDDING_DIMENSIONS = int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "256"))

# SiliconFlow Embedding（国内可用）
SILICONFLOW_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"

# 集合名称
COLLECTION_NAME = "dnd_phb_zh"
