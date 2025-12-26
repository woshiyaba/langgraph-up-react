"""PDF 索引构建器.

将 DND 规则书 PDF 解析、分块、向量化，存入 ChromaDB。
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_PROVIDER,
    OPENAI_EMBEDDING_MODEL,
    PDF_PATH,
    SILICONFLOW_API_BASE,
    SILICONFLOW_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


def get_embeddings():
    """根据配置获取 Embedding 模型.
    
    支持:
    - openai: OpenAI text-embedding-3-small
    - siliconflow: 硅基流动 BGE 模型（国内推荐）
    """
    if EMBEDDING_PROVIDER == "siliconflow":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=SILICONFLOW_EMBEDDING_MODEL,
            openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
            openai_api_base=SILICONFLOW_API_BASE,
        )
    else:
        # 默认使用 OpenAI
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def load_pdf(pdf_path: Path) -> list[Document]:
    """加载 PDF 文件.
    
    Args:
        pdf_path: PDF 文件路径
        
    Returns:
        Document 列表，每页一个 Document
    """
    from langchain_community.document_loaders import PyPDFLoader
    
    logger.info(f"📖 正在加载 PDF: {pdf_path}")
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
    
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    
    logger.info(f"✅ 加载完成: {len(documents)} 页")
    return documents


def clean_text(text: str) -> str:
    """清理文本，去除噪音.
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 去除页眉页脚常见模式（可根据实际 PDF 调整）
    text = re.sub(r'第\s*\d+\s*页', '', text)
    text = re.sub(r'Page\s*\d+', '', text, flags=re.IGNORECASE)
    # 去除首尾空白
    text = text.strip()
    return text


def preprocess_documents(documents: list[Document]) -> list[Document]:
    """预处理文档，清理文本并添加元数据.
    
    Args:
        documents: 原始 Document 列表
        
    Returns:
        预处理后的 Document 列表
    """
    processed = []
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        if len(cleaned_content) < 50:  # 跳过太短的页面
            continue
        
        # 保留并增强元数据
        metadata = doc.metadata.copy()
        metadata["source_type"] = "dnd_phb"
        metadata["language"] = "zh"
        
        processed.append(Document(
            page_content=cleaned_content,
            metadata=metadata
        ))
    
    logger.info(f"📝 预处理完成: {len(processed)} 个有效文档")
    return processed


def split_documents(documents: list[Document]) -> list[Document]:
    """将文档分块.
    
    使用针对中文优化的分隔符。
    
    Args:
        documents: Document 列表
        
    Returns:
        分块后的 Document 列表
    """
    # 中文友好的分隔符（按优先级）
    separators = [
        "\n\n",      # 段落
        "\n",        # 换行
        "。",        # 句号
        "！",        # 感叹号
        "？",        # 问号
        "；",        # 分号
        "，",        # 逗号
        " ",         # 空格
        ""           # 字符级别（最后手段）
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        length_function=len,
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(f"✂️ 分块完成: {len(chunks)} 个文档块")
    return chunks


def build_index(
    pdf_path: Optional[Path] = None,
    persist_directory: Optional[Path] = None,
    force_rebuild: bool = False
) -> "Chroma":
    """构建向量索引.
    
    Args:
        pdf_path: PDF 文件路径，默认使用配置中的路径
        persist_directory: 持久化目录，默认使用配置中的路径
        force_rebuild: 是否强制重建索引
        
    Returns:
        ChromaDB 向量数据库实例
    """
    from langchain_chroma import Chroma
    
    pdf_path = pdf_path or PDF_PATH
    persist_directory = persist_directory or CHROMA_PERSIST_DIR
    
    # 检查是否已存在索引
    if persist_directory.exists() and not force_rebuild:
        logger.info(f"📂 发现已有索引: {persist_directory}")
        logger.info("如需重建，请使用 force_rebuild=True")
        return Chroma(
            persist_directory=str(persist_directory),
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME
        )
    
    # 确保目录存在
    persist_directory.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 开始构建索引...")
    
    # 1. 加载 PDF
    documents = load_pdf(pdf_path)
    
    # 2. 预处理
    processed_docs = preprocess_documents(documents)
    
    # 3. 分块
    chunks = split_documents(processed_docs)
    
    # 4. 获取 Embedding 模型
    embeddings = get_embeddings()
    
    # 5. 创建向量数据库
    logger.info("🔄 正在向量化并存储（这可能需要几分钟）...")
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=COLLECTION_NAME
    )
    
    logger.info(f"✅ 索引构建完成!")
    logger.info(f"   - 文档块数量: {len(chunks)}")
    logger.info(f"   - 存储位置: {persist_directory}")
    
    return vectordb


def get_index_stats(persist_directory: Optional[Path] = None) -> dict:
    """获取索引统计信息.
    
    Args:
        persist_directory: 持久化目录
        
    Returns:
        统计信息字典
    """
    from langchain_chroma import Chroma
    
    persist_directory = persist_directory or CHROMA_PERSIST_DIR
    
    if not persist_directory.exists():
        return {"exists": False, "error": "索引不存在"}
    
    try:
        vectordb = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME
        )
        
        # 获取集合信息
        collection = vectordb._collection
        count = collection.count()
        
        return {
            "exists": True,
            "document_count": count,
            "collection_name": COLLECTION_NAME,
            "persist_directory": str(persist_directory)
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}


# ============================================================
# CLI 入口
# ============================================================

def main():
    """命令行入口."""
    import argparse
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(
        description="DND 规则书 PDF 索引构建工具"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(PDF_PATH),
        help=f"PDF 文件路径 (默认: {PDF_PATH})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(CHROMA_PERSIST_DIR),
        help=f"索引输出目录 (默认: {CHROMA_PERSIST_DIR})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重建索引（覆盖已有索引）"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="仅显示索引统计信息"
    )
    
    args = parser.parse_args()
    
    if args.stats:
        stats = get_index_stats(Path(args.output))
        print("\n📊 索引统计信息:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        return
    
    # 构建索引
    build_index(
        pdf_path=Path(args.pdf),
        persist_directory=Path(args.output),
        force_rebuild=args.force
    )


if __name__ == "__main__":
    main()

