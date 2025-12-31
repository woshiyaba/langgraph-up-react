"""PDF 索引构建器.

将 DND 规则书 PDF 解析、分块、向量化，存入 ChromaDB。
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from src.rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_PROVIDER,
    OPENAI_EMBEDDING_MODEL,
    PDF_PATH,
    SILICONFLOW_API_BASE,
    SILICONFLOW_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()


class DashScopeEmbeddings(Embeddings):
    """DashScope Embeddings 包装类，支持 text-embedding-v4 和 dimensions 参数.
    
    使用 OpenAI 兼容客户端调用 DashScope API。
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v4",
        dimensions: Optional[int] = None
    ):
        """初始化 DashScope Embeddings.
        
        Args:
            api_key: DashScope API Key
            base_url: API 基础 URL
            model: 模型名称
            dimensions: 向量维度（仅 text-embedding-v4 支持）
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.dimensions = dimensions
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文档.
        
        DashScope API 限制每批最多 10 个文本，需要分批处理。
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        # DashScope API 限制：每批最多 10 个文本
        BATCH_SIZE = 10
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            
            # 构建请求参数
            params = {
                "model": self.model,
                "input": batch_texts,
            }
            # 如果指定了 dimensions 且模型支持，则添加该参数
            if self.dimensions is not None and self.model == "text-embedding-v4":
                params["dimensions"] = self.dimensions
            
            try:
                resp = self.client.embeddings.create(**params)
                batch_embeddings = [item.embedding for item in resp.data]
                all_embeddings.extend(batch_embeddings)
                
                # 记录进度（每 50 个文本记录一次）
                processed = min(i + BATCH_SIZE, len(texts))
                if processed % 50 == 0 or processed >= len(texts):
                    logger.info(f"📊 向量化进度: {processed}/{len(texts)} ({processed*100//len(texts)}%)")
            except Exception as e:
                logger.error(f"DashScope Embedding 调用失败 (批次 {i//BATCH_SIZE + 1}): {e}")
                raise
        
        return all_embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询文本.
        
        Args:
            text: 查询文本
            
        Returns:
            向量
        """
        return self.embed_documents([text])[0]


def get_embeddings():
    """根据配置获取 Embedding 模型.
    
    支持:
    - openai: OpenAI text-embedding-3-small
    - siliconflow: 硅基流动 BGE 模型（国内推荐）
    - dashscope: 通义千问 Embedding 模型（通过兼容模式，支持 text-embedding-v4 和 dimensions）
    """
    from langchain_openai import OpenAIEmbeddings
    
    if EMBEDDING_PROVIDER == "siliconflow":
        return OpenAIEmbeddings(
            model=SILICONFLOW_EMBEDDING_MODEL,
            openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
            openai_api_base=SILICONFLOW_API_BASE,
        )
    else:
        api_base = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        model = OPENAI_EMBEDDING_MODEL
        
        # 如果使用 DashScope 兼容模式，使用自定义的 DashScopeEmbeddings
        if api_base and "dashscope" in api_base.lower():
            # 使用 DashScope 自定义 Embeddings 类，支持 text-embedding-v4 和 dimensions
            logger.info(f"使用 DashScope Embeddings: model={model}, dimensions={EMBEDDING_DIMENSIONS}")
            return DashScopeEmbeddings(
                api_key=api_key,
                base_url=api_base,
                model=model,
                dimensions=EMBEDDING_DIMENSIONS if model == "text-embedding-v4" else None
            )
        
        # 其他情况使用标准的 OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model,
            openai_api_base=api_base,
            openai_api_key=api_key
        )


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
        "\n\n",  # 段落
        "\n",  # 换行
        "。",  # 句号
        "！",  # 感叹号
        "？",  # 问号
        "；",  # 分号
        "，",  # 逗号
        " ",  # 空格
        ""  # 字符级别（最后手段）
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    
    # 过滤无效的文档块：确保内容是非空字符串
    valid_chunks = []
    for chunk in chunks:
        content = chunk.page_content
        # 确保内容是字符串类型且非空
        if isinstance(content, str) and content.strip():
            valid_chunks.append(chunk)
        else:
            logger.warning(f"跳过无效文档块: 类型={type(content)}, 长度={len(content) if content else 0}")
    
    logger.info(f"✂️ 分块完成: {len(valid_chunks)} 个有效文档块 (过滤了 {len(chunks) - len(valid_chunks)} 个无效块)")
    return valid_chunks


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

    # 4. 再次验证文档块的有效性（双重保险）
    valid_chunks = []
    for chunk in chunks:
        if isinstance(chunk.page_content, str) and chunk.page_content.strip():
            valid_chunks.append(chunk)
    
    if len(valid_chunks) < len(chunks):
        logger.warning(f"过滤了 {len(chunks) - len(valid_chunks)} 个无效文档块")
        chunks = valid_chunks
    
    if not chunks:
        raise ValueError("没有有效的文档块可以向量化！请检查 PDF 内容和分块配置。")
    
    # 5. 获取 Embedding 模型
    embeddings = get_embeddings()

    # 6. 创建向量数据库
    logger.info("🔄 正在向量化并存储（这可能需要几分钟）...")
    logger.info(f"   将处理 {len(chunks)} 个文档块...")
    
    # 验证 embedding 模型配置
    api_base = os.getenv("OPENAI_API_BASE", "")
    if api_base and "dashscope" in api_base.lower():
        logger.info(f"✅ 使用 DashScope Embeddings (model={OPENAI_EMBEDDING_MODEL}, dimensions={EMBEDDING_DIMENSIONS})")

    try:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_directory),
            collection_name=COLLECTION_NAME
        )
    except Exception as e:
        error_msg = str(e)
        if "contents is neither str nor list of str" in error_msg or "InvalidParameter" in error_msg:
            logger.error("❌ Embedding API 参数格式错误！")
            logger.error("   可能的原因：")
            logger.error("   1. API Key 不正确或未设置")
            logger.error("   2. 模型名称不正确")
            logger.error("   3. dimensions 参数值不正确（应为 256, 512, 或 1024）")
            logger.error("   建议解决方案：")
            logger.error("   - 检查 DASHSCOPE_API_KEY 环境变量")
            logger.error("   - 确认模型名称正确（text-embedding-v4）")
            logger.error("   - 检查 RAG_EMBEDDING_DIMENSIONS 环境变量（可选值：256, 512, 1024）")
        raise

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
