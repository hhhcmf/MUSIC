import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import fasttext
from transformers import BertTokenizer, BertModel
import torch
import openai
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import redis

# 初始化 Redis 缓存，减少重复嵌入计算
cache = redis.Redis(host='localhost', port=6379, db=0)

# ========== 定义枚举和类型别名 ==========
class EmbedModelType(str, Enum):
    FASTTEXT = "fasttext"
    BERT = "bert"
    OPENAI = "openai"
    MOCK = "mock"

TextEmbedder = Callable[[List[str]], List[List[float]]]

# ========== 定义嵌入函数 ==========
def load_embedder(model_type: EmbedModelType) -> TextEmbedder:
    """根据模型类型加载嵌入器"""
    if model_type == EmbedModelType.FASTTEXT:
        return fasttext_text_embedder
    elif model_type == EmbedModelType.BERT:
        return bert_text_embedder
    elif model_type == EmbedModelType.OPENAI:
        return openai_text_embedder
    elif model_type == EmbedModelType.MOCK:
        return mock_text_embedder
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# 加载fastText模型
fasttext_model = fasttext.load_model('cc.en.300.bin')  # 请确保模型路径正确

def fasttext_text_embedder(texts: List[str]) -> List[List[float]]:
    """使用fastText模型生成嵌入"""
    return [fasttext_model.get_sentence_vector(text).tolist() for text in texts]

# 加载BERT模型和分词器
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def bert_text_embedder(texts: List[str]) -> List[List[float]]:
    """使用BERT模型生成嵌入"""
    embeddings = []
    for text in texts:
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
        embeddings.append(cls_embedding)
    return embeddings

# OpenAI 嵌入生成函数
def openai_text_embedder(texts: List[str]) -> List[List[float]]:
    """使用 OpenAI 嵌入模型生成嵌入"""
    openai.api_key = os.getenv("OPENAI_API_KEY")  # 确保已设置 API 密钥
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"  # 使用 OpenAI 的嵌入模型
    )
    return [embedding["embedding"] for embedding in response["data"]]

# Mock 嵌入生成器
def mock_text_embedder(texts: List[str]) -> List[List[float]]:
    """Mock嵌入模型，生成随机嵌入"""
    return [np.random.rand(300).tolist() for _ in texts]

# ========== 定义文档和向量存储类 ==========
class VectorStoreDocument:
    """表示存储的文档"""
    def __init__(self, id: str, text: str, vector: List[float], attributes: Optional[Dict[str, Any]] = None):
        self.id = id
        self.text = text
        self.vector = vector
        self.attributes = attributes or {}

class LanceDBVectorStore:
    """LanceDB 向量存储实现"""
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.documents = []

    def connect(self):
        print(f"Connected to LanceDB collection: {self.collection_name}")

    def load_documents(self, documents: List[VectorStoreDocument], overwrite: bool = True):
        """批量加载文档"""
        if overwrite:
            self.documents = documents
        else:
            self.documents.extend(documents)
        print(f"Loaded {len(documents)} documents into collection '{self.collection_name}'")

    def similarity_search(self, query_embedding: List[float], k: int = 10) -> List[Dict]:
        """基于向量的相似性搜索"""
        results = []
        for doc in self.documents[:k]:  # 简化的模拟相似性搜索
            results.append({
                "id": doc.id,
                "text": doc.text,
                "score": np.random.random(),  # 假设相似性分数
                "content_type": doc.attributes.get("content_type")
            })
        return results

# ========== 定义批量嵌入和加载函数 ==========
async def batch_embed_and_load_async(
    texts: List[str],
    content_type: str,
    vector_store: LanceDBVectorStore,
    embedder: TextEmbedder,
    batch_size: int = 10
):
    """批量嵌入并加载到向量数据库，支持异步处理和缓存"""
    documents = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = await asyncio.gather(*[async_get_embedding(text, embedder) for text in batch_texts])

        for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
            doc_id = f"{content_type}_{i + j}"
            document = VectorStoreDocument(id=doc_id, text=text, vector=embedding, attributes={"content_type": content_type})
            documents.append(document)

    vector_store.load_documents(documents)

async def async_get_embedding(text: str, embedder: TextEmbedder) -> List[float]:
    """使用缓存的异步嵌入生成"""
    hash_key = hashlib.md5(text.encode()).hexdigest()
    cached_embedding = cache.get(hash_key)
    if cached_embedding:
        return eval(cached_embedding.decode("utf-8"))
    embedding = embedder([text])[0]
    cache.set(hash_key, str(embedding))
    return embedding

# ========== 示例数据和主函数 ==========
def create_sample_texts() -> Dict[str, List[str]]:
    """生成不同内容类型的样例文本"""
    return {
        "entity_name": ["Entity name example 1", "Entity name example 2"],
        "entity_description": ["This is a description of an entity.", "Another entity description."],
        "text": ["Full text content example.", "Another full text content."],
        "text_block": ["Text block example.", "Another text block."]
    }

async def main():
    # 创建不同内容类型的集合
    collections = {
        "entity_name": LanceDBVectorStore("entity_name_collection"),
        "entity_description": LanceDBVectorStore("entity_description_collection"),
        "text": LanceDBVectorStore("text_collection"),
        "text_block": LanceDBVectorStore("text_block_collection")
    }

    # 连接每个集合
    for store in collections.values():
        store.connect()

    # 选择嵌入模型（可以是 EmbedModelType.FASTTEXT、EmbedModelType.BERT、EmbedModelType.OPENAI 或 EmbedModelType.MOCK）
    embed_model_type = EmbedModelType.MOCK  # 假设这里选择 OpenAI 嵌入
    embedder = load_embedder(embed_model_type)

    # 创建示例数据并批量加载
    sample_texts = create_sample_texts()
    tasks = [
        batch_embed_and_load_async(texts, content_type, collections[content_type], embedder, batch_size=10)
        for content_type, texts in sample_texts.items()
    ]
    await asyncio.gather(*tasks)

    # 执行查询，搜索实体名称集合中的相似文档
    query_text = "entity name"
    query_embedding = await async_get_embedding(query_text, embedder)
    print("\nSearching for similar 'entity_name' documents:")
    results = collections["entity_name"].similarity_search(query_embedding, k=2)
    for result in results:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
