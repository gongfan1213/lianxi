#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyDE RAG演示程序
基于假设性文档嵌入的检索增强生成系统
"""

import os
import numpy as np
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from openai import OpenAI

# 配置OpenAI客户端（使用本地模型）
client = OpenAI(
    base_url="http://localhost:11434/v1",  # 本地Ollama服务
    api_key="ollama"  # 本地服务不需要真实API密钥
)

@dataclass
class DocumentChunk:
    """文档块数据结构"""
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class SimpleVectorStore:
    """简单的向量存储实现"""
    
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []
    
    def add_item(self, text: str, embedding: List[float], metadata: Optional[Dict] = None):
        """添加项目到向量存储"""
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """相似性搜索"""
        if not self.vectors:
            return []
        
        query_vector = np.array(query_embedding)
        similarities = []
        
        for i, vector in enumerate(self.vectors):
            # 计算余弦相似度
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            similarities.append((i, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })
        
        return results

def create_embeddings(texts: List[str], model: str = "llama2") -> List[List[float]]:
    """创建文本嵌入"""
    if not texts:
        return []
    
    all_embeddings = []
    
    for text in texts:
        try:
            # 使用本地Ollama模型生成嵌入
            response = client.embeddings.create(
                model=model,
                input=text
            )
            embedding = response.data[0].embedding
            all_embeddings.append(embedding)
        except Exception as e:
            print(f"嵌入生成失败: {e}")
            # 生成随机嵌入作为备用
            embedding = np.random.rand(384).tolist()
            all_embeddings.append(embedding)
    
    return all_embeddings

def generate_hypothetical_document(query: str, desired_length: int = 500) -> str:
    """生成假设性文档"""
    system_prompt = f"""你是一个专业的文档创建者。
给定一个问题，生成一个详细回答该问题的文档。
文档应该大约{desired_length}个字符，并提供深入、信息丰富的答案。
写得像来自该主题权威来源的文档。包含具体细节、事实和解释。
不要提及这是假设性文档 - 直接写内容。"""

    user_prompt = f"问题: {query}\n\n生成一个完全回答这个问题的文档:"
    
    try:
        response = client.chat.completions.create(
            model="llama2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"假设性文档生成失败: {e}")
        # 返回一个简单的假设性文档
        return f"这是一个关于'{query}'的详细文档。它包含了相关的信息、事实和解释，旨在全面回答用户的问题。"

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """文本分块"""
    chunks = []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "start_pos": i,
                    "end_pos": i + len(chunk_text)
                }
            })
    
    return chunks

def generate_response(query: str, relevant_chunks: List[Dict]) -> str:
    """生成最终响应"""
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
    
    try:
        response = client.chat.completions.create(
            model="llama2",
            messages=[
                {"role": "system", "content": "你是一个有用的助手。基于提供的上下文回答问题。"},
                {"role": "user", "content": f"上下文:\n{context}\n\n问题: {query}"}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"响应生成失败: {e}")
        return f"基于检索到的内容，我可以回答您的问题：{query}。相关信息包括：{context[:200]}..."

def hyde_rag(query: str, vector_store: SimpleVectorStore, k: int = 5) -> Dict:
    """HyDE RAG实现"""
    print(f"\n=== 使用HyDE处理查询: {query} ===\n")
    
    # 步骤1：生成假设性文档
    print("生成假设性文档...")
    hypothetical_doc = generate_hypothetical_document(query)
    print(f"生成了{len(hypothetical_doc)}字符的假设性文档")
    
    # 步骤2：为假设性文档创建嵌入
    print("为假设性文档创建嵌入...")
    hypothetical_embedding = create_embeddings([hypothetical_doc])[0]
    
    # 步骤3：基于假设性文档检索相似块
    print(f"检索{k}个最相似的块...")
    retrieved_chunks = vector_store.similarity_search(hypothetical_embedding, k=k)
    
    # 步骤4：生成最终响应
    print("生成最终响应...")
    response = generate_response(query, retrieved_chunks)
    
    return {
        "query": query,
        "hypothetical_document": hypothetical_doc,
        "retrieved_chunks": retrieved_chunks,
        "response": response
    }

def standard_rag(query: str, vector_store: SimpleVectorStore, k: int = 5) -> Dict:
    """标准RAG实现"""
    print(f"\n=== 使用标准RAG处理查询: {query} ===\n")
    
    # 步骤1：为查询创建嵌入
    print("为查询创建嵌入...")
    query_embedding = create_embeddings([query])[0]
    
    # 步骤2：基于查询嵌入检索相似块
    print(f"检索{k}个最相似的块...")
    retrieved_chunks = vector_store.similarity_search(query_embedding, k=k)
    
    # 步骤3：生成最终响应
    print("生成最终响应...")
    response = generate_response(query, retrieved_chunks)
    
    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "response": response
    }

def create_sample_documents() -> List[str]:
    """创建示例文档"""
    documents = [
        """
        人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。
        这些任务包括学习、推理、问题解决、感知和语言理解。AI技术已经在各个领域得到广泛应用，
        包括医疗保健、金融、教育、交通和娱乐。机器学习是AI的一个重要子领域，它使计算机能够
        从数据中学习并改进性能，而无需明确编程。
        """,
        
        """
        机器学习算法可以分为三大类：监督学习、无监督学习和强化学习。监督学习使用标记的训练数据
        来学习输入和输出之间的映射关系。无监督学习在没有标记数据的情况下发现数据中的模式。
        强化学习通过与环境交互来学习最优策略。深度学习是机器学习的一个子集，使用多层神经网络
        来处理复杂的数据模式。
        """,
        
        """
        自然语言处理（NLP）是AI的一个分支，专注于计算机理解和生成人类语言的能力。NLP技术包括
        文本分析、机器翻译、情感分析、问答系统和聊天机器人。近年来，大型语言模型如GPT和BERT
        在NLP任务中取得了突破性进展，能够生成高质量的人类语言文本并理解复杂的语言结构。
        """,
        
        """
        计算机视觉是AI的另一个重要分支，致力于使计算机能够理解和解释视觉信息。计算机视觉技术
        包括图像识别、物体检测、人脸识别、医学图像分析和自动驾驶。卷积神经网络（CNN）是计算机
        视觉中最常用的深度学习架构，特别适合处理图像数据。
        """,
        
        """
        人工智能的伦理考虑包括偏见和公平性、透明度和可解释性、隐私和数据保护、问责制和责任感。
        确保AI系统公平、非歧视性且不会延续训练数据中存在的现有偏见至关重要。透明度对于建立
        信任和问责制也很重要，而隐私保护则需要负责任的数据处理和隐私保护技术。
        """
    ]
    
    return [doc.strip() for doc in documents]

def process_documents(documents: List[str]) -> SimpleVectorStore:
    """处理文档并创建向量存储"""
    print("处理文档并创建向量存储...")
    
    vector_store = SimpleVectorStore()
    
    for i, doc in enumerate(documents):
        # 分块处理文档
        chunks = chunk_text(doc, chunk_size=500, overlap=100)
        
        for j, chunk in enumerate(chunks):
            # 为每个块创建嵌入
            embeddings = create_embeddings([chunk["text"]])
            if embeddings:
                vector_store.add_item(
                    text=chunk["text"],
                    embedding=embeddings[0],
                    metadata={
                        "doc_id": i,
                        "chunk_id": j,
                        **chunk["metadata"]
                    }
                )
    
    print(f"向量存储创建完成，包含{len(vector_store.texts)}个块")
    return vector_store

def visualize_comparison(query: str, hyde_result: Dict, standard_result: Dict):
    """可视化比较结果"""
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    # 查询
    axs[0].text(0.5, 0.5, f"查询:\n\n{query}", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, wrap=True)
    axs[0].set_title("用户查询")
    axs[0].axis('off')
    
    # 假设性文档
    hypothetical_doc = hyde_result["hypothetical_document"]
    shortened_doc = hypothetical_doc[:300] + "..." if len(hypothetical_doc) > 300 else hypothetical_doc
    axs[1].text(0.5, 0.5, f"假设性文档:\n\n{shortened_doc}", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=10, wrap=True)
    axs[1].set_title("HyDE生成的假设性文档")
    axs[1].axis('off')
    
    # 检索结果比较
    hyde_chunks = [chunk["text"][:80] + "..." for chunk in hyde_result["retrieved_chunks"]]
    std_chunks = [chunk["text"][:80] + "..." for chunk in standard_result["retrieved_chunks"]]
    
    comparison_text = "HyDE检索结果:\n\n"
    for i, chunk in enumerate(hyde_chunks):
        comparison_text += f"{i+1}. {chunk}\n\n"
    
    comparison_text += "\n标准RAG检索结果:\n\n"
    for i, chunk in enumerate(std_chunks):
        comparison_text += f"{i+1}. {chunk}\n\n"
    
    axs[2].text(0.5, 0.5, comparison_text, 
                horizontalalignment='center', verticalalignment='center',
                fontsize=8, wrap=True)
    axs[2].set_title("检索结果比较")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def compare_responses(query: str, hyde_response: str, standard_response: str) -> str:
    """比较两种方法的响应"""
    comparison = f"""
查询: {query}

HyDE RAG响应:
{hyde_response}

标准RAG响应:
{standard_response}

比较分析:
HyDE方法通过生成假设性文档来增强查询的语义表达，这通常能够：
1. 提供更丰富的上下文信息
2. 更好地匹配文档的语言风格
3. 捕获更复杂的语义关系

标准RAG方法直接使用原始查询进行检索，优点是：
1. 计算成本更低
2. 响应速度更快
3. 实现更简单

选择哪种方法取决于具体的应用场景和性能要求。
"""
    return comparison

def main():
    """主函数"""
    print("HyDE RAG演示程序")
    print("=" * 50)
    
    # 创建示例文档
    documents = create_sample_documents()
    print(f"创建了{len(documents)}个示例文档")
    
    # 处理文档并创建向量存储
    vector_store = process_documents(documents)
    
    # 测试查询
    test_queries = [
        "什么是机器学习？",
        "人工智能的伦理考虑有哪些？",
        "自然语言处理的主要技术是什么？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"测试查询 {i}: {query}")
        print(f"{'='*60}")
        
        # 运行HyDE RAG
        hyde_result = hyde_rag(query, vector_store)
        
        # 运行标准RAG
        standard_result = standard_rag(query, vector_store)
        
        # 显示结果
        print("\n" + "="*50)
        print("HyDE RAG响应:")
        print("="*50)
        print(hyde_result["response"])
        
        print("\n" + "="*50)
        print("标准RAG响应:")
        print("="*50)
        print(standard_result["response"])
        
        # 比较结果
        comparison = compare_responses(query, hyde_result["response"], standard_result["response"])
        print("\n" + "="*50)
        print("方法比较:")
        print("="*50)
        print(comparison)
        
        # 可视化结果
        try:
            visualize_comparison(query, hyde_result, standard_result)
        except Exception as e:
            print(f"可视化失败: {e}")
        
        # 等待用户输入继续
        if i < len(test_queries):
            input("\n按回车键继续下一个查询...")
    
    print("\n" + "="*60)
    print("演示完成！")
    print("HyDE RAG通过生成假设性文档来提升检索质量。")
    print("="*60)

if __name__ == "__main__":
    main() 