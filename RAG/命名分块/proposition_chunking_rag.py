#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命题分块（Proposition Chunking）RAG系统实现

这个文件实现了一个完整的命题分块RAG系统，用于提高检索增强生成的准确性。
与传统分块方法不同，命题分块将文档分解为原子性的、自包含的事实陈述，
从而提供更精确的检索结果。

主要特点：
1. 将文档分解为原子性的事实陈述
2. 创建更小、更精确的检索单元
3. 实现查询与相关内容之间的精确匹配
4. 过滤低质量或不完整的命题

作者：AI助手
日期：2024年
"""

import os
import numpy as np
import json
import re
import requests
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# 配置部分
# ============================================================================

@dataclass
class APIConfig:
    """API配置类"""
    api_key: str = "a2xxxxxxxxxxa"  # 你的API KEY
    base_url: str = "hxxxxxxxxt"
    model_name: str = "gpt-4-o-mini"  # 使用的模型名称
    api_version: str = "2024-05-01-preview"
    embedding_model: str = "text-embedding-3-small"  # 嵌入模型

# 全局配置
config = APIConfig()

# ============================================================================
# API调用函数
# ============================================================================

def call_llm_api(messages: List[Dict], temperature: float = 0.0) -> str:
    """
    调用LLM API进行文本生成
    
    Args:
        messages: 消息列表，包含system和user消息
        temperature: 生成温度，控制随机性
        
    Returns:
        str: 生成的文本响应
    """
    url = f"{config.base_url}/deployments/{config.model_name}/chat/completions?api-version={config.api_version}"
    headers = {
        'Content-Type': 'application/json',
        'api-key': config.api_key
    }
    
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            print(f"API调用错误: {response.status_code}, {response.text}")
            return ""
    except Exception as e:
        print(f"API调用异常: {e}")
        return ""

def call_embedding_api(texts: List[str]) -> List[List[float]]:
    """
    调用嵌入API生成文本向量
    
    Args:
        texts: 要嵌入的文本列表
        
    Returns:
        List[List[float]]: 文本向量列表
    """
    url = f"{config.base_url}/deployments/{config.embedding_model}/embeddings?api-version={config.api_version}"
    headers = {
        'Content-Type': 'application/json',
        'api-key': config.api_key
    }
    
    # 处理单个文本的情况
    if isinstance(texts, str):
        texts = [texts]
    
    payload = {
        "input": texts
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        else:
            print(f"嵌入API调用错误: {response.status_code}, {response.text}")
            return []
    except Exception as e:
        print(f"嵌入API调用异常: {e}")
        return []

# ============================================================================
# 文档处理函数
# ============================================================================

def extract_text_from_file(file_path: str) -> str:
    """
    从文件中提取文本内容
    
    Args:
        file_path: 文件路径（支持txt、md等文本文件）
        
    Returns:
        str: 提取的文本内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"文件读取错误: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict]:
    """
    将文本分割成重叠的块
    
    Args:
        text: 输入文本
        chunk_size: 每个块的大小（字符数）
        overlap: 块之间的重叠字符数
        
    Returns:
        List[Dict]: 包含文本和元数据的块列表
    """
    chunks = []
    
    # 按指定大小和重叠度遍历文本
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:  # 确保不添加空块
            chunks.append({
                "text": chunk,
                "chunk_id": len(chunks) + 1,
                "start_char": i,
                "end_char": i + len(chunk)
            })
    
    print(f"创建了 {len(chunks)} 个文本块")
    return chunks

# ============================================================================
# 向量存储实现
# ============================================================================

class SimpleVectorStore:
    """
    简单的向量存储实现，使用NumPy进行相似度计算
    """
    
    def __init__(self):
        """初始化向量存储"""
        self.vectors = []  # 存储向量
        self.texts = []    # 存储文本
        self.metadata = [] # 存储元数据
    
    def add_item(self, text: str, embedding: List[float], metadata: Optional[Dict] = None):
        """
        向向量存储添加单个项目
        
        Args:
            text: 文本内容
            embedding: 嵌入向量
            metadata: 元数据字典
        """
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def add_items(self, texts: List[str], embeddings: List[List[float]], 
                  metadata_list: Optional[List[Dict]] = None):
        """
        向向量存储添加多个项目
        
        Args:
            texts: 文本列表
            embeddings: 嵌入向量列表
            metadata_list: 元数据列表
        """
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(texts))]
        
        for text, embedding, metadata in zip(texts, embeddings, metadata_list):
            self.add_item(text, embedding, metadata)
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        基于查询向量进行相似度搜索
        
        Args:
            query_embedding: 查询向量
            k: 返回结果数量
            
        Returns:
            List[Dict]: 最相似的k个项目
        """
        if not self.vectors:
            return []
        
        query_vector = np.array(query_embedding)
        
        # 计算余弦相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 余弦相似度计算
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 收集前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })
        
        return results

# ============================================================================
# 命题生成和处理
# ============================================================================

def generate_propositions(chunk: Dict) -> List[str]:
    """
    从文本块生成原子性的、自包含的命题
    
    Args:
        chunk: 包含内容和元数据的文本块
        
    Returns:
        List[str]: 生成的命题列表
    """
    system_prompt = """请将以下文本分解为简单、自包含的命题。
确保每个命题满足以下标准：

1. 表达单一事实：每个命题应该陈述一个具体的事实或主张
2. 无需上下文即可理解：命题应该是自包含的，无需额外上下文即可理解
3. 使用完整名称而非代词：避免代词或模糊引用，使用完整的实体名称
4. 包含相关日期/限定词：如果适用，包含必要的日期、时间和限定词以使事实精确
5. 包含一个主谓关系：专注于单一主语及其对应的动作或属性，避免连词或多重从句

仅输出命题列表，不要包含任何额外的文本或解释。"""

    user_prompt = f"要转换为命题的文本：\n\n{chunk['text']}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_llm_api(messages, temperature=0)
    
    # 从响应中提取命题
    raw_propositions = response.strip().split('\n')
    
    # 清理命题（移除编号、项目符号等）
    clean_propositions = []
    for prop in raw_propositions:
        # 移除编号（1.、2.等）和项目符号
        cleaned = re.sub(r'^\s*(\d+\.|\-|\*)\s*', '', prop).strip()
        if cleaned and len(cleaned) > 10:  # 简单过滤空或很短的命题
            clean_propositions.append(cleaned)
    
    return clean_propositions

def evaluate_proposition(proposition: str, original_text: str) -> Dict:
    """
    评估命题质量，基于准确性、清晰度、完整性和简洁性
    
    Args:
        proposition: 要评估的命题
        original_text: 用于比较的原始文本
        
    Returns:
        Dict: 每个评估维度的分数
    """
    system_prompt = """你是评估从文本中提取的命题质量的专家。
对给定命题按以下标准评分（1-10分）：

- 准确性：命题在多大程度上反映了原始文本中的信息
- 清晰度：无需额外上下文即可理解命题的难易程度
- 完整性：命题是否包含必要的细节（日期、限定词等）
- 简洁性：命题是否简洁而不丢失重要信息

响应必须是有效的JSON格式，包含每个标准的数值分数：
{"accuracy": X, "clarity": X, "completeness": X, "conciseness": X}"""

    user_prompt = f"""命题：{proposition}

原始文本：{original_text}

请以JSON格式提供评估分数。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_llm_api(messages, temperature=0)
    
    # 解析JSON响应
    try:
        scores = json.loads(response.strip())
        return scores
    except json.JSONDecodeError:
        # 如果JSON解析失败，返回默认分数
        return {
            "accuracy": 5,
            "clarity": 5,
            "completeness": 5,
            "conciseness": 5
        }

# ============================================================================
# 完整的命题处理管道
# ============================================================================

def process_document_into_propositions(file_path: str, chunk_size: int = 800, 
                                     chunk_overlap: int = 100,
                                     quality_thresholds: Optional[Dict] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    将文档处理为质量检查的命题
    
    Args:
        file_path: 文件路径
        chunk_size: 每个块的大小（字符数）
        chunk_overlap: 块之间的重叠字符数
        quality_thresholds: 命题质量阈值
        
    Returns:
        Tuple[List[Dict], List[Dict]]: 原始块和命题块
    """
    # 设置默认质量阈值
    if quality_thresholds is None:
        quality_thresholds = {
            "accuracy": 7,
            "clarity": 7,
            "completeness": 7,
            "conciseness": 7
        }
    
    # 从文件提取文本
    text = extract_text_from_file(file_path)
    if not text:
        print("无法提取文本内容")
        return [], []
    
    # 从提取的文本创建块
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # 初始化存储所有命题的列表
    all_propositions = []
    
    print("从块生成命题...")
    for i, chunk in enumerate(chunks):
        print(f"处理块 {i+1}/{len(chunks)}...")
        
        # 为当前块生成命题
        chunk_propositions = generate_propositions(chunk)
        print(f"生成了 {len(chunk_propositions)} 个命题")
        
        # 处理每个生成的命题
        for prop in chunk_propositions:
            proposition_data = {
                "text": prop,
                "source_chunk_id": chunk["chunk_id"],
                "source_text": chunk["text"]
            }
            all_propositions.append(proposition_data)
    
    # 评估生成命题的质量
    print("\n评估命题质量...")
    quality_propositions = []
    
    for i, prop in enumerate(all_propositions):
        if i % 10 == 0:  # 每10个命题更新一次状态
            print(f"评估命题 {i+1}/{len(all_propositions)}...")
        
        # 评估当前命题的质量
        scores = evaluate_proposition(prop["text"], prop["source_text"])
        prop["quality_scores"] = scores
        
        # 检查命题是否通过质量阈值
        passes_quality = True
        for metric, threshold in quality_thresholds.items():
            if scores.get(metric, 0) < threshold:
                passes_quality = False
                break
        
        if passes_quality:
            quality_propositions.append(prop)
        else:
            print(f"命题未通过质量检查：{prop['text'][:50]}...")
    
    print(f"\n质量过滤后保留了 {len(quality_propositions)}/{len(all_propositions)} 个命题")
    
    return chunks, quality_propositions

# ============================================================================
# 向量存储构建
# ============================================================================

def build_vector_stores(chunks: List[Dict], propositions: List[Dict]) -> Tuple[SimpleVectorStore, SimpleVectorStore]:
    """
    为基于块和基于命题的方法构建向量存储
    
    Args:
        chunks: 原始文档块
        propositions: 质量过滤的命题
        
    Returns:
        Tuple[SimpleVectorStore, SimpleVectorStore]: 块和命题向量存储
    """
    # 为块创建向量存储
    chunk_store = SimpleVectorStore()
    
    # 提取块文本并创建嵌入
    chunk_texts = [chunk["text"] for chunk in chunks]
    print(f"为 {len(chunk_texts)} 个块创建嵌入...")
    chunk_embeddings = call_embedding_api(chunk_texts)
    
    # 将块添加到向量存储，包含元数据
    chunk_metadata = [{"chunk_id": chunk["chunk_id"], "type": "chunk"} for chunk in chunks]
    chunk_store.add_items(chunk_texts, chunk_embeddings, chunk_metadata)
    
    # 为命题创建向量存储
    prop_store = SimpleVectorStore()
    
    # 提取命题文本并创建嵌入
    prop_texts = [prop["text"] for prop in propositions]
    print(f"为 {len(prop_texts)} 个命题创建嵌入...")
    prop_embeddings = call_embedding_api(prop_texts)
    
    # 将命题添加到向量存储，包含元数据
    prop_metadata = [
        {
            "type": "proposition",
            "source_chunk_id": prop["source_chunk_id"],
            "quality_scores": prop["quality_scores"]
        }
        for prop in propositions
    ]
    prop_store.add_items(prop_texts, prop_embeddings, prop_metadata)
    
    return chunk_store, prop_store

# ============================================================================
# 查询和检索函数
# ============================================================================

def retrieve_from_store(query: str, vector_store: SimpleVectorStore, k: int = 5) -> List[Dict]:
    """
    基于查询从向量存储检索相关项目
    
    Args:
        query: 用户查询
        vector_store: 要搜索的向量存储
        k: 要检索的结果数量
        
    Returns:
        List[Dict]: 检索到的项目，包含分数和元数据
    """
    # 创建查询嵌入
    query_embedding = call_embedding_api(query)
    if not query_embedding:
        return []
    
    # 在向量存储中搜索最相似的k个项目
    results = vector_store.similarity_search(query_embedding[0], k=k)
    
    return results

def compare_retrieval_approaches(query: str, chunk_store: SimpleVectorStore, 
                                prop_store: SimpleVectorStore, k: int = 5) -> Dict:
    """
    比较基于块和基于命题的检索方法
    
    Args:
        query: 用户查询
        chunk_store: 基于块的向量存储
        prop_store: 基于命题的向量存储
        k: 从每个存储检索的结果数量
        
    Returns:
        Dict: 比较结果
    """
    print(f"\n=== 查询：{query} ===")
    
    # 从基于命题的向量存储检索结果
    print("\n使用基于命题的方法检索...")
    prop_results = retrieve_from_store(query, prop_store, k)
    
    # 从基于块的向量存储检索结果
    print("使用基于块的方法检索...")
    chunk_results = retrieve_from_store(query, chunk_store, k)
    
    # 显示基于命题的结果
    print("\n=== 基于命题的结果 ===")
    for i, result in enumerate(prop_results):
        print(f"{i+1}) {result['text']} (分数：{result['similarity']:.4f})")
    
    # 显示基于块的结果
    print("\n=== 基于块的结果 ===")
    for i, result in enumerate(chunk_results):
        # 截断文本以保持输出可管理
        truncated_text = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
        print(f"{i+1}) {truncated_text} (分数：{result['similarity']:.4f})")
    
    # 返回比较结果
    return {
        "query": query,
        "proposition_results": prop_results,
        "chunk_results": chunk_results
    }

# ============================================================================
# 响应生成和评估
# ============================================================================

def generate_response(query: str, results: List[Dict], result_type: str = "proposition") -> str:
    """
    基于检索结果生成响应
    
    Args:
        query: 用户查询
        results: 检索到的项目
        result_type: 结果类型（'proposition' 或 'chunk'）
        
    Returns:
        str: 生成的响应
    """
    # 将检索到的文本组合成单个上下文字符串
    context = "\n\n".join([result["text"] for result in results])
    
    system_prompt = f"""你是一个基于检索信息回答问题的AI助手。
你的答案应该基于从知识库检索到的以下{result_type}。
如果检索到的信息无法回答问题，请承认这一限制。"""

    user_prompt = f"""查询：{query}

检索到的{result_type}：
{context}

请基于检索到的信息回答查询。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_llm_api(messages, temperature=0.2)
    return response

def evaluate_responses(query: str, prop_response: str, chunk_response: str, 
                      reference_answer: Optional[str] = None) -> str:
    """
    评估和比较两种方法的响应
    
    Args:
        query: 用户查询
        prop_response: 基于命题方法的响应
        chunk_response: 基于块方法的响应
        reference_answer: 用于比较的参考答案
        
    Returns:
        str: 评估分析
    """
    system_prompt = """你是信息检索系统的专家评估员。
比较对同一查询的两个响应，一个来自基于命题的检索，
另一个来自基于块的检索。

基于以下标准评估它们：
1. 准确性：哪个响应提供了更准确的信息？
2. 相关性：哪个响应更好地解决了特定查询？
3. 简洁性：哪个响应在保持完整性的同时更简洁？
4. 清晰度：哪个响应更容易理解？

具体说明每种方法的优缺点。"""

    user_prompt = f"""查询：{query}

基于命题检索的响应：
{prop_response}

基于块检索的响应：
{chunk_response}"""

    if reference_answer:
        user_prompt += f"""

参考答案（用于事实检查）：
{reference_answer}"""

    user_prompt += """

请详细比较这两个响应，突出哪种方法表现更好以及原因。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_llm_api(messages, temperature=0)
    return response

# ============================================================================
# 完整的端到端评估管道
# ============================================================================

def generate_overall_analysis(results: List[Dict]) -> str:
    """
    生成基于命题与块方法的整体分析
    
    Args:
        results: 每个测试查询的结果
        
    Returns:
        str: 整体分析
    """
    system_prompt = """你是信息检索系统的专家。
基于多个测试查询，提供比较基于命题的检索与基于块的检索
用于RAG（检索增强生成）系统的整体分析。

重点关注：
1. 基于命题的检索何时表现更好
2. 基于块的检索何时表现更好
3. 每种方法的整体优缺点
4. 何时使用每种方法的建议"""

    # 为每个查询创建评估摘要
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"查询 {i+1}：{result['query']}\n"
        evaluations_summary += f"评估摘要：{result['evaluation'][:200]}...\n\n"

    user_prompt = f"""基于以下对基于命题与基于块检索在{len(results)}个查询中的评估，
提供这两种方法的整体分析：

{evaluations_summary}

请提供基于命题与基于块检索用于RAG系统的相对优缺点综合分析。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_llm_api(messages, temperature=0)
    return response

def run_proposition_chunking_evaluation(file_path: str, test_queries: List[str], 
                                       reference_answers: Optional[List[str]] = None) -> Dict:
    """
    运行命题分块与标准分块的完整评估
    
    Args:
        file_path: 文件路径
        test_queries: 测试查询列表
        reference_answers: 查询的参考答案
        
    Returns:
        Dict: 评估结果
    """
    print("=== 开始命题分块评估 ===\n")
    
    # 将文档处理为命题和块
    chunks, propositions = process_document_into_propositions(file_path)
    
    # 为块和命题构建向量存储
    chunk_store, prop_store = build_vector_stores(chunks, propositions)
    
    # 初始化存储每个查询结果的列表
    results = []
    
    # 为每个查询运行测试
    for i, query in enumerate(test_queries):
        print(f"\n\n=== 测试查询 {i+1}/{len(test_queries)} ===")
        print(f"查询：{query}")
        
        # 从基于块和基于命题的方法获取检索结果
        retrieval_results = compare_retrieval_approaches(query, chunk_store, prop_store)
        
        # 基于检索到的基于命题的结果生成响应
        print("\n基于基于命题的结果生成响应...")
        prop_response = generate_response(
            query,
            retrieval_results["proposition_results"],
            "proposition"
        )
        
        # 基于检索到的基于块的结果生成响应
        print("基于基于块的结果生成响应...")
        chunk_response = generate_response(
            query,
            retrieval_results["chunk_results"],
            "chunk"
        )
        
        # 获取参考答案（如果可用）
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        
        # 评估生成的响应
        print("\n评估响应...")
        evaluation = evaluate_responses(query, prop_response, chunk_response, reference)
        
        # 编译当前查询的结果
        query_result = {
            "query": query,
            "proposition_results": retrieval_results["proposition_results"],
            "chunk_results": retrieval_results["chunk_results"],
            "proposition_response": prop_response,
            "chunk_response": chunk_response,
            "reference_answer": reference,
            "evaluation": evaluation
        }
        
        # 将结果添加到整体结果列表
        results.append(query_result)
        
        # 打印当前查询的响应和评估
        print("\n=== 基于命题的响应 ===")
        print(prop_response)
        
        print("\n=== 基于块的响应 ===")
        print(chunk_response)
        
        print("\n=== 评估 ===")
        print(evaluation)
    
    # 生成评估的整体分析
    print("\n\n=== 生成整体分析 ===")
    overall_analysis = generate_overall_analysis(results)
    print("\n" + overall_analysis)
    
    # 返回评估结果、整体分析以及命题和块的数量
    return {
        "results": results,
        "overall_analysis": overall_analysis,
        "proposition_count": len(propositions),
        "chunk_count": len(chunks)
    }

# ============================================================================
# 主函数和示例用法
# ============================================================================

def create_sample_text_file():
    """创建示例文本文件用于测试"""
    sample_text = """
人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。

AI的主要类型包括：
1. 狭义AI：设计用于执行特定任务的AI系统
2. 通用AI：能够执行任何人类可以完成的智力任务的AI系统

机器学习是AI的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。

深度学习是机器学习的一个分支，使用神经网络来模拟人脑的工作方式。

AI在各个领域都有应用，包括：
- 医疗保健：疾病诊断、药物发现
- 金融：风险评估、欺诈检测
- 交通：自动驾驶汽车
- 教育：个性化学习

AI的发展带来了伦理考虑，包括：
- 偏见和公平性
- 隐私保护
- 就业影响
- 安全性和控制

确保AI系统的透明度和可解释性对于建立信任至关重要。

AI的未来发展需要平衡技术进步与伦理考虑。
    """
    
    with open("sample_ai_text.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    print("已创建示例文本文件：sample_ai_text.txt")

def main():
    """主函数 - 运行命题分块评估示例"""
    print("=== 命题分块RAG系统演示 ===\n")
    
    # 创建示例文本文件
    create_sample_text_file()
    
    # 文件路径
    file_path = "sample_ai_text.txt"
    
    # 定义测试查询
    test_queries = [
        "什么是人工智能？",
        "AI的主要类型有哪些？",
        "AI的伦理考虑包括什么？",
        "机器学习与深度学习的关系是什么？"
    ]
    
    # 参考答案（可选）
    reference_answers = [
        "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
        "AI的主要类型包括狭义AI（执行特定任务）和通用AI（执行任何人类智力任务）。",
        "AI的伦理考虑包括偏见和公平性、隐私保护、就业影响、安全性和控制。",
        "机器学习是AI的子集，深度学习是机器学习的分支，使用神经网络模拟人脑工作方式。"
    ]
    
    try:
        # 运行评估
        evaluation_results = run_proposition_chunking_evaluation(
            file_path=file_path,
            test_queries=test_queries,
            reference_answers=reference_answers
        )
        
        # 打印最终统计信息
        print("\n" + "="*50)
        print("最终统计信息")
        print("="*50)
        print(f"处理的块数量：{evaluation_results['chunk_count']}")
        print(f"生成的命题数量：{evaluation_results['proposition_count']}")
        print(f"测试查询数量：{len(test_queries)}")
        
        # 保存结果到文件
        with open("evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        print("\n评估结果已保存到 evaluation_results.json")
        
    except Exception as e:
        print(f"运行评估时发生错误：{e}")

if __name__ == "__main__":
    main() 