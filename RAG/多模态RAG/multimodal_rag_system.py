#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态RAG系统实现

这个文件实现了一个完整的多模态RAG系统，能够同时处理文本和图像内容。
系统从PDF文档中提取文本和图像，为图像生成描述，并将所有内容整合到统一的
知识库中进行检索和问答。

主要功能：
1. 从PDF文档提取文本和图像
2. 使用LLaVA模型为图像生成描述
3. 将文本和图像描述统一向量化
4. 支持多模态查询和响应生成
5. 与纯文本RAG进行性能对比

作者：AI助手
日期：2024年
"""

import os
import io
import numpy as np
import json
import base64
import re
import tempfile
import shutil
import requests
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

# ============================================================================
# 配置部分
# ============================================================================

@dataclass
class APIConfig:
    """API配置类"""
    api_key: str = "xxxx"
    base_url: str = "https:xxxxxxxst"
    model_name: str = "gpt-4-o-mini"
    api_version: str = "2024-05-01-preview"
    embedding_model: str = "text-embedding-3-small"
    vision_model: str = "llava-hf/llava-1.5-7b-hf"  # LLaVA多模态模型

# 全局配置
config = APIConfig()

# ============================================================================
# API调用函数
# ============================================================================

def call_llm_api(messages: List[Dict], temperature: float = 0.0, max_tokens: int = 4000) -> str:
    """
    调用LLM API进行文本生成
    
    Args:
        messages: 消息列表
        temperature: 生成温度
        max_tokens: 最大token数
        
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
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            print(f"LLM API调用错误: {response.status_code}, {response.text}")
            return ""
    except Exception as e:
        print(f"LLM API调用异常: {e}")
        return ""

def call_vision_api(messages: List[Dict], max_tokens: int = 300) -> str:
    """
    调用视觉API进行图像分析
    
    Args:
        messages: 包含图像的消息列表
        max_tokens: 最大token数
        
    Returns:
        str: 生成的图像描述
    """
    url = f"{config.base_url}/deployments/{config.vision_model}/chat/completions?api-version={config.api_version}"
    headers = {
        'Content-Type': 'application/json',
        'api-key': config.api_key
    }
    
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            print(f"视觉API调用错误: {response.status_code}, {response.text}")
            return ""
    except Exception as e:
        print(f"视觉API调用异常: {e}")
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

def extract_content_from_pdf(pdf_path: str, output_dir: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    从PDF文件中提取文本和图像
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 图像保存目录
        
    Returns:
        Tuple[List[Dict], List[Dict]]: 文本数据和图像数据
    """
    # 创建临时目录用于保存图像
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.mkdtemp()
        output_dir = temp_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    text_data = []  # 存储提取的文本数据
    image_paths = []  # 存储提取的图像路径
    
    print(f"从 {pdf_path} 提取内容...")
    
    try:
        # 使用PyMuPDF提取内容
        import fitz  # PyMuPDF
        
        with fitz.open(pdf_path) as pdf_file:
            # 遍历PDF的每一页
            for page_number in range(len(pdf_file)):
                page = pdf_file[page_number]
                
                # 提取页面文本
                text = page.get_text().strip()
                if text:
                    text_data.append({
                        "content": text,
                        "metadata": {
                            "source": pdf_path,
                            "page": page_number + 1,
                            "type": "text"
                        }
                    })
                
                # 提取页面图像
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]  # 图像的XREF
                    base_image = pdf_file.extract_image(xref)
                    
                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # 保存图像到输出目录
                        img_filename = f"page_{page_number+1}_img_{img_index+1}.{image_ext}"
                        img_path = os.path.join(output_dir, img_filename)
                        
                        with open(img_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        image_paths.append({
                            "path": img_path,
                            "metadata": {
                                "source": pdf_path,
                                "page": page_number + 1,
                                "image_index": img_index + 1,
                                "type": "image"
                            }
                        })
        
        print(f"提取了 {len(text_data)} 个文本段落和 {len(image_paths)} 个图像")
        return text_data, image_paths
    
    except Exception as e:
        print(f"提取内容时发生错误: {e}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise

def chunk_text(text_data: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    将文本数据分割成重叠的块
    
    Args:
        text_data: 从PDF提取的文本数据
        chunk_size: 每个块的大小（字符数）
        overlap: 块之间的重叠字符数
        
    Returns:
        List[Dict]: 分块后的文本数据
    """
    chunked_data = []
    
    for item in text_data:
        text = item["content"]
        metadata = item["metadata"]
        
        # 如果文本太短，直接添加
        if len(text) < chunk_size / 2:
            chunked_data.append({
                "content": text,
                "metadata": metadata
            })
            continue
        
        # 创建重叠的块
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
        
        # 为每个块添加更新的元数据
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_count"] = len(chunks)
            
            chunked_data.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
    
    print(f"创建了 {len(chunked_data)} 个文本块")
    return chunked_data

# ============================================================================
# 图像处理函数
# ============================================================================

def encode_image(image_path: str) -> str:
    """
    将图像文件编码为base64
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        str: base64编码的图像
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode('utf-8')

def generate_image_caption(image_path: str) -> str:
    """
    使用LLaVA模型为图像生成描述
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        str: 生成的图像描述
    """
    # 检查文件是否存在且为图像
    if not os.path.exists(image_path):
        return "错误：图像文件未找到"
    
    try:
        # 打开并验证图像
        Image.open(image_path)
        
        # 将图像编码为base64
        base64_image = encode_image(image_path)
        
        # 创建API请求来生成描述
        messages = [
            {
                "role": "system",
                "content": "你是一个专门描述学术论文图像的助手。"
                "为图像提供详细的描述，捕捉关键信息。"
                "如果图像包含图表、表格或图表，请清楚地描述其内容和目的。"
                "你的描述应该针对未来人们询问这些内容时的检索进行优化。"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "详细描述这个图像，重点关注其学术内容："},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # 调用视觉API生成描述
        caption = call_vision_api(messages, max_tokens=300)
        return caption
    
    except Exception as e:
        return f"生成描述时发生错误: {str(e)}"

def process_images(image_paths: List[Dict]) -> List[Dict]:
    """
    处理所有图像并生成描述
    
    Args:
        image_paths: 提取的图像路径列表
        
    Returns:
        List[Dict]: 包含描述的图像数据
    """
    image_data = []
    
    print(f"为 {len(image_paths)} 个图像生成描述...")
    for i, img_item in enumerate(image_paths):
        print(f"处理图像 {i+1}/{len(image_paths)}...")
        img_path = img_item["path"]
        metadata = img_item["metadata"]
        
        # 为图像生成描述
        caption = generate_image_caption(img_path)
        
        # 将图像数据与描述添加到列表
        image_data.append({
            "content": caption,
            "metadata": metadata,
            "image_path": img_path
        })
    
    return image_data

# ============================================================================
# 向量存储实现
# ============================================================================

class MultiModalVectorStore:
    """
    多模态内容的简单向量存储实现
    """
    
    def __init__(self):
        """初始化向量存储"""
        self.vectors = []
        self.contents = []
        self.metadata = []
    
    def add_item(self, content: str, embedding: List[float], metadata: Optional[Dict] = None):
        """
        向向量存储添加单个项目
        
        Args:
            content: 内容（文本或图像描述）
            embedding: 嵌入向量
            metadata: 元数据
        """
        self.vectors.append(np.array(embedding))
        self.contents.append(content)
        self.metadata.append(metadata or {})
    
    def add_items(self, items: List[Dict], embeddings: List[List[float]]):
        """
        向向量存储添加多个项目
        
        Args:
            items: 内容项目列表
            embeddings: 嵌入向量列表
        """
        for item, embedding in zip(items, embeddings):
            self.add_item(
                content=item["content"],
                embedding=embedding,
                metadata=item.get("metadata", {})
            )
    
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
        
        # 使用余弦相似度计算相似性
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "content": self.contents[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })
        
        return results

# ============================================================================
# 完整的处理管道
# ============================================================================

def process_document(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[MultiModalVectorStore, Dict]:
    """
    处理文档用于多模态RAG
    
    Args:
        pdf_path: PDF文件路径
        chunk_size: 每个块的大小（字符数）
        chunk_overlap: 块之间的重叠字符数
        
    Returns:
        Tuple[MultiModalVectorStore, Dict]: 向量存储和文档信息
    """
    # 为提取的图像创建目录
    image_dir = "extracted_images"
    os.makedirs(image_dir, exist_ok=True)
    
    # 从PDF提取文本和图像
    text_data, image_paths = extract_content_from_pdf(pdf_path, image_dir)
    
    # 对提取的文本进行分块
    chunked_text = chunk_text(text_data, chunk_size, chunk_overlap)
    
    # 处理提取的图像以生成描述
    image_data = process_images(image_paths)
    
    # 合并所有内容项目（文本块和图像描述）
    all_items = chunked_text + image_data
    
    # 提取用于嵌入的内容
    contents = [item["content"] for item in all_items]
    
    # 为所有内容创建嵌入
    print("为所有内容创建嵌入...")
    embeddings = call_embedding_api(contents)
    
    # 构建向量存储并添加项目及其嵌入
    vector_store = MultiModalVectorStore()
    vector_store.add_items(all_items, embeddings)
    
    # 准备文档信息，包含文本块和图像描述的数量
    doc_info = {
        "text_count": len(chunked_text),
        "image_count": len(image_data),
        "total_items": len(all_items),
    }
    
    # 打印添加项目的摘要
    print(f"向向量存储添加了 {len(all_items)} 个项目（{len(chunked_text)} 个文本块，{len(image_data)} 个图像描述）")
    
    # 返回向量存储和文档信息
    return vector_store, doc_info

# ============================================================================
# 查询处理和响应生成
# ============================================================================

def query_multimodal_rag(query: str, vector_store: MultiModalVectorStore, k: int = 5) -> Dict:
    """
    查询多模态RAG系统
    
    Args:
        query: 用户查询
        vector_store: 包含文档内容的向量存储
        k: 检索结果数量
        
    Returns:
        Dict: 查询结果和生成的响应
    """
    print(f"\n=== 处理查询: {query} ===\n")
    
    # 为查询生成嵌入
    query_embedding = call_embedding_api(query)
    if not query_embedding:
        return {"error": "无法生成查询嵌入"}
    
    # 从向量存储检索相关内容
    results = vector_store.similarity_search(query_embedding[0], k=k)
    
    # 分离文本和图像结果
    text_results = [r for r in results if r["metadata"].get("type") == "text"]
    image_results = [r for r in results if r["metadata"].get("type") == "image"]
    
    print(f"检索到 {len(results)} 个相关项目（{len(text_results)} 个文本，{len(image_results)} 个图像描述）")
    
    # 使用检索到的内容生成响应
    response = generate_response(query, results)
    
    return {
        "query": query,
        "results": results,
        "response": response,
        "text_results_count": len(text_results),
        "image_results_count": len(image_results)
    }

def generate_response(query: str, results: List[Dict]) -> str:
    """
    基于查询和检索结果生成响应
    
    Args:
        query: 用户查询
        results: 检索到的内容
        
    Returns:
        str: 生成的响应
    """
    # 从检索结果格式化上下文
    context = ""
    
    for i, result in enumerate(results):
        # 确定内容类型（文本或图像描述）
        content_type = "文本" if result["metadata"].get("type") == "text" else "图像描述"
        # 从元数据获取页码
        page_num = result["metadata"].get("page", "未知")
        
        # 将内容类型和页码添加到上下文
        context += f"[第{page_num}页的{content_type}]\n"
        # 将实际内容添加到上下文
        context += result["content"]
        context += "\n\n"
    
    # 系统消息指导AI助手
    system_message = """你是一个专门回答包含文本和图像的文档问题的AI助手。
你获得了来自文档的相关文本段落和图像描述。使用这些信息为查询提供全面、准确的响应。
如果信息来自图像或图表，请在回答中提及这一点。
如果检索到的信息不能完全回答查询，请承认这些限制。"""

    # 包含查询和格式化上下文的用户消息
    user_message = f"""查询：{query}

检索到的内容：
{context}

请基于检索到的内容回答查询。
"""
    
    # 使用OpenAI API生成响应
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    response = call_llm_api(messages, temperature=0.1)
    return response

# ============================================================================
# 与纯文本RAG的对比评估
# ============================================================================

def build_text_only_store(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> MultiModalVectorStore:
    """
    构建纯文本向量存储用于对比
    
    Args:
        pdf_path: PDF文件路径
        chunk_size: 每个块的大小（字符数）
        chunk_overlap: 块之间的重叠字符数
        
    Returns:
        MultiModalVectorStore: 纯文本向量存储
    """
    # 从PDF提取文本（重用函数但忽略图像）
    text_data, _ = extract_content_from_pdf(pdf_path, None)
    
    # 对文本进行分块
    chunked_text = chunk_text(text_data, chunk_size, chunk_overlap)
    
    # 提取用于嵌入的内容
    contents = [item["content"] for item in chunked_text]
    
    # 创建嵌入
    print("为纯文本内容创建嵌入...")
    embeddings = call_embedding_api(contents)
    
    # 构建向量存储
    vector_store = MultiModalVectorStore()
    vector_store.add_items(chunked_text, embeddings)
    
    print(f"向纯文本向量存储添加了 {len(chunked_text)} 个文本项目")
    return vector_store

def compare_responses(query: str, mm_response: str, text_response: str, reference: Optional[str] = None) -> str:
    """
    比较多模态和纯文本响应
    
    Args:
        query: 用户查询
        mm_response: 多模态响应
        text_response: 纯文本响应
        reference: 参考答案
        
    Returns:
        str: 比较分析
    """
    # 评估者的系统提示
    system_prompt = """你是比较两个RAG系统的专家评估员：
1. 多模态RAG：从文本和图像描述中检索
2. 纯文本RAG：仅从文本中检索

基于以下标准评估哪个响应更好地回答查询：
- 准确性和正确性
- 信息的完整性
- 与查询的相关性
- 来自视觉元素的独特信息（对于多模态）"""

    # 包含查询和响应的用户提示
    user_prompt = f"""查询：{query}

多模态RAG响应：
{mm_response}

纯文本RAG响应：
{text_response}"""

    if reference:
        user_prompt += f"""

参考答案：
{reference}"""

        user_prompt += """

比较这些响应并解释哪个更好地回答查询以及原因。
注意多模态响应中来自图像的任何特定信息。
"""

    # 生成比较
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_llm_api(messages, temperature=0)
    return response

def generate_overall_analysis(results: List[Dict]) -> str:
    """
    生成多模态与纯文本RAG的整体分析
    
    Args:
        results: 每个查询的评估结果
        
    Returns:
        str: 整体分析
    """
    # 评估者的系统提示
    system_prompt = """你是RAG系统的专家评估员。基于多个测试查询提供比较
多模态RAG（文本+图像）与纯文本RAG的整体分析。

重点关注：
1. 多模态RAG优于纯文本RAG的查询类型
2. 整合图像信息的具体优势
3. 多模态方法的任何缺点或限制
4. 何时使用每种方法的整体建议"""

    # 创建评估摘要
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"查询 {i+1}：{result['query']}\n"
        evaluations_summary += f"多模态检索了 {result['multimodal_results']['text_count']} 个文本块和 {result['multimodal_results']['image_count']} 个图像描述\n"
        evaluations_summary += f"比较摘要：{result['comparison'][:200]}...\n\n"

    # 包含评估摘要的用户提示
    user_prompt = f"""基于以下对多模态与纯文本RAG在{len(results)}个查询中的评估，
提供这两种方法的整体分析：

{evaluations_summary}

请提供多模态RAG与纯文本RAG相对优缺点的综合分析，
特别关注图像信息如何贡献（或未贡献）响应质量。"""

    # 生成整体分析
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_llm_api(messages, temperature=0)
    return response

def evaluate_multimodal_vs_textonly(pdf_path: str, test_queries: List[str], 
                                   reference_answers: Optional[List[str]] = None) -> Dict:
    """
    比较多模态RAG与纯文本RAG
    
    Args:
        pdf_path: PDF文件路径
        test_queries: 测试查询列表
        reference_answers: 参考答案
        
    Returns:
        Dict: 评估结果
    """
    print("=== 评估多模态RAG与纯文本RAG ===\n")
    
    # 为多模态RAG处理文档
    print("\n为多模态RAG处理文档...")
    mm_vector_store, mm_doc_info = process_document(pdf_path)
    
    # 构建纯文本存储
    print("\n为纯文本RAG处理文档...")
    text_vector_store = build_text_only_store(pdf_path)
    
    # 为每个查询运行评估
    results = []
    
    for i, query in enumerate(test_queries):
        print(f"\n\n=== 评估查询 {i+1}：{query} ===")
        
        # 获取参考答案（如果可用）
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        
        # 运行多模态RAG
        print("\n运行多模态RAG...")
        mm_result = query_multimodal_rag(query, mm_vector_store)
        
        # 运行纯文本RAG
        print("\n运行纯文本RAG...")
        text_result = query_multimodal_rag(query, text_vector_store)
        
        # 比较响应
        comparison = compare_responses(query, mm_result["response"], text_result["response"], reference)
        
        # 添加到结果
        results.append({
            "query": query,
            "multimodal_response": mm_result["response"],
            "textonly_response": text_result["response"],
            "multimodal_results": {
                "text_count": mm_result["text_results_count"],
                "image_count": mm_result["image_results_count"]
            },
            "reference_answer": reference,
            "comparison": comparison
        })
    
    # 生成整体分析
    overall_analysis = generate_overall_analysis(results)
    
    return {
        "results": results,
        "overall_analysis": overall_analysis,
        "multimodal_doc_info": mm_doc_info
    }

# ============================================================================
# 演示和测试函数
# ============================================================================

def create_sample_pdf_content():
    """创建示例PDF内容用于演示"""
    # 这里应该创建一个包含文本和图像的示例PDF
    # 由于复杂性，我们创建一个模拟的文本文件
    sample_content = """
    人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。
    
    机器学习是AI的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。
    
    深度学习是机器学习的一个分支，使用神经网络来模拟人脑的工作方式。
    
    图表1显示了不同AI模型的性能比较：
    - 传统机器学习：准确率75%
    - 深度学习：准确率92%
    - 强化学习：准确率88%
    
    AI在各个领域都有应用，包括医疗保健、金融、交通和教育。
    
    图2展示了AI在医疗诊断中的应用流程。
    
    AI的发展带来了伦理考虑，包括偏见和公平性、隐私保护、就业影响、安全性和控制。
    
    确保AI系统的透明度和可解释性对于建立信任至关重要。
    """
    
    with open("sample_content.txt", "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    print("已创建示例内容文件：sample_content.txt")

def demonstrate_multimodal_rag():
    """演示多模态RAG系统"""
    print("=== 多模态RAG系统演示 ===\n")
    
    # 创建示例内容
    create_sample_pdf_content()
    
    # 模拟PDF处理（实际使用中需要真实的PDF文件）
    print("注意：这是一个演示版本，使用模拟数据")
    print("在实际使用中，需要提供包含文本和图像的PDF文件")
    
    # 模拟测试查询
    test_queries = [
        "什么是人工智能？",
        "图表1显示了什么信息？",
        "AI在医疗诊断中如何应用？",
        "深度学习与传统机器学习有什么区别？"
    ]
    
    print(f"\n示例查询：")
    for i, query in enumerate(test_queries):
        print(f"  {i+1}. {query}")
    
    print("\n系统特点：")
    print("1. 能够处理包含文本和图像的PDF文档")
    print("2. 使用LLaVA模型为图像生成详细描述")
    print("3. 将文本和图像描述统一向量化")
    print("4. 支持多模态查询和响应生成")
    print("5. 与纯文本RAG进行性能对比")
    
    print("\n技术优势：")
    print("- 访问图表和图表中的信息")
    print("- 理解补充文本的表格和图表")
    print("- 创建更全面的知识库")
    print("- 回答依赖视觉数据的问题")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("多模态RAG系统")
    print("=" * 60)
    
    # 演示系统
    demonstrate_multimodal_rag()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("\n使用说明：")
    print("1. 准备包含文本和图像的PDF文档")
    print("2. 配置API密钥和模型参数")
    print("3. 调用process_document()处理文档")
    print("4. 使用query_multimodal_rag()进行查询")
    print("5. 使用evaluate_multimodal_vs_textonly()进行性能对比")

if __name__ == "__main__":
    main() 