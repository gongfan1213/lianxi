#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命题分块（Proposition Chunking）演示版本

这是一个简化的演示版本，展示命题分块的核心概念和基本实现。
用于理解命题分块与传统分块方法的区别。

作者：AI助手
日期：2024年
"""

import json
import re
from typing import List, Dict, Tuple

# ============================================================================
# 模拟API调用（实际使用时替换为真实API）
# ============================================================================

def mock_llm_call(prompt: str) -> str:
    """
    模拟LLM调用，返回预设的响应
    在实际使用中，这里应该调用真实的LLM API
    """
    # 模拟命题生成的响应
    if "分解为简单、自包含的命题" in prompt:
        return """
1. 人工智能是计算机科学的一个分支
2. AI旨在创建能够执行通常需要人类智能的任务的系统
3. 机器学习是AI的一个子集
4. 机器学习使计算机能够在没有明确编程的情况下学习和改进
5. 深度学习是机器学习的一个分支
6. 深度学习使用神经网络来模拟人脑的工作方式
7. AI在医疗保健领域有应用
8. AI在金融领域有应用
9. AI在交通领域有应用
10. AI在教育领域有应用
11. AI的发展带来了伦理考虑
12. 确保AI系统的透明度和可解释性对于建立信任至关重要
        """
    
    # 模拟质量评估的响应
    elif "评估分数" in prompt:
        return '{"accuracy": 8, "clarity": 9, "completeness": 7, "conciseness": 8}'
    
    # 模拟响应生成的响应
    elif "基于检索到的信息回答查询" in prompt:
        return "基于检索到的信息，人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。"
    
    return "默认响应"

def mock_embedding_call(texts: List[str]) -> List[List[float]]:
    """
    模拟嵌入API调用，返回随机向量
    在实际使用中，这里应该调用真实的嵌入API
    """
    import random
    embeddings = []
    for text in texts:
        # 生成1536维的随机向量（模拟text-embedding-3-small）
        embedding = [random.uniform(-1, 1) for _ in range(1536)]
        # 归一化向量
        norm = sum(x*x for x in embedding) ** 0.5
        embedding = [x/norm for x in embedding]
        embeddings.append(embedding)
    return embeddings

# ============================================================================
# 核心函数实现
# ============================================================================

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[Dict]:
    """
    传统分块方法：按字符数量分割文本
    
    Args:
        text: 输入文本
        chunk_size: 每个块的大小
        overlap: 块之间的重叠
        
    Returns:
        List[Dict]: 文本块列表
    """
    chunks = []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append({
                "text": chunk,
                "chunk_id": len(chunks) + 1,
                "start_char": i,
                "end_char": i + len(chunk)
            })
    
    return chunks

def generate_propositions(chunk: Dict) -> List[str]:
    """
    从文本块生成原子性命题
    
    Args:
        chunk: 文本块
        
    Returns:
        List[str]: 生成的命题列表
    """
    system_prompt = """请将以下文本分解为简单、自包含的命题。
确保每个命题满足以下标准：
1. 表达单一事实
2. 无需上下文即可理解
3. 使用完整名称而非代词
4. 包含相关日期/限定词（如果适用）
5. 包含一个主谓关系

仅输出命题列表，不要包含任何额外的文本或解释。"""

    user_prompt = f"要转换为命题的文本：\n\n{chunk['text']}"
    
    # 模拟API调用
    response = mock_llm_call(system_prompt + "\n" + user_prompt)
    
    # 清理命题
    raw_propositions = response.strip().split('\n')
    clean_propositions = []
    
    for prop in raw_propositions:
        # 移除编号和项目符号
        cleaned = re.sub(r'^\s*(\d+\.|\-|\*)\s*', '', prop).strip()
        if cleaned and len(cleaned) > 10:
            clean_propositions.append(cleaned)
    
    return clean_propositions

def evaluate_proposition(proposition: str, original_text: str) -> Dict:
    """
    评估命题质量
    
    Args:
        proposition: 要评估的命题
        original_text: 原始文本
        
    Returns:
        Dict: 质量分数
    """
    system_prompt = """你是评估从文本中提取的命题质量的专家。
对给定命题按以下标准评分（1-10分）：
- 准确性：命题在多大程度上反映了原始文本中的信息
- 清晰度：无需额外上下文即可理解命题的难易程度
- 完整性：命题是否包含必要的细节
- 简洁性：命题是否简洁而不丢失重要信息

响应必须是有效的JSON格式：
{"accuracy": X, "clarity": X, "completeness": X, "conciseness": X}"""

    user_prompt = f"""命题：{proposition}
原始文本：{original_text}
请以JSON格式提供评估分数。"""

    response = mock_llm_call(system_prompt + "\n" + user_prompt)
    
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return {"accuracy": 5, "clarity": 5, "completeness": 5, "conciseness": 5}

# ============================================================================
# 向量存储实现
# ============================================================================

class SimpleVectorStore:
    """简单的向量存储实现"""
    
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []
    
    def add_items(self, texts: List[str], embeddings: List[List[float]], 
                  metadata_list: List[Dict] = None):
        """添加项目到向量存储"""
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(texts))]
        
        for text, embedding, metadata in zip(texts, embeddings, metadata_list):
            self.vectors.append(embedding)
            self.texts.append(text)
            self.metadata.append(metadata)
    
    def similarity_search(self, query_embedding: List[float], k: int = 3) -> List[Dict]:
        """相似度搜索"""
        if not self.vectors:
            return []
        
        # 计算余弦相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 简化的相似度计算
            similarity = sum(a * b for a, b in zip(query_embedding, vector))
            similarities.append((i, similarity))
        
        # 排序并返回前k个结果
        similarities.sort(key=lambda x: x[1], reverse=True)
        
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
# 演示函数
# ============================================================================

def demonstrate_proposition_chunking():
    """演示命题分块的核心概念"""
    
    print("=== 命题分块（Proposition Chunking）演示 ===\n")
    
    # 示例文本
    sample_text = """
    人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。
    机器学习是AI的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。
    深度学习是机器学习的一个分支，使用神经网络来模拟人脑的工作方式。
    AI在各个领域都有应用，包括医疗保健、金融、交通和教育。
    AI的发展带来了伦理考虑，包括偏见和公平性、隐私保护、就业影响、安全性和控制。
    确保AI系统的透明度和可解释性对于建立信任至关重要。
    """
    
    print("原始文本：")
    print(sample_text.strip())
    print("\n" + "="*50)
    
    # 1. 传统分块
    print("\n1. 传统分块方法：")
    chunks = chunk_text(sample_text, chunk_size=150, overlap=30)
    print(f"生成了 {len(chunks)} 个文本块：")
    for i, chunk in enumerate(chunks):
        print(f"  块 {i+1}: {chunk['text'][:50]}...")
    
    print("\n" + "="*50)
    
    # 2. 命题生成
    print("\n2. 命题生成：")
    all_propositions = []
    for chunk in chunks:
        propositions = generate_propositions(chunk)
        all_propositions.extend(propositions)
        print(f"  从块 {chunk['chunk_id']} 生成了 {len(propositions)} 个命题")
    
    print(f"\n总共生成了 {len(all_propositions)} 个命题：")
    for i, prop in enumerate(all_propositions[:5]):  # 只显示前5个
        print(f"  {i+1}. {prop}")
    
    print("\n" + "="*50)
    
    # 3. 质量评估
    print("\n3. 质量评估：")
    quality_propositions = []
    for prop in all_propositions:
        scores = evaluate_proposition(prop, sample_text)
        print(f"  命题: {prop[:50]}...")
        print(f"    准确性: {scores['accuracy']}, 清晰度: {scores['clarity']}, "
              f"完整性: {scores['completeness']}, 简洁性: {scores['conciseness']}")
        
        # 简单的质量过滤（平均分>7）
        avg_score = sum(scores.values()) / len(scores)
        if avg_score > 7:
            quality_propositions.append(prop)
            print(f"    ✓ 通过质量检查")
        else:
            print(f"    ✗ 未通过质量检查")
        print()
    
    print(f"质量过滤后保留了 {len(quality_propositions)}/{len(all_propositions)} 个命题")
    
    print("\n" + "="*50)
    
    # 4. 向量存储比较
    print("\n4. 向量存储比较：")
    
    # 创建传统分块的向量存储
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = mock_embedding_call(chunk_texts)
    chunk_store = SimpleVectorStore()
    chunk_store.add_items(chunk_texts, chunk_embeddings, 
                         [{"type": "chunk", "chunk_id": chunk["chunk_id"]} for chunk in chunks])
    
    # 创建命题的向量存储
    prop_embeddings = mock_embedding_call(quality_propositions)
    prop_store = SimpleVectorStore()
    prop_store.add_items(quality_propositions, prop_embeddings,
                        [{"type": "proposition"} for _ in quality_propositions])
    
    # 测试查询
    test_query = "什么是人工智能？"
    print(f"测试查询: {test_query}")
    
    # 查询嵌入
    query_embedding = mock_embedding_call([test_query])[0]
    
    # 传统分块检索
    chunk_results = chunk_store.similarity_search(query_embedding, k=2)
    print(f"\n传统分块检索结果（前2个）:")
    for i, result in enumerate(chunk_results):
        print(f"  {i+1}. {result['text'][:80]}... (相似度: {result['similarity']:.3f})")
    
    # 命题检索
    prop_results = prop_store.similarity_search(query_embedding, k=3)
    print(f"\n命题检索结果（前3个）:")
    for i, result in enumerate(prop_results):
        print(f"  {i+1}. {result['text']} (相似度: {result['similarity']:.3f})")
    
    print("\n" + "="*50)
    
    # 5. 总结
    print("\n5. 方法比较总结：")
    print("传统分块方法：")
    print("  ✓ 实现简单，计算效率高")
    print("  ✗ 可能切断语义单元")
    print("  ✗ 检索结果可能包含不相关信息")
    print("  ✗ 上下文信息可能丢失")
    
    print("\n命题分块方法：")
    print("  ✓ 保持语义完整性")
    print("  ✓ 提供更精确的检索")
    print("  ✓ 每个命题都是自包含的")
    print("  ✓ 支持质量过滤")
    print("  ✗ 实现复杂度较高")
    print("  ✗ 计算成本较高")

def demonstrate_quality_filtering():
    """演示质量过滤的效果"""
    
    print("\n\n=== 质量过滤演示 ===\n")
    
    # 示例命题
    sample_propositions = [
        "人工智能是计算机科学的一个分支",
        "AI很好",
        "机器学习使计算机能够在没有明确编程的情况下学习和改进",
        "深度学习使用神经网络",
        "AI在各个领域都有应用，包括医疗保健、金融、交通和教育",
        "这很重要",
        "确保AI系统的透明度和可解释性对于建立信任至关重要"
    ]
    
    print("原始命题：")
    for i, prop in enumerate(sample_propositions):
        print(f"  {i+1}. {prop}")
    
    print("\n质量评估结果：")
    quality_propositions = []
    
    for prop in sample_propositions:
        # 模拟质量评估
        if len(prop) < 15:
            scores = {"accuracy": 3, "clarity": 4, "completeness": 2, "conciseness": 6}
        elif "很好" in prop or "重要" in prop:
            scores = {"accuracy": 4, "clarity": 5, "completeness": 3, "conciseness": 7}
        else:
            scores = {"accuracy": 8, "clarity": 9, "completeness": 7, "conciseness": 8}
        
        avg_score = sum(scores.values()) / len(scores)
        print(f"  '{prop}' - 平均分: {avg_score:.1f}")
        
        if avg_score > 6:
            quality_propositions.append(prop)
            print(f"    ✓ 保留")
        else:
            print(f"    ✗ 过滤")
    
    print(f"\n质量过滤后保留了 {len(quality_propositions)}/{len(sample_propositions)} 个命题")
    print("\n保留的命题：")
    for i, prop in enumerate(quality_propositions):
        print(f"  {i+1}. {prop}")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("命题分块（Proposition Chunking）RAG系统演示")
    print("=" * 60)
    
    # 演示核心概念
    demonstrate_proposition_chunking()
    
    # 演示质量过滤
    demonstrate_quality_filtering()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("\n关键要点：")
    print("1. 命题分块将文档分解为原子性事实，而不是按字符数量分割")
    print("2. 每个命题都是自包含的，无需额外上下文即可理解")
    print("3. 质量评估确保只有高质量的命题被保留")
    print("4. 相比传统分块，命题分块提供更精确的检索结果")
    print("5. 适用于需要高精度信息检索的应用场景")

if __name__ == "__main__":
    main() 