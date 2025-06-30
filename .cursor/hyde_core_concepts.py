#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyDE RAG核心概念讲解
简化版本，专注于核心算法原理
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# 核心概念1：传统RAG vs HyDE RAG
# ============================================================================

def traditional_rag_flow():
    """
    传统RAG流程演示
    
    流程：
    查询 → 向量化 → 检索 → 生成答案
    
    问题：
    - 短查询语义信息不足
    - 词汇不匹配
    - 上下文缺失
    """
    print("=== 传统RAG流程 ===")
    print("1. 用户查询: '什么是机器学习？'")
    print("2. 直接向量化查询")
    print("3. 在文档库中检索相似内容")
    print("4. 基于检索结果生成答案")
    print("问题: 短查询语义信息不足，检索效果有限\n")

def hyde_rag_flow():
    """
    HyDE RAG流程演示
    
    流程：
    查询 → 生成假设性文档 → 向量化文档 → 检索 → 生成答案
    
    优势：
    - 语义信息丰富
    - 词汇匹配更好
    - 上下文完整
    """
    print("=== HyDE RAG流程 ===")
    print("1. 用户查询: '什么是机器学习？'")
    print("2. 生成假设性文档: '机器学习是AI分支，包含监督学习、无监督学习...'")
    print("3. 向量化假设性文档")
    print("4. 在文档库中检索相似内容")
    print("5. 基于检索结果生成答案")
    print("优势: 假设性文档提供丰富语义信息，检索效果更好\n")

# ============================================================================
# 核心概念2：假设性文档生成
# ============================================================================

def demonstrate_hypothetical_document_generation():
    """
    演示假设性文档生成的核心思想
    """
    print("=== 假设性文档生成演示 ===")
    
    # 原始查询
    query = "什么是机器学习？"
    print(f"原始查询: {query}")
    print(f"查询长度: {len(query)} 字符")
    print(f"语义信息: 有限\n")
    
    # 生成的假设性文档
    hypothetical_doc = """
    机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并改进性能，而无需明确编程。
    机器学习算法可以分为监督学习、无监督学习和强化学习三大类。监督学习使用标记的训练数据
    来学习输入和输出之间的映射关系，适用于分类和回归问题。无监督学习在没有标记数据的情况下
    发现数据中的模式，常用于聚类和降维。强化学习通过与环境交互来学习最优策略，在游戏、
    机器人控制和自动驾驶等领域有重要应用。深度学习是机器学习的一个子集，使用多层神经网络
    来处理复杂的数据模式，在图像识别、自然语言处理和语音识别等任务中取得了突破性进展。
    """.strip()
    
    print(f"假设性文档: {hypothetical_doc}")
    print(f"文档长度: {len(hypothetical_doc)} 字符")
    print(f"语义信息: 丰富，包含相关概念、技术分类、应用领域等\n")
    
    return query, hypothetical_doc

# ============================================================================
# 核心概念3：向量化与相似度计算
# ============================================================================

def demonstrate_vectorization():
    """
    演示向量化和相似度计算
    """
    print("=== 向量化与相似度计算演示 ===")
    
    # 示例文档
    documents = [
        "机器学习是人工智能的重要分支，使计算机能从数据中学习。",
        "深度学习使用神经网络处理复杂数据模式。",
        "自然语言处理专注于计算机理解人类语言。",
        "计算机视觉致力于理解和解释视觉信息。"
    ]
    
    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer(max_features=50)
    
    # 向量化文档
    doc_vectors = vectorizer.fit_transform(documents).toarray()
    
    print("文档向量化结果:")
    for i, doc in enumerate(documents):
        print(f"文档{i+1}: {doc}")
        print(f"向量维度: {doc_vectors[i].shape}")
        print(f"非零元素: {np.count_nonzero(doc_vectors[i])}")
        print()
    
    return vectorizer, doc_vectors

def demonstrate_similarity_calculation(vectorizer, doc_vectors):
    """
    演示相似度计算
    """
    print("=== 相似度计算演示 ===")
    
    # 查询向量化
    query = "什么是机器学习？"
    query_vector = vectorizer.transform([query]).toarray()
    
    print(f"查询: {query}")
    print(f"查询向量维度: {query_vector.shape}")
    print()
    
    # 计算余弦相似度
    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    
    print("相似度计算结果:")
    for i, similarity in enumerate(similarities):
        print(f"文档{i+1}相似度: {similarity:.4f}")
    
    # 找到最相似的文档
    most_similar_idx = np.argmax(similarities)
    print(f"\n最相似的文档: 文档{most_similar_idx + 1}")
    print(f"相似度分数: {similarities[most_similar_idx]:.4f}")
    print()

# ============================================================================
# 核心概念4：HyDE vs 标准RAG对比
# ============================================================================

def compare_hyde_vs_standard():
    """
    对比HyDE和标准RAG的检索效果
    """
    print("=== HyDE vs 标准RAG对比 ===")
    
    # 示例文档库
    documents = [
        "机器学习是人工智能的重要分支，使计算机能从数据中学习。",
        "深度学习使用神经网络处理复杂数据模式。",
        "自然语言处理专注于计算机理解人类语言。",
        "计算机视觉致力于理解和解释视觉信息。"
    ]
    
    # 向量化文档库
    vectorizer = TfidfVectorizer(max_features=50)
    doc_vectors = vectorizer.fit_transform(documents).toarray()
    
    # 测试查询
    query = "什么是机器学习？"
    
    print(f"测试查询: {query}")
    print()
    
    # 标准RAG：直接使用查询
    print("--- 标准RAG ---")
    query_vector = vectorizer.transform([query]).toarray()
    standard_similarities = cosine_similarity(query_vector, doc_vectors)[0]
    
    for i, similarity in enumerate(standard_similarities):
        print(f"文档{i+1}相似度: {similarity:.4f}")
    
    standard_best = np.argmax(standard_similarities)
    print(f"最佳匹配: 文档{standard_best + 1}")
    print()
    
    # HyDE RAG：使用假设性文档
    print("--- HyDE RAG ---")
    hypothetical_doc = """
    机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并改进性能，而无需明确编程。
    机器学习算法可以分为监督学习、无监督学习和强化学习三大类。监督学习使用标记的训练数据
    来学习输入和输出之间的映射关系，适用于分类和回归问题。
    """.strip()
    
    print(f"假设性文档: {hypothetical_doc[:100]}...")
    
    hyde_vector = vectorizer.transform([hypothetical_doc]).toarray()
    hyde_similarities = cosine_similarity(hyde_vector, doc_vectors)[0]
    
    for i, similarity in enumerate(hyde_similarities):
        print(f"文档{i+1}相似度: {similarity:.4f}")
    
    hyde_best = np.argmax(hyde_similarities)
    print(f"最佳匹配: 文档{hyde_best + 1}")
    print()
    
    # 比较结果
    print("--- 对比结果 ---")
    print(f"标准RAG最佳相似度: {standard_similarities[standard_best]:.4f}")
    print(f"HyDE RAG最佳相似度: {hyde_similarities[hyde_best]:.4f}")
    
    improvement = hyde_similarities[hyde_best] - standard_similarities[standard_best]
    print(f"改进幅度: {improvement:.4f}")
    
    if improvement > 0:
        print("HyDE方法表现更好！")
    else:
        print("标准RAG方法表现更好。")
    print()

# ============================================================================
# 核心概念5：算法复杂度分析
# ============================================================================

def analyze_algorithm_complexity():
    """
    分析算法复杂度
    """
    print("=== 算法复杂度分析 ===")
    
    print("标准RAG复杂度:")
    print("- 查询向量化: O(V) - V为词汇表大小")
    print("- 相似度计算: O(N×D) - N为文档数，D为向量维度")
    print("- 总复杂度: O(V + N×D)")
    print()
    
    print("HyDE RAG复杂度:")
    print("- 假设性文档生成: O(L) - L为生成文档长度")
    print("- 文档向量化: O(V)")
    print("- 相似度计算: O(N×D)")
    print("- 总复杂度: O(L + V + N×D)")
    print()
    
    print("复杂度对比:")
    print("- HyDE增加了文档生成步骤")
    print("- 但检索质量显著提升")
    print("- 在质量要求高的场景下，额外开销是值得的")
    print()

# ============================================================================
# 核心概念6：实际应用场景
# ============================================================================

def demonstrate_use_cases():
    """
    演示实际应用场景
    """
    print("=== 实际应用场景 ===")
    
    scenarios = [
        {
            "name": "学术论文检索",
            "query": "深度学习在医疗影像中的应用",
            "hyde_advantage": "生成包含技术细节、应用案例、评估指标的假设性文档"
        },
        {
            "name": "技术文档搜索",
            "query": "如何优化数据库性能",
            "hyde_advantage": "生成包含优化策略、工具、最佳实践的假设性文档"
        },
        {
            "name": "法律文档检索",
            "query": "知识产权保护",
            "hyde_advantage": "生成包含法律条款、案例、程序步骤的假设性文档"
        }
    ]
    
    for scenario in scenarios:
        print(f"场景: {scenario['name']}")
        print(f"查询: {scenario['query']}")
        print(f"HyDE优势: {scenario['hyde_advantage']}")
        print()

# ============================================================================
# 主函数：完整演示
# ============================================================================

def main():
    """
    主函数：完整演示HyDE核心概念
    """
    print("HyDE RAG核心概念讲解")
    print("=" * 60)
    
    # 1. 流程对比
    traditional_rag_flow()
    hyde_rag_flow()
    
    # 2. 假设性文档生成演示
    query, hypothetical_doc = demonstrate_hypothetical_document_generation()
    
    # 3. 向量化演示
    vectorizer, doc_vectors = demonstrate_vectorization()
    
    # 4. 相似度计算演示
    demonstrate_similarity_calculation(vectorizer, doc_vectors)
    
    # 5. HyDE vs 标准RAG对比
    compare_hyde_vs_standard()
    
    # 6. 算法复杂度分析
    analyze_algorithm_complexity()
    
    # 7. 实际应用场景
    demonstrate_use_cases()
    
    print("=" * 60)
    print("核心概念讲解完成！")
    print("HyDE通过生成假设性文档来弥合查询与文档之间的语义差距。")

if __name__ == "__main__":
    main() 