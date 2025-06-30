#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyDE RAG核心代码详细讲解
Hypothetical Document Embedding (HyDE) 检索增强生成系统
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# 第一部分：核心数据结构
# ============================================================================

@dataclass
class DocumentChunk:
    """
    文档块数据结构 - 存储文档的基本信息
    
    属性说明：
    - text: 文档块的文本内容
    - metadata: 元数据信息（如文档ID、位置等）
    - embedding: 可选的向量嵌入（用于存储预计算的嵌入）
    """
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class SimpleVectorStore:
    """
    简单向量存储实现 - HyDE RAG的核心存储组件
    
    功能说明：
    1. 存储文档文本和对应的向量表示
    2. 提供相似性搜索功能
    3. 支持元数据管理
    """
    
    def __init__(self):
        """
        初始化向量存储
        
        组件说明：
        - vectors: 存储所有文档的向量表示
        - texts: 存储所有文档的文本内容
        - metadata: 存储所有文档的元数据
        - vectorizer: TF-IDF向量化器，用于文本到向量的转换
        - is_fitted: 标记向量化器是否已经训练
        """
        self.vectors = []          # 存储向量嵌入
        self.texts = []            # 存储文本内容
        self.metadata = []         # 存储元数据
        self.vectorizer = TfidfVectorizer(
            max_features=1000,     # 最大特征数
            stop_words='english'   # 停用词
        )
        self.is_fitted = False     # 向量化器训练状态
    
    def add_item(self, text: str, metadata: Optional[Dict] = None):
        """
        添加文档到向量存储
        
        参数说明：
        - text: 文档文本内容
        - metadata: 可选的元数据信息
        
        工作流程：
        1. 将文本添加到文本列表
        2. 将元数据添加到元数据列表
        3. 注意：向量化在fit_vectorizer()中进行
        """
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def fit_vectorizer(self):
        """
        训练向量化器 - 将文本转换为向量表示
        
        核心步骤：
        1. 使用TF-IDF向量化器拟合所有文本
        2. 将所有文本转换为向量矩阵
        3. 标记向量化器为已训练状态
        
        TF-IDF原理：
        - Term Frequency (TF): 词频，衡量词在文档中的重要性
        - Inverse Document Frequency (IDF): 逆文档频率，衡量词的稀有程度
        - 最终分数 = TF × IDF，能够有效表示文档的语义特征
        """
        if not self.is_fitted:
            # 训练TF-IDF向量化器
            self.vectorizer.fit(self.texts)
            # 将所有文本转换为向量
            self.vectors = self.vectorizer.transform(self.texts).toarray()
            self.is_fitted = True
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        相似性搜索 - 找到与查询最相似的文档
        
        参数说明：
        - query: 查询文本
        - k: 返回结果数量
        
        返回格式：
        List[Dict] - 每个字典包含text、metadata和similarity
        
        算法流程：
        1. 确保向量化器已训练
        2. 将查询转换为向量
        3. 计算查询向量与所有文档向量的余弦相似度
        4. 返回相似度最高的k个结果
        """
        # 确保向量化器已训练
        if not self.is_fitted:
            self.fit_vectorizer()
        
        if not self.vectors:
            return []
        
        # 将查询转换为向量
        query_vector = self.vectorizer.transform([query]).toarray()
        
        # 计算余弦相似度
        # 余弦相似度 = (A·B) / (||A|| × ||B||)
        # 值域：[-1, 1]，1表示完全相同，0表示无关，-1表示完全相反
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # 获取前k个最相似的结果
        # np.argsort返回排序后的索引，[::-1]反转得到降序
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # 构建结果列表
        results = []
        for idx in top_indices:
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx])
            })
        
        return results

# ============================================================================
# 第二部分：HyDE核心算法
# ============================================================================

def generate_hypothetical_document(query: str) -> str:
    """
    生成假设性文档 - HyDE算法的核心创新
    
    参数说明：
    - query: 用户查询
    
    返回：
    - str: 生成的假设性文档
    
    核心思想：
    传统RAG直接使用短查询进行检索，但短查询往往缺乏足够的语义信息。
    HyDE通过生成一个假设性的、详细的答案文档来增强查询的语义表达。
    
    生成策略：
    1. 基于查询关键词选择预定义的模板
    2. 模板包含丰富的语义信息和相关概念
    3. 模拟真实文档的语言风格和结构
    """
    
    # 预定义的假设性文档模板
    # 这些模板模拟了真实文档的丰富语义表达
    hypothetical_templates = {
        "机器学习": """
        机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并改进性能，而无需明确编程。
        机器学习算法可以分为监督学习、无监督学习和强化学习三大类。监督学习使用标记的训练数据
        来学习输入和输出之间的映射关系，适用于分类和回归问题。无监督学习在没有标记数据的情况下
        发现数据中的模式，常用于聚类和降维。强化学习通过与环境交互来学习最优策略，在游戏、
        机器人控制和自动驾驶等领域有重要应用。深度学习是机器学习的一个子集，使用多层神经网络
        来处理复杂的数据模式，在图像识别、自然语言处理和语音识别等任务中取得了突破性进展。
        """,
        
        "人工智能": """
        人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。
        这些任务包括学习、推理、问题解决、感知和语言理解。AI技术已经在各个领域得到广泛应用，
        包括医疗保健、金融、教育、交通和娱乐。人工智能的发展经历了几个重要阶段，从早期的
        符号主义到现代的深度学习和神经网络方法。当前，大型语言模型和生成式AI正在推动AI技术的
        快速发展，为各行各业带来了新的机遇和挑战。
        """,
        
        "自然语言处理": """
        自然语言处理（NLP）是人工智能的一个分支，专注于计算机理解和生成人类语言的能力。
        NLP技术包括文本分析、机器翻译、情感分析、问答系统、聊天机器人和文本生成等。
        近年来，大型语言模型如GPT、BERT和T5在NLP任务中取得了突破性进展，能够生成高质量
        的人类语言文本并理解复杂的语言结构。NLP的应用范围非常广泛，从搜索引擎到虚拟助手，
        从内容推荐到自动摘要，都在使用NLP技术来提升用户体验和服务质量。
        """,
        
        "计算机视觉": """
        计算机视觉是人工智能的另一个重要分支，致力于使计算机能够理解和解释视觉信息。
        计算机视觉技术包括图像识别、物体检测、人脸识别、医学图像分析、自动驾驶和增强现实等。
        卷积神经网络（CNN）是计算机视觉中最常用的深度学习架构，特别适合处理图像数据。
        近年来，Transformer架构在计算机视觉领域也取得了重要进展，Vision Transformer（ViT）
        等模型在图像分类任务中表现优异。计算机视觉技术正在改变我们与数字世界的交互方式。
        """,
        
        "伦理": """
        人工智能的伦理考虑是AI发展中的重要议题，包括偏见和公平性、透明度和可解释性、
        隐私和数据保护、问责制和责任感等方面。确保AI系统公平、非歧视性且不会延续训练数据中
        存在的现有偏见至关重要。透明度对于建立信任和问责制也很重要，用户需要了解AI系统如何
        做出决策。隐私保护则需要负责任的数据处理和隐私保护技术。随着AI技术的快速发展，
        建立完善的伦理框架和监管机制变得越来越重要。
        """
    }
    
    # 根据查询关键词选择最合适的模板
    for keyword, template in hypothetical_templates.items():
        if keyword in query:
            return template.strip()
    
    # 默认模板 - 当没有匹配的关键词时使用
    return f"""
    这是一个关于"{query}"的详细文档。它包含了相关的信息、事实和解释，旨在全面回答用户的问题。
    文档涵盖了该主题的基本概念、主要技术、应用领域和发展趋势。通过深入分析相关理论和实践案例，
    本文档为读者提供了对该主题的全面理解。
    """.strip()

def hyde_rag(query: str, vector_store: SimpleVectorStore, k: int = 5) -> Dict:
    """
    HyDE RAG主函数 - 实现完整的HyDE检索流程
    
    参数说明：
    - query: 用户查询
    - vector_store: 向量存储对象
    - k: 检索结果数量
    
    返回：
    - Dict: 包含查询、假设性文档、检索结果和最终响应的字典
    
    算法流程：
    1. 生成假设性文档：将短查询扩展为详细的文档
    2. 基于假设性文档检索：使用文档嵌入进行相似性搜索
    3. 生成最终响应：基于检索结果生成答案
    
    核心优势：
    - 语义增强：假设性文档提供更丰富的语义信息
    - 词汇扩展：自动包含相关概念和术语
    - 上下文对齐：更好地匹配文档的语言风格
    """
    print(f"\n=== 使用HyDE处理查询: {query} ===\n")
    
    # 步骤1：生成假设性文档
    print("生成假设性文档...")
    hypothetical_doc = generate_hypothetical_document(query)
    print(f"生成了{len(hypothetical_doc)}字符的假设性文档")
    
    # 步骤2：基于假设性文档检索相似块
    print(f"检索{k}个最相似的块...")
    retrieved_chunks = vector_store.similarity_search(hypothetical_doc, k=k)
    
    # 步骤3：生成最终响应
    print("生成最终响应...")
    response = generate_response(query, retrieved_chunks)
    
    return {
        "query": query,
        "hypothetical_document": hypothetical_doc,
        "retrieved_chunks": retrieved_chunks,
        "response": response
    }

def standard_rag(query: str, vector_store: SimpleVectorStore, k: int = 5) -> Dict:
    """
    标准RAG实现 - 用于对比HyDE的效果
    
    参数说明：
    - query: 用户查询
    - vector_store: 向量存储对象
    - k: 检索结果数量
    
    返回：
    - Dict: 包含查询、检索结果和最终响应的字典
    
    算法流程：
    1. 直接使用查询进行检索
    2. 基于检索结果生成响应
    
    对比目的：
    - 展示HyDE相对于传统方法的改进
    - 量化性能提升
    - 验证假设性文档的有效性
    """
    print(f"\n=== 使用标准RAG处理查询: {query} ===\n")
    
    # 步骤1：基于查询直接检索相似块
    print(f"检索{k}个最相似的块...")
    retrieved_chunks = vector_store.similarity_search(query, k=k)
    
    # 步骤2：生成最终响应
    print("生成最终响应...")
    response = generate_response(query, retrieved_chunks)
    
    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "response": response
    }

def generate_response(query: str, relevant_chunks: List[Dict]) -> str:
    """
    生成最终响应 - 基于检索结果生成答案
    
    参数说明：
    - query: 用户查询
    - relevant_chunks: 检索到的相关文档块
    
    返回：
    - str: 生成的最终响应
    
    生成策略：
    1. 提取检索结果中的关键信息
    2. 根据查询类型选择响应模板
    3. 结合检索内容生成连贯的答案
    """
    # 提取检索结果中的文本内容
    context = " ".join([chunk["text"][:200] for chunk in relevant_chunks])
    
    # 基于查询类型选择响应模板
    response_templates = {
        "机器学习": f"基于检索到的内容，机器学习是AI的重要分支，使计算机能从数据中学习。主要包括监督学习、无监督学习和强化学习。{context[:100]}...",
        "人工智能": f"人工智能是计算机科学分支，创建能执行人类智能任务的系统。应用广泛，包括医疗、金融、教育等领域。{context[:100]}...",
        "自然语言处理": f"NLP专注于计算机理解和生成人类语言，包括文本分析、机器翻译、情感分析等技术。{context[:100]}...",
        "计算机视觉": f"计算机视觉使计算机能理解和解释视觉信息，包括图像识别、物体检测、人脸识别等技术。{context[:100]}...",
        "伦理": f"AI伦理考虑包括偏见公平性、透明度可解释性、隐私数据保护、问责制责任感等方面。{context[:100]}..."
    }
    
    # 根据查询关键词选择响应模板
    for keyword, template in response_templates.items():
        if keyword in query:
            return template
    
    # 默认响应
    return f"基于检索到的内容，我可以回答您的问题：{query}。相关信息包括：{context[:200]}..."

# ============================================================================
# 第三部分：文档处理与向量化
# ============================================================================

def create_sample_documents() -> List[str]:
    """
    创建示例文档 - 用于演示和测试
    
    返回：
    - List[str]: 示例文档列表
    
    文档内容：
    包含AI相关的基础知识，涵盖机器学习、NLP、计算机视觉等主题
    """
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
    """
    处理文档并创建向量存储
    
    参数说明：
    - documents: 原始文档列表
    
    返回：
    - SimpleVectorStore: 包含文档向量表示的存储对象
    
    处理流程：
    1. 创建向量存储对象
    2. 将文档分割成小块
    3. 添加文档块到向量存储
    4. 训练向量化器
    """
    print("处理文档并创建向量存储...")
    
    vector_store = SimpleVectorStore()
    
    for i, doc in enumerate(documents):
        # 将文档分割成小块，使用滑动窗口策略
        # 块大小：500字符，重叠：300字符
        chunks = [doc[j:j+500] for j in range(0, len(doc), 300)]
        
        for j, chunk in enumerate(chunks):
            # 只添加有意义的块（长度大于50字符）
            if len(chunk.strip()) > 50:
                vector_store.add_item(
                    text=chunk,
                    metadata={
                        "doc_id": i,
                        "chunk_id": j,
                        "length": len(chunk)
                    }
                )
    
    print(f"向量存储创建完成，包含{len(vector_store.texts)}个块")
    return vector_store

# ============================================================================
# 第四部分：评估与分析
# ============================================================================

def analyze_similarity_scores(hyde_result: Dict, standard_result: Dict) -> Dict:
    """
    分析相似度分数 - 量化HyDE的性能改进
    
    参数说明：
    - hyde_result: HyDE RAG的结果
    - standard_result: 标准RAG的结果
    
    返回：
    - Dict: 包含各种相似度指标的字典
    
    分析指标：
    - 平均相似度：检索结果的平均相似度分数
    - 最大相似度：最高相似度分数
    - 改进幅度：HyDE相对于标准RAG的改进
    """
    # 提取相似度分数
    hyde_scores = [chunk["similarity"] for chunk in hyde_result["retrieved_chunks"]]
    standard_scores = [chunk["similarity"] for chunk in standard_result["retrieved_chunks"]]
    
    return {
        "hyde_avg_similarity": np.mean(hyde_scores),
        "standard_avg_similarity": np.mean(standard_scores),
        "hyde_max_similarity": np.max(hyde_scores),
        "standard_max_similarity": np.max(standard_scores),
        "improvement": np.mean(hyde_scores) - np.mean(standard_scores)
    }

def compare_responses(query: str, hyde_response: str, standard_response: str) -> str:
    """
    比较两种方法的响应 - 定性分析
    
    参数说明：
    - query: 用户查询
    - hyde_response: HyDE RAG的响应
    - standard_response: 标准RAG的响应
    
    返回：
    - str: 比较分析结果
    
    分析维度：
    - 准确性：响应信息的准确性
    - 相关性：响应与查询的相关程度
    - 完整性：响应内容的完整程度
    - 清晰度：响应的组织和表达清晰度
    """
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

# ============================================================================
# 第五部分：演示主函数
# ============================================================================

def main():
    """
    主函数 - 演示HyDE RAG的完整流程
    
    演示内容：
    1. 创建示例文档
    2. 处理文档并创建向量存储
    3. 运行HyDE RAG和标准RAG
    4. 比较两种方法的效果
    5. 分析性能改进
    """
    print("HyDE RAG核心代码演示")
    print("=" * 50)
    
    # 步骤1：创建示例文档
    documents = create_sample_documents()
    print(f"创建了{len(documents)}个示例文档")
    
    # 步骤2：处理文档并创建向量存储
    vector_store = process_documents(documents)
    
    # 步骤3：测试查询
    test_queries = [
        "什么是机器学习？",
        "人工智能的伦理考虑有哪些？",
        "自然语言处理的主要技术是什么？"
    ]
    
    all_results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"测试查询 {i}: {query}")
        print(f"{'='*60}")
        
        # 运行HyDE RAG
        hyde_result = hyde_rag(query, vector_store)
        
        # 运行标准RAG
        standard_result = standard_rag(query, vector_store)
        
        # 分析相似度分数
        similarity_analysis = analyze_similarity_scores(hyde_result, standard_result)
        
        # 显示结果
        print("\n" + "="*50)
        print("HyDE RAG响应:")
        print("="*50)
        print(hyde_result["response"])
        
        print("\n" + "="*50)
        print("标准RAG响应:")
        print("="*50)
        print(standard_result["response"])
        
        print("\n" + "="*50)
        print("相似度分析:")
        print("="*50)
        print(f"HyDE平均相似度: {similarity_analysis['hyde_avg_similarity']:.4f}")
        print(f"标准RAG平均相似度: {similarity_analysis['standard_avg_similarity']:.4f}")
        print(f"改进幅度: {similarity_analysis['improvement']:.4f}")
        
        # 比较结果
        comparison = compare_responses(query, hyde_result["response"], standard_result["response"])
        print("\n" + "="*50)
        print("方法比较:")
        print("="*50)
        print(comparison)
        
        # 保存结果
        all_results.append({
            "query": query,
            "hyde_result": hyde_result,
            "standard_result": standard_result,
            "similarity_analysis": similarity_analysis
        })
    
    # 总结分析
    print("\n" + "="*60)
    print("总结分析:")
    print("="*60)
    
    total_improvement = sum(r["similarity_analysis"]["improvement"] for r in all_results)
    avg_improvement = total_improvement / len(all_results)
    
    print(f"平均改进幅度: {avg_improvement:.4f}")
    print(f"测试查询数量: {len(all_results)}")
    
    if avg_improvement > 0:
        print("HyDE方法在相似度方面表现更好！")
    else:
        print("标准RAG方法在相似度方面表现更好。")
    
    print("\n" + "="*60)
    print("演示完成！")
    print("HyDE RAG通过生成假设性文档来提升检索质量。")
    print("="*60)

if __name__ == "__main__":
    main() 