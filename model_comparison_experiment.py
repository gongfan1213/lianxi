#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph RAG模型对比实验

用于比较不同模型组合在Graph RAG系统中的表现差异

作者：AI助手
日期：2024年
"""

import numpy as np
import json
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 实验配置
# ============================================================================

@dataclass
class ModelConfig:
    """模型配置"""
    embedding_model: str
    concept_model: str
    response_model: str
    description: str

# 定义不同的模型组合
MODEL_CONFIGS = [
    ModelConfig(
        embedding_model="text-embedding-ada-002",
        concept_model="gpt-3.5-turbo",
        response_model="gpt-3.5-turbo",
        description="基础配置"
    ),
    ModelConfig(
        embedding_model="text-embedding-3-small",
        concept_model="gpt-3.5-turbo",
        response_model="gpt-3.5-turbo",
        description="新嵌入模型"
    ),
    ModelConfig(
        embedding_model="text-embedding-ada-002",
        concept_model="gpt-4",
        response_model="gpt-3.5-turbo",
        description="GPT-4概念提取"
    ),
    ModelConfig(
        embedding_model="text-embedding-ada-002",
        concept_model="gpt-3.5-turbo",
        response_model="gpt-4",
        description="GPT-4响应生成"
    ),
    ModelConfig(
        embedding_model="text-embedding-3-small",
        concept_model="gpt-4",
        response_model="gpt-4",
        description="全GPT-4配置"
    )
]

# ============================================================================
# 模拟API调用（实际使用时替换为真实API）
# ============================================================================

def mock_embedding_call(texts: List[str], model: str) -> List[List[float]]:
    """模拟不同嵌入模型的调用"""
    embeddings = []
    
    # 模拟不同模型的特性
    if "ada-002" in model:
        dimension = 1536
        # 模拟ada-002的特性：更稳定的语义表示
        base_embedding = np.random.normal(0, 0.1, dimension)
    elif "3-small" in model:
        dimension = 1536
        # 模拟3-small的特性：更精细的语义区分
        base_embedding = np.random.normal(0, 0.15, dimension)
    else:
        dimension = 1536
        base_embedding = np.random.normal(0, 0.1, dimension)
    
    for text in texts:
        # 基于文本内容生成不同的嵌入
        text_seed = hash(text) % 10000
        np.random.seed(text_seed)
        
        embedding = base_embedding + np.random.normal(0, 0.05, dimension)
        # 归一化
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding.tolist())
    
    return embeddings

def mock_concept_extraction(text: str, model: str) -> List[str]:
    """模拟不同模型的概念提取"""
    # 模拟不同模型的概念提取能力
    if "gpt-4" in model:
        # GPT-4提取更多、更准确的概念
        concepts = [
            "transformer", "attention mechanism", "neural network", 
            "machine learning", "natural language processing", "deep learning",
            "self-attention", "encoder-decoder", "parallel processing"
        ]
    else:
        # GPT-3.5提取较少、较基础的概念
        concepts = [
            "transformer", "neural network", "machine learning", 
            "natural language processing"
        ]
    
    # 根据文本内容选择相关概念
    relevant_concepts = []
    for concept in concepts:
        if any(word in text.lower() for word in concept.split()):
            relevant_concepts.append(concept)
    
    return relevant_concepts[:5]  # 限制概念数量

def mock_response_generation(query: str, context: str, model: str) -> str:
    """模拟不同模型的响应生成"""
    if "gpt-4" in model:
        # GPT-4生成更详细、更准确的响应
        return f"基于提供的上下文，{query}的答案是：Transformer模型通过自注意力机制处理序列数据，相比RNN具有更好的并行性和长距离依赖捕获能力。GPT-4生成的详细分析。"
    else:
        # GPT-3.5生成较简洁的响应
        return f"根据上下文，{query}的答案是：Transformer模型使用注意力机制，比RNN更有效。GPT-3.5生成的简洁回答。"

# ============================================================================
# 实验函数
# ============================================================================

def run_single_experiment(config: ModelConfig, test_data: Dict) -> Dict:
    """运行单个模型配置的实验"""
    print(f"\n运行实验: {config.description}")
    print(f"配置: 嵌入={config.embedding_model}, 概念={config.concept_model}, 响应={config.response_model}")
    
    start_time = time.time()
    
    # 1. 嵌入生成
    print("1. 生成嵌入...")
    embeddings = mock_embedding_call(test_data["texts"], config.embedding_model)
    
    # 2. 概念提取
    print("2. 提取概念...")
    all_concepts = []
    for text in test_data["texts"]:
        concepts = mock_concept_extraction(text, config.concept_model)
        all_concepts.append(concepts)
    
    # 3. 图构建
    print("3. 构建图...")
    graph_stats = build_mock_graph(test_data["texts"], all_concepts, embeddings)
    
    # 4. 图遍历
    print("4. 图遍历...")
    query_embedding = mock_embedding_call([test_data["query"]], config.embedding_model)[0]
    traversal_results = mock_graph_traversal(test_data["texts"], embeddings, query_embedding)
    
    # 5. 响应生成
    print("5. 生成响应...")
    context = "\n".join([test_data["texts"][i] for i in traversal_results["relevant_nodes"]])
    response = mock_response_generation(test_data["query"], context, config.response_model)
    
    end_time = time.time()
    
    return {
        "config": config,
        "execution_time": end_time - start_time,
        "graph_stats": graph_stats,
        "traversal_results": traversal_results,
        "response": response,
        "metrics": calculate_metrics(response, test_data.get("reference_answer", ""))
    }

def build_mock_graph(texts: List[str], concepts: List[List[str]], embeddings: List[List[float]]) -> Dict:
    """构建模拟的知识图谱"""
    nodes = len(texts)
    edges = 0
    
    # 计算边数（基于概念重叠）
    for i in range(nodes):
        for j in range(i + 1, nodes):
            shared_concepts = set(concepts[i]).intersection(set(concepts[j]))
            if len(shared_concepts) > 0:
                edges += 1
    
    # 计算平均度
    avg_degree = (2 * edges) / nodes if nodes > 0 else 0
    
    return {
        "nodes": nodes,
        "edges": edges,
        "avg_degree": avg_degree,
        "density": edges / (nodes * (nodes - 1) / 2) if nodes > 1 else 0
    }

def mock_graph_traversal(texts: List[str], embeddings: List[List[float]], query_embedding: List[float]) -> Dict:
    """模拟图遍历"""
    # 计算相似度
    similarities = []
    for i, embedding in enumerate(embeddings):
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        similarities.append((i, similarity))
    
    # 排序并选择top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = min(3, len(similarities))
    relevant_nodes = [node for node, _ in similarities[:top_k]]
    
    return {
        "relevant_nodes": relevant_nodes,
        "similarities": similarities[:top_k],
        "path_length": len(relevant_nodes)
    }

def calculate_metrics(response: str, reference: str) -> Dict:
    """计算评估指标"""
    if not reference:
        return {
            "response_length": len(response),
            "has_keywords": any(keyword in response.lower() for keyword in ["transformer", "attention", "neural"])
        }
    
    # 简单的相似度计算（实际应使用更复杂的指标）
    response_words = set(response.lower().split())
    reference_words = set(reference.lower().split())
    
    if len(reference_words) == 0:
        similarity = 0
    else:
        intersection = response_words.intersection(reference_words)
        similarity = len(intersection) / len(reference_words)
    
    return {
        "similarity": similarity,
        "response_length": len(response),
        "reference_length": len(reference),
        "has_keywords": any(keyword in response.lower() for keyword in ["transformer", "attention", "neural"])
    }

# ============================================================================
# 结果分析和可视化
# ============================================================================

def analyze_results(results: List[Dict]) -> Dict:
    """分析实验结果"""
    analysis = {
        "model_comparison": {},
        "performance_metrics": {},
        "recommendations": []
    }
    
    # 比较不同模型配置
    for result in results:
        config = result["config"]
        key = f"{config.embedding_model}_{config.concept_model}_{config.response_model}"
        
        analysis["model_comparison"][key] = {
            "description": config.description,
            "execution_time": result["execution_time"],
            "graph_nodes": result["graph_stats"]["nodes"],
            "graph_edges": result["graph_stats"]["edges"],
            "graph_density": result["graph_stats"]["density"],
            "traversal_path_length": result["traversal_results"]["path_length"],
            "response_length": result["metrics"]["response_length"],
            "similarity": result["metrics"].get("similarity", 0),
            "has_keywords": result["metrics"]["has_keywords"]
        }
    
    # 性能指标分析
    execution_times = [r["execution_time"] for r in results]
    similarities = [r["metrics"].get("similarity", 0) for r in results]
    graph_densities = [r["graph_stats"]["density"] for r in results]
    
    analysis["performance_metrics"] = {
        "fastest_model": results[np.argmin(execution_times)]["config"].description,
        "best_similarity": results[np.argmax(similarities)]["config"].description,
        "densest_graph": results[np.argmax(graph_densities)]["config"].description,
        "avg_execution_time": np.mean(execution_times),
        "avg_similarity": np.mean(similarities),
        "avg_graph_density": np.mean(graph_densities)
    }
    
    # 生成建议
    if len(similarities) > 0:
        best_similarity_idx = np.argmax(similarities)
        fastest_idx = np.argmin(execution_times)
        
        analysis["recommendations"] = [
            f"最佳质量模型: {results[best_similarity_idx]['config'].description}",
            f"最快执行模型: {results[fastest_idx]['config'].description}",
            f"平均执行时间: {np.mean(execution_times):.2f}秒",
            f"平均相似度: {np.mean(similarities):.3f}"
        ]
    
    return analysis

def visualize_results(results: List[Dict], analysis: Dict):
    """可视化实验结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 执行时间对比
    configs = [r["config"].description for r in results]
    times = [r["execution_time"] for r in results]
    
    axes[0, 0].bar(configs, times, color='skyblue')
    axes[0, 0].set_title('执行时间对比')
    axes[0, 0].set_ylabel('时间 (秒)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 相似度对比
    similarities = [r["metrics"].get("similarity", 0) for r in results]
    
    axes[0, 1].bar(configs, similarities, color='lightgreen')
    axes[0, 1].set_title('响应相似度对比')
    axes[0, 1].set_ylabel('相似度')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 图密度对比
    densities = [r["graph_stats"]["density"] for r in results]
    
    axes[1, 0].bar(configs, densities, color='lightcoral')
    axes[1, 0].set_title('图密度对比')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 综合评分
    # 综合评分 = 相似度 * 0.6 + (1/执行时间) * 0.3 + 图密度 * 0.1
    max_time = max(times)
    normalized_times = [1 - (t / max_time) for t in times]
    composite_scores = [s * 0.6 + nt * 0.3 + d * 0.1 for s, nt, d in zip(similarities, normalized_times, densities)]
    
    axes[1, 1].bar(configs, composite_scores, color='gold')
    axes[1, 1].set_title('综合评分对比')
    axes[1, 1].set_ylabel('综合评分')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def print_detailed_comparison(results: List[Dict], analysis: Dict):
    """打印详细的对比结果"""
    print("\n" + "="*80)
    print("Graph RAG模型对比实验结果")
    print("="*80)
    
    print("\n1. 模型配置对比:")
    print("-" * 60)
    for i, result in enumerate(results):
        config = result["config"]
        print(f"\n配置 {i+1}: {config.description}")
        print(f"  嵌入模型: {config.embedding_model}")
        print(f"  概念提取: {config.concept_model}")
        print(f"  响应生成: {config.response_model}")
        print(f"  执行时间: {result['execution_time']:.2f}秒")
        print(f"  图节点数: {result['graph_stats']['nodes']}")
        print(f"  图边数: {result['graph_stats']['edges']}")
        print(f"  图密度: {result['graph_stats']['density']:.3f}")
        print(f"  遍历路径长度: {result['traversal_results']['path_length']}")
        print(f"  响应长度: {result['metrics']['response_length']}")
        if 'similarity' in result['metrics']:
            print(f"  相似度: {result['metrics']['similarity']:.3f}")
    
    print("\n2. 性能指标总结:")
    print("-" * 60)
    metrics = analysis["performance_metrics"]
    print(f"  最快模型: {metrics['fastest_model']}")
    print(f"  最佳质量: {metrics['best_similarity']}")
    print(f"  最密图谱: {metrics['densest_graph']}")
    print(f"  平均执行时间: {metrics['avg_execution_time']:.2f}秒")
    print(f"  平均相似度: {metrics['avg_similarity']:.3f}")
    print(f"  平均图密度: {metrics['avg_graph_density']:.3f}")
    
    print("\n3. 建议:")
    print("-" * 60)
    for rec in analysis["recommendations"]:
        print(f"  • {rec}")

# ============================================================================
# 主实验函数
# ============================================================================

def run_model_comparison_experiment():
    """运行完整的模型对比实验"""
    print("Graph RAG模型对比实验")
    print("="*50)
    
    # 测试数据
    test_data = {
        "texts": [
            "Transformer模型是自然语言处理领域的重要突破，它使用自注意力机制来处理序列数据。",
            "自注意力机制允许模型同时关注输入序列的所有位置，捕获长距离依赖关系。",
            "相比传统的RNN和LSTM，Transformer具有更好的并行性和训练效率。",
            "BERT、GPT等模型都是基于Transformer架构构建的预训练语言模型。",
            "注意力机制通过计算查询、键、值之间的相似度来分配权重。"
        ],
        "query": "Transformer模型相比RNN有什么优势？",
        "reference_answer": "Transformer模型相比RNN具有以下优势：1) 更好的并行性，可以同时处理所有位置；2) 更强的长距离依赖捕获能力；3) 更高的训练效率；4) 避免了RNN的梯度消失问题。"
    }
    
    # 运行所有模型配置的实验
    results = []
    for config in MODEL_CONFIGS:
        try:
            result = run_single_experiment(config, test_data)
            results.append(result)
        except Exception as e:
            print(f"实验失败 {config.description}: {e}")
    
    # 分析结果
    analysis = analyze_results(results)
    
    # 可视化结果
    visualize_results(results, analysis)
    
    # 打印详细对比
    print_detailed_comparison(results, analysis)
    
    return results, analysis

# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 运行实验
    results, analysis = run_model_comparison_experiment()
    
    print("\n" + "="*80)
    print("实验完成！")
    print("\n关键发现：")
    print("1. 不同嵌入模型影响语义表示的精度")
    print("2. 不同概念提取模型影响图结构的质量")
    print("3. 不同响应生成模型影响最终答案的质量")
    print("4. 模型组合需要根据具体任务进行优化")
    print("5. 需要在效果和效率之间找到平衡") 