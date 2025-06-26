#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合检索核心算法演示

专注于展示融合检索的核心算法实现，包括：
1. 得分标准化
2. 加权融合
3. 结果排序

作者：AI助手
日期：2024年
"""

import numpy as np
import random
from typing import List, Dict

# ============================================================================
# 核心算法实现
# ============================================================================

def normalize_scores(scores: List[float], epsilon: float = 1e-8) -> List[float]:
    """
    标准化得分（Min-Max标准化）
    
    参数：
    - scores: 原始得分列表
    - epsilon: 避免除零的小值
    
    返回：
    - 标准化后的得分列表（范围0-1）
    """
    if not scores:
        return []
    
    scores_array = np.array(scores)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    
    # 避免除零错误
    if max_score == min_score:
        return [1.0] * len(scores)  # 如果所有得分相同，都设为1
    
    # Min-Max标准化
    normalized = (scores_array - min_score) / (max_score - min_score + epsilon)
    return normalized.tolist()

def fusion_combine_scores(vector_scores: List[float], bm25_scores: List[float], 
                         alpha: float = 0.5) -> List[float]:
    """
    融合两种得分
    
    参数：
    - vector_scores: 向量搜索得分
    - bm25_scores: BM25搜索得分
    - alpha: 向量搜索权重（0-1）
    
    返回：
    - 融合后的得分列表
    """
    # 标准化得分
    norm_vector = normalize_scores(vector_scores)
    norm_bm25 = normalize_scores(bm25_scores)
    
    # 加权融合
    combined = [alpha * v + (1 - alpha) * b for v, b in zip(norm_vector, norm_bm25)]
    
    return combined

def fusion_retrieval_core(vector_scores: List[float], bm25_scores: List[float], 
                         alpha: float = 0.5) -> List[Dict]:
    """
    融合检索核心算法
    
    参数：
    - vector_scores: 向量搜索得分
    - bm25_scores: BM25搜索得分
    - alpha: 向量搜索权重
    
    返回：
    - 包含融合得分和排序的结果列表
    """
    # 标准化得分
    norm_vector = normalize_scores(vector_scores)
    norm_bm25 = normalize_scores(bm25_scores)
    
    # 融合得分
    combined_scores = fusion_combine_scores(vector_scores, bm25_scores, alpha)
    
    # 创建结果列表
    results = []
    for i in range(len(vector_scores)):
        results.append({
            "index": i,
            "vector_score": vector_scores[i],
            "bm25_score": bm25_scores[i],
            "norm_vector_score": norm_vector[i],
            "norm_bm25_score": norm_bm25[i],
            "combined_score": combined_scores[i]
        })
    
    # 按融合得分排序（降序）
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    
    return results

# ============================================================================
# 演示函数
# ============================================================================

def demonstrate_normalization():
    """演示得分标准化"""
    print("=== 得分标准化演示 ===\n")
    
    # 示例1：不同范围的得分
    vector_scores = [0.8, 0.6, 0.9, 0.3, 0.7]  # 范围0.3-0.9
    bm25_scores = [2.5, 8.1, 4.2, 9.3, 1.8]    # 范围1.8-9.3
    
    print("原始得分：")
    print(f"  向量得分: {vector_scores}")
    print(f"  BM25得分: {bm25_scores}")
    
    # 标准化
    norm_vector = normalize_scores(vector_scores)
    norm_bm25 = normalize_scores(bm25_scores)
    
    print("\n标准化后得分：")
    print(f"  标准化向量得分: {[round(score, 3) for score in norm_vector]}")
    print(f"  标准化BM25得分: {[round(score, 3) for score in norm_bm25]}")
    
    print("\n标准化效果：")
    print(f"  向量得分范围: {min(vector_scores):.1f} - {max(vector_scores):.1f}")
    print(f"  BM25得分范围: {min(bm25_scores):.1f} - {max(bm25_scores):.1f}")
    print(f"  标准化后范围: 0.0 - 1.0")
    
    # 示例2：相同得分的情况
    print("\n\n示例2：相同得分的情况")
    same_scores = [0.5, 0.5, 0.5, 0.5]
    norm_same = normalize_scores(same_scores)
    print(f"  原始得分: {same_scores}")
    print(f"  标准化后: {norm_same}")

def demonstrate_fusion():
    """演示融合算法"""
    print("\n\n=== 融合算法演示 ===\n")
    
    # 示例数据
    vector_scores = [0.8, 0.6, 0.9, 0.3, 0.7]
    bm25_scores = [2.5, 8.1, 4.2, 9.3, 1.8]
    
    print("原始得分：")
    print(f"  向量得分: {vector_scores}")
    print(f"  BM25得分: {bm25_scores}")
    
    # 不同alpha值的融合结果
    alphas = [0.3, 0.5, 0.7]
    
    for alpha in alphas:
        print(f"\n融合结果 (alpha={alpha}):")
        results = fusion_retrieval_core(vector_scores, bm25_scores, alpha)
        
        print(f"  排名结果:")
        for i, result in enumerate(results):
            print(f"    排名{i+1}: 文档{result['index']+1} "
                  f"(向量:{result['norm_vector_score']:.3f}, "
                  f"BM25:{result['norm_bm25_score']:.3f}, "
                  f"融合:{result['combined_score']:.3f})")

def demonstrate_alpha_impact():
    """演示alpha参数的影响"""
    print("\n\n=== Alpha参数影响演示 ===\n")
    
    # 创建对比明显的示例
    vector_scores = [0.9, 0.1, 0.8, 0.2]  # 向量搜索偏好文档0和2
    bm25_scores = [0.1, 0.9, 0.2, 0.8]    # BM25偏好文档1和3
    
    print("原始得分：")
    print(f"  向量得分: {vector_scores}")
    print(f"  BM25得分: {bm25_scores}")
    
    # 测试不同alpha值
    test_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("\n不同alpha值的排序结果：")
    for alpha in test_alphas:
        results = fusion_retrieval_core(vector_scores, bm25_scores, alpha)
        top_doc = results[0]['index'] + 1
        
        print(f"  alpha={alpha:.2f}: 最佳文档 = 文档{top_doc} "
              f"(融合得分: {results[0]['combined_score']:.3f})")
    
    print("\n分析：")
    print("  - alpha=0.0: 纯BM25搜索，偏好文档1和3")
    print("  - alpha=1.0: 纯向量搜索，偏好文档0和2")
    print("  - alpha=0.5: 平衡融合，综合考虑两种方法")

def demonstrate_edge_cases():
    """演示边界情况处理"""
    print("\n\n=== 边界情况处理演示 ===\n")
    
    # 情况1：所有得分相同
    print("情况1：所有得分相同")
    same_vector = [0.5, 0.5, 0.5]
    same_bm25 = [0.3, 0.3, 0.3]
    
    results = fusion_retrieval_core(same_vector, same_bm25, 0.5)
    print(f"  向量得分: {same_vector}")
    print(f"  BM25得分: {same_bm25}")
    print(f"  标准化向量: {[round(r['norm_vector_score'], 3) for r in results]}")
    print(f"  标准化BM25: {[round(r['norm_bm25_score'], 3) for r in results]}")
    print(f"  融合得分: {[round(r['combined_score'], 3) for r in results]}")
    
    # 情况2：空列表
    print("\n情况2：空列表处理")
    empty_vector = []
    empty_bm25 = []
    
    try:
        results = fusion_retrieval_core(empty_vector, empty_bm25, 0.5)
        print(f"  结果: {results}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # 情况3：只有一个得分
    print("\n情况3：只有一个得分")
    single_vector = [0.8]
    single_bm25 = [0.6]
    
    results = fusion_retrieval_core(single_vector, single_bm25, 0.5)
    print(f"  向量得分: {single_vector}")
    print(f"  BM25得分: {single_bm25}")
    print(f"  标准化向量: {[round(r['norm_vector_score'], 3) for r in results]}")
    print(f"  标准化BM25: {[round(r['norm_bm25_score'], 3) for r in results]}")
    print(f"  融合得分: {[round(r['combined_score'], 3) for r in results]}")

def demonstrate_mathematical_properties():
    """演示数学性质"""
    print("\n\n=== 数学性质演示 ===\n")
    
    # 测试单调性
    print("1. 单调性测试：")
    vector_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    bm25_scores = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    results = fusion_retrieval_core(vector_scores, bm25_scores, 0.5)
    
    print("  原始排序（向量得分）: ", [i+1 for i in range(len(vector_scores))])
    print("  融合后排序: ", [r['index']+1 for r in results])
    
    # 测试权重影响
    print("\n2. 权重影响测试：")
    test_scores = [0.8, 0.6, 0.9, 0.3]
    bm25_test = [0.2, 0.8, 0.4, 0.9]
    
    for alpha in [0.0, 0.5, 1.0]:
        results = fusion_retrieval_core(test_scores, bm25_test, alpha)
        print(f"  alpha={alpha}: 排序 = {[r['index']+1 for r in results]}")
    
    # 测试范围保持
    print("\n3. 范围保持测试：")
    results = fusion_retrieval_core(vector_scores, bm25_scores, 0.5)
    combined_scores = [r['combined_score'] for r in results]
    
    print(f"  融合得分范围: {min(combined_scores):.3f} - {max(combined_scores):.3f}")
    print(f"  是否在[0,1]范围内: {all(0 <= score <= 1 for score in combined_scores)}")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("融合检索核心算法演示")
    print("=" * 60)
    
    # 演示标准化
    demonstrate_normalization()
    
    # 演示融合算法
    demonstrate_fusion()
    
    # 演示alpha参数影响
    demonstrate_alpha_impact()
    
    # 演示边界情况
    demonstrate_edge_cases()
    
    # 演示数学性质
    demonstrate_mathematical_properties()
    
    print("\n" + "=" * 60)
    print("核心算法演示完成！")
    print("\n关键要点：")
    print("1. 标准化确保不同范围的得分可以公平比较")
    print("2. 加权融合通过alpha参数控制两种方法的影响")
    print("3. 融合算法保持数学性质（单调性、范围保持等）")
    print("4. 边界情况处理确保算法的鲁棒性")
    print("5. Alpha参数的选择直接影响最终排序结果")

if __name__ == "__main__":
    main() 