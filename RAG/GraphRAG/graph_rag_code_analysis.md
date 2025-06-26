# Graph RAG（图增强检索生成）代码详细分析

## 1. 概述

Graph RAG是一种先进的RAG技术，它将知识组织为关联图结构而非扁平文档集合。这种方法能够：
- 保留信息间的关联关系
- 支持关联概念遍历检索
- 提升复杂多维度查询的处理能力
- 通过可视化知识路径增强可解释性

## 2. 系统架构

### 2.1 整体架构图

```
Graph RAG系统架构
├── 文档处理模块
│   ├── PDF文本提取
│   └── 文本分块
├── 知识图谱构建模块
│   ├── 概念提取
│   ├── 嵌入生成
│   └── 图结构构建
├── 图遍历模块
│   ├── 相似度计算
│   ├── 优先级队列遍历
│   └── 路径记录
├── 响应生成模块
│   ├── 上下文整合
│   └── 响应生成
└── 可视化模块
    ├── 图结构可视化
    └── 遍历路径可视化
```

### 2.2 数据流

```
PDF文档 → 文本提取 → 文本分块 → 概念提取 → 嵌入生成 → 图构建 → 图遍历 → 响应生成
```

## 3. 核心组件详解

### 3.1 文档处理模块

#### 3.1.1 PDF文本提取

```python
def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容
    """
    print(f"Extracting text from {pdf_path}...")
    pdf_document = fitz.open(pdf_path)  # 使用PyMuPDF打开PDF
    text = ""
    
    # 遍历PDF的每一页
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]  # 获取页面对象
        text += page.get_text()  # 提取页面文本并追加
    
    return text
```

**技术要点：**
- 使用PyMuPDF库进行PDF处理
- 逐页提取文本内容
- 支持多种PDF格式

#### 3.1.2 文本分块

```python
def chunk_text(text, chunk_size=1000, overlap=200):
    """
    将文本分割成重叠的块
    """
    chunks = []
    
    # 按指定大小和重叠度遍历文本
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "index": len(chunks),
                "start_pos": i,
                "end_pos": i + len(chunk_text)
            })
    
    print(f"Created {len(chunks)} text chunks")
    return chunks
```

**分块策略：**
- 固定大小分块（默认1000字符）
- 重叠分块（默认200字符重叠）
- 保持位置信息

### 3.2 知识图谱构建模块

#### 3.2.1 概念提取

```python
def extract_concepts(text):
    """
    使用OpenAI API从文本中提取关键概念
    """
    system_message = """Extract key concepts and entities from the provided text.
Return ONLY a list of 5-10 key terms, entities, or concepts that are most important in this text.
Format your response as a JSON array of strings."""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Extract key concepts from:\n\n{text[:3000]}"}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    try:
        concepts_json = json.loads(response.choices[0].message.content)
        concepts = concepts_json.get("concepts", [])
        if not concepts and "concepts" not in concepts_json:
            # 尝试获取响应中的任何数组
            for key, value in concepts_json.items():
                if isinstance(value, list):
                    concepts = value
                    break
        return concepts
    except (json.JSONDecodeError, AttributeError):
        # 如果JSON解析失败，使用正则表达式提取
        content = response.choices[0].message.content
        matches = re.findall(r'\[(.*?)\]', content, re.DOTALL)
        if matches:
            items = re.findall(r'"([^"]*)"', matches[0])
            return items
        return []
```

**概念提取特点：**
- 使用LLM进行智能概念提取
- 返回5-10个关键概念
- 支持JSON格式和正则表达式解析
- 限制文本长度以适应API限制

#### 3.2.2 嵌入生成

```python
def create_embeddings(texts, model="text-embedding-ada-002"):
    """
    为给定文本创建嵌入向量
    """
    if not texts:
        return []
    
    # 批处理处理（OpenAI API限制）
    batch_size = 100
    all_embeddings = []
    
    # 分批处理输入文本
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings
```

**嵌入生成特点：**
- 支持批处理以提高效率
- 使用OpenAI的嵌入模型
- 处理空输入情况

#### 3.2.3 知识图谱构建

```python
def build_knowledge_graph(chunks):
    """
    从文本块构建知识图谱
    """
    print("Building knowledge graph...")
    
    # 创建图
    graph = nx.Graph()
    
    # 提取块文本
    texts = [chunk["text"] for chunk in chunks]
    
    # 为所有块创建嵌入
    print("Creating embeddings for chunks...")
    embeddings = create_embeddings(texts)
    
    # 添加节点到图
    print("Adding nodes to the graph...")
    for i, chunk in enumerate(chunks):
        print(f"Extracting concepts for chunk {i+1}/{len(chunks)}...")
        concepts = extract_concepts(chunk["text"])
        
        # 添加带属性的节点
        graph.add_node(i,
                      text=chunk["text"],
                      concepts=concepts,
                      embedding=embeddings[i])
    
    # 基于共享概念连接节点
    print("Creating edges between nodes...")
    for i in range(len(chunks)):
        node_concepts = set(graph.nodes[i]["concepts"])
        
        for j in range(i + 1, len(chunks)):
            # 计算概念重叠
            other_concepts = set(graph.nodes[j]["concepts"])
            shared_concepts = node_concepts.intersection(other_concepts)
            
            # 如果共享概念，添加边
            if shared_concepts:
                # 使用嵌入计算语义相似度
                similarity = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                
                # 基于概念重叠和语义相似度计算边权重
                concept_score = len(shared_concepts) / min(len(node_concepts), len(other_concepts))
                edge_weight = 0.7 * similarity + 0.3 * concept_score
                
                # 只添加有显著关系的边
                if edge_weight > 0.6:
                    graph.add_edge(i, j,
                                  weight=edge_weight,
                                  similarity=similarity,
                                  shared_concepts=list(shared_concepts))
    
    print(f"Knowledge graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph, embeddings
```

**图构建算法：**
1. **节点创建**：每个文本块作为一个节点，包含文本、概念和嵌入
2. **边创建**：基于共享概念和语义相似度创建边
3. **权重计算**：边权重 = 0.7 × 语义相似度 + 0.3 × 概念重叠分数
4. **过滤**：只保留权重 > 0.6 的边

### 3.3 图遍历模块

#### 3.3.1 图遍历算法

```python
def traverse_graph(query, graph, embeddings, top_k=5, max_depth=3):
    """
    遍历知识图谱以找到查询的相关信息
    """
    print(f"Traversing graph for query: {query}")
    
    # 获取查询嵌入
    query_embedding = create_embeddings(query)
    
    # 计算查询与所有节点的相似度
    similarities = []
    for i, node_embedding in enumerate(embeddings):
        similarity = np.dot(query_embedding, node_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding))
        similarities.append((i, similarity))
    
    # 按相似度排序（降序）
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 获取前k个最相似节点作为起始点
    starting_nodes = [node for node, _ in similarities[:top_k]]
    print(f"Starting traversal from {len(starting_nodes)} nodes")
    
    # 初始化遍历
    visited = set()  # 跟踪已访问节点
    traversal_path = []  # 存储遍历路径
    results = []  # 存储结果
    
    # 使用优先级队列进行遍历
    queue = []
    for node in starting_nodes:
        heapq.heappush(queue, (-similarities[node][1], node))  # 负值用于最大堆
    
    # 使用修改的广度优先搜索进行图遍历
    while queue and len(results) < (top_k * 3):  # 限制结果数量
        _, node = heapq.heappop(queue)
        
        if node in visited:
            continue
        
        # 标记为已访问
        visited.add(node)
        traversal_path.append(node)
        
        # 将当前节点的文本添加到结果中
        results.append({
            "text": graph.nodes[node]["text"],
            "concepts": graph.nodes[node]["concepts"],
            "node_id": node
        })
        
        # 如果未达到最大深度，探索邻居
        if len(traversal_path) < max_depth:
            neighbors = [(neighbor, graph[node][neighbor]["weight"])
                        for neighbor in graph.neighbors(node)
                        if neighbor not in visited]
            
            # 基于边权重将邻居添加到队列
            for neighbor, weight in sorted(neighbors, key=lambda x: x[1], reverse=True):
                heapq.heappush(queue, (-weight, neighbor))
    
    print(f"Graph traversal found {len(results)} relevant chunks")
    return results, traversal_path
```

**遍历算法特点：**
1. **起始点选择**：基于查询相似度选择top-k个起始节点
2. **优先级队列**：使用堆进行优先级遍历
3. **深度限制**：最大遍历深度为3
4. **结果限制**：最多返回top_k × 3个结果
5. **路径记录**：记录完整的遍历路径

### 3.4 响应生成模块

```python
def generate_response(query, context_chunks):
    """
    使用检索到的上下文生成响应
    """
    # 从每个块中提取文本
    context_texts = [chunk["text"] for chunk in context_chunks]
    
    # 将提取的文本组合成单个上下文字符串
    combined_context = "\n\n---\n\n".join(context_texts)
    
    # 定义上下文的最大允许长度（OpenAI限制）
    max_context = 14000
    
    # 如果组合上下文超过最大长度，截断
    if len(combined_context) > max_context:
        combined_context = combined_context[:max_context] + "... [truncated]"
    
    # 定义系统消息指导AI助手
    system_message = """You are a helpful AI assistant. Answer the user's question based on the provided context.
If the information is not in the context, say so. Refer to specific parts of the context in your answer when possible."""
    
    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {query}"}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content
```

**响应生成特点：**
- 上下文长度限制（14000字符）
- 使用分隔符组织多个块
- 指导模型基于上下文回答
- 支持截断处理

### 3.5 可视化模块

```python
def visualize_graph_traversal(graph, traversal_path):
    """
    可视化知识图谱和遍历路径
    """
    plt.figure(figsize=(12, 10))
    
    # 定义节点颜色，默认为浅蓝色
    node_color = ['lightblue'] * graph.number_of_nodes()
    
    # 用浅绿色高亮遍历路径节点
    for node in traversal_path:
        node_color[node] = 'lightgreen'
    
    # 用绿色高亮起始节点，红色高亮结束节点
    if traversal_path:
        node_color[traversal_path[0]] = 'green'
        node_color[traversal_path[-1]] = 'red'
    
    # 使用弹簧布局为所有节点创建位置
    pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
    
    # 绘制图节点
    nx.draw_networkx_nodes(graph, pos, node_color=node_color, node_size=500, alpha=0.8)
    
    # 绘制边，宽度与权重成正比
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], width=weight*2, alpha=0.6)
    
    # 用红色虚线绘制遍历路径
    traversal_edges = [(traversal_path[i], traversal_path[i+1])
                      for i in range(len(traversal_path)-1)]
    
    nx.draw_networkx_edges(graph, pos, edgelist=traversal_edges,
                          width=3, alpha=0.8, edge_color='red',
                          style='dashed', arrows=True)
    
    # 为每个节点添加标签，显示第一个概念
    labels = {}
    for node in graph.nodes():
        concepts = graph.nodes[node]['concepts']
        label = concepts[0] if concepts else f"Node {node}"
        labels[node] = f"{node}: {label}"
    
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    
    plt.title("Knowledge Graph with Traversal Path")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
```

**可视化特点：**
- 节点颜色编码：绿色=起始，红色=结束，浅绿色=遍历路径
- 边宽度与权重成正比
- 遍历路径用红色虚线显示
- 节点标签显示第一个概念

## 4. 完整管道

### 4.1 Graph RAG管道

```python
def graph_rag_pipeline(pdf_path, query, chunk_size=1000, chunk_overlap=200, top_k=3):
    """
    从文档到答案的完整Graph RAG管道
    """
    # 从PDF文档提取文本
    text = extract_text_from_pdf(pdf_path)
    
    # 将提取的文本分割成重叠的块
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # 从文本块构建知识图谱
    graph, embeddings = build_knowledge_graph(chunks)
    
    # 遍历知识图谱以找到查询的相关信息
    relevant_chunks, traversal_path = traverse_graph(query, graph, embeddings, top_k)
    
    # 基于查询和相关块生成响应
    response = generate_response(query, relevant_chunks)
    
    # 可视化图遍历路径
    visualize_graph_traversal(graph, traversal_path)
    
    # 返回查询、响应、相关块、遍历路径和图
    return {
        "query": query,
        "response": response,
        "relevant_chunks": relevant_chunks,
        "traversal_path": traversal_path,
        "graph": graph
    }
```

### 4.2 评估函数

```python
def evaluate_graph_rag(pdf_path, test_queries, reference_answers=None):
    """
    在多个测试查询上评估Graph RAG
    """
    # 从PDF提取文本
    text = extract_text_from_pdf(pdf_path)
    
    # 将文本分割成块
    chunks = chunk_text(text)
    
    # 构建知识图谱（对所有查询只做一次）
    graph, embeddings = build_knowledge_graph(chunks)
    
    results = []
    
    for i, query in enumerate(test_queries):
        print(f"\n\n=== Evaluating Query {i+1}/{len(test_queries)} ===")
        print(f"Query: {query}")
        
        # 遍历图以找到相关信息
        relevant_chunks, traversal_path = traverse_graph(query, graph, embeddings)
        
        # 生成响应
        response = generate_response(query, relevant_chunks)
        
        # 如果可用，与参考答案比较
        reference = None
        comparison = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
            comparison = compare_with_reference(response, reference, query)
        
        # 为当前查询追加结果
        results.append({
            "query": query,
            "response": response,
            "reference_answer": reference,
            "comparison": comparison,
            "traversal_path_length": len(traversal_path),
            "relevant_chunks_count": len(relevant_chunks)
        })
        
        # 显示结果
        print(f"\nResponse: {response}\n")
        if comparison:
            print(f"Comparison: {comparison}\n")
    
    # 返回评估结果和图统计
    return {
        "results": results,
        "graph_stats": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        }
    }
```

## 5. 关键技术实现

### 5.1 概念提取策略

```python
# 概念提取的多种策略
def extract_concepts_advanced(text):
    """
    高级概念提取策略
    """
    # 策略1：使用LLM提取
    concepts = extract_concepts(text)
    
    # 策略2：使用关键词提取
    keywords = extract_keywords(text)
    
    # 策略3：使用命名实体识别
    entities = extract_entities(text)
    
    # 合并所有概念
    all_concepts = list(set(concepts + keywords + entities))
    
    return all_concepts[:10]  # 限制为10个概念
```

### 5.2 图权重计算

```python
def calculate_edge_weight(node1_concepts, node2_concepts, similarity):
    """
    计算边权重的多种策略
    """
    # 策略1：概念重叠 + 语义相似度
    shared_concepts = set(node1_concepts).intersection(set(node2_concepts))
    concept_score = len(shared_concepts) / min(len(node1_concepts), len(node2_concepts))
    weight1 = 0.7 * similarity + 0.3 * concept_score
    
    # 策略2：Jaccard相似度
    jaccard = len(shared_concepts) / len(set(node1_concepts).union(set(node2_concepts)))
    weight2 = 0.5 * similarity + 0.5 * jaccard
    
    # 策略3：加权平均
    weight3 = (similarity + concept_score + jaccard) / 3
    
    return weight1  # 使用策略1
```

### 5.3 遍历算法优化

```python
def optimized_traverse_graph(query, graph, embeddings, top_k=5, max_depth=3):
    """
    优化的图遍历算法
    """
    # 1. 多起始点策略
    starting_nodes = select_starting_nodes(query, graph, embeddings, top_k)
    
    # 2. 动态深度调整
    max_depth = adjust_depth_based_on_complexity(query)
    
    # 3. 路径多样性
    diverse_paths = find_diverse_paths(graph, starting_nodes, max_depth)
    
    # 4. 结果去重和排序
    results = deduplicate_and_rank_results(diverse_paths)
    
    return results
```

## 6. 性能优化

### 6.1 批处理优化

```python
def batch_process_chunks(chunks, batch_size=10):
    """
    批处理块以提高效率
    """
    results = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # 并行处理批次
        batch_results = process_batch_parallel(batch)
        results.extend(batch_results)
    
    return results
```

### 6.2 缓存机制

```python
class GraphCache:
    """
    图缓存机制
    """
    def __init__(self):
        self.embeddings_cache = {}
        self.concepts_cache = {}
        self.graph_cache = {}
    
    def get_or_compute_embedding(self, text):
        """获取或计算嵌入"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        embedding = create_embeddings([text])[0]
        self.embeddings_cache[text] = embedding
        return embedding
```

## 7. 评估指标

### 7.1 图质量指标

- **节点数**：图中的节点数量
- **边数**：图中的边数量
- **平均度**：节点的平均连接数
- **连通性**：图的连通组件数量

### 7.2 检索质量指标

- **遍历路径长度**：找到答案所需的遍历步数
- **相关块数量**：检索到的相关文本块数量
- **响应质量**：与参考答案的匹配度

## 8. 应用场景

### 8.1 适用场景

- **复杂查询**：涉及多个概念和关系的查询
- **知识发现**：需要发现隐含关联的场景
- **可解释性要求**：需要解释检索路径的应用
- **多跳推理**：需要多步推理的复杂问题

### 8.2 不适用场景

- **简单查询**：单一概念的直接查询
- **实时要求**：需要快速响应的场景
- **资源限制**：计算资源有限的环境

## 9. 总结

Graph RAG通过图结构组织知识，实现了：

1. **关系保持**：显式保留概念间的关联关系
2. **智能遍历**：基于图结构进行智能信息检索
3. **可解释性**：通过可视化展示检索路径
4. **复杂处理**：更好地处理多概念复杂查询

这种方法的优势在于能够捕获文档中隐含的概念关系，提供更全面和相关的检索结果。 