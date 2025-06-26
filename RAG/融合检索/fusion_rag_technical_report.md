# 融合检索（Fusion Retrieval）RAG系统技术报告

## 1. 概述

### 1.1 什么是融合检索？

融合检索是一种先进的RAG技术，它结合了语义向量搜索和关键词BM25检索的优势。传统的RAG系统通常只使用向量搜索，但这种方法存在局限性：

- **向量搜索**：擅长语义相似性，但可能遗漏精确的关键词匹配
- **关键词搜索**：擅长精确匹配，但缺乏语义理解能力
- **不同查询**：在不同检索方法下表现各异

融合检索通过以下方式实现优势互补：
1. 同时执行基于向量和基于关键词的检索
2. 对两种方法的得分进行标准化处理
3. 通过加权公式将二者结合
4. 基于组合得分对文档进行排序

### 1.2 核心优势

- **信息完整性**：同时捕捉语义相似性和精确关键词匹配
- **检索精度**：通过融合算法提高检索准确性
- **适应性**：能够适应不同类型的查询需求
- **鲁棒性**：减少单一方法的局限性

## 2. 系统架构

### 2.1 整体架构图

```
融合检索RAG系统架构
├── 文档处理模块
│   ├── PDF文本提取
│   ├── 文本清理
│   └── 文本分块
├── 向量检索模块
│   ├── 嵌入生成
│   ├── 向量存储
│   └── 相似度计算
├── 关键词检索模块
│   ├── BM25索引构建
│   ├── 关键词匹配
│   └── 得分计算
├── 融合算法模块
│   ├── 得分标准化
│   ├── 加权融合
│   └── 结果排序
└── 响应生成模块
    ├── 上下文整合
    ├── 响应生成
    └── 结果评估
```

### 2.2 数据流

```
PDF文档 → 文本提取 → 文本清理 → 文本分块 → 向量化 → 向量存储
     ↓
  关键词索引 → BM25检索 → 得分计算
     ↓
    融合算法 → 结果排序 → 响应生成
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

#### 3.1.2 文本清理

```python
def clean_text(text):
    """
    清理文本，移除多余空白和特殊字符
    """
    # 将多个空白字符替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 修复常见OCR问题
    text = text.replace('\\t', ' ')
    text = text.replace('\\n', ' ')
    
    # 移除首尾空白并确保单词间单空格
    text = ' '.join(text.split())
    
    return text
```

**清理策略：**
- 正则表达式处理多余空白
- 修复OCR常见问题
- 标准化文本格式

#### 3.1.3 文本分块

```python
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    将文本分割成重叠的块
    """
    chunks = []
    
    # 按指定大小和重叠度遍历文本
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]  # 提取指定大小的块
        if chunk:  # 确保不添加空块
            chunk_data = {
                "text": chunk,
                "metadata": {
                    "start_char": i,
                    "end_char": i + len(chunk)
                }
            }
            chunks.append(chunk_data)
    
    print(f"Created {len(chunks)} text chunks")
    return chunks
```

**分块策略：**
- 固定大小分块（默认1000字符）
- 重叠分块（默认200字符重叠）
- 保持元数据信息

### 3.2 向量检索模块

#### 3.2.1 嵌入生成

```python
def create_embeddings(texts, model="text-embedding-ada-002"):
    """
    为给定文本创建嵌入向量
    """
    # 处理字符串和列表输入
    input_texts = texts if isinstance(texts, list) else [texts]
    
    # 批处理处理（OpenAI API限制）
    batch_size = 100
    all_embeddings = []
    
    # 分批处理输入文本
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]
        
        # 为当前批次创建嵌入
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        
        # 从响应中提取嵌入
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    # 如果输入是字符串，返回第一个嵌入
    if isinstance(texts, str):
        return all_embeddings[0]
    
    # 否则返回所有嵌入
    return all_embeddings
```

**技术要点：**
- 支持批处理以提高效率
- 处理单个和多个文本输入
- 使用OpenAI的嵌入模型

#### 3.2.2 向量存储

```python
class SimpleVectorStore:
    """
    使用NumPy的简单向量存储实现
    """
    def __init__(self):
        self.vectors = []  # 存储嵌入向量
        self.texts = []    # 存储文本内容
        self.metadata = [] # 存储元数据
    
    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储添加单个项目
        """
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def add_items(self, items, embeddings):
        """
        向向量存储添加多个项目
        """
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            self.add_item(
                text=item["text"],
                embedding=embedding,
                metadata={**item.get("metadata", {}), "index": i}
            )
    
    def similarity_search_with_scores(self, query_embedding, k=5):
        """
        查找与查询嵌入最相似的项目，返回相似度得分
        """
        if not self.vectors:
            return []
        
        query_vector = np.array(query_embedding)
        
        # 使用余弦相似度计算相似性
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = cosine_similarity([query_vector], [vector])[0][0]
            similarities.append((i, similarity))
        
        # 按相似度排序（降序）
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个结果及得分
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })
        
        return results
```

**存储特点：**
- 使用NumPy进行高效计算
- 支持余弦相似度搜索
- 返回带得分的搜索结果

### 3.3 关键词检索模块

#### 3.3.1 BM25索引构建

```python
def create_bm25_index(chunks):
    """
    从给定块创建BM25索引
    """
    # 从每个块提取文本
    texts = [chunk["text"] for chunk in chunks]
    
    # 通过空格分割对每个文档进行分词
    tokenized_docs = [text.split() for text in texts]
    
    # 使用分词文档创建BM25索引
    bm25 = BM25Okapi(tokenized_docs)
    
    print(f"Created BM25 index with {len(texts)} documents")
    return bm25
```

**BM25算法特点：**
- 基于概率的检索模型
- 考虑词频和文档长度
- 适合关键词精确匹配

#### 3.3.2 BM25搜索

```python
def bm25_search(bm25, chunks, query, k=5):
    """
    使用查询搜索BM25索引
    """
    # 通过分割将查询分词
    query_tokens = query.split()
    
    # 获取查询词对索引文档的BM25得分
    scores = bm25.get_scores(query_tokens)
    
    # 初始化结果列表
    results = []
    
    # 遍历得分和对应块
    for i, score in enumerate(scores):
        metadata = chunks[i].get("metadata", {}).copy()
        metadata["index"] = i
        
        results.append({
            "text": chunks[i]["text"],
            "metadata": metadata,
            "bm25_score": float(score)
        })
    
    # 按BM25得分降序排序
    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    
    # 返回前k个结果
    return results[:k]
```

**搜索特点：**
- 支持多词查询
- 返回带BM25得分的搜索结果
- 按得分排序

### 3.4 融合算法模块

#### 3.4.1 融合检索核心算法

```python
def fusion_retrieval(query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
    """
    执行融合检索，结合基于向量和BM25的搜索
    """
    print(f"Performing fusion retrieval for query: {query}")
    
    # 定义小epsilon避免除零
    epsilon = 1e-8
    
    # 获取向量搜索结果
    query_embedding = create_embeddings(query)
    vector_results = vector_store.similarity_search_with_scores(query_embedding, k=len(chunks))
    
    # 获取BM25搜索结果
    bm25_results = bm25_search(bm25_index, chunks, query, k=len(chunks))
    
    # 创建字典映射文档索引到得分
    vector_scores_dict = {result["metadata"]["index"]: result["similarity"] for result in vector_results}
    bm25_scores_dict = {result["metadata"]["index"]: result["bm25_score"] for result in bm25_results}
    
    # 确保所有文档都有两种方法的得分
    all_docs = vector_store.get_all_documents()
    combined_results = []
    
    for i, doc in enumerate(all_docs):
        vector_score = vector_scores_dict.get(i, 0.0)
        bm25_score = bm25_scores_dict.get(i, 0.0)
        combined_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "index": i
        })
    
    # 提取得分作为数组
    vector_scores = np.array([doc["vector_score"] for doc in combined_results])
    bm25_scores = np.array([doc["bm25_score"] for doc in combined_results])
    
    # 标准化得分
    norm_vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    norm_bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)
    
    # 计算组合得分
    combined_scores = alpha * norm_vector_scores + (1 - alpha) * norm_bm25_scores
    
    # 将组合得分添加到结果中
    for i, score in enumerate(combined_scores):
        combined_results[i]["combined_score"] = float(score)
    
    # 按组合得分排序（降序）
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # 返回前k个结果
    top_results = combined_results[:k]
    
    print(f"Retrieved {len(top_results)} documents with fusion retrieval")
    return top_results
```

**融合算法特点：**
- **得分标准化**：使用Min-Max标准化确保两种得分在相同范围内
- **加权融合**：通过alpha参数控制向量搜索和BM25的权重
- **组合排序**：基于融合得分进行最终排序

### 3.5 响应生成模块

#### 3.5.1 响应生成

```python
def generate_response(query, context):
    """
    基于查询和上下文生成响应
    """
    # 定义系统提示词指导AI助手
    system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context.
    If the context doesn't contain relevant information to answer the question fully, acknowledge this limitation."""
    
    # 格式化用户提示词
    user_prompt = f"""Context:
    {context}
    
    Question: {query}
    
    Please answer the question based on the provided context."""
    
    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    
    return response.choices[0].message.content
```

## 4. 评估和对比

### 4.1 检索方法对比

```python
def compare_retrieval_methods(query, chunks, vector_store, bm25_index, k=5, alpha=0.5, reference_answer=None):
    """
    比较不同检索方法对查询的效果
    """
    print(f"\n=== Comparing retrieval methods for query: {query} ===\n")
    
    # 运行仅向量RAG
    print("\nRunning vector-only RAG...")
    vector_result = vector_only_rag(query, vector_store, k)
    
    # 运行仅BM25 RAG
    print("\nRunning BM25-only RAG...")
    bm25_result = bm25_only_rag(query, chunks, bm25_index, k)
    
    # 运行融合RAG
    print("\nRunning fusion RAG...")
    fusion_result = answer_with_fusion_rag(query, chunks, vector_store, bm25_index, k, alpha)
    
    # 比较不同检索方法的响应
    print("\nComparing responses...")
    comparison = evaluate_responses(
        query,
        vector_result["response"],
        bm25_result["response"],
        fusion_result["response"],
        reference_answer
    )
    
    return {
        "query": query,
        "vector_result": vector_result,
        "bm25_result": bm25_result,
        "fusion_result": fusion_result,
        "comparison": comparison
    }
```

### 4.2 响应评估

```python
def evaluate_responses(query, vector_response, bm25_response, fusion_response, reference_answer=None):
    """
    评估不同检索方法的响应
    """
    system_prompt = """You are an expert evaluator of RAG systems. Compare responses from three different retrieval approaches:
    1. Vector-based retrieval: Uses semantic similarity for document retrieval
    2. BM25 keyword retrieval: Uses keyword matching for document retrieval
    3. Fusion retrieval: Combines both vector and keyword approaches
    
    Evaluate the responses based on:
    - Relevance to the query
    - Factual correctness
    - Comprehensiveness
    - Clarity and coherence"""
    
    user_prompt = f"""Query: {query}
    
    Vector-based response:
    {vector_response}
    
    BM25 keyword response:
    {bm25_response}
    
    Fusion response:
    {fusion_response}
    """
    
    if reference_answer:
        user_prompt += f"""
            Reference answer:
            {reference_answer}
        """
    
    user_prompt += """
    Please provide a detailed comparison of these three responses. Which approach performed best for this query and why?
    Be specific about the strengths and weaknesses of each approach for this particular query.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content
```

## 5. 关键技术实现

### 5.1 得分标准化

```python
# 标准化得分
norm_vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)
norm_bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)
```

**标准化目的：**
- 确保两种得分在相同范围内（0-1）
- 避免因得分范围不同导致的权重偏差
- 使用epsilon避免除零错误

### 5.2 加权融合

```python
# 计算组合得分
combined_scores = alpha * norm_vector_scores + (1 - alpha) * norm_bm25_scores
```

**融合策略：**
- alpha控制向量搜索权重
- (1-alpha)控制BM25权重
- 默认alpha=0.5表示等权重

### 5.3 批处理优化

```python
# 批处理处理（OpenAI API限制）
batch_size = 100
all_embeddings = []

for i in range(0, len(input_texts), batch_size):
    batch = input_texts[i:i + batch_size]
    response = client.embeddings.create(model=model, input=batch)
    batch_embeddings = [item.embedding for item in response.data]
    all_embeddings.extend(batch_embeddings)
```

**优化目的：**
- 避免API调用限制
- 提高处理效率
- 减少网络开销

## 6. 评估指标

### 6.1 检索质量指标

- **相关性**：检索结果与查询的相关程度
- **准确性**：检索结果的事实正确性
- **完整性**：检索结果的信息完整程度
- **清晰度**：检索结果的可理解性

### 6.2 性能指标

- **检索速度**：不同方法的检索时间
- **内存使用**：向量存储和BM25索引的内存占用
- **可扩展性**：处理大规模文档的能力

## 7. 应用场景

### 7.1 适用场景

- **学术论文检索**：需要精确术语匹配和语义理解
- **技术文档搜索**：包含专业术语和概念解释
- **知识库问答**：需要全面的信息检索
- **研究助手**：支持复杂查询和深度理解

### 7.2 不适用场景

- **实时搜索**：计算复杂度较高
- **简单关键词匹配**：BM25已足够
- **纯语义搜索**：向量搜索已足够

## 8. 部署和配置

### 8.1 环境要求

```bash
pip install PymuPDF rank_bm25 numpy scikit-learn openai
```

### 8.2 配置参数

```python
# 推荐配置
config = {
    "chunk_size": 1000,        # 文本块大小
    "chunk_overlap": 200,      # 块重叠大小
    "retrieval_k": 5,          # 检索结果数量
    "alpha": 0.5,              # 向量搜索权重
    "batch_size": 100,         # 批处理大小
    "epsilon": 1e-8            # 避免除零的小值
}
```

### 8.3 性能优化建议

1. **并行处理**：并行执行向量搜索和BM25搜索
2. **缓存机制**：缓存嵌入向量和搜索结果
3. **索引优化**：使用高效的向量数据库和BM25索引
4. **预计算**：预计算常用查询的结果

## 9. 未来发展方向

### 9.1 技术改进

1. **多模态融合**：结合文本、图像等多种模态
2. **动态权重**：根据查询类型动态调整alpha值
3. **深度学习融合**：使用神经网络学习最优融合策略
4. **实时学习**：根据用户反馈调整检索策略

### 9.2 功能扩展

1. **多语言支持**：支持多种语言的融合检索
2. **个性化检索**：根据用户偏好调整检索策略
3. **增量更新**：支持知识库的增量更新
4. **交互式检索**：支持多轮交互式检索

## 10. 总结

融合检索RAG系统通过结合向量搜索和关键词搜索的优势，显著提升了检索质量。主要优势包括：

- **信息完整性**：同时捕捉语义相似性和精确关键词匹配
- **检索精度**：通过融合算法提高检索准确性
- **适应性**：能够适应不同类型的查询需求
- **鲁棒性**：减少单一方法的局限性

该系统特别适用于需要高精度信息检索的应用场景，为RAG技术的发展提供了新的思路和方法。 