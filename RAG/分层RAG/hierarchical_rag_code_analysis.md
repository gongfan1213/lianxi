# 分层索引RAG系统核心代码详细分析

## 1. 系统概述

`main.py`实现了一个基于分层索引的RAG（检索增强生成）系统，通过双层检索策略提升检索效果：
- **摘要层检索**：首先通过文档摘要定位相关章节
- **细节层检索**：再从相关章节中检索具体细节

## 2. 核心架构组件

### 2.1 文档处理模块

#### 2.1.1 PDF文本提取

```python
def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容，按页面组织
    
    核心功能：
    1. 使用PyMuPDF库打开PDF文件
    2. 逐页提取文本内容
    3. 过滤掉内容过少的页面（少于50字符）
    4. 保留页面元数据（源文件路径、页码）
    """
    print(f"Extracting text from {pdf_path}...")
    pdf = fitz.open(pdf_path)  # 使用PyMuPDF打开PDF
    pages = []

    # 遍历PDF的每一页
    for page_num in range(len(pdf)):
        page = pdf[page_num]  # 获取当前页面
        text = page.get_text()  # 提取页面文本

        # 跳过内容过少的页面
        if len(text.strip()) > 50:
            pages.append({
                "text": text,
                "metadata": {
                    "source": pdf_path,  # 源文件路径
                    "page": page_num + 1  # 页码（从1开始）
                }
            })

    print(f"Extracted {len(pages)} pages with content")
    return pages
```

**技术要点：**
- 使用PyMuPDF库进行PDF处理
- 按页面组织文本，保留页面结构
- 过滤无效页面，提高处理效率
- 保留完整的元数据信息

#### 2.1.2 文本分块处理

```python
def chunk_text(text, metadata, chunk_size=1000, overlap=200):
    """
    将文本分割成重叠的块，同时保留元数据
    
    核心算法：
    1. 按指定大小和重叠度分割文本
    2. 为每个块添加位置信息
    3. 保留原始元数据
    4. 标记块类型（非摘要）
    """
    chunks = []

    # 按指定步长遍历文本
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]  # 提取当前块

        # 跳过过小的块
        if chunk_text and len(chunk_text.strip()) > 50:
            # 复制元数据并添加块特定信息
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(chunks),  # 块索引
                "start_char": i,  # 起始字符位置
                "end_char": i + len(chunk_text),  # 结束字符位置
                "is_summary": False  # 标记为非摘要
            })

            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

    return chunks
```

**技术要点：**
- 使用滑动窗口算法进行文本分块
- 保留块间的重叠，避免信息丢失
- 记录每个块的精确位置信息
- 维护完整的元数据链

### 2.2 向量存储实现

#### 2.2.1 简单向量存储类

```python
class SimpleVectorStore:
    """
    基于NumPy的简单向量存储实现
    
    核心功能：
    1. 存储文本、向量和元数据
    2. 支持余弦相似度搜索
    3. 支持结果过滤
    4. 提供top-k检索
    """
    def __init__(self):
        self.vectors = []  # 存储向量嵌入
        self.texts = []    # 存储文本内容
        self.metadata = [] # 存储元数据

    def add_item(self, text, embedding, metadata=None):
        """添加项目到向量存储"""
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        基于余弦相似度的向量搜索
        
        核心算法：
        1. 计算查询向量与所有存储向量的余弦相似度
        2. 应用可选的过滤函数
        3. 按相似度排序
        4. 返回top-k结果
        """
        if not self.vectors:
            return []

        query_vector = np.array(query_embedding)
        similarities = []

        # 计算所有向量的相似度
        for i, vector in enumerate(self.vectors):
            # 应用过滤函数
            if filter_func and not filter_func(self.metadata[i]):
                continue

            # 计算余弦相似度
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            similarities.append((i, similarity))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回top-k结果
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

**技术要点：**
- 使用NumPy进行高效的向量计算
- 实现余弦相似度算法
- 支持灵活的结果过滤
- 提供结构化的搜索结果

### 2.3 嵌入生成模块

```python
def create_embeddings(texts, model="text-embedding-ada-002"):
    """
    为给定文本创建向量嵌入
    
    核心功能：
    1. 批量处理文本（OpenAI API限制）
    2. 调用OpenAI嵌入API
    3. 提取嵌入向量
    4. 处理空输入情况
    """
    if not texts:
        return []

    batch_size = 100  # OpenAI API批处理大小限制
    all_embeddings = []

    # 分批处理文本
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # 调用OpenAI API创建嵌入
        response = client.embeddings.create(
            model=model,
            input=batch
        )

        # 提取嵌入向量
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
```

**技术要点：**
- 处理API调用限制（批处理）
- 错误处理和边界情况
- 支持不同的嵌入模型
- 高效的批量处理

### 2.4 摘要生成模块

```python
def generate_page_summary(page_text):
    """
    为页面生成简洁摘要
    
    核心功能：
    1. 使用GPT-3.5-turbo生成摘要
    2. 控制输入长度（token限制）
    3. 设置合适的温度参数
    4. 提供详细的系统提示
    """
    system_prompt = """You are an expert summarization system.
    Create a detailed summary of the provided text.
    Focus on capturing the main topics, key information, and important facts.
    Your summary should be comprehensive enough to understand what the page contains
    but more concise than the original."""

    # 截断过长的输入
    max_tokens = 6000
    truncated_text = page_text[:max_tokens] if len(page_text) > max_tokens else page_text

    # 调用OpenAI API生成摘要
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please summarize this text:\n\n{truncated_text}"}
        ],
        temperature=0.3  # 较低温度确保一致性
    )

    return response.choices[0].message.content
```

**技术要点：**
- 使用专门的摘要系统提示
- 控制输入长度避免token超限
- 设置合适的温度参数
- 确保摘要质量和一致性

## 3. 分层处理核心算法

### 3.1 分层文档处理

```python
def process_document_hierarchically(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    将文档处理成分层索引
    
    核心流程：
    1. 提取PDF页面
    2. 为每页生成摘要
    3. 为每页创建详细块
    4. 为摘要和详细块创建嵌入
    5. 构建两个向量存储
    """
    # 提取页面
    pages = extract_text_from_pdf(pdf_path)

    # 生成页面摘要
    print("Generating page summaries...")
    summaries = []
    for i, page in enumerate(pages):
        print(f"Summarizing page {i+1}/{len(pages)}...")
        summary_text = generate_page_summary(page["text"])

        # 创建摘要元数据
        summary_metadata = page["metadata"].copy()
        summary_metadata.update({"is_summary": True})

        summaries.append({
            "text": summary_text,
            "metadata": summary_metadata
        })

    # 创建详细块
    detailed_chunks = []
    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            page["metadata"],
            chunk_size,
            chunk_overlap
        )
        detailed_chunks.extend(page_chunks)

    print(f"Created {len(detailed_chunks)} detailed chunks")

    # 创建嵌入
    print("Creating embeddings for summaries...")
    summary_texts = [summary["text"] for summary in summaries]
    summary_embeddings = create_embeddings(summary_texts)

    print("Creating embeddings for detailed chunks...")
    chunk_texts = [chunk["text"] for chunk in detailed_chunks]
    chunk_embeddings = create_embeddings(chunk_texts)

    # 构建向量存储
    summary_store = SimpleVectorStore()
    detailed_store = SimpleVectorStore()

    # 添加摘要到摘要存储
    for i, summary in enumerate(summaries):
        summary_store.add_item(
            text=summary["text"],
            embedding=summary_embeddings[i],
            metadata=summary["metadata"]
        )

    # 添加块到详细存储
    for i, chunk in enumerate(detailed_chunks):
        detailed_store.add_item(
            text=chunk["text"],
            embedding=chunk_embeddings[i],
            metadata=chunk["metadata"]
        )

    print(f"Created vector stores with {len(summaries)} summaries and {len(detailed_chunks)} chunks")
    return summary_store, detailed_store
```

**技术要点：**
- 双层索引结构：摘要层 + 详细层
- 并行处理摘要和详细块
- 保持元数据一致性
- 高效的向量存储构建

### 3.2 分层检索算法

```python
def retrieve_hierarchically(query, summary_store, detailed_store, k_summaries=3, k_chunks=5):
    """
    使用分层索引进行信息检索
    
    核心算法：
    1. 在摘要层检索相关页面
    2. 基于相关页面过滤详细块
    3. 在详细层检索具体信息
    4. 关联摘要和详细信息
    """
    print(f"Performing hierarchical retrieval for query: {query}")

    # 创建查询嵌入
    query_embedding = create_embeddings(query)

    # 第一步：检索相关摘要
    summary_results = summary_store.similarity_search(
        query_embedding,
        k=k_summaries
    )

    print(f"Retrieved {len(summary_results)} relevant summaries")

    # 收集相关页面
    relevant_pages = [result["metadata"]["page"] for result in summary_results]

    # 创建页面过滤函数
    def page_filter(metadata):
        return metadata["page"] in relevant_pages

    # 第二步：从相关页面检索详细块
    detailed_results = detailed_store.similarity_search(
        query_embedding,
        k=k_chunks * len(relevant_pages),
        filter_func=page_filter
    )

    print(f"Retrieved {len(detailed_results)} detailed chunks from relevant pages")

    # 为每个结果添加对应的摘要信息
    for result in detailed_results:
        page = result["metadata"]["page"]
        matching_summaries = [s for s in summary_results if s["metadata"]["page"] == page]
        if matching_summaries:
            result["summary"] = matching_summaries[0]["text"]

    return detailed_results
```

**技术要点：**
- 两步检索策略：摘要→详细
- 基于页面过滤的精确检索
- 关联摘要和详细信息
- 可配置的检索参数

## 4. 响应生成模块

### 4.1 上下文响应生成

```python
def generate_response(query, retrieved_chunks):
    """
    基于查询和检索块生成响应
    
    核心功能：
    1. 整合检索到的上下文
    2. 使用GPT-3.5-turbo生成响应
    3. 包含页面引用
    4. 确保响应准确性
    """
    # 准备上下文部分
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks):
        page_num = chunk["metadata"]["page"]
        context_parts.append(f"[Page {page_num}]: {chunk['text']}")

    # 组合上下文
    context = "\n\n".join(context_parts)

    # 系统提示
    system_message = """You are a helpful AI assistant answering questions based on the provided context.
Use the information from the context to answer the user's question accurately.
If the context doesn't contain relevant information, acknowledge that.
Include page numbers when referencing specific information."""

    # 生成响应
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.2  # 低温度确保准确性
    )

    return response.choices[0].message.content
```

**技术要点：**
- 结构化上下文组织
- 包含页面引用信息
- 使用专门的系统提示
- 控制生成温度确保准确性

## 5. 完整RAG管道

### 5.1 分层RAG管道

```python
def hierarchical_rag(query, pdf_path, chunk_size=1000, chunk_overlap=200,
                    k_summaries=3, k_chunks=5, regenerate=False):
    """
    完整的分层RAG管道
    
    核心流程：
    1. 检查缓存，决定是否重新处理文档
    2. 处理文档创建分层索引
    3. 执行分层检索
    4. 生成响应
    5. 返回完整结果
    """
    # 缓存文件名
    summary_store_file = f"{os.path.basename(pdf_path)}_summary_store.pkl"
    detailed_store_file = f"{os.path.basename(pdf_path)}_detailed_store.pkl"

    # 检查是否需要重新处理
    if regenerate or not os.path.exists(summary_store_file) or not os.path.exists(detailed_store_file):
        print("Processing document and creating vector stores...")
        summary_store, detailed_store = process_document_hierarchically(
            pdf_path, chunk_size, chunk_overlap
        )

        # 保存向量存储
        with open(summary_store_file, 'wb') as f:
            pickle.dump(summary_store, f)

        with open(detailed_store_file, 'wb') as f:
            pickle.dump(detailed_store, f)
    else:
        # 加载现有向量存储
        print("Loading existing vector stores...")
        with open(summary_store_file, 'rb') as f:
            summary_store = pickle.load(f)

        with open(detailed_store_file, 'rb') as f:
            detailed_store = pickle.load(f)

    # 执行分层检索
    retrieved_chunks = retrieve_hierarchically(
        query, summary_store, detailed_store, k_summaries, k_chunks
    )

    # 生成响应
    response = generate_response(query, retrieved_chunks)

    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks,
        "summary_count": len(summary_store.texts),
        "detailed_count": len(detailed_store.texts)
    }
```

**技术要点：**
- 智能缓存机制
- 完整的端到端流程
- 可配置的参数
- 详细的结果统计

### 5.2 标准RAG对比

```python
def standard_rag(query, pdf_path, chunk_size=1000, chunk_overlap=200, k=15):
    """
    标准RAG管道（非分层）
    
    核心流程：
    1. 直接分块处理文档
    2. 创建单一向量存储
    3. 执行直接检索
    4. 生成响应
    """
    # 提取页面
    pages = extract_text_from_pdf(pdf_path)

    # 直接创建块
    chunks = []
    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            page["metadata"],
            chunk_size,
            chunk_overlap
        )
        chunks.extend(page_chunks)

    print(f"Created {len(chunks)} chunks for standard RAG")

    # 创建向量存储
    store = SimpleVectorStore()

    # 创建嵌入
    print("Creating embeddings for chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    # 添加块到存储
    for i, chunk in enumerate(chunks):
        store.add_item(
            text=chunk["text"],
            embedding=embeddings[i],
            metadata=chunk["metadata"]
        )

    # 检索
    query_embedding = create_embeddings(query)
    retrieved_chunks = store.similarity_search(query_embedding, k=k)
    print(f"Retrieved {len(retrieved_chunks)} chunks with standard RAG")

    # 生成响应
    response = generate_response(query, retrieved_chunks)

    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks
    }
```

**技术要点：**
- 单一层次的处理
- 直接向量检索
- 与分层RAG的对比基准
- 相同的响应生成机制

## 6. 评估系统

### 6.1 响应比较

```python
def compare_responses(query, hierarchical_response, standard_response, reference=None):
    """
    比较分层和标准RAG的响应
    
    评估维度：
    1. 准确性：哪个响应提供更准确的信息
    2. 全面性：哪个响应更好地覆盖查询的所有方面
    3. 连贯性：哪个响应有更好的逻辑流程
    4. 页面引用：哪个响应更好地使用页面引用
    """
    system_prompt = """You are an expert evaluator of information retrieval systems.
Compare the two responses to the same query, one generated using hierarchical retrieval
and the other using standard retrieval.

Evaluate them based on:
1. Accuracy: Which response provides more factually correct information?
2. Comprehensiveness: Which response better covers all aspects of the query?
3. Coherence: Which response has better logical flow and organization?
4. Page References: Does either response make better use of page references?

Be specific in your analysis of the strengths and weaknesses of each approach."""

    user_prompt = f"""Query: {query}

Response from Hierarchical RAG:
{hierarchical_response}

Response from Standard RAG:
{standard_response}"""

    if reference:
        user_prompt += f"""

Reference Answer:
{reference}"""

    user_prompt += """

Please provide a detailed comparison of these two responses, highlighting which approach performed better and why."""

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

**技术要点：**
- 多维度评估标准
- 使用GPT进行自动评估
- 包含参考答案对比
- 结构化的评估结果

## 7. 系统优势分析

### 7.1 分层索引的优势

1. **上下文保持**：
   - 摘要层保留文档的整体结构
   - 避免信息碎片化

2. **检索效率**：
   - 两步检索减少计算量
   - 基于页面过滤提高精度

3. **结果质量**：
   - 更好的相关性排序
   - 更完整的上下文信息

### 7.2 技术实现亮点

1. **智能缓存**：
   - 避免重复处理文档
   - 提高系统响应速度

2. **灵活配置**：
   - 可调整的块大小和重叠
   - 可配置的检索参数

3. **完整评估**：
   - 自动化的响应比较
   - 多维度评估指标

## 8. 应用场景

### 8.1 适用场景

- **长文档处理**：需要保持文档结构的场景
- **精确检索**：需要高精度信息检索的应用
- **上下文敏感**：需要完整上下文理解的查询

### 8.2 性能考虑

- **计算成本**：需要额外的摘要生成开销
- **存储需求**：需要存储摘要和详细两层索引
- **响应时间**：两步检索可能增加延迟

## 9. 总结

这个分层索引RAG系统通过以下核心技术实现了高效的文档检索：

1. **双层索引结构**：摘要层 + 详细层
2. **两步检索策略**：先定位相关页面，再检索具体信息
3. **智能缓存机制**：避免重复处理，提高效率
4. **完整评估体系**：自动化的质量评估和对比

该系统特别适合处理长文档和需要保持上下文结构的应用场景，在检索精度和响应质量方面都有显著提升。 