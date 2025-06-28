# CRAG (Corrective RAG) 核心代码详解

## 概述

CRAG (Corrective RAG) 是一种高级的检索增强生成方法，它能够动态评估检索到的信息质量，并在必要时通过网络搜索来纠正检索过程。相比传统RAG，CRAG具有以下优势：

- 在使用检索内容前先评估其相关性
- 根据相关性动态切换知识源
- 当本地知识不足时，通过网络搜索纠正检索
- 在适当时机组合多个信息源

## 核心架构组件

### 1. 文档处理模块

#### 1.1 PDF文本提取
```python
def extract_text_from_pdf(pdf_path):
    """从PDF文件中提取文本内容"""
    pdf = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text += page.get_text()
    return text
```

**核心要点：**
- 使用PyMuPDF (fitz) 库处理PDF文件
- 逐页提取文本内容
- 返回完整的文档文本

#### 1.2 文本分块处理
```python
def chunk_text(text, chunk_size=1000, overlap=200):
    """将文本分割成重叠的块"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "start_pos": i,
                    "end_pos": i + len(chunk_text),
                    "source_type": "document"
                }
            })
    return chunks
```

**核心要点：**
- 使用滑动窗口方法分割文本
- 设置重叠区域保持上下文连续性
- 为每个块添加元数据信息
- 重叠大小 = chunk_size - overlap

### 2. 向量存储实现

#### 2.1 简单向量存储类
```python
class SimpleVectorStore:
    def __init__(self):
        self.vectors = []      # 存储向量
        self.texts = []        # 存储文本
        self.metadata = []     # 存储元数据
    
    def add_item(self, text, embedding, metadata=None):
        """添加单个项目到向量存储"""
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def similarity_search(self, query_embedding, k=5):
        """基于余弦相似度搜索最相似的项目"""
        if not self.vectors:
            return []
        
        query_vector = np.array(query_embedding)
        similarities = []
        
        for i, vector in enumerate(self.vectors):
            # 计算余弦相似度
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            similarities.append((i, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个结果
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

**核心要点：**
- 使用NumPy数组存储向量
- 实现余弦相似度计算
- 支持批量添加和检索
- 返回带相似度分数的结果

### 3. 嵌入向量生成

#### 3.1 批量嵌入生成
```python
def create_embeddings(texts, model="text-embedding-3-small"):
    """使用OpenAI API创建文本嵌入向量"""
    input_texts = texts if isinstance(texts, list) else [texts]
    
    batch_size = 100  # 批处理大小
    all_embeddings = []
    
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]
        
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings[0] if isinstance(texts, str) else all_embeddings
```

**核心要点：**
- 支持单个文本和批量文本处理
- 使用批处理避免API限制
- 自动处理API响应格式
- 返回标准化的向量格式

### 4. 相关性评估模块

#### 4.1 文档相关性评估
```python
def evaluate_document_relevance(query, document):
    """评估文档与查询的相关性"""
    system_prompt = """
    You are an expert at evaluating document relevance. 
    Rate how relevant the given document is to the query on a scale from 0 to 1.
    0 means completely irrelevant, 1 means perfectly relevant.
    Provide ONLY the score as a float between 0 and 1.
    """
    
    user_prompt = f"Query: {query}\n\nDocument: {document}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=5
        )
        
        score_text = response.choices[0].message.content.strip()
        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
        if score_match:
            return float(score_match.group(1))
        return 0.5
    except Exception as e:
        print(f"Error evaluating document relevance: {e}")
        return 0.5
```

**核心要点：**
- 使用LLM评估文档相关性
- 返回0-1之间的标准化分数
- 包含错误处理和默认值
- 使用正则表达式解析分数

### 5. 网络搜索模块

#### 5.1 DuckDuckGo搜索实现
```python
def duck_duck_go_search(query, num_results=3):
    """使用DuckDuckGo进行网络搜索"""
    encoded_query = quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
    
    try:
        response = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        data = response.json()
        
        results_text = ""
        sources = []
        
        # 添加摘要信息
        if data.get("AbstractText"):
            results_text += f"{data['AbstractText']}\n\n"
            sources.append({
                "title": data.get("AbstractSource", "Wikipedia"),
                "url": data.get("AbstractURL", "")
            })
        
        # 添加相关主题
        for topic in data.get("RelatedTopics", [])[:num_results]:
            if "Text" in topic and "FirstURL" in topic:
                results_text += f"{topic['Text']}\n\n"
                sources.append({
                    "title": topic.get("Text", "").split(" - ")[0],
                    "url": topic.get("FirstURL", "")
                })
        
        return results_text, sources
    
    except Exception as e:
        print(f"Error performing web search: {e}")
        return "Failed to retrieve search results.", []
```

**核心要点：**
- 使用DuckDuckGo API进行搜索
- 提取摘要和相关主题信息
- 保存来源元数据
- 包含错误处理和备用方案

#### 5.2 查询重写功能
```python
def rewrite_search_query(query):
    """重写查询以优化网络搜索效果"""
    system_prompt = """
    You are an expert at creating effective search queries.
    Rewrite the given query to make it more suitable for a web search engine.
    Focus on keywords and facts, remove unnecessary words, and make it concise.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original query: {query}\n\nRewritten query:"}
            ],
            temperature=0.3,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error rewriting search query: {e}")
        return query
```

**核心要点：**
- 使用LLM优化搜索查询
- 提取关键词和事实
- 提高搜索结果的准确性
- 包含错误处理机制

### 6. 知识精炼模块

#### 6.1 知识提取和精炼
```python
def refine_knowledge(text):
    """从文本中提取和精炼关键信息"""
    system_prompt = """
    Extract the key information from the following text as a set of clear, concise bullet points.
    Focus on the most relevant facts and important details.
    Format your response as a bulleted list with each point on a new line starting with "• ".
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text to refine:\n\n{text}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error refining knowledge: {e}")
        return text
```

**核心要点：**
- 提取文本中的关键信息
- 格式化为清晰的要点列表
- 提高信息的可读性和可用性
- 包含错误处理机制

### 7. 核心CRAG流程

#### 7.1 主要CRAG处理函数
```python
def crag_process(query, vector_store, k=3):
    """运行CRAG核心流程"""
    print(f"\n=== Processing query with CRAG: {query} ===\n")
    
    # 步骤1: 创建查询嵌入并检索文档
    print("Retrieving initial documents...")
    query_embedding = create_embeddings(query)
    retrieved_docs = vector_store.similarity_search(query_embedding, k=k)
    
    # 步骤2: 评估文档相关性
    print("Evaluating document relevance...")
    relevance_scores = []
    for doc in retrieved_docs:
        score = evaluate_document_relevance(query, doc["text"])
        relevance_scores.append(score)
        doc["relevance"] = score
        print(f"Document scored {score:.2f} relevance")
    
    # 步骤3: 基于最佳相关性分数确定行动
    max_score = max(relevance_scores) if relevance_scores else 0
    best_doc_idx = relevance_scores.index(max_score) if relevance_scores else -1
    
    sources = []
    final_knowledge = ""
    
    # 步骤4: 执行相应的知识获取策略
    if max_score > 0.7:
        # 情况1: 高相关性 - 直接使用文档
        print(f"High relevance ({max_score:.2f}) - Using document directly")
        best_doc = retrieved_docs[best_doc_idx]["text"]
        final_knowledge = best_doc
        sources.append({"title": "Document", "url": ""})
        
    elif max_score < 0.3:
        # 情况2: 低相关性 - 使用网络搜索
        print(f"Low relevance ({max_score:.2f}) - Performing web search")
        web_results, web_sources = perform_web_search(query)
        final_knowledge = refine_knowledge(web_results)
        sources.extend(web_sources)
        
    else:
        # 情况3: 中等相关性 - 结合文档和网络搜索
        print(f"Medium relevance ({max_score:.2f}) - Combining document with web search")
        best_doc = retrieved_docs[best_doc_idx]["text"]
        refined_doc = refine_knowledge(best_doc)
        
        web_results, web_sources = perform_web_search(query)
        refined_web = refine_knowledge(web_results)
        
        final_knowledge = f"From document:\n{refined_doc}\n\nFrom web search:\n{refined_web}"
        sources.append({"title": "Document", "url": ""})
        sources.extend(web_sources)
    
    # 步骤5: 生成最终响应
    print("Generating final response...")
    response = generate_response(query, final_knowledge, sources)
    
    return {
        "query": query,
        "response": response,
        "retrieved_docs": retrieved_docs,
        "relevance_scores": relevance_scores,
        "max_relevance": max_score,
        "final_knowledge": final_knowledge,
        "sources": sources
    }
```

**核心要点：**
- **动态决策机制**: 根据相关性分数选择不同的知识获取策略
- **三种处理模式**:
  - 高相关性 (>0.7): 直接使用文档内容
  - 低相关性 (<0.3): 完全依赖网络搜索
  - 中等相关性 (0.3-0.7): 结合文档和网络搜索
- **知识融合**: 在中等相关性情况下，智能组合多个信息源
- **来源追踪**: 记录所有信息来源用于归因

### 8. 响应生成模块

#### 8.1 智能响应生成
```python
def generate_response(query, knowledge, sources):
    """基于查询和知识生成响应"""
    # 格式化来源信息
    sources_text = ""
    for source in sources:
        title = source.get("title", "Unknown Source")
        url = source.get("url", "")
        if url:
            sources_text += f"- {title}: {url}\n"
        else:
            sources_text += f"- {title}\n"
    
    system_prompt = """
    You are a helpful AI assistant. Generate a comprehensive, informative response to the query based on the provided knowledge.
    Include all relevant information while keeping your answer clear and concise.
    If the knowledge doesn't fully answer the query, acknowledge this limitation.
    Include source attribution at the end of your response.
    """
    
    user_prompt = f"""
    Query: {query}
    
    Knowledge:
    {knowledge}
    
    Sources:
    {sources_text}
    
    Please provide an informative response to the query based on this information.
    Include the sources at the end of your response.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"I apologize, but I encountered an error while generating a response to your query: '{query}'. The error was: {str(e)}"
```

**核心要点：**
- 使用GPT-4生成高质量响应
- 包含完整的来源归因
- 处理知识不足的情况
- 包含错误处理机制

### 9. 评估和比较模块

#### 9.1 响应质量评估
```python
def evaluate_crag_response(query, response, reference_answer=None):
    """评估CRAG响应质量"""
    system_prompt = """
    You are an expert at evaluating the quality of responses to questions.
    Please evaluate the provided response based on the following criteria:
    
    1. Relevance (0-10): How directly does the response address the query?
    2. Accuracy (0-10): How factually correct is the information?
    3. Completeness (0-10): How thoroughly does the response answer all aspects of the query?
    4. Clarity (0-10): How clear and easy to understand is the response?
    5. Source Quality (0-10): How well does the response cite relevant sources?
    
    Return your evaluation as a JSON object with scores for each criterion and a brief explanation for each score.
    Also include an "overall_score" (0-10) and a brief "summary" of your evaluation.
    """
    
    user_prompt = f"""
    Query: {query}
    
    Response to evaluate:
    {response}
    """
    
    if reference_answer:
        user_prompt += f"""
    Reference answer (for comparison):
    {reference_answer}
    """
    
    try:
        evaluation_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        evaluation = json.loads(evaluation_response.choices[0].message.content)
        return evaluation
    except Exception as e:
        print(f"Error evaluating response: {e}")
        return {
            "error": str(e),
            "overall_score": 0,
            "summary": "Evaluation failed due to an error."
        }
```

**核心要点：**
- 多维度评估响应质量
- 支持与参考答案比较
- 返回结构化的评估结果
- 包含详细的评分说明

#### 9.2 CRAG与标准RAG比较
```python
def compare_crag_vs_standard_rag(query, vector_store, reference_answer=None):
    """比较CRAG与标准RAG的性能"""
    # 运行CRAG流程
    print("\n=== Running CRAG ===")
    crag_result = crag_process(query, vector_store)
    crag_response = crag_result["response"]
    
    # 运行标准RAG
    print("\n=== Running standard RAG ===")
    query_embedding = create_embeddings(query)
    retrieved_docs = vector_store.similarity_search(query_embedding, k=3)
    combined_text = "\n\n".join([doc["text"] for doc in retrieved_docs])
    standard_sources = [{"title": "Document", "url": ""}]
    standard_response = generate_response(query, combined_text, standard_sources)
    
    # 评估两种方法
    print("\n=== Evaluating CRAG response ===")
    crag_eval = evaluate_crag_response(query, crag_response, reference_answer)
    
    print("\n=== Evaluating standard RAG response ===")
    standard_eval = evaluate_crag_response(query, standard_response, reference_answer)
    
    # 比较方法
    print("\n=== Comparing approaches ===")
    comparison = compare_responses(query, crag_response, standard_response, reference_answer)
    
    return {
        "query": query,
        "crag_response": crag_response,
        "standard_response": standard_response,
        "reference_answer": reference_answer,
        "crag_evaluation": crag_eval,
        "standard_evaluation": standard_eval,
        "comparison": comparison
    }
```

**核心要点：**
- 并行运行CRAG和标准RAG
- 使用相同的评估标准
- 提供详细的性能比较
- 支持多查询批量评估

## 核心算法流程总结

### CRAG的核心创新点：

1. **智能相关性评估**: 使用LLM评估检索文档的相关性，而不是仅依赖向量相似度
2. **动态知识源切换**: 根据相关性分数自动选择最合适的信息源
3. **知识融合策略**: 在适当时机智能组合多个信息源
4. **网络搜索集成**: 当本地知识不足时，自动进行网络搜索补充
5. **质量保证机制**: 通过多层评估确保最终响应的质量

### 关键技术特点：

- **模块化设计**: 每个组件都可以独立测试和优化
- **错误处理**: 完善的异常处理机制确保系统稳定性
- **可扩展性**: 易于添加新的知识源和评估方法
- **可解释性**: 详细的调试信息帮助理解系统决策过程

这个CRAG实现展示了如何通过智能评估和动态决策来改进传统RAG系统，使其能够更好地处理各种查询场景。 