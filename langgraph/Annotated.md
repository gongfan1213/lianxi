`Annotated` 是 Python 类型提示系统中的一个重要工具，让我详细解释一下它的作用：

## Annotated 的基本概念

`Annotated` 是 Python 3.9+ 引入的类型提示功能，它允许我们为类型添加额外的元数据，而不影响运行时的类型检查。

## 语法格式
```python
Annotated[类型, 元数据1, 元数据2, ...]
```

## 在代码中的具体作用

在你的代码中：
```python
messages: Annotated[list, add_messages]
```

这里的 `Annotated[list, add_messages]` 表示：
- **类型**：`list` - 这是一个列表类型
- **元数据**：`add_messages` - 这是 LangGraph 框架提供的特殊函数

## LangGraph 中的特殊用途

在 LangGraph 框架中，`Annotated` 被用来：

1. **状态合并策略**：`add_messages` 告诉 LangGraph 如何合并消息列表
   - 当状态更新时，新的消息会被添加到现有消息列表的末尾
   - 而不是替换整个消息列表

2. **框架识别**：LangGraph 通过这个注解识别这是一个需要特殊处理的消息字段

## 实际效果对比

**不使用 Annotated：**
```python
messages: list  # 普通列表，会被完全替换
```

**使用 Annotated：**
```python
messages: Annotated[list, add_messages]  # 列表会被追加合并
```

## 其他常见的 Annotated 用法

```python
# 添加验证信息
age: Annotated[int, "must be positive"]

# 添加描述信息
name: Annotated[str, "user's full name"]

# 添加约束条件
score: Annotated[float, "range: 0.0 to 100.0"]
```

## 总结

在你的代码中，`Annotated[list, add_messages]` 的作用是：
- 告诉 LangGraph 这是一个消息列表
- 指定使用 `add_messages` 函数来合并消息（追加而不是替换）
- 确保对话历史能够正确累积和保存

这样设计使得 LangGraph 能够智能地处理状态更新，保持对话的连续性和完整性。