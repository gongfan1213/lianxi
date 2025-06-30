from typing import Annotated

from langchain_tavily import TavilySearch  # 导入Tavily搜索工具
from langchain_core.messages import ToolMessage  # 导入工具消息类型
from langchain_core.tools import InjectedToolCallId, tool  # 导入工具相关装饰器和类型
from typing_extensions import TypedDict  # 导入类型字典

from langgraph.checkpoint.memory import MemorySaver  # 导入内存保存器，用于状态持久化
from langgraph.graph import StateGraph, START, END  # 导入状态图相关组件
from langgraph.graph.message import add_messages  # 导入消息添加函数
from langgraph.prebuilt import ToolNode, tools_condition  # 导入预构建的工具节点和条件
from langgraph.types import Command, interrupt  # 导入命令和中断类型

# 定义状态类型，包含消息列表、姓名和生日
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表，使用add_messages注解
    name: str  # 姓名
    birthday: str  # 生日

@tool  # 使用@tool装饰器定义工具函数
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """请求人工协助验证信息"""
    # 使用interrupt中断执行，等待人工输入
    human_response = interrupt(
        {
            "question": "Is this correct?",  # 询问是否正确
            "name": name,  # 传递姓名
            "birthday": birthday,  # 传递生日
        },
    )
    
    # 检查人工响应，如果以'y'开头则认为正确
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name  # 使用原始姓名
        verified_birthday = birthday  # 使用原始生日
        response = "Correct"  # 响应消息
    else:
        # 如果人工认为不正确，使用人工提供的修正信息
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # 构建状态更新，包含验证后的姓名、生日和工具消息
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)  # 返回命令对象，用于更新状态


# 创建Tavily搜索工具实例，限制最大结果数为2
tool = TavilySearch(max_results=2)
# 定义工具列表，包含搜索工具和人工协助工具
tools = [tool, human_assistance]
# 将工具绑定到LLM
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    """聊天机器人节点函数，处理用户消息并调用工具"""
    # 使用绑定工具的LLM处理消息
    message = llm_with_tools.invoke(state["messages"])
    # 断言确保工具调用数量不超过1个
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}  # 返回包含新消息的状态更新

# 创建状态图构建器
graph_builder = StateGraph(State)
# 添加聊天机器人节点
graph_builder.add_node("chatbot", chatbot)

# 创建工具节点，用于执行工具调用
tool_node = ToolNode(tools=tools)
# 添加工具节点到图中
graph_builder.add_node("tools", tool_node)

# 添加条件边：从聊天机器人到工具节点（当有工具调用时）
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,  # 使用预定义的工具条件
)
# 添加边：从工具节点回到聊天机器人
graph_builder.add_edge("tools", "chatbot")
# 添加边：从开始节点到聊天机器人
graph_builder.add_edge(START, "chatbot")

# 创建内存保存器，用于状态持久化
memory = MemorySaver()
# 编译图，并设置检查点保存器
graph = graph_builder.compile(checkpointer=memory)