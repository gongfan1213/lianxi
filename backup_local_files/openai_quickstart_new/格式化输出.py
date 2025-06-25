import os
import json
import re
import random
import time

# 1. System Prompt
SYSTEM_PROMPT = """
你是一个资深的小红书爆款文案专家，擅长结合最新潮流和产品卖点，创作引人入胜、高互动、高转化的笔记文案。
你的任务是根据用户提供的产品和需求，生成包含标题、正文、相关标签和表情符号的完整小红书笔记。
请始终采用'Thought-Action-Observation'模式进行推理和行动。文案风格需活泼、真诚、富有感染力。当完成任务后，请以JSON格式直接输出最终文案，格式如下：
```json
{
  "title": "小红书标题",
  "body": "小红书正文",
  "hashtags": ["#标签1", "#标签2", "#标签3", "#标签4", "#标签5"],
  "emojis": ["✨", "🔥", "💖"]
}
```
在生成文案前，请务必先思考并收集足够的信息。
"""

# 2. Tools定义
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索互联网上的实时信息，用于获取最新新闻、流行趋势、用户评价、行业报告等。请确保搜索关键词精确，避免宽泛的查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的关键词或问题，例如'最新小红书美妆趋势'或'深海蓝藻保湿面膜 用户评价'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_product_database",
            "description": "查询内部产品数据库，获取指定产品的详细卖点、成分、适用人群、使用方法等信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "要查询的产品名称，例如'深海蓝藻保湿面膜'"
                    }
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_emoji",
            "description": "根据提供的文本内容，生成一组适合小红书风格的表情符号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "文案的关键内容或情感，例如'惊喜效果'、'补水保湿'"
                    }
                },
                "required": ["context"]
            }
        }
    }
]

# 3. 模拟工具实现

def mock_search_web(query: str) -> str:
    print(f"[Tool Call] 模拟搜索网页：{query}")
    time.sleep(0.5)
    if "小红书美妆趋势" in query:
        return "近期小红书美妆流行'多巴胺穿搭'、'早C晚A'护肤理念、'伪素颜'妆容，热门关键词有#氛围感、#抗老、#屏障修复。"
    elif "保湿面膜" in query:
        return "小红书保湿面膜热门话题：沙漠干皮救星、熬夜急救面膜、水光肌养成。用户痛点：卡粉、泛红、紧绷感。"
    elif "深海蓝藻保湿面膜" in query:
        return "关于深海蓝藻保湿面膜的用户评价：普遍反馈补水效果好，吸收快，对敏感肌友好。有用户提到价格略高，但效果值得。"
    else:
        return f"未找到关于 '{query}' 的特定信息，但市场反馈通常关注产品成分、功效和用户体验。"

def mock_query_product_database(product_name: str) -> str:
    print(f"[Tool Call] 模拟查询产品数据库：{product_name}")
    time.sleep(0.3)
    if "深海蓝藻保湿面膜" in product_name:
        return "深海蓝藻保湿面膜：核心成分为深海蓝藻提取物，富含多糖和氨基酸，能深层补水、修护肌肤屏障、舒缓敏感泛红。质地清爽不粘腻，适合所有肤质，尤其适合干燥、敏感肌。规格：25ml*5片。"
    elif "美白精华" in product_name:
        return "美白精华：核心成分是烟酰胺和VC衍生物，主要功效是提亮肤色、淡化痘印、改善暗沉。质地轻薄易吸收，适合需要均匀肤色的人群。"
    else:
        return f"产品数据库中未找到关于 '{product_name}' 的详细信息。"

def mock_generate_emoji(context: str) -> list:
    print(f"[Tool Call] 模拟生成表情符号，上下文：{context}")
    time.sleep(0.1)
    if "补水" in context or "水润" in context or "保湿" in context:
        return ["💦", "💧", "🌊", "✨"]
    elif "惊喜" in context or "哇塞" in context or "爱了" in context:
        return ["💖", "😍", "🤩", "💯"]
    elif "熬夜" in context or "疲惫" in context:
        return ["😭", "😮‍💨", "😴", "💡"]
    elif "好物" in context or "推荐" in context:
        return ["✅", "👍", "⭐", "🛍️"]
    else:
        return random.sample(["✨", "🔥", "💖", "💯", "🎉", "👍", "🤩", "💧", "🌿"], k=min(5, len(context.split())))

available_tools = {
    "search_web": mock_search_web,
    "query_product_database": mock_query_product_database,
    "generate_emoji": mock_generate_emoji,
}

# 4. Agent主流程（本地模拟ReAct）
def generate_rednote(product_name: str, tone_style: str = "活泼甜美", max_iterations: int = 5) -> str:
    print(f"\n🚀 启动小红书文案生成助手，产品：{product_name}，风格：{tone_style}\n")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请为产品「{product_name}」生成一篇小红书爆款文案。要求：语气{tone_style}，包含标题、正文、至少5个相关标签和5个表情符号。请以完整的JSON格式输出，并确保JSON内容用markdown代码块包裹（例如：```json{{...}}```）。"}
    ]
    iteration_count = 0
    final_response = None
    # 本地模拟ReAct流程
    while iteration_count < max_iterations:
        iteration_count += 1
        print(f"-- Iteration {iteration_count} --")
        # 1. 模拟模型决策：先查产品数据库，再查表情，再输出
        if iteration_count == 1:
            # 决定调用query_product_database
            tool_result = mock_query_product_database(product_name)
            print(f"Observation: 工具返回结果：{tool_result}")
            messages.append({"role": "tool", "content": tool_result, "name": "query_product_database"})
        elif iteration_count == 2:
            # 决定调用generate_emoji
            tool_result = mock_generate_emoji(f"补水保湿、惊喜效果、适合敏感肌")
            print(f"Observation: 工具返回结果：{tool_result}")
            messages.append({"role": "tool", "content": str(tool_result), "name": "generate_emoji"})
        elif iteration_count == 3:
            # 直接输出最终文案
            # 这里用一个固定的示例输出
            response_content = '''```json
{
  "title": "💦深海蓝藻保湿面膜｜敏感肌救星！一夜回春的补水神器✨",
  "body": "姐妹们！我终于找到了我的本命面膜——深海蓝藻保湿面膜！💧\\n\\n作为一个常年干燥+敏感肌的人，这款面膜简直就是我的救星！🌊核心成分是深海蓝藻提取物，富含多糖和氨基酸，敷完脸立马水润润的，而且一点都不粘腻！\\n\\n🌟使用感：\\n- 精华液超多，敷完还能涂脖子和手臂！\\n- 质地清爽，敏感肌完全无负担！\\n- 第二天起床皮肤还是水嫩嫩的，上妆超服帖！\\n\\n💖适合人群：\\n- 干燥肌、敏感肌的姐妹必入！\\n- 熬夜党、换季皮肤不稳定的宝子们！\\n\\n真的是一夜回春的效果，我已经回购第三次了！姐妹们冲鸭！✨",
  "hashtags": ["#敏感肌救星", "#深海蓝藻保湿面膜", "#补水神器", "#护肤必备", "#面膜推荐"],
  "emojis": ["💦", "💧", "🌊", "✨", "💖"]
}
```'''
            print(f"[模型生成结果] {response_content}")
            json_string_match = re.search(r"```json\s*(\{.*\})\s*```", response_content, re.DOTALL)
            if json_string_match:
                extracted_json_content = json_string_match.group(1)
                try:
                    final_response = json.loads(extracted_json_content)
                    print("Agent: 任务完成，成功解析最终JSON文案。")
                    return json.dumps(final_response, ensure_ascii=False, indent=2)
                except json.JSONDecodeError as e:
                    print(f"Agent: 提取到JSON块但解析失败: {e}")
                    print(f"尝试解析的字符串:\n{extracted_json_content}")
            break
        else:
            print("Agent: 未知响应，可能需要更多交互。")
            break
    print("\n⚠️ Agent 达到最大迭代次数或未能生成最终文案。请检查Prompt或增加迭代次数。")
    return "未能成功生成文案。"

# 5. Markdown格式化输出

def format_rednote_for_markdown(json_string: str) -> str:
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        return f"错误：无法解析 JSON 字符串 - {e}\n原始字符串：\n{json_string}"
    title = data.get("title", "无标题")
    body = data.get("body", "")
    hashtags = data.get("hashtags", [])
    markdown_output = f"## {title}\n\n"
    markdown_output += f"{body}\n\n"
    if hashtags:
        hashtag_string = " ".join(hashtags)
        markdown_output += f"{hashtag_string}\n"
    return markdown_output.strip()

# 6. 示例调用
if __name__ == "__main__":
    product_name_1 = "深海蓝藻保湿面膜"
    tone_style_1 = "活泼甜美"
    result_1 = generate_rednote(product_name_1, tone_style_1)
    print("\n--- 生成的文案 1 (JSON) ---")
    print(result_1)
    markdown_note = format_rednote_for_markdown(result_1)
    print("\n--- 格式化后的小红书文案 (Markdown) ---")
    print(markdown_note) 