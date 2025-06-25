import os
import json
import re
import random
import time
import requests

# 1. 环境准备与API配置（与 test_hkbuapi.py 保持一致）
API_KEY = os.getenv("DEEPSEEK_API_KEY") or "xxxxxxxxx"
BASE_URL = "https://genai.hkbu.edu.hk/general/rest"
MODEL_NAME = "deepseek-r1"  # 与 test_hkbuapi.py 保持一致
API_VERSION = "2024-05-01-preview"

# 2. System Prompt
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

# 3. Tools定义（保留但不传给API，仅本地mock用）
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

# 4. 模拟工具实现（保留）
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

# 5. API请求封装（test_hkbuapi.py风格，不带function call）
def call_llm_api(messages):
    url = f"{BASE_URL}/deployments/{MODEL_NAME}/chat/completions?api-version={API_VERSION}"
    headers = {
        'Content-Type': 'application/json',
        'api-key': API_KEY
    }
    payload = {"messages": messages}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        print(f"API Error: {response.status_code}, {response.text}")
        return None

# 6. Agent主流程（只用基础对话API）
def generate_rednote(product_name: str, tone_style: str = "活泼甜美") -> str:
    print(f"\n🚀 启动小红书文案生成助手，产品：{product_name}，风格：{tone_style}\n")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请为产品「{product_name}」生成一篇小红书爆款文案。要求：语气{tone_style}，包含标题、正文、至少5个相关标签和5个表情符号。请以完整的JSON格式输出，并确保JSON内容用markdown代码块包裹（例如：```json{{...}}```）。"}
    ]
    reply = call_llm_api(messages)
    if reply:
        print(f"[模型生成结果] {reply}")
        json_string_match = re.search(r"```json\s*(\{.*\})\s*```", reply, re.DOTALL)
        if json_string_match:
            extracted_json_content = json_string_match.group(1)
            try:
                final_response = json.loads(extracted_json_content)
                print("Agent: 任务完成，成功解析最终JSON文案。")
                return json.dumps(final_response, ensure_ascii=False, indent=2)
            except json.JSONDecodeError as e:
                print(f"Agent: 提取到JSON块但解析失败: {e}")
                print(f"尝试解析的字符串:\n{extracted_json_content}")
        else:
            try:
                final_response = json.loads(reply)
                print("Agent: 任务完成，直接解析最终JSON文案。")
                return json.dumps(final_response, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                print("Agent: 生成了非JSON格式内容或非Markdown JSON块，可能还在思考或出错。")
    else:
        print("API未返回内容，或发生错误。")
        return "未能成功生成文案。"

# 7. Markdown格式化输出
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

# 8. 示例调用与评估建议
if __name__ == "__main__":
    product_name_1 = "深海蓝藻保湿面膜"
    tone_style_1 = "活泼甜美"
    result_1 = generate_rednote(product_name_1, tone_style_1)
    print("\n--- 生成的文案 1 (JSON) ---")
    print(result_1)
    markdown_note = format_rednote_for_markdown(result_1)
    print("\n--- 格式化后的小红书文案 (Markdown) ---")
    print(markdown_note)
    # 评估与优化建议：
    # - 可根据实际业务数据（点赞、评论、转化等）持续优化Prompt和工具
    # - 可扩展更多工具，如敏感词检测、竞品分析、RAG知识库等
    # - 可将结果自动推送到内容平台或进行A/B测试 


# admin@DESKTOP-KNJMU3O MINGW64 /d/lianxi
# $ python rednote_agent.py

# 🚀 启动小红书文案生成助手，产品：深海蓝藻保湿面膜，风格：活泼甜美

# -- Iteration 1 --
# Agent: 决定调用工具...
# Agent Action: 调用工具 'query_product_database'，参数：{'product_name': '深海蓝藻保
# 湿面膜'}
# [Tool Call] 模拟查询产品数据库：深海蓝藻保湿面膜
# Observation: 工具返回结果：深海蓝藻保湿面膜：核心成分为深海蓝藻提取物，富含多糖和氨
# 基酸，能深层补水、修护肌肤屏障、舒缓敏感泛红。质地清爽不粘腻，适合所有肤质，尤其适合
# 干燥、敏感肌。规格：25ml*5片。
# Agent Action: 调用工具 'generate_emoji'，参数：{'context': '补水保湿'}
# [Tool Call] 模拟生成表情符号，上下文：补水保湿
# Observation: 工具返回结果：['💦', '💧', '🌊', '✨']
# -- Iteration 2 --
# API Error: 400, {"error":{"message":"'content' is a required property - 'messages.2'","type":"invalid_request_error","param":null,"code":null}}
# [Tool Call] 模拟生成表情符号，上下文：补水保湿、惊喜效果、适合敏感肌
# Observation: 工具返回结果：['💦', '💧', '🌊', '✨']
# -- Iteration 3 --
# API Error: 400, {"error":{"message":"'content' is a required property - 'messages.2'","type":"invalid_request_error","param":null,"code":null}}
# [模型生成结果] ```json
# {
#   "title": "💦深海蓝藻保湿面膜｜敏感肌救星！一夜回春的补水神器✨",
#   "body": "姐妹们！我终于找到了我的本命面膜——深海蓝藻保湿面膜！💧\n\n作为一个常年干 
# 燥+敏感肌的人，这款面膜简直就是我的救星！🌊核心成分是深海蓝藻提取物，富含多糖和氨基 
# 酸，敷完脸立马水润润的，而且一点都不粘腻！\n\n🌟使用感：\n- 精华液超多，敷完还能涂脖
# 子和手臂！\n- 质地清爽，敏感肌完全无负担！\n- 第二天起床皮肤还是水嫩嫩的，上妆超服帖
# ！\n\n💖适合人群：\n- 干燥肌、敏感肌的姐妹必入！\n- 熬夜党、换季皮肤不稳定的宝子们！
# \n\n真的是一夜回春的效果，我已经回购第三次了！姐妹们冲鸭！✨",
#   "hashtags": ["#敏感肌救星", "#深海蓝藻保湿面膜", "#补水神器", "#护肤必备", "#面膜 
# 推荐"],
#   "emojis": ["💦", "💧", "🌊", "✨", "💖"]
# }
# ```
# Agent: 任务完成，成功解析最终JSON文案。

# --- 生成的文案 1 (JSON) ---
# {
#   "title": "💦深海蓝藻保湿面膜｜敏感肌救星！一夜回春的补水神器✨",
#   "body": "姐妹们！我终于找到了我的本命面膜——深海蓝藻保湿面膜！💧\n\n作为一个常年干 
# 燥+敏感肌的人，这款面膜简直就是我的救星！🌊核心成分是深海蓝藻提取物，富含多糖和氨基 
# 酸，敷完脸立马水润润的，而且一点都不粘腻！\n\n🌟使用感：\n- 精华液超多，敷完还能涂脖
# 子和手臂！\n- 质地清爽，敏感肌完全无负担！\n- 第二天起床皮肤还是水嫩嫩的，上妆超服帖
# ！\n\n💖适合人群：\n- 干燥肌、敏感肌的姐妹必入！\n- 熬夜党、换季皮肤不稳定的宝子们！
# \n\n真的是一夜回春的效果，我已经回购第三次了！姐妹们冲鸭！✨",
#   "hashtags": [
#     "#敏感肌救星",
#     "#深海蓝藻保湿面膜",
#     "#补水神器",
#     "#护肤必备",
#     "#面膜推荐"
#   ],
#   "emojis": [
#     "💦",
#     "💧",
#     "🌊",
#     "✨",
#     "💖"
#   ]
# }

# --- 格式化后的小红书文案 (Markdown) ---
# ## 💦深海蓝藻保湿面膜｜敏感肌救星！一夜回春的补水神器✨

# 姐妹们！我终于找到了我的本命面膜——深海蓝藻保湿面膜！💧

# 作为一个常年干燥+敏感肌的人，这款面膜简直就是我的救星！🌊核心成分是深海蓝藻提取物， 
# 富含多糖和氨基酸，敷完脸立马水润润的，而且一点都不粘腻！

# 🌟使用感：
# - 精华液超多，敷完还能涂脖子和手臂！
# - 质地清爽，敏感肌完全无负担！
# - 第二天起床皮肤还是水嫩嫩的，上妆超服帖！

# 💖适合人群：
# - 干燥肌、敏感肌的姐妹必入！
# - 熬夜党、换季皮肤不稳定的宝子们！

# 真的是一夜回春的效果，我已经回购第三次了！姐妹们冲鸭！✨

# #敏感肌救星 #深海蓝藻保湿面膜 #补水神器 #护肤必备 #面膜推荐
# (base) 

# admin@DESKTOP-KNJMU3O MINGW64 /d/lianxi
# $ python rednote_agent.py

# 🚀 启动小红书文案生成助手，产品：深海蓝藻保湿面膜，风格：活泼甜美

# [模型生成结果] <think>
# 首先，用户要求我为产品“深海蓝藻保湿面膜”生成一篇小红书爆款文案。文案需要包括标题、正
# 文、至少5个相关标签和5个表情符号。输出必须是完整的JSON格式，并用markdown代码块包裹。

# 关键要求：
# - 语气活泼甜美：文案要充满活力、亲切、可爱，使用轻松的语言，吸引年轻女性用户。      
# - 包含标题：一个吸引眼球的标题。
# - 正文：详细描述产品，突出卖点，引发互动。
# - 至少5个相关标签：例如，#保湿面膜、#护肤推荐等。
# - 至少5个表情符号：如✨、💧、💖等，增强情感表达。
# - 输出格式：JSON对象，包含"title"、"body"、"hashtags"和"emojis"键值对，并用```json{...}```包裹。

# 产品是“深海蓝藻保湿面膜”，我需要突出其卖点：
# - 深海蓝藻：强调其天然、海洋成分，可能具有保湿、修复、抗氧化等功效。
# - 保湿：核心功能是保湿，适合干燥肌肤。
# - 其他可能卖点：如深层补水、提亮肤色、舒缓肌肤等（基于常见面膜特性）。

# 作为爆款文案专家，我需要结合最新潮流：
# - 小红书流行趋势：真实分享、个人体验、高互动元素（如提问、呼吁行动）。
# - 结构：开头吸引注意，中间分享使用体验和效果，结尾呼吁互动（如点赞、收藏、评论）。  

# 使用'Thought-Action-Observation'模式进行推理：
# - Thought：思考产品信息、目标受众、潮流元素。
# - Action：基于思考，生成文案内容。
# - Observation：确保文案符合要求，并确保在输出前完成推理。

# 推理步骤：
# 1. **Thought**：产品是深海蓝藻保湿面膜，卖点是深海蓝藻的保湿和修复功效。目标受众是年
# 轻女性，关注护肤、天然成分。潮流是强调成分、真实体验、高转化。
# 2. **Action**：创建标题：活泼甜美，吸引点击。例如，“✨深海蓝藻面膜，我的肌肤喝饱水啦 
# ！💧”
# 3. **Action**：正文结构：
#    - 开头：分享个人故事或问题（如肌肤干燥）。
#    - 中间：描述使用体验、效果、成分好处。
#    - 结尾：呼吁行动（如推荐、提问）。
#    - 使用活泼语言：如“姐妹们”、“超爱”、“绝了”。
# 4. **Action**：标签：至少5个相关标签，如#保湿面膜、#深海蓝藻、#护肤推荐、#面膜测评、
# #天然护肤。
# 5. **Action**：表情符号：至少5个，如✨、💧、💖、🌟、😍。
# 6. **Observation**：检查语气是否活泼甜美；确保JSON格式正确；标签和表情符号数量达标。

# 正文内容草稿：
# - 开头： “姐妹们，最近我的皮肤干到爆皮，还好发现了这款深海蓝藻保湿面膜！✨”
# - 中间： “用了之后，肌肤瞬间水润透亮，深海蓝藻成分超级补水，还能抗氧化哦～💧”       
# - 结尾： “强烈推荐给干皮姐妹！你们用过吗？评论区告诉我吧～💖”

# 标签建议：
# - #保湿面膜
# - #深海蓝藻
# - #护肤推荐
# - #面膜测评
# - #天然护肤
# - 可以加更多，确保至少5个。

# 表情符号：
# - ✨（闪亮）
# - 💧（水滴，表示保湿）
# - 💖（爱心，表示喜爱）
# - 😍（爱慕）
# - 🌟（星星，表示优秀）
# - 确保至少5个。

# 最终JSON结构：
# {
#   "title": "标题",
#   "body": "正文内容",
#   "hashtags": ["#标签1", "#标签2", ...],
#   "emojis": ["✨", "💧", ...]
# }

# 输出时用```json{...}```包裹。

# 确保正文生动：使用第一人称，分享真实感。
# </think>
# ```json
# {
#   "title": "✨深海蓝藻面膜，干皮救星来啦！肌肤喝饱水超Q弹💧",
#   "body": "姐妹们，最近换季我的脸干到起皮，化妆都卡粉了😭！还好闺蜜安利了这款深海蓝 
# 藻保湿面膜，用了一次就爱上啦～\n\n🌟 成分超天然：深海蓝藻精华超强锁水，敷上脸冰冰凉 
# 凉的，像给肌肤做SPA！15分钟后揭下来，脸蛋水润到发光✨，摸起来软软弹弹的，干纹瞬间拜拜
# ～\n\n💖 真实体验：我每周用2次，现在上妆超服帖，素颜都透亮！敏感肌也友好哦，温和不刺
# 激～\n\n姐妹们快冲！评论区告诉我你们的保湿秘诀吧～一起变美美哒😍 #护肤日常",        
#   "hashtags": ["#保湿面膜", "#深海蓝藻护肤", "#干皮救星", "#面膜推荐", "#天然成分", 
# "#护肤分享"],
#   "emojis": ["✨", "💧", "💖", "🌟", "😍"]
# }
# ```
# Agent: 提取到JSON块但解析失败: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
# 尝试解析的字符串:
# {...}```包裹。

# 产品是“深海蓝藻保湿面膜”，我需要突出其卖点：
# - 深海蓝藻：强调其天然、海洋成分，可能具有保湿、修复、抗氧化等功效。
# - 保湿：核心功能是保湿，适合干燥肌肤。
# - 其他可能卖点：如深层补水、提亮肤色、舒缓肌肤等（基于常见面膜特性）。

# 作为爆款文案专家，我需要结合最新潮流：
# - 小红书流行趋势：真实分享、个人体验、高互动元素（如提问、呼吁行动）。
# - 结构：开头吸引注意，中间分享使用体验和效果，结尾呼吁互动（如点赞、收藏、评论）。  

# 使用'Thought-Action-Observation'模式进行推理：
# - Thought：思考产品信息、目标受众、潮流元素。
# - Action：基于思考，生成文案内容。
# - Observation：确保文案符合要求，并确保在输出前完成推理。

# 推理步骤：
# 1. **Thought**：产品是深海蓝藻保湿面膜，卖点是深海蓝藻的保湿和修复功效。目标受众是年
# 轻女性，关注护肤、天然成分。潮流是强调成分、真实体验、高转化。
# 2. **Action**：创建标题：活泼甜美，吸引点击。例如，“✨深海蓝藻面膜，我的肌肤喝饱水啦 
# ！💧”
# 3. **Action**：正文结构：
#    - 开头：分享个人故事或问题（如肌肤干燥）。
#    - 中间：描述使用体验、效果、成分好处。
#    - 结尾：呼吁行动（如推荐、提问）。
#    - 使用活泼语言：如“姐妹们”、“超爱”、“绝了”。
# 4. **Action**：标签：至少5个相关标签，如#保湿面膜、#深海蓝藻、#护肤推荐、#面膜测评、
# #天然护肤。
# 5. **Action**：表情符号：至少5个，如✨、💧、💖、🌟、😍。
# 6. **Observation**：检查语气是否活泼甜美；确保JSON格式正确；标签和表情符号数量达标。

# 正文内容草稿：
# - 开头： “姐妹们，最近我的皮肤干到爆皮，还好发现了这款深海蓝藻保湿面膜！✨”
# - 中间： “用了之后，肌肤瞬间水润透亮，深海蓝藻成分超级补水，还能抗氧化哦～💧”       
# - 结尾： “强烈推荐给干皮姐妹！你们用过吗？评论区告诉我吧～💖”

# 标签建议：
# - #保湿面膜
# - #深海蓝藻
# - #护肤推荐
# - #面膜测评
# - #天然护肤
# - 可以加更多，确保至少5个。

# 表情符号：
# - ✨（闪亮）
# - 💧（水滴，表示保湿）
# - 💖（爱心，表示喜爱）
# - 😍（爱慕）
# - 🌟（星星，表示优秀）
# - 确保至少5个。

# 最终JSON结构：
# {
#   "title": "标题",
#   "body": "正文内容",
#   "hashtags": ["#标签1", "#标签2", ...],
#   "emojis": ["✨", "💧", ...]
# }

# 输出时用```json{...}```包裹。

# 确保正文生动：使用第一人称，分享真实感。
# </think>
# ```json
# {
#   "title": "✨深海蓝藻面膜，干皮救星来啦！肌肤喝饱水超Q弹💧",
#   "body": "姐妹们，最近换季我的脸干到起皮，化妆都卡粉了😭！还好闺蜜安利了这款深海蓝 
# 藻保湿面膜，用了一次就爱上啦～\n\n🌟 成分超天然：深海蓝藻精华超强锁水，敷上脸冰冰凉 
# 凉的，像给肌肤做SPA！15分钟后揭下来，脸蛋水润到发光✨，摸起来软软弹弹的，干纹瞬间拜拜
# ～\n\n💖 真实体验：我每周用2次，现在上妆超服帖，素颜都透亮！敏感肌也友好哦，温和不刺
# 激～\n\n姐妹们快冲！评论区告诉我你们的保湿秘诀吧～一起变美美哒😍 #护肤日常",        
#   "hashtags": ["#保湿面膜", "#深海蓝藻护肤", "#干皮救星", "#面膜推荐", "#天然成分", 
# "#护肤分享"],
#   "emojis": ["✨", "💧", "💖", "🌟", "😍"]
# }

# --- 生成的文案 1 (JSON) ---
# None
# Traceback (most recent call last):
#   File "D:\lianxi\rednote_agent.py", line 198, in <module>
#     markdown_note = format_rednote_for_markdown(result_1)
#   File "D:\lianxi\rednote_agent.py", line 178, in format_rednote_for_markdown       
#     data = json.loads(json_string)
#   File "C:\Users\admin\miniconda3\lib\json\__init__.py", line 339, in loads
#     raise TypeError(f'the JSON object must be str, bytes or bytearray, '
# TypeError: the JSON object must be str, bytes or bytearray, not NoneType
# (base) 