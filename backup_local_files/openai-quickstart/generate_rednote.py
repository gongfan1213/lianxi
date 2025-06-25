import json
import re

def generate_rednote(product_name: str, tone_style: str = "活泼甜美", max_iterations: int = 5) -> str:
    """
    使用 DeepSeek Agent 生成小红书爆款文案。
    
    Args:
        product_name (str): 要生成文案的产品名称。
        tone_style (str): 文案的语气和风格，如"活泼甜美"、"知性"、"搞怪"等。
        max_iterations (int): Agent 最大迭代次数，防止无限循环。
        
    Returns:
        str: 生成的爆款文案（JSON 格式字符串）。
    """
    
    print(f"\n🚀 启动小红书文案生成助手，产品：{product_name}，风格：{tone_style}\n")
    
    # 存储对话历史，包括系统提示词和用户请求
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请为产品「{product_name}」生成一篇小红书爆款文案。要求：语气{tone_style}，包含标题、正文、至少5个相关标签和5个表情符号。请以完整的JSON格式输出，并确保JSON内容用markdown代码块包裹（例如：```json{{...}}```）。"}
    ]
    
    iteration_count = 0
    final_response = None
    
    while iteration_count < max_iterations:
        iteration_count += 1
        print(f"-- Iteration {iteration_count} --")
        
        try:
            # 调用 DeepSeek API，传入对话历史和工具定义
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS_DEFINITION, # 告知模型可用的工具
                tool_choice="auto" # 允许模型自动决定是否使用工具
            )

            response_message = response.choices[0].message
            
            # **ReAct模式：处理工具调用**
            if response_message.tool_calls: # 如果模型决定调用工具
                print("Agent: 决定调用工具...")
                messages.append(response_message) # 将工具调用信息添加到对话历史
                
                tool_outputs = []
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    # 确保参数是合法的JSON字符串，即使工具不要求参数，也需要传递空字典
                    function_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                    print(f"Agent Action: 调用工具 '{function_name}'，参数：{function_args}")
                    
                    # 查找并执行对应的模拟工具函数
                    if function_name in available_tools:
                        tool_function = available_tools[function_name]
                        tool_result = tool_function(**function_args)
                        print(f"Observation: 工具返回结果：{tool_result}")
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": str(tool_result) # 工具结果作为字符串返回
                        })
                    else:
                        error_message = f"错误：未知的工具 '{function_name}'"
                        print(error_message)
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": error_message
                        })
                messages.extend(tool_outputs) # 将工具执行结果作为 Observation 添加到对话历史
                
            # **ReAct 模式：处理最终内容**
            elif response_message.content: # 如果模型直接返回内容（通常是最终答案）
                print(f"[模型生成结果] {response_message.content}")
                
                # --- START: 添加 JSON 提取和解析逻辑 ---
                json_string_match = re.search(r"```json\s*(\{.*\})\s*```", response_message.content, re.DOTALL)
                
                if json_string_match:
                    extracted_json_content = json_string_match.group(1)
                    try:
                        final_response = json.loads(extracted_json_content)
                        print("Agent: 任务完成，成功解析最终JSON文案。")
                        return json.dumps(final_response, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError as e:
                        print(f"Agent: 提取到JSON块但解析失败: {e}")
                        print(f"尝试解析的字符串:\n{extracted_json_content}")
                        messages.append(response_message) # 解析失败，继续对话
                else:
                    # 如果没有匹配到 ```json 块，尝试直接解析整个 content
                    try:
                        final_response = json.loads(response_message.content)
                        print("Agent: 任务完成，直接解析最终JSON文案。")
                        return json.dumps(final_response, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        print("Agent: 生成了非JSON格式内容或非Markdown JSON块，可能还在思考或出错。")
                        messages.append(response_message) # 非JSON格式，继续对话
                # --- END: 添加 JSON 提取和解析逻辑 ---
            else:
                print("Agent: 未知响应，可能需要更多交互。")
                break
                
        except Exception as e:
            print(f"调用 DeepSeek API 时发生错误: {e}")
            break
    
    print("\n⚠️ Agent 达到最大迭代次数或未能生成最终文案。请检查Prompt或增加迭代次数。")
    return "未能成功生成文案。" 