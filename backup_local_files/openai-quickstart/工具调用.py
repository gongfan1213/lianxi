import requests

# 配置参数（来自模型.md）
api_key = "xxxxxxx"
base_url = "hxxxxt"
model_name = "gpt-4-o-mini"  # 按模型.md推荐
api_version = "2024-05-01-preview"

def send_messages(messages, tools=None):
    """
    向 API 发送消息，支持 function call（如果API支持）。
    """
    url = f"{base_url}/deployments/{model_name}/chat/completions?api-version={api_version}"
    headers = {
        'Content-Type': 'application/json',
        'api-key': api_key
    }
    payload = {
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # 兼容function call和普通对话
        message = data["choices"][0]["message"]
        return message
    else:
        raise RuntimeError(f"Error: {response.status_code}, {response.text}")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply a location first",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. Shanghai",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

def main():
    messages = [{"role": "user", "content": "How's the weather in Shanghai?"}]
    print(f"User>\t {messages[0]['content']}")
    # 第一次请求，尝试function call
    message = send_messages(messages, tools=tools)
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        tool = tool_calls[0]
        print("Tool call detected:", tool)
        # 模拟工具调用结果（假设查到24℃）
        tool_result = "24℃"
        messages.append({
            "role": "assistant",
            "content": "今天天气40度很热很热很热",  # 必须有content字段
            "tool_calls": tool_calls
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool["id"],
            "content": tool_result
        })
        # 再次请求，获得最终回复
        message = send_messages(messages)
        print(f"Model>\t {message['content']}")
    else:
        print("Model>\t", message.get("content", ""))

if __name__ == "__main__":
    main()
# admin@DESKTOP-KNJMU3O MINGW64 /d/lianxi
# $ python functioncall.py
# User>    How's the weather in Shanghai?
# Tool call detected: {'id': 'call_oFnyqPH9UPVhjHIhXfUqYLRl', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"location":"Shanghai"}'}}
# Model>   当前上海的天气是24℃。如果你需要更具体的天气信息，比如湿度或风速，请告诉我！
# (base) 