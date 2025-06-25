# OpenAI TTS（Text-To-Speech）文字配音 API 快速入门技术文档

## 1. TTS API 简介
OpenAI TTS（文本到语音）API 提供高质量的多语言文字配音服务，支持多种声音风格和输出格式，适用于播报、配音、语音助手、内容创作等场景。

## 2. 支持的功能与模型
- 文本转语音（Text-to-Speech, TTS）
- 支持 6 种官方声音：alloy, echo, fable, onyx, nova, shimmer
- 支持多种语言（对英语优化）
- 支持流式输出，适合实时语音场景
- 支持多种音频格式输出
- 模型ID：'tts-1'（标准），'tts-1-hd'（高清）

## 3. 支持的声音与语言
- 声音：alloy, echo, fable, onyx, nova, shimmer
- 语言：支持全球主流语言（详见官方文档），对英语发音优化，中文、日语、法语等均可用
- 声音试听：https://platform.openai.com/docs/guides/text-to-speech/voice-options

## 4. 输出格式说明
- 默认：mp3
- 其他支持：opus、aac、flac、wav、pcm
  - Opus：适合互联网流媒体、低延迟
  - AAC：数字音频压缩，兼容主流平台
  - FLAC：无损压缩，适合存档
  - WAV：未压缩，低延迟
  - PCM：原始样本，24kHz 16位小端

## 5. 主要参数详解
- `model`：'tts-1' 或 'tts-1-hd'
- `input`：要合成的文本，最长4096字符
- `voice`：声音风格，见上
- `response_format`：输出格式，见上
- `speed`：语速，0.25~4.0，默认1.0

## 6. 典型代码示例
### 基本配音
```python
from openai import OpenAI
client = OpenAI()
speech_file_path = "./audio/liyunlong.mp3"
with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="echo",
    input="二营长！你他娘的意大利炮呢？给我拉来！"
) as response:
    response.stream_to_file(speech_file_path)
```

### 更换音色
```python
speech_file_path = "./audio/quewang.mp3"
with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="onyx",
    input="周三早上11点，雀王争霸赛，老地方23号房，经典三缺一！"
) as response:
    response.stream_to_file(speech_file_path)
```

### 新闻播报
```python
speech_file_path = "./audio/boyin.mp3"
with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="onyx",
    input="""
    上海F1赛车时隔五年回归 首位中国车手周冠宇：我渴望站上领奖台
    2024年4月17日
    ...
    """
) as response:
    response.stream_to_file(speech_file_path)
```

## 7. 返回值结构说明
- 返回为音频文件内容（如 mp3、wav 等），可直接保存或流式播放
- 推荐用 `with ... as response: response.stream_to_file(path)` 保存

## 8. 常见问题与注意事项
- 文本最长4096字符，超长需分段
- 选择合适的 voice 和 response_format 以适配不同场景
- 语速 speed 可调节，适合不同听众
- 中文、日语等非英语语言支持良好，但部分声音风格对英文优化
- 若需流式播放，建议用 streaming_response

## 9. 进阶用法与组合应用
- 可与 Whisper 语音识别结合，实现"音频-文本-音频"多语种配音
- 可用于自动播报、虚拟主播、语音机器人等
- 可批量生成多语言音频内容

## 10. 结论与参考链接
OpenAI TTS API 提供高质量、灵活的文字配音能力，适合多种AI语音场景。建议结合实际需求选择参数和声音。

- 官方文档：https://platform.openai.com/docs/guides/text-to-speech
- 声音试听：https://platform.openai.com/docs/guides/text-to-speech/voice-options 