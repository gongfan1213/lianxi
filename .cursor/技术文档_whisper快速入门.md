# Whisper 语音识别与翻译 API 快速入门技术文档

## 1. Whisper API 简介
Whisper 是 OpenAI 基于开源 Whisper large-v2 模型的语音到文本（Speech-to-Text, STT）API服务，支持多语言音频的自动转录与翻译，适用于语音助手、字幕生成、音频检索等场景。

## 2. 支持的功能与模型
- **转录（transcriptions）**：将音频转为原语言文本。
- **翻译（translations）**：将音频内容翻译并转录为英文。
- **模型ID**：目前仅支持 `whisper-1`（Whisper V2）。

## 3. 语音转录（transcriptions）API用法
- **接口功能**：输入音频文件，返回音频原语言的转录文本。
- **主要参数：**
  - `file`：音频文件对象（如 open('xxx.mp3', 'rb')）
  - `model`：'whisper-1'
  - `language`：可选，ISO-639-1 语言代码（如 'zh'、'en'），指定可提升准确率
  - `prompt`：可选，指导模型风格或补全
  - `response_format`：可选，json/text/srt/verbose_json/vtt
  - `temperature`：可选，采样温度，0-1
  - `timestamp_granularities[]`：可选，时间戳粒度，需 verbose_json 格式

**Python 示例：**
```python
from openai import OpenAI
client = OpenAI()
audio_file = open("./audio/liyunlong.mp3", "rb")
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)
print(transcription.text)
```

## 4. 语音翻译（translations）API用法
- **接口功能**：输入音频文件，返回英文翻译文本。
- **主要参数：**
  - `file`：音频文件对象
  - `model`：'whisper-1'
  - `prompt`：可选，英文提示
  - `response_format`：可选，json/text/srt/verbose_json/vtt
  - `temperature`：可选，采样温度

**Python 示例：**
```python
audio_file = open("./audio/liyunlong.mp3", "rb")
translation = client.audio.translations.create(
    model="whisper-1",
    file=audio_file,
    prompt="Translate into English",
)
print(translation.text)
```

## 5. 支持的音频格式与限制
- 支持：mp3, mp4, mpeg, mpga, m4a, wav, webm, flac, ogg
- 文件大小：最大 25MB

## 6. 典型代码示例
### 中文转录
```python
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("./audio/liyunlong.mp3", "rb"),
    language="zh"
)
print(transcription.text)
```

### 中文识别+翻译
```python
translation = client.audio.translations.create(
    model="whisper-1",
    file=open("./audio/liyunlong.mp3", "rb"),
    prompt="Translate into English"
)
print(translation.text)
```

### Whisper + TTS 英文配音
```python
speech_file_path = "./audio/liyunlong_en.mp3"
with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="onyx",
    input=translation.text
) as response:
    response.stream_to_file(speech_file_path)
```

## 7. 返回值结构说明
- **转录对象（Transcription Object）**
```json
{
  "text": "...音频内容转录..."
}
```
- **详细转录对象（Verbose Transcription Object）**
```json
{
  "task": "transcribe",
  "language": "english",
  "duration": 8.47,
  "text": "...",
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.32, "text": "...", ...},
    ...
  ]
}
```

## 8. 常见问题与注意事项
- 文件需为二进制对象（open(..., 'rb')），不能直接传文件名字符串。
- 文件大小不能超过 25MB。
- `language` 参数建议指定，提升准确率。
- `response_format` 默认为 json，若需字幕等格式可选 srt/vtt。
- `timestamp_granularities` 仅 verbose_json 格式下支持 word/segment 粒度。
- Whisper 适合多语言，但翻译仅支持转为英文。

## 9. 进阶用法与组合应用
- 可结合 TTS（Text-to-Speech）实现语音翻译配音。
- 可批量处理音频，自动生成字幕、摘要、检索等。
- 可与 RAG、知识库等结合，实现多模态智能问答。

## 10. 结论与参考链接
Whisper API 提供了高效、易用的语音识别与翻译能力，适合多种AI应用场景。建议结合实际业务需求，灵活选择参数与输出格式。

- 官方文档：https://platform.openai.com/docs/guides/speech-to-text
- Whisper 开源项目：https://github.com/openai/whisper 