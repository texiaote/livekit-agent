# AI 中译英翻译助手 (Chinese-to-English Translation Agent)

一个基于 LiveKit Agents 框架构建的实时语音翻译助手，专门用于将中文语音翻译成英文。

## 功能特性

- **实时语音识别**: 使用 Cartesia 的 ink-whisper 模型进行中文语音识别
- **智能翻译**: 集成 DeepSeek Chat 模型，提供高质量的中译英翻译
- **自然语音合成**: 使用 Cartesia TTS 生成自然流畅的英文语音
- **多语言检测**: 支持多语言环境下的语音检测和处理
- **噪音消除**: 内置 LiveKit Cloud 增强噪音消除功能
- **性能监控**: 集成使用情况统计和性能指标收集

## 技术架构

### 核心组件

- **LLM**: DeepSeek Chat 模型 (通过 OpenAI 兼容接口)
- **STT**: Cartesia ink-whisper 语音识别
- **TTS**: Cartesia 语音合成
- **VAD**: Silero 语音活动检测
- **Turn Detection**: 多语言模型支持

### 工作流程

1. 用户说中文 → Cartesia STT 识别
2. 文本传递给 DeepSeek Chat → 生成英文翻译
3. 英文文本 → Cartesia TTS 合成语音
4. 播放英文语音给用户

## 安装和配置

### 环境要求

- Python >= 3.13
- LiveKit 账户和 API 密钥
- DeepSeek API 密钥
- Cartesia API 密钥

### 安装依赖

```bash
# 使用 uv 安装依赖
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 环境配置

创建 `.env.local` 文件并配置以下环境变量：

```env
# LiveKit 配置
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# DeepSeek API 配置
OPENAI_API_KEY=your-deepseek-api-key

# Cartesia API 配置
CARTESIA_API_KEY=your-cartesia-api-key
```

## 使用方法

### 启动 Agent

```bash
python agent.py dev
```

### 连接到房间

Agent 启动后，用户可以通过 LiveKit 客户端连接到房间开始使用翻译服务。

### 使用流程

1. 进入房间后，Agent 会用中文欢迎用户
2. 用户说中文，Agent 自动识别并翻译成英文
3. Agent 用英文语音播放翻译结果

## 自定义配置

### 修改翻译指令

在 `AITranslatorAssistant` 类中修改 `instructions` 参数：

```python
instructions="""你现在是一个优秀的AI中译英翻译，你的职责是当用户说出的是中文的内容时，你将对应对的中文内容翻译成英文"""
```

### 更换语音模型

修改 TTS 配置中的 voice ID：

```python
tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532")
```

### 调整 LLM 模型

可以更换为其他兼容 OpenAI API 的模型：

```python
llm=openai.LLM(model="deepseek-chat", base_url="https://api.deepseek.com")
```

## 开发和调试

### 日志配置

Agent 包含详细的日志记录，包括：
- STT 识别结果
- 用户语音转录
- 性能指标统计

### 性能监控

内置使用情况收集器，可以监控：
- 模型调用次数
- 响应时间
- 资源使用情况

## 部署

### 本地开发

```bash
python agent.py dev
```

### 生产部署

```bash
python agent.py start
```

## 故障排除

### 常见问题

1. **语音识别不准确**: 检查麦克风设置和网络连接
2. **翻译质量问题**: 调整 LLM 的 instructions 参数
3. **语音合成延迟**: 检查 Cartesia API 配置和网络状况

### 调试模式

启用详细日志：

```python
logging.basicConfig(level=logging.DEBUG)
```

## 许可证

本项目基于 MIT 许可证开源。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。

## 相关链接

- [LiveKit Agents 文档](https://docs.livekit.io/agents/)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)
- [Cartesia 文档](https://docs.cartesia.ai/)