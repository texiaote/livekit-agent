import logging
import re

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics, )
from livekit.plugins import cartesia, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# 创建日志记录器
logger = logging.getLogger("agent")

# 加载环境变量配置文件
load_dotenv(".env.local")


def clean_llm_output(text: str) -> str:
    """
    清理 LLM 输出文本，去除 <think></think> 标签和标点符号
    只保留单词用于 TTS 合成
    """
    if not text:
        return ""
    
    # 去除 <think></think> 标签及其内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 去除所有标点符号，只保留字母、数字和空格
    text = re.sub(r'[^\w\s]', '', text)
    
    # 去除多余的空格
    text = ' '.join(text.split())
    
    return text.strip()


def is_chinese_text(text: str) -> bool:
    """
    检测文本是否包含中文字符
    """
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(chinese_pattern.search(text))


class AITranslatorAssistant(Agent):
    """AI 中译英翻译助手类"""
    
    def __init__(self) -> None:
        super().__init__(
            # 设置 AI 助手的基本指令：专门负责中文到英文的实时语音翻译
            instructions="""You are a Chinese-to-English translation assistant. Your ONLY job is to translate Chinese input into English.

CRITICAL RULES:
1. You MUST ALWAYS respond in English, never in Chinese
2. When user speaks Chinese, translate it to English immediately
3. Do NOT repeat the Chinese text
4. Do NOT add explanations or prefixes like "The translation is" or "In English"
5. Output ONLY the English translation
6. Do NOT use any tags like <think></think>
7. Keep responses concise and natural

Examples:
User: "你好，很高兴见到你" 
You: "Hello nice to meet you"

User: "今天天气真不错"
You: "The weather is really nice today"

User: "我想喝水"
You: "I want to drink water"

REMEMBER: Always respond in English only. Never output Chinese characters.""",
        )
    
    async def say(self, message: str, **kwargs) -> None:
        """重写 say 方法，在发送到 TTS 前清理文本并检查语言"""
        # 记录原始 LLM 输出
        logger.info(f"LLM 原始输出: '{message}'")
        
        # 检查是否包含中文字符
        if is_chinese_text(message):
            logger.error(f"警告：LLM 输出包含中文字符，应该是英文翻译！原文: '{message}'")
            # 强制使用英文默认响应
            cleaned_message = "I apologize but I need to translate that to English"
        else:
            # 清理文本：去除 <think></think> 标签和标点符号
            cleaned_message = clean_llm_output(message)
            
            # 如果清理后的文本为空，使用默认响应
            if not cleaned_message:
                cleaned_message = "Sorry I could not translate that"
                logger.warning("清理后文本为空，使用默认响应")
        
        # 记录清理后的文本
        logger.info(f"清理后的文本: '{cleaned_message}'")
        
        # 调用父类的 say 方法发送清理后的文本到 TTS
        await super().say(cleaned_message, **kwargs)
    
    async def on_enter(self) -> None:
        """当用户进入房间时的英文欢迎消息"""
        await self.say("Hello I am your Chinese to English translation assistant Please speak in Chinese and I will translate it to English for you")



def prewarm(proc: JobProcess):
    """预热函数：在 Agent 启动前预加载模型以提高响应速度"""
    # 加载 Silero VAD (语音活动检测) 模型，用于检测用户是否在说话
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Agent 的主入口函数"""
    # 日志设置 - 在所有日志条目中添加房间名称作为上下文
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # 创建语音 AI 管道，整合 LLM、STT、TTS 和语音检测功能
    session = AgentSession(
        # LLM (大语言模型) - Agent 的"大脑"，处理用户输入并生成翻译响应
        # 使用 DeepSeek Chat 模型进行中译英翻译
        llm=openai.LLM(model="deepseek-chat", base_url="https://api.deepseek.com"),

        # STT (语音转文本) - Agent 的"耳朵"，将用户的中文语音转换为文本
        # 使用 Cartesia 的 ink-whisper 模型进行语音识别
        stt=cartesia.STT(
            model="ink-whisper"
        ),
        # TTS (文本转语音) - Agent 的"嘴巴"，将翻译后的英文文本转换为语音
        # 使用指定的语音 ID 生成自然的英文语音
        tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        # 多语言轮次检测 - 判断用户何时开始和结束说话
        turn_detection=MultilingualModel(),
        # VAD (语音活动检测) - 检测是否有语音输入
        vad=ctx.proc.userdata["vad"],
        # 预生成模式 - 允许 LLM 在等待用户说话结束前就开始生成响应，提高响应速度
        preemptive_generation=True,
    )

    # 如果要使用实时模型而不是语音管道，可以使用以下配置：
    # session = AgentSession(
    #     # 实时模型提供更低延迟的语音交互体验
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # 记录语音转文本的最终结果
    @session.on("user_speech_committed")
    def _on_user_speech_committed(ev):
        logger.info(f"STT 最终结果: {ev.user_transcript}")

    # 记录实时语音转录结果（可能包含部分识别结果）
    @session.on("user_transcript_received")
    def _on_user_transcript_received(ev):
        logger.info(f"用户说话内容: {ev.transcript}")

    # 记录 LLM 生成的翻译结果
    @session.on("agent_speech_committed")
    def _on_agent_speech_committed(ev):
        logger.info(f"LLM 翻译输出: {ev.agent_transcript}")

    # 记录 LLM 实时生成的内容（可能包含部分生成结果）
    @session.on("agent_transcript_received")
    def _on_agent_transcript_received(ev):
        cleaned_text = clean_llm_output(ev.transcript)
        logger.info(f"LLM 实时输出: '{ev.transcript}' -> 清理后: '{cleaned_text}' (原长度: {len(ev.transcript)}, 清理后长度: {len(cleaned_text)})")

    # 监听 LLM 开始生成响应的事件
    @session.on("agent_started_speaking")
    def _on_agent_started_speaking(ev):
        logger.info("Agent 开始生成语音响应")

    # 监听 LLM 停止生成响应的事件
    @session.on("agent_stopped_speaking")
    def _on_agent_stopped_speaking(ev):
        logger.info("Agent 停止生成语音响应")

    # 处理误判中断：有时背景噪音可能会误触发中断，这些被认为是假阳性中断
    # 当检测到误判时，恢复 Agent 的语音输出
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("检测到误判中断，恢复语音输出")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # 性能指标收集 - 用于监控管道性能和资源使用情况
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        # 记录性能指标到日志
        metrics.log_metrics(ev.metrics)
        # 收集使用情况统计
        usage_collector.collect(ev.metrics)

    async def log_usage():
        """在 Agent 关闭时输出使用情况摘要"""
        summary = usage_collector.get_summary()
        logger.info(f"使用情况统计: {summary}")

    # 注册关闭回调，确保在 Agent 停止时记录使用统计
    ctx.add_shutdown_callback(log_usage)

    # # 可选：为会话添加虚拟头像
    # # 其他头像提供商请参考文档
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # 头像 ID，参考 Hedra 文档
    # )
    # # 启动头像并等待其加入房间
    # await avatar.start(session, room=ctx.room)

    # 启动会话 - 初始化语音管道并预热模型
    await session.start(
        agent=AITranslatorAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud 增强噪音消除功能
            # - 如果是自托管部署，可以省略此参数
            # - 对于电话应用，建议使用 `BVCTelephony` 以获得最佳效果
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # 加入房间并连接到用户
    await ctx.connect()


if __name__ == "__main__":
    # 启动 LiveKit Agent 应用
    # entrypoint_fnc: 主入口函数
    # prewarm_fnc: 预热函数，用于提前加载模型
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
