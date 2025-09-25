import logging

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


class AITranslatorAssistant(Agent):
    """AI 中译英翻译助手类"""
    
    def __init__(self) -> None:
        super().__init__(
            # 设置 AI 助手的基本指令：专门负责中文到英文的翻译
            instructions="""你现在是一个优秀的AI中译英翻译，你的职责是当用户说出的是中文的内容时，你将对应对的中文内容翻译成英文""",
        )

    async def on_enter(self) -> None:
        """当用户进入房间时的欢迎消息"""
        await self.session.generate_reply(
            instructions="用中文的方式，告诉用户你是优秀的中译英翻译，你可以帮助他将他说的中文内容翻译成英文")

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
