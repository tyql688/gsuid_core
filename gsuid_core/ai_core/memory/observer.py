"""观察者管道（Observer Pipeline）

Observer 是整个记忆系统的"被动感知层"：AI 可以读取所有消息以构建认知，
但不需要因此回复任何一条。它与 AI 的发言决策完全正交——即使 Persona 配置为
纯静默模式，记忆依然在后台积累。

使用 queue.Queue（线程安全）传递观察记录，支持 IngestionWorker 在独立线程
的事件循环中运行，避免 LLM 调用阻塞主事件循环导致 WebSocket 心跳超时。
"""

import queue as sync_queue
from typing import Optional
from datetime import datetime, timezone
from dataclasses import dataclass

from gsuid_core.logger import logger

# 全局消息队列（线程安全，支持跨线程通信）
_observation_queue: sync_queue.Queue = sync_queue.Queue(maxsize=10_000)


@dataclass
class ObservationRecord:
    """Observer Pipeline 的最小数据单元"""

    raw_content: str
    speaker_id: str
    group_id: Optional[str]  # 原始群组 ID（如 "789012"）
    scope_key: str  # 格式化后的 Scope Key（如 "group:789012"）
    timestamp: datetime
    message_type: str  # "group_msg" | "private_msg"


def _should_observe(
    content: str,
    speaker_id: str,
    bot_self_id: str,
    observer_blacklist: list[str],
    group_id: Optional[str],
) -> bool:
    """判断该消息是否值得入队"""
    # 过滤自身消息
    if speaker_id == bot_self_id:
        return False
    # 过滤黑名单群组
    if group_id and group_id in observer_blacklist:
        return False
    # 过滤过短内容（纯表情、单字回复等噪声）
    stripped = content.strip()
    if len(stripped) < 5:
        return False
    # 过滤纯图片/文件消息（无文字）
    if not stripped or (stripped.startswith("[图片]") and len(stripped) < 10):
        return False
    return True


async def observe(
    content: str,
    speaker_id: str,
    group_id: Optional[str],
    bot_self_id: str,
    observer_blacklist: list[str],
    message_type: str = "group_msg",
) -> None:
    """向观察队列投递一条消息记录。

    此函数应在 handler.py 中以 asyncio.create_task() 调用，不 await。

    Example:
        asyncio.create_task(
            memory_observer.observe(
                content=msg.raw_text,
                speaker_id=msg.user_id,
                group_id=msg.group_id,
                bot_self_id=bot_self_id,
                observer_blacklist=memory_config.observer_blacklist,
            )
        )
    """
    from .scope import ScopeType, make_scope_key

    if not _should_observe(content, speaker_id, bot_self_id, observer_blacklist, group_id):
        return

    record = ObservationRecord(
        raw_content=content,
        speaker_id=speaker_id,
        group_id=group_id,
        scope_key=make_scope_key(
            ScopeType.GROUP if group_id else ScopeType.USER_GLOBAL,
            group_id if group_id else speaker_id,
        ),
        timestamp=datetime.now(timezone.utc),
        message_type=message_type,
    )

    try:
        _observation_queue.put_nowait(record)
        # 上报观察入队统计
        try:
            from gsuid_core.ai_core.statistics import statistics_manager

            statistics_manager.record_memory_observation()
        except Exception:
            pass
    except sync_queue.Full:
        # 队列满时丢弃最老的一条，保证新消息不丢失
        try:
            _observation_queue.get_nowait()
            _observation_queue.put_nowait(record)
        except Exception:
            logger.warning("Memory observation queue overflow, dropping message")


def get_observation_queue() -> sync_queue.Queue:
    """供 IngestionWorker 获取队列引用（线程安全的 queue.Queue）"""
    return _observation_queue
