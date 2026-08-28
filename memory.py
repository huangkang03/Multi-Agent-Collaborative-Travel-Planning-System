"""
短期记忆（Short-Term Memory）-- agent 会话内对话数据的管理与保存。

对话数据的保存机制分三层：
1. 上下文内记忆：完整消息列表（system + 多轮 user/assistant/tool）常驻内存，
   每轮请求整体发给模型--模型"记得"对话靠的就是这个；
2. 有界窗口：消息超过上限时按「用户轮次边界」裁剪旧消息。只在 user
   消息处下刀，system 永远保留，assistant(tool_calls) 和它对应的
   tool 消息永远锁在同一轮内被一起保留或裁掉，不破坏 function calling
   协议（孤儿 tool 消息会让 API 直接报错）；
3. 会话文件：每次 append 自动把消息列表写入 sessions/<agent>_<时间戳>
   .json，进程退出数据不丢，load() 可恢复继续对话。
   短期 = 会话/任务级，与跨会话沉淀用户偏好的长期记忆相区别。

纯基础设施：不实例化模型客户端，与 llm.py 同级，供各 agent 使用。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent
SESSION_DIR = _ROOT / "sessions"

# 窗口上限（消息条数，含 system）。一轮对话约 3~6 条消息，60 条约 10+ 轮
DEFAULT_MAX_MESSAGES = 60


class ShortTermMemory:
    """单个 agent 会话的短期记忆：内存消息窗口 + 会话文件。"""

    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        session_dir: Path = SESSION_DIR,
        persist: bool = True,
    ):
        self.agent_name = agent_name
        self.max_messages = max_messages
        self.session_dir = Path(session_dir)
        self.persist = persist
        # agent 附带的结构化数据（如 analyst 的 preference），随会话保存/恢复
        self.data: dict = {}
        self._messages: list = [{"role": "system", "content": system_prompt}]
        self._created_at = datetime.now()
        self._session_file: Optional[Path] = None
        if self.persist:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._session_file = self.session_dir / (
                f"{agent_name}_{self._created_at:%Y%m%d_%H%M%S}.json"
            )
            self._save()

    # ---------- 消息操作 ----------

    def append(self, message: dict) -> None:
        """追加一条消息；随后做窗口裁剪并落盘。"""
        self._messages.append(message)
        self._trim()
        if self.persist:
            self._save()

    @property
    def messages(self) -> list:
        """当前窗口的快照（发给模型用；浅拷贝防止调用方误改内部列表）。"""
        return list(self._messages)

    @property
    def session_file(self) -> Optional[Path]:
        """本会话的落盘文件路径（persist=False 时为 None）。"""
        return self._session_file

    def reset(self) -> None:
        """开始新会话：清空对话（system 保留）和附带数据，换新会话文件。"""
        self._messages = [self._messages[0]]
        self.data = {}
        if self.persist:
            self._created_at = datetime.now()
            self._session_file = self.session_dir / (
                f"{self.agent_name}_{self._created_at:%Y%m%d_%H%M%S}.json"
            )
            self._save()

    def save_data(self, **kwargs) -> None:
        """更新附带的会话数据（如 preference）并落盘。"""
        self.data.update(kwargs)
        if self.persist:
            self._save()

    # ---------- 窗口裁剪 ----------

    def _trim(self) -> None:
        """超出上限时裁掉最旧的消息，只在 user 消息边界处下刀。"""
        if len(self._messages) <= self.max_messages:
            return
        # 候选起点：砍到只剩 max_messages 条（索引 0 是 system，跳过）
        start = 1 + (len(self._messages) - self.max_messages)
        # 向后推进到最近的 user 消息：一轮以 user 开头，
        # assistant(tool_calls) 和它的 tool 消息都在同一轮内，
        # 在这里切不会产生没有配对的孤儿 tool 消息
        while (
            start < len(self._messages)
            and self._messages[start].get("role") != "user"
        ):
            start += 1
        if start < len(self._messages):
            self._messages = [self._messages[0]] + self._messages[start:]

    # ---------- 会话文件 ----------

    def _save(self) -> None:
        if not self.persist or self._session_file is None:
            return
        payload = {
            "agent": self.agent_name,
            "created_at": self._created_at.isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "message_count": len(self._messages),
            "messages": self._messages,
            "data": self.data,
        }
        self._session_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        session_file,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        persist: bool = True,
    ) -> "ShortTermMemory":
        """从会话文件恢复记忆（继续写回原文件，可接着对话）。"""
        session_file = Path(session_file)
        raw = json.loads(session_file.read_text(encoding="utf-8"))
        memory = cls(
            agent_name=raw["agent"],
            system_prompt=raw["messages"][0]["content"],
            max_messages=max_messages,
            persist=False,  # 先禁止构造器另建新文件
        )
        memory._messages = raw["messages"]
        memory.data = raw.get("data", {})
        try:
            memory._created_at = datetime.fromisoformat(raw["created_at"])
        except (KeyError, ValueError):
            pass
        memory._session_file = session_file  # 继续写回原会话文件
        memory.persist = persist
        memory._sanitize_tail()
        if persist:
            memory._save()
        return memory

    def _sanitize_tail(self) -> None:
        """防御：会话若在工具调用中途崩溃，末尾可能残留没有 tool 结果的
        assistant(tool_calls)，这种状态会让下一次请求直接报协议错误，
        把不完整的尾部连同其后的消息一起截掉。"""
        for i in range(len(self._messages) - 1, -1, -1):
            message = self._messages[i]
            if message.get("role") == "assistant" and message.get("tool_calls"):
                ids = {c.get("id") for c in message["tool_calls"]}
                answered = {
                    m.get("tool_call_id")
                    for m in self._messages[i + 1 :]
                    if m.get("role") == "tool"
                }
                if not ids <= answered:
                    self._messages = self._messages[:i]
                break

    @staticmethod
    def latest_session_file(agent_name: str, session_dir=SESSION_DIR) -> Optional[Path]:
        """找该 agent 最近的一个会话文件；没有则返回 None。"""
        files = sorted(Path(session_dir).glob(f"{agent_name}_*.json"))
        return files[-1] if files else None
