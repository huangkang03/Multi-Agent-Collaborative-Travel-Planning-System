"""
Web 服务 -- 多智能体旅行规划系统的浏览器交互入口。

架构：
- 前端 web/index.html（原生 JS，EventSource 收 SSE）
- 本文件 FastAPI 后端：
  GET  /               首页
  GET  /api/stream     SSE 事件流（连接时先推全量 state，页面刷新可恢复）
  POST /api/chat       用户说话 -> analyst 处理（线程池，流式 delta 走 SSE）
  POST /api/pipeline   需求收齐后启动第二、三环（selector + planner 后台线程）
  POST /api/reset      新会话（清空对话重新收集需求）

单用户单会话（与 CLI 语义一致）：启动时自动 resume 最近 analyst 会话；
agent 是同步阻塞代码，全部经 asyncio.to_thread 跑在线程池里，
事件经 EventBus（asyncio.Queue 扇出）推送回浏览器。

运行：python server.py  然后浏览器打开 http://127.0.0.1:8000
"""

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Windows 控制台默认 GBK 编码，输出含中文/emoji 时会崩溃，改为 UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI  # noqa: E402  # 需要在 sys.path 调整之后再导入
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from sse_starlette.sse import EventSourceResponse  # noqa: E402

from Agents.analyst import AnalystAgent  # noqa: E402
from Agents.api_selector import ApiSelectorAgent  # noqa: E402
from Agents.planner import PlannerAgent  # noqa: E402

ROOT = Path(__file__).resolve().parent
WEB_INDEX = ROOT / "web" / "index.html"

# 工具日志在内存里最多保留多少条（SSE 时间线展示用）
TOOL_LOG_LIMIT = 120


def sse_event(name: str, data: str) -> dict:
    """sse-starlette 的事件格式：dict 才会生成 event:/data: 两行。"""
    return {"event": name, "data": data}


# ---------- 事件总线 ----------


class EventBus:
    """SSE 事件总线：工作线程 -> 主事件循环 -> 各浏览器连接。"""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: dict = {}  # id(queue) -> asyncio.Queue

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[id(queue)] = queue
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.pop(id(queue), None)

    def _fanout(self, name: str, data: str) -> None:
        for queue in list(self._subscribers.values()):
            queue.put_nowait((name, data))

    def publish(self, name: str, data: dict) -> None:
        """事件循环线程内调用。"""
        self._fanout(name, json.dumps(data, ensure_ascii=False))

    def thread_publish(self, name: str, data: dict) -> None:
        """工作线程里调用：安全地转回主事件循环再扇出。"""
        if self._loop is not None and self._loop.is_running():
            payload = (name, json.dumps(data, ensure_ascii=False))
            self._loop.call_soon_threadsafe(self._fanout, *payload)


bus = EventBus()


# ---------- 全局状态（单会话） ----------


class AppState:
    def __init__(self):
        self.analyst: Optional[AnalystAgent] = None
        self.plan: Optional[dict] = None
        self.itinerary_file: Optional[str] = None
        # 正在进行的操作："chat" / "pipeline"，None 表示空闲
        self.busy_reason: Optional[str] = None
        # selector / planner 两环的展示状态
        self.stages = {"selector": "idle", "planner": "idle"}
        # planner 的工具调用日志（内存，重启不保留）
        self.tool_log: list = []

    @property
    def busy(self) -> bool:
        return self.busy_reason is not None

    def snapshot(self) -> dict:
        """全量状态（连接 / 重连 / reset 时推给浏览器恢复界面）。"""
        messages = []
        preference = None
        if self.analyst and self.analyst.memory:
            for m in self.analyst.memory.messages:
                if m.get("role") in ("user", "assistant") and m.get("content"):
                    messages.append({"role": m["role"], "content": m["content"]})
            preference = self.analyst.preference
        complete = bool(preference and preference.get("success")
                        and not preference.get("missing_fields"))

        # 服务重启后从磁盘恢复已有行程
        itinerary_markdown = None
        if self.itinerary_file and Path(self.itinerary_file).exists():
            itinerary_markdown = Path(self.itinerary_file).read_text(encoding="utf-8")
        elif complete and preference.get("city"):
            legacy = ROOT / f"itinerary_{str(preference['city']).strip()}.md"
            if legacy.exists():
                itinerary_markdown = legacy.read_text(encoding="utf-8")
                self.itinerary_file = str(legacy)

        apis = []
        if self.plan and self.plan.get("apis"):
            apis = [
                {
                    "method": a.get("method"),
                    "path": a.get("path"),
                    "summary": a.get("summary"),
                    "purpose": a.get("purpose"),
                }
                for a in self.plan["apis"]
            ]
        # 有行程说明 planner 已完成（重启恢复场景）
        if itinerary_markdown and self.stages["planner"] == "idle":
            self.stages["planner"] = "done"
            self.stages["selector"] = "done"

        return {
            "messages": messages,
            "preference": preference,
            "complete": complete,
            "busy": self.busy_reason,
            "stages": dict(self.stages, analyst="done" if complete else "idle"),
            "apis": apis,
            "tool_log": self.tool_log,
            "itinerary": itinerary_markdown,
            "itinerary_file": self.itinerary_file,
        }


state = AppState()

app = FastAPI(title="多智能体旅行规划系统")


@app.on_event("startup")
async def startup() -> None:
    bus.bind_loop(asyncio.get_running_loop())
    # 自动恢复最近一次分析师会话（等价 CLI 的 --resume）
    state.analyst = AnalystAgent(resume=True)
    print("多智能体旅行规划系统已启动：http://127.0.0.1:8000")


# ---------- 页面与 SSE ----------


@app.get("/")
async def index():
    return FileResponse(WEB_INDEX)


@app.get("/api/stream")
async def stream():
    queue = bus.subscribe()

    async def event_generator():
        try:
            # 连接即发送全量状态（页面刷新 / 断线重连后恢复界面）
            yield sse_event("state", json.dumps(state.snapshot(), ensure_ascii=False))
            while True:
                name, data = await queue.get()
                yield sse_event(name, data)
        finally:
            bus.unsubscribe(queue)

    return EventSourceResponse(event_generator())


# ---------- 对话（第一环） ----------


class ChatIn(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(inp: ChatIn):
    if state.busy:
        return JSONResponse({"error": "系统正忙，请稍候"}, status_code=409)
    if not inp.message.strip():
        return JSONResponse({"error": "消息不能为空"}, status_code=400)

    state.busy_reason = "chat"

    def on_delta(text: str) -> None:
        bus.thread_publish("chat_delta", {"text": text})

    try:
        # agent 是同步阻塞代码，放线程池；流式片段经 SSE 推给浏览器
        reply = await asyncio.to_thread(state.analyst.chat, inp.message, on_delta)
        preference = state.analyst.preference
        bus.publish("chat_done", {"reply": reply, "preference": preference})
        return {"reply": reply, "preference": preference}
    except Exception as e:
        bus.publish("error", {"message": f"对话处理失败：{e}"})
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        state.busy_reason = None


# ---------- 规划流水线（第二、三环） ----------


@app.post("/api/pipeline")
async def pipeline():
    if state.busy:
        return JSONResponse({"error": "系统正忙，请稍候"}, status_code=409)
    if not (state.analyst and state.analyst.is_complete):
        return JSONResponse({"error": "需求尚未收集完整，请先完成对话"}, status_code=400)

    state.busy_reason = "pipeline"
    preference = state.analyst.preference
    asyncio.create_task(run_pipeline(preference))
    return {"started": True, "preference": preference}


async def run_pipeline(preference: dict) -> None:
    """后台执行：API 选择器（kimi-k3）-> 行程规划师（qwen）。"""
    try:
        # --- 第二环：API 选择 ---
        state.stages["selector"] = "running"
        state.plan = None
        state.tool_log = []
        bus.publish("stage", {"stage": "selector", "status": "running"})

        def selector_event(_stage: str, detail: str) -> None:
            bus.thread_publish("progress", {"stage": "selector", "detail": detail})

        def run_selector() -> dict:
            return ApiSelectorAgent().analyze(preference, on_event=selector_event)

        plan = await asyncio.to_thread(run_selector)
        if not plan.get("success"):
            state.stages["selector"] = "error"
            bus.publish("stage", {"stage": "selector", "status": "error",
                                  "detail": plan.get("error", "")})
            bus.publish("error", {"message": f"API 选择失败：{plan.get('error')}"})
            return

        state.plan = plan
        state.stages["selector"] = "done"
        bus.publish("stage", {"stage": "selector", "status": "done"})
        bus.publish("apis", {"apis": [
            {
                "method": a.get("method"),
                "path": a.get("path"),
                "summary": a.get("summary"),
                "purpose": a.get("purpose"),
            }
            for a in plan["apis"]
        ]})

        # --- 第三环：行程规划 ---
        state.stages["planner"] = "running"
        bus.publish("stage", {"stage": "planner", "status": "running"})

        planner = PlannerAgent()

        def planner_event(kind: str, data: dict) -> None:
            if kind == "tool":
                state.tool_log = (state.tool_log + [data])[-TOOL_LOG_LIMIT:]
            bus.thread_publish("planner_event", {"kind": kind, **data})

        def planner_delta(text: str) -> None:
            bus.thread_publish("itinerary_delta", {"text": text})

        def run_planner() -> Optional[str]:
            return planner.plan(plan, on_event=planner_event, on_delta=planner_delta)

        itinerary = await asyncio.to_thread(run_planner)
        if itinerary:
            state.itinerary_file = str(planner.itinerary_file)
            state.stages["planner"] = "done"
            bus.publish("itinerary", {"markdown": itinerary,
                                      "file": state.itinerary_file})
            bus.publish("stage", {"stage": "planner", "status": "done"})
        else:
            state.stages["planner"] = "error"
            bus.publish("stage", {"stage": "planner", "status": "error",
                                  "detail": "行程规划失败"})
            bus.publish("error", {"message": "行程规划失败，请重试"})
    except Exception as e:
        bus.publish("error", {"message": f"规划流程异常：{e}"})
    finally:
        state.busy_reason = None


# ---------- 新会话 ----------


@app.post("/api/reset")
async def reset():
    if state.busy:
        return JSONResponse({"error": "系统正忙，请稍候"}, status_code=409)
    state.analyst.reset()
    state.plan = None
    state.itinerary_file = None
    state.stages = {"selector": "idle", "planner": "idle"}
    state.tool_log = []
    bus.publish("state", state.snapshot())
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
