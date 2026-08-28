"""
旅游需求分析师 agent（Analyst Agent）—— 多智能体旅行规划系统的第一环。

职责：与用户多轮对话，收集旅游倾向信息（目标城市、天数、市内出行方式、
氛围偏好--热闹/清静）。

实现方式：把 ANALYSE_TOOLS 传给模型做 function calling，
收到工具调用后执行 analyse_travel_preference(...) 完成
校验与封装；根据返回的 missing_fields 决定追问用户还是结束收集。
信息齐全后，结构化需求保存在 agent.preference 里，供下游 agent
（行程规划等）使用。

对话数据由 ShortTermMemory（memory.py）管理：消息窗口常驻内存、超出
上限按用户轮次边界裁剪、每条消息自动写入 sessions/analyst_*.json；
进程重启后用 resume=True 恢复最近会话继续对话。
"""

import json
import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

# Windows 控制台默认 GBK 编码，模型回复含 emoji 时 print 会崩溃，改为 UTF-8
sys.stdout.reconfigure(encoding="utf-8")

# Agents/ 下的脚本直接运行时，把项目根目录加进 sys.path 才能导入 llm/tools 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import MODEL, create_client  # noqa: E402  # 需要在 sys.path 调整之后再导入
from memory import ShortTermMemory  # noqa: E402
from tools.analyse import (  # noqa: E402
    ANALYSE_TOOLS,
    TOOL_NAME,
    WEATHER_TOOL_NAME,
    WEATHER_TOOLS,
    analyse_travel_preference,
    get_weather,
)


SYSTEM_PROMPT = (
    "你是旅游需求分析师，在多智能体旅行规划系统中负责收集用户的旅游需求，"
    "包括：目标城市、旅游天数、出发日期、市内出行方式（漫步/驾车/打车，"
    "步行和 citywalk 算漫步、自驾算驾车、出租车和网约车算打车）、"
    "氛围偏好（热闹/清静，喜欢人多繁华、烟火气、热门景点算热闹，"
    "喜欢安静人少、小众清幽算清静。必须是用户明确表达的，"
    "不要仅凭出行方式、场景或活动推断，"
    "比如「在西湖边漫步」不代表清静）。\n"
    "关于日期：用户提到「明天」「后天」「下周五」「几号」等出发时间时，"
    "把它换算成具体日期（YYYY-MM-DD）填入 start_date；今天出发填今天的日期；"
    "用户没提日期就不要问，填 null 即可。\n"
    "工作方式：\n"
    "1. 每当用户给出新信息，就调用 analyse_travel_preference 工具抽取结构化信息；"
    "抽取时要汇总整个对话中已知的信息，而不只是用户最新一句；"
    "确实没提到的字段填 null，不要猜测或编造。\n"
    "2. 查看工具返回的 missing_fields：仍有缺失时，友好地向用户追问，"
    "可以一次把缺失项都问出来。\n"
    "3. 五项信息都齐了，就用一两句话向用户确认需求，不要再继续闲聊。\n"
    "4. 用户询问目的地天气时，调用 get_weather 工具（默认实况，"
    "问未来几天就传 forecast=true），根据返回结果简要回答。\n"
    "用中文回复，保持简短自然。"
)

# agent 可用的全部工具：需求抽取 + 天气查询
ALL_TOOLS = ANALYSE_TOOLS + WEATHER_TOOLS

# 单轮 chat 内工具调用的最大轮数，防止模型陷入无限调用
MAX_TOOL_ROUNDS = 8

EXIT_WORDS = {"退出", "exit", "quit", "再见", "拜拜"}


class AnalystAgent:
    """基于 function calling 的旅游需求分析师。"""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        resume: bool = False,
        persist: bool = True,
    ):
        self.client = client or create_client()
        if resume:
            latest = ShortTermMemory.latest_session_file("analyst")
            if latest is None:
                self.memory = ShortTermMemory(
                    "analyst", SYSTEM_PROMPT, persist=persist
                )
            else:
                self.memory = ShortTermMemory.load(latest, persist=persist)
        else:
            self.memory = ShortTermMemory(
                "analyst", SYSTEM_PROMPT, persist=persist
            )

    def reset(self) -> None:
        """清空对话历史，开始一次新的需求收集（换新会话文件）。"""
        self.memory.reset()

    @property
    def preference(self) -> Optional[dict]:
        """最近一次工具抽取出的结构化需求（随会话一起保存/恢复）。"""
        return self.memory.data.get("preference")

    @property
    def is_complete(self) -> bool:
        """各字段是否已收集齐全（可以交给下游 agent）。"""
        return (
            self.preference is not None
            and bool(self.preference.get("success"))
            and not self.preference.get("missing_fields")
        )

    def chat(self, user_input: str, on_delta=None) -> str:
        """
        处理一轮用户输入，返回助手的回复文本。

        on_delta(chunk) 存在时，流式正文逐段交给回调（Web 端推给浏览器）；
        否则流式打印到终端（CLI 行为）。
        内部是一个 function calling 循环：模型请求调用工具 ->
        执行 analyse_travel_preference 并把结果作为 tool 消息回传 ->
        模型基于结果给出追问或确认，直到不再调用工具为止。
        """
        self.memory.append({"role": "user", "content": user_input})

        for _ in range(MAX_TOOL_ROUNDS):
            message = self._ask_model(on_delta)

            if not message.tool_calls:
                reply = message.content or ""
                self.memory.append({"role": "assistant", "content": reply})
                return reply

            # 先记录模型的工具调用，再逐个执行，结果以 tool 消息回传给模型
            self.memory.append(message.model_dump(exclude_none=True))
            for tool_call in message.tool_calls:
                result = self._execute_tool(tool_call)
                self.memory.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        # 超过轮数上限还没结束：兜底返回，同时补一条 assistant 消息，
        # 保证消息历史里 tool 消息后面跟着 assistant，协议依然合法
        fallback = "抱歉，我这边处理出了点问题，请换个说法再试一次。"
        self.memory.append({"role": "assistant", "content": fallback})
        return fallback

    def _ask_model(self, on_delta=None) -> ChatCompletionMessage:
        """请求一次模型；流式输出回复正文，返回完整消息（可能含 tool_calls）。"""
        with self.client.chat.completions.stream(
            model=MODEL,
            messages=self.memory.messages,
            tools=ALL_TOOLS,
            # 对话和工具调用不需要深度思考，关掉更快更省
            extra_body={"enable_thinking": False},
        ) as stream:
            printed = False
            for event in stream:
                if event.type == "content.delta":
                    if on_delta is not None:
                        on_delta(event.delta)
                    else:
                        print(event.delta, end="", flush=True)
                    printed = True
            if printed and on_delta is None:
                print()  # 终端流式输出结束后换行
            return stream.get_final_completion().choices[0].message

    def _execute_tool(self, tool_call) -> dict:
        """执行模型请求的工具调用，返回结果 dict（会回传给模型）。"""
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            return {"success": False, "error": "工具调用参数不是合法的 JSON"}

        name = tool_call.function.name
        if name == TOOL_NAME:
            result = analyse_travel_preference(
                city=arguments.get("city"),
                days=arguments.get("days"),
                start_date=arguments.get("start_date"),
                travel_mode=arguments.get("travel_mode"),
                atmosphere=arguments.get("atmosphere"),
            )
            if result.get("success"):
                self.memory.save_data(preference=result)
            return result
        if name == WEATHER_TOOL_NAME:
            return get_weather(
                city=arguments.get("city", ""),
                forecast=bool(arguments.get("forecast", False)),
            )
        return {"success": False, "error": f"未知工具：{name}"}

    def run(self) -> None:
        """命令行交互入口：多轮对话收集需求，收集齐全后输出结构化结果。"""
        print("=" * 20 + " 旅游需求分析师 " + "=" * 20)
        if self.memory.session_file:
            print(f"会话记录：{self.memory.session_file}")
        if len(self.memory.messages) > 1:
            print(f"（已恢复上次会话，{len(self.memory.messages)} 条消息）")
        print("和我聊聊你的旅行想法吧（目的地、天数、出行和氛围偏好），输入「退出」结束。")
        while True:
            try:
                user_input = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n分析师：再见，想规划旅行时随时来找我！")
                break
            if not user_input:
                continue
            if user_input in EXIT_WORDS:
                print("分析师：再见，想规划旅行时随时来找我！")
                break

            print("分析师：", end="", flush=True)
            self.chat(user_input)

            if self.is_complete:
                print("=" * 20 + " 结构化需求（交给下游 agent） " + "=" * 20)
                print(json.dumps(self.preference, ensure_ascii=False, indent=2))
                break


if __name__ == "__main__":
    # 加 --resume 参数可恢复最近一次会话继续对话
    AnalystAgent(resume="--resume" in sys.argv).run()
