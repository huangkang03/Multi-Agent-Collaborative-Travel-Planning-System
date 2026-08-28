"""
行程规划师 agent（Planner Agent）-- 多智能体旅行规划系统的第三环（终环）。

职责：接收 api_plan.json（结构化需求 + 选中的 API 及用途），按计划真实调用
高德 Web 服务接口收集数据（城市坐标、POI、路线、天气），最终产出逐日行程。

实现方式：function calling 循环（同 analyst 模式）--只暴露 api_plan 选中的
接口对应的本地封装（tools/amap.py 的 run_tool + analyse 的 get_weather），
模型按计划里的调用顺序与用途取数，数据足够后输出最终行程
（markdown，流式打印）。行程保存为项目根目录的 itinerary_<城市>.md，
规划过程由短期记忆落盘 sessions/planner_*.json。
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

# Windows 控制台默认 GBK 编码，模型回复含中文/emoji 时会崩溃，改为 UTF-8
sys.stdout.reconfigure(encoding="utf-8")

# Agents/ 下的脚本直接运行时，把项目根目录加进 sys.path 才能导入 llm/tools 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import MODEL, create_client  # noqa: E402  # 需要在 sys.path 调整之后再导入
from memory import ShortTermMemory  # noqa: E402
from tools.amap import PATH_TOOL_MAP, SCHEMA_BY_NAME, run_tool  # noqa: E402
from tools.analyse import (  # noqa: E402
    WEATHER_TOOL_NAME,
    WEATHER_TOOLS,
    get_weather,
)

_ROOT = Path(__file__).resolve().parent.parent
API_PLAN_PATH = _ROOT / "api_plan.json"

# 规划要查坐标/搜景点/排路线/看天气，工具轮数上限比对话类 agent 高
MAX_TOOL_ROUNDS = 40

# 氛围偏好 -> 搜索关键词指引
_ATMOSPHERE_HINTS = {
    "清静": "关键词偏向小众清幽：园林、公园、博物馆、古镇、寺庙、安静街巷、绿道",
    "热闹": "关键词偏热门人气：热门商圈、地标、步行街、夜市、美食街、网红景点",
}

# 出行方式 -> 路线工具指引
_MODE_HINTS = {
    "漫步": "用 amap_route_walking 查相邻点位步行路线，相邻点位优先控制在 2 公里内",
    "驾车": "用 amap_route_driving 查驾车路线与耗时，点位可跨区域安排",
    "打车": "用 amap_route_driving 查车行距离耗时，作为打车参考，点位可跨区域安排",
}


class PlannerAgent:
    """按 api_plan 真实调用高德接口、产出逐日行程的规划师。"""

    def __init__(self, client: Optional[OpenAI] = None, persist: bool = True):
        self.client = client or create_client()
        self.persist = persist
        # 每次 plan() 新建一份会话记忆（规划过程落盘 sessions/planner_*.json）
        self.memory: Optional[ShortTermMemory] = None
        # 成功产出行程后指向 itinerary_<城市>.md
        self.itinerary_file: Optional[Path] = None
        # plan() 期间生效的进度/流式回调（Web 端注入）
        self._on_event = None
        self._on_delta = None

    # ---------- 工具选择与提示词 ----------

    def _select_tools(self, apis: list) -> tuple:
        """把计划选中的 OpenAPI path 映射到本地工具。

        返回 (工具定义列表, 跳过的 path 列表)。
        """
        tools, seen, skipped = [], set(), []
        for api in apis:
            name = PATH_TOOL_MAP.get(api["path"])
            if name is None:
                skipped.append(api["path"])
                continue
            if name in seen:
                continue
            if name == WEATHER_TOOL_NAME:
                tools.append(WEATHER_TOOLS[0])
            else:
                schema = SCHEMA_BY_NAME.get(name)
                if schema is None:
                    skipped.append(api["path"])
                    continue
                tools.append(schema)
            seen.add(name)
        return tools, skipped

    def _build_system_prompt(self, preference: dict, apis: list) -> str:
        """生成规划师的 system prompt：需求 + 本次计划选中的工具及用途。"""
        city = preference.get("city") or "目的地"
        days = preference.get("days") or ""
        mode = preference.get("travel_mode") or ""
        atmosphere = preference.get("atmosphere") or ""

        # 日期与星期：出发日期 + 逐日排期，周几用于闭馆日提醒
        date_lines = ""
        start_date = preference.get("start_date")
        if start_date:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d").date()
            except ValueError:
                start = None
        else:
            start = None
        if start is not None:
            week_names = "一二三四五六日"
            day_lines = []
            for offset in range(int(days or 1)):
                d = start + timedelta(days=offset)
                day_lines.append(
                    f"Day {offset + 1} = {d.isoformat()}（周{week_names[d.weekday()]}）"
                )
            date_lines = (
                f"\n出发日期：{start.isoformat()}（周{week_names[start.weekday()]}），"
                f"共 {days} 天。逐日对应：\n"
                + "\n".join(day_lines)
                + "\n注意：周一多数博物馆闭馆；把行程日期和闭馆日核对，"
                "如某天撞上闭馆日要调整当天安排。\n"
            )
        else:
            date_lines = "\n出发日期：用户未指定，按 Day 1/Day 2 相对日期编排即可。\n"

        tool_lines = [
            f"- {PATH_TOOL_MAP[api['path']]}（{api['summary']}）：{api['purpose']}"
            for api in apis
            if PATH_TOOL_MAP.get(api["path"])
        ]
        atmosphere_hint = _ATMOSPHERE_HINTS.get(
            atmosphere, "按用户氛围偏好组织搜索关键词"
        )
        mode_hint = _MODE_HINTS.get(mode, "按出行方式选择对应的路线工具")

        return (
            "你是行程规划师，在多智能体旅行规划系统中负责产出最终的逐日行程。\n"
            f"\n用户需求：{city} {days} 天，市内出行{mode}，氛围偏好{atmosphere}。"
            + date_lines
            + "\n\n本次规划可用的工具（用途来自上游 API 选择器的决策，按建议顺序）：\n"
            + "\n".join(tool_lines)
            + "\n\n工作流程：\n"
            "1. 先用 amap_geocode 把城市名转成坐标。\n"
            f"2. 搜索每天的候选景点：{atmosphere_hint}。\n"
            "3. 搜索美食（本地菜、小吃、茶馆/咖啡馆）。\n"
            "4. 每天挑 2~4 个点位，围绕点位用周边搜索安排三餐。\n"
            f"5. 相邻点位查路线：{mode_hint}。\n"
            "6. 用 get_weather(city=城市名, forecast=True) 查天气预报，"
            "把逐日天气对应到具体日期。\n"
            "7. 信息足够后（每天点位、衔接路线、天气齐备）立即停止调用工具，"
            "直接输出最终行程。\n"
            "\n最终行程用 markdown 输出，格式：\n"
            f"# {city}{days}日行程\n"
            "## Day 1（M月D日 周X）：当日主题/区域\n"
            "- 09:00 地点名（地址）--一句话玩法\n"
            "- 步行 15 分钟（1.2km）\n"
            "- 12:00 午餐：店名（地址，评分）\n"
            "……每天如此；行程末尾附「天气提示」「出行说明」两节。\n"
            "\n写作要求：\n"
            "- 地点、地址、评分、距离、耗时必须来自工具返回的数据，不得编造；\n"
            "- 同一天的点位相对集中，行程紧凑但不赶；\n"
            "- 直接输出 markdown 正文，不要包裹在代码块里。"
        )

    # ---------- 主流程 ----------

    def plan(
        self, api_plan: dict, on_event=None, on_delta=None
    ) -> Optional[str]:
        """
        执行完整规划：按计划调用高德接口取数，产出最终行程。

        api_plan 为 api_selector 的输出（preference + apis + details）。
        on_event(kind, data) 接收进度事件（Web 端推送）：
          kind="start"  data={"tools": 工具数}
          kind="tool"   data={"name", "note"}   每次高德/天气调用
          kind="saved"  data={"file"}           行程落盘
        on_delta(chunk) 接收行程正文的流式片段；两者缺省时打印到终端。
        返回行程 markdown 文本；失败返回 None。成功后写入 itinerary_<城市>.md。
        """
        self._on_event = on_event
        self._on_delta = on_delta
        preference = api_plan.get("preference", {})
        apis = api_plan.get("apis", [])
        tools, skipped = self._select_tools(apis)
        if skipped:
            self._report_text(f"提示：计划中以下接口暂无本地封装，已跳过：{skipped}")
        if not tools:
            self._report_text("错误：计划里没有任何可调用的工具，无法规划。")
            return None

        prompt = self._build_system_prompt(preference, apis)
        self.memory = ShortTermMemory("planner", prompt, persist=self.persist)
        self.memory.append(
            {
                "role": "user",
                "content": (
                    f"请为上述需求制定 {preference.get('days')} 天的完整行程，"
                    "先调用工具收集真实数据，再输出最终行程。"
                ),
            }
        )
        self._emit("start", {"tools": len(tools)})

        itinerary = self._loop(tools)
        if itinerary:
            self.itinerary_file = self._save_itinerary(itinerary, preference)
        return itinerary

    def _emit(self, kind: str, data: dict) -> None:
        """进度事件：有回调交给回调，否则按终端样式打印。"""
        if self._on_event is not None:
            self._on_event(kind, data)
        elif kind == "start":
            print("=" * 20 + f" 行程规划师（{data['tools']} 个工具） " + "=" * 20)
        elif kind == "tool":
            print(f"  [工具] {data['name']}：{data.get('note', '')}")
        elif kind == "saved":
            print(f"行程已保存：{data['file']}")

    def _report_text(self, text: str) -> None:
        """规划过程中的普通提示文本。"""
        if self._on_event is not None:
            self._on_event("text", {"message": text})
        else:
            print(text)

    def _loop(self, tools: list) -> Optional[str]:
        """function calling 循环：取数直到模型输出最终行程。"""
        for _ in range(MAX_TOOL_ROUNDS):
            message = self._ask_model(tools)

            if not message.tool_calls:
                reply = self._clean(message.content or "")
                self.memory.append({"role": "assistant", "content": reply})
                return reply

            # 先记录模型的工具调用，再逐个执行，结果以 tool 消息回传给模型
            self.memory.append(message.model_dump(exclude_none=True))
            for tool_call in message.tool_calls:
                result = self._execute_tool(tool_call)
                self._emit("tool", self._tool_note(tool_call, result))
                self.memory.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        # 工具轮数用尽：追加收尾指令，不带工具再问一次，逼出最终行程
        self._report_text("（已达到工具调用上限，基于现有信息收尾）")
        return self._force_final()

    @staticmethod
    def _tool_note(tool_call, result: dict) -> dict:
        """把工具调用压缩成一条进度备注（Web 端时间线展示用）。"""
        name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        brief = {}
        if arguments.get("keywords"):
            brief["keywords"] = arguments["keywords"]
        if arguments.get("city"):
            brief["city"] = arguments["city"]
        if arguments.get("address"):
            brief["address"] = arguments["address"]
        note = "、".join(f"{k}={v}" for k, v in brief.items()) or "调用成功"
        if not result.get("success"):
            note = f"失败：{result.get('error', '')[:60]}"
        elif result.get("pois"):
            note += f"，返回 {result['count']} 个地点"
        return {"name": name, "note": note}

    def _force_final(self) -> Optional[str]:
        """不带工具请求一次，让模型基于已收集的信息直接输出行程。"""
        self.memory.append(
            {
                "role": "user",
                "content": (
                    "已达到工具调用上限。请立即基于已收集的全部信息，"
                    "直接输出完整行程（markdown），不要再调用工具。"
                ),
            }
        )
        message = self._ask_model(tools=None)
        reply = self._clean(message.content or "")
        if not reply:
            return None
        self.memory.append({"role": "assistant", "content": reply})
        return reply

    # ---------- 模型与工具执行 ----------

    def _ask_model(self, tools: Optional[list] = None) -> ChatCompletionMessage:
        """请求一次模型；流式输出回复正文，返回完整消息（可能含 tool_calls）。"""
        request = {
            "model": MODEL,
            "messages": self.memory.messages,
            # 规划的取数和写作不需要深度思考，关掉更快更省
            "extra_body": {"enable_thinking": False},
        }
        if tools:
            request["tools"] = tools
        with self.client.chat.completions.stream(**request) as stream:
            printed = False
            for event in stream:
                if event.type == "content.delta":
                    if self._on_delta is not None:
                        self._on_delta(event.delta)
                    else:
                        print(event.delta, end="", flush=True)
                    printed = True
            if printed and self._on_delta is None:
                print()  # 终端流式输出结束后换行
            return stream.get_final_completion().choices[0].message

    def _execute_tool(self, tool_call) -> dict:
        """执行模型请求的工具调用（高德接口 / 天气），返回结果 dict。"""
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            return {"success": False, "error": "工具调用参数不是合法的 JSON"}
        name = tool_call.function.name
        if name == WEATHER_TOOL_NAME:
            return get_weather(
                city=arguments.get("city", ""),
                forecast=bool(arguments.get("forecast", False)),
            )
        return run_tool(name, arguments)

    # ---------- 结果保存 ----------

    @staticmethod
    def _clean(itinerary: str) -> str:
        """去掉模型可能加的 markdown 代码围栏。"""
        text = itinerary.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].rstrip()
        return text + "\n" if text else text

    def _save_itinerary(self, itinerary: str, preference: dict) -> Path:
        """行程写入 itinerary_<城市>.md，并登记到会话数据。"""
        city = (preference.get("city") or "行程").strip()
        path = _ROOT / f"itinerary_{city}.md"
        path.write_text(itinerary, encoding="utf-8")
        self.memory.save_data(itinerary_file=str(path))
        self._emit("saved", {"file": str(path)})
        return path


if __name__ == "__main__":
    # 直接运行：读取项目根目录的 api_plan.json 执行规划
    if not API_PLAN_PATH.exists():
        print(f"找不到 {API_PLAN_PATH}，请先运行 run.py 或 Agents/api_selector.py。")
        sys.exit(1)
    api_plan = json.loads(API_PLAN_PATH.read_text(encoding="utf-8"))
    PlannerAgent().plan(api_plan)
