"""
API 选择 agent（Api Selector Agent）-- 多智能体旅行规划系统的第二环。

职责：接收分析师（analyst）产出的结构化旅游需求（城市、天数、出行方式、
氛围偏好），读取 mcp_config.json 配置的高德地图 API 文档服务器
（apifox-mcp-server，MCP 协议），获取全部可用 API 清单，
再由 kimi-k3 深度思考模型阅读清单后决定本次行程规划需要调用哪些 API，
并把这些 API 的参数文档（OpenAPI Spec 的 $ref 明细）一并取回，
供下游执行 agent 实际调用高德接口时使用。

流程：
1. 读取 mcp_config.json，按配置通过 stdio 启动 MCP 服务器；
2. list_tools 动态发现工具（注意：apifox 服务器的工具名每次会话
   都带不同的随机后缀，不能写死名字）；
3. 调用 read_project_oas_* 读取 OpenAPI Spec 主文件，得到 API 目录
   （每个接口的路径 + 中文名称）；
4. kimi-k3 思考模型阅读 API 目录 + 旅游需求（开启深度思考），
   强制 function calling 调用 select_amap_apis，
   输出选中的 API、各自用途和调用顺序；
5. 对选中的 API 批量拉取 $ref 参数明细，连同选择结果保存到 api_plan.json。
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# Windows 控制台默认 GBK 编码，输出含中文/emoji 时会崩溃，改为 UTF-8
sys.stdout.reconfigure(encoding="utf-8")

# Agents/ 下的脚本直接运行时，把项目根目录加进 sys.path 才能导入 llm/tools 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import KIMI_MODEL, create_kimi_client  # noqa: E402  # 需要在 sys.path 调整之后再导入

# 项目根目录下的固定路径
_ROOT = Path(__file__).resolve().parent.parent
MCP_CONFIG_PATH = _ROOT / "mcp_config.json"
API_PLAN_PATH = _ROOT / "api_plan.json"


_SELECT_SYSTEM_PROMPT = (
    "你是高德地图 API 选择器，在多智能体旅行规划系统中负责决定"
    "行程规划需要调用哪些高德地图 API。\n"
    "你会收到：1) 用户的结构化旅游需求（城市、天数、市内出行方式、氛围偏好）；"
    "2) 高德地图可用 API 清单（HTTP 方法、路径、中文名称）。\n"
    "选择原则：\n"
    "- 只选行程规划真正需要的 API，宁缺毋滥；功能相同的新旧版本"
    "（如 v3/v5）只选一个，优先新版本。\n"
    "- 按调用顺序排列：一般先用地理编码把城市名转成坐标，"
    "再用关键字/周边搜索找景点和美食，按出行方式选对应的路径规划 API，"
    "天气查询用于行程建议。\n"
    "- 市内出行方式对应：漫步->步行路径规划，驾车->驾车路径规划，"
    "打车->驾车路径规划（预估车距车时）。\n"
    "- 氛围偏好影响搜索关键字（热闹选热门商圈地标，清静选小众公园绿地），"
    "不改变 API 选择本身。\n"
    "- 每个 API 用一句话说明 purpose（在本行程规划中的用途）。"
)

# 选择结果的结构化工具定义：强制模型以该格式返回选中的 API
SELECT_TOOL = {
    "type": "function",
    "function": {
        "name": "select_amap_apis",
        "description": "从高德地图 API 清单中，选择规划本次行程需要调用的 API",
        "parameters": {
            "type": "object",
            "properties": {
                "apis": {
                    "type": "array",
                    "description": "选中的 API，按建议的调用顺序排列",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": (
                                    "API 的路径，必须来自清单中的 path；"
                                    "只要路径本身，不要带 HTTP 方法"
                                ),
                            },
                            "purpose": {
                                "type": "string",
                                "description": "该 API 在本次行程规划中的用途，一句话",
                            },
                        },
                        "required": ["path", "purpose"],
                    },
                }
            },
            "required": ["apis"],
        },
    },
}


class ApiSelectorAgent:
    """读取 MCP 上的高德地图 API 文档，由模型决定要调用哪些 API。"""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        mcp_config_path: Path = MCP_CONFIG_PATH,
    ):
        # 独立实例化的 kimi-k3 客户端（深度思考决策），与 analyst 互不共享
        self.client = client or create_kimi_client()
        self.mcp_config_path = Path(mcp_config_path)

    # ---------- 1. MCP 侧：读配置、连服务器、读 API 目录 ----------

    def _load_mcp_server(self) -> StdioServerParameters:
        """从 mcp_config.json 读取服务器启动参数。"""
        config = json.loads(self.mcp_config_path.read_text(encoding="utf-8"))
        servers = config.get("mcpServers", {})
        if not servers:
            raise RuntimeError(f"{self.mcp_config_path} 里没有 mcpServers 配置")
        # 优先选名字带「高德」或 amap 的服务器，否则用第一个
        name = next(
            (n for n in servers if "高德" in n or "amap" in n.lower()),
            next(iter(servers)),
        )
        cfg = servers[name]
        return StdioServerParameters(
            command=cfg["command"], args=cfg.get("args", [])
        )

    @staticmethod
    async def _find_tools(session: ClientSession) -> tuple:
        """发现 apifox MCP 服务器上的两个工具名（名字每次会话都变）。

        返回 (读 Spec 主文件的工具, 读 $ref 明细的工具)。
        """
        tools = await session.list_tools()
        names = [t.name for t in tools.tools]
        spec_tool = next(
            (n for n in names if n.startswith("read_project_oas") and "ref" not in n),
            None,
        )
        ref_tool = next((n for n in names if "ref" in n), None)
        if spec_tool is None or ref_tool is None:
            raise RuntimeError("MCP 服务器上找不到读取 OpenAPI Spec 的工具")
        return spec_tool, ref_tool

    @staticmethod
    def _parse_catalog(spec: dict) -> list:
        """把 OpenAPI Spec 主文件解析成 API 目录列表。"""
        catalog = []
        for path, item in spec.get("paths", {}).items():
            ref = item.get("$ref")
            for method, op in item.items():
                if method == "$ref":
                    continue
                catalog.append(
                    {
                        "method": method,
                        "path": path,
                        "summary": op.get("summary", ""),
                        "ref": ref,
                    }
                )
        return catalog

    # ---------- 2. 模型侧：阅读目录并决定调用哪些 API ----------

    def _select_apis(self, catalog: list, preference: dict) -> dict:
        """让模型阅读 API 目录和旅游需求，返回选中的 API 列表。"""
        catalog_text = "\n".join(
            f"- {item['method'].upper()} {item['path']}：{item['summary']}"
            for item in catalog
        )
        messages = [
            {"role": "system", "content": _SELECT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "结构化旅游需求：\n"
                    + json.dumps(preference, ensure_ascii=False, indent=2)
                    + "\n\n可用 API 清单：\n"
                    + catalog_text
                ),
            },
        ]
        try:
            completion = self.client.chat.completions.create(
                model=KIMI_MODEL,
                messages=messages,
                tools=[SELECT_TOOL],
                # 强制调用选择工具，保证拿到结构化结果
                tool_choice={
                    "type": "function",
                    "function": {"name": SELECT_TOOL["function"]["name"]},
                },
                # API 选择是核心决策环节，开启 kimi-k3 深度思考
                extra_body={"enable_thinking": True},
            )
        except Exception as e:
            return {"success": False, "error": f"模型调用失败：{e}"}

        tool_calls = getattr(completion.choices[0].message, "tool_calls", None)
        if not tool_calls:
            return {"success": False, "error": "模型未调用选择工具"}
        try:
            arguments = json.loads(tool_calls[0].function.arguments)
        except json.JSONDecodeError:
            return {"success": False, "error": "选择结果不是合法的 JSON"}
        chosen = arguments.get("apis", [])

        # 校验：模型给出的 path 必须真实存在于目录，顺便补全 method/summary/ref
        by_path = {item["path"]: item for item in catalog}
        apis, invalid = [], []
        for entry in chosen:
            raw_path = (entry.get("path") or "").strip()
            # 模型偶尔会把 HTTP 方法也写进 path（如 "GET /v3/geocode/geo"），先剥掉
            if raw_path.upper().startswith(("GET ", "POST ", "PUT ", "DELETE ")):
                raw_path = raw_path.split(" ", 1)[1].strip()
            base = by_path.get(raw_path)
            if base is None:
                invalid.append(entry.get("path"))
                continue
            apis.append(
                {
                    "method": base["method"],
                    "path": base["path"],
                    "summary": base["summary"],
                    "ref": base["ref"],
                    "purpose": entry.get("purpose", ""),
                }
            )
        if invalid:
            print(f"忽略清单中不存在的 API：{invalid}")
        if not apis:
            return {"success": False, "error": "模型没有选出任何有效 API"}
        return {"success": True, "apis": apis}

    # ---------- 3. 整体编排 ----------

    async def analyze_async(self, preference: dict, on_event=None) -> dict:
        """完整流程（异步版）：连 MCP -> 读目录 -> 模型决策 -> 拉参数明细。

        on_event(stage, detail) 存在时把阶段进度交给回调（Web 端推送），
        否则打印到终端（CLI 行为）。
        """

        def report(stage: str, detail: str) -> None:
            if on_event is not None:
                on_event(stage, detail)
            else:
                print(detail)

        server = self._load_mcp_server()
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                spec_tool, ref_tool = await self._find_tools(session)

                result = await session.call_tool(spec_tool, {})
                spec = json.loads(result.content[0].text)
                catalog = self._parse_catalog(spec)
                report(
                    "catalog", f"已从 MCP 服务器读取 API 目录：{len(catalog)} 个接口"
                )

                selection = self._select_apis(catalog, preference)
                if not selection.get("success"):
                    return selection

                # 批量拉取选中 API 的参数明细（$ref 引用文件）
                details = {}
                refs = [api["ref"] for api in selection["apis"] if api.get("ref")]
                if refs:
                    ref_result = await session.call_tool(ref_tool, {"path": refs})
                    raw = json.loads(ref_result.content[0].text)
                    for ref, detail_text in raw.items():
                        try:
                            details[ref] = json.loads(detail_text)
                        except (json.JSONDecodeError, TypeError):
                            details[ref] = detail_text

                return {
                    "success": True,
                    "preference": preference,
                    "catalog_size": len(catalog),
                    "apis": selection["apis"],
                    "details": details,
                }

    def analyze(self, preference: dict, on_event=None) -> dict:
        """同步入口：完整执行「发现目录 -> 模型决策 -> 拉取明细」。"""
        try:
            return asyncio.run(self.analyze_async(preference, on_event=on_event))
        except (RuntimeError, OSError) as e:
            # MCP 服务器启动/连接/协议层错误
            return {"success": False, "error": f"MCP 服务器连接失败：{e}"}

    def run(self, preference: dict, save_path: Path = API_PLAN_PATH) -> Optional[dict]:
        """命令行入口：执行完整流程，打印选中的 API，结果存 api_plan.json。"""
        print("=" * 20 + " API 选择器 " + "=" * 20)
        print(f"旅游需求：{json.dumps(preference, ensure_ascii=False)}")
        plan = self.analyze(preference)
        if not plan.get("success"):
            print("失败：" + str(plan.get("error", "未知错误")))
            return None
        print("=" * 20 + f" 选中的 API（{len(plan['apis'])} 个，按调用顺序） " + "=" * 20)
        for i, api in enumerate(plan["apis"], 1):
            print(f"{i}. {api['method'].upper()} {api['path']}  {api['summary']}")
            print(f"   用途：{api['purpose']}")
        save_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"参数明细已保存：{save_path}（供下游执行 agent 使用）")
        return plan


if __name__ == "__main__":
    # 用法：python api_selector.py [城市] [天数] [出行方式] [氛围]
    # 不带参数时用示例需求；实际使用时传入 analyst 产出的 preference dict
    argv = sys.argv[1:]
    if len(argv) == 4:
        demo = {
            "success": True,
            "city": argv[0],
            "days": int(argv[1]),
            "travel_mode": argv[2],
            "atmosphere": argv[3],
            "missing_fields": [],
        }
    else:
        demo = {
            "success": True,
            "city": "杭州",
            "days": 3,
            "travel_mode": "漫步",
            "atmosphere": "清静",
            "missing_fields": [],
        }
    ApiSelectorAgent().run(preference=demo)
