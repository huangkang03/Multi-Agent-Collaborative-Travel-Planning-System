"""
多智能体旅行规划系统 -- 总入口（编排层）。

串联三个 agent：
1. 需求分析师（Agents/analyst.py，qwen3.8-flash）
   与用户多轮对话收集结构化旅游需求（城市/天数/出行方式/氛围偏好），
   对话数据由短期记忆管理（sessions/analyst_*.json，可跨进程恢复）；
2. API 选择器（Agents/api_selector.py，kimi-k3 深度思考）
   读取 mcp_config.json 连接高德地图 API 文档服务器（MCP），
   从 37 个接口中决定本次行程需要调用的 API 及调用顺序，
   结果连同参数明细落盘 api_plan.json；
3. 行程规划师（Agents/planner.py，qwen3.8-flash）
   按计划真实调用高德 Web 服务接口（坐标/POI/路线/天气），
   产出逐日行程，保存 itinerary_<城市>.md。

用法：
    python run.py            # 新会话，从需求收集开始走完整流程
    python run.py --resume   # 恢复最近一次分析师会话继续（可改需求）
                             # 若恢复的需求已完整，可选择直接复用进入下一环
"""

import json
import sys
from pathlib import Path
from typing import Optional

# run.py 在项目根目录，直接导入同级的包
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows 控制台默认 GBK 编码，输出含中文/emoji 时会崩溃，改为 UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from Agents.analyst import AnalystAgent  # noqa: E402  # 需要在 sys.path 调整之后再导入
from Agents.api_selector import ApiSelectorAgent  # noqa: E402
from Agents.planner import PlannerAgent  # noqa: E402


def collect_preference(resume: bool) -> Optional[dict]:
    """第一环：需求分析师收集结构化需求。未收集完整时返回 None。"""
    analyst = AnalystAgent(resume=resume)

    # --resume 且上次已收集完整：让用户决定复用还是继续修改
    if resume and analyst.is_complete:
        print("=" * 20 + " 已恢复完整需求 " + "=" * 20)
        print(json.dumps(analyst.preference, ensure_ascii=False, indent=2))
        choice = input("直接用这份需求进入 API 选择？(y=使用 / 回车=继续对话修改)：").strip().lower()
        if choice in ("y", "yes", "是"):
            return analyst.preference

    analyst.run()  # 交互循环：收集齐全自动结束，用户退出则中断

    if not analyst.is_complete:
        print("\n需求未收集完整，本次不进入 API 选择。")
        print("对话已保存，下次运行 python run.py --resume 可接着聊。")
        return None
    return analyst.preference


def select_apis(preference: dict) -> Optional[dict]:
    """第二环：kimi-k3 读 MCP 上的 API 文档，决定要调用的接口。"""
    try:
        selector = ApiSelectorAgent()
        return selector.run(preference)  # 内部已打印选中的 API 并保存 api_plan.json
    except KeyboardInterrupt:
        print("\nAPI 选择被中断。")
        return None


def make_itinerary(plan: dict) -> Optional[Path]:
    """第三环：行程规划师按计划调用高德真实接口，产出逐日行程。"""
    try:
        planner = PlannerAgent()
        itinerary = planner.plan(plan)
        return planner.itinerary_file if itinerary else None
    except KeyboardInterrupt:
        print("\n行程规划被中断。")
        return None


def main() -> None:
    resume = "--resume" in sys.argv
    print("#" * 20 + " 多智能体旅行规划系统 " + "#" * 20)
    print("环节：需求分析师(qwen) -> API 选择器(kimi-k3) -> 行程规划师(qwen)")
    print()

    preference = collect_preference(resume)
    if preference is None:
        return

    print()
    plan = select_apis(preference)
    if plan is None:
        return

    print()
    itinerary_file = make_itinerary(plan)
    if itinerary_file is None:
        return

    print()
    print("#" * 20 + " 全流程完成 " + "#" * 20)
    print(f"需求     ：{preference['city']} {preference['days']} 天，"
          f"市内{preference['travel_mode']}，偏好{preference['atmosphere']}")
    print(f"选中 API ：{len(plan['apis'])} 个（api_plan.json）")
    print(f"行程文件 ：{itinerary_file}")
    print(f"行程质量 ：地点/地址/评分/距离/耗时均来自高德真实接口返回")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。会话数据已保存，可随时 python run.py --resume 继续。")
