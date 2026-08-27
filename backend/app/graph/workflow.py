from langgraph.graph import StateGraph, END
from app.graph.state import TripState
from app.agents.attraction_agent import recommend_attractions
from app.agents.hotel_agent import recommend_hotels
from app.services.llm_service import llm_service

# ----- 节点定义 -----
def attraction_node(state: TripState) -> dict:
    result = recommend_attractions(state["destination"], state["preferences"])
    return {"attractions": result}

def hotel_node(state: TripState) -> dict:
    result = recommend_hotels(state["destination"], state["preferences"])
    return {"hotels": result}

def compiler_node(state: TripState) -> dict:
    attractions_text = ""
    hotels_text = ""

    if state.get("attractions") and "recommendations" in state["attractions"]:
        items = state["attractions"]["recommendations"]
        attractions_text = "\n".join([f"- {item['name']}（{item['reason']}）" for item in items])

    if state.get("hotels") and "hotels" in state["hotels"]:
        items = state["hotels"]["hotels"]
        hotels_text = "\n".join([f"- {item['name']}（{item['reason']}）" for item in items])

    prompt = f"""
用户将去「{state['destination']}」旅游，偏好「{state['preferences']}」。
请根据以下信息生成一段简短、友好的行程安排（100字左右）：

【推荐景点】
{attractions_text if attractions_text else "暂无"}

【推荐酒店】
{hotels_text if hotels_text else "用户未要求推荐酒店"}

请用自然、亲切的语气输出。
"""
    final_text = llm_service.chat(prompt)
    return {"final_plan": final_text}

# ----- 路由决策函数（关键新增）-----
def route_after_attractions(state: TripState) -> str:
    """
    条件边逻辑：
    - 如果用户要求包含酒店（include_hotel=True），则去酒店节点
    - 否则，直接跳到编译器（跳过酒店）
    """
    if state.get("include_hotel", True):  # 默认为 True
        return "hotels"
    else:
        return "compiler"

# ----- 构建图 -----
def build_trip_graph():
    workflow = StateGraph(TripState)

    # 添加节点
    workflow.add_node("attractions", attraction_node)
    workflow.add_node("hotels", hotel_node)
    workflow.add_node("compiler", compiler_node)

    # 设置入口
    workflow.set_entry_point("attractions")

    # 顺序边：景点 -> （条件路由） -> 酒店 或 编译器
    workflow.add_conditional_edges(
        "attractions",           # 从景点节点出发
        route_after_attractions, # 调用路由函数
        {
            "hotels": "hotels",  # 如果返回 "hotels"，去酒店节点
            "compiler": "compiler"  # 如果返回 "compiler"，去编译器
        }
    )

    # 固定边：酒店 -> 编译器，编译器 -> END
    workflow.add_edge("hotels", "compiler")
    workflow.add_edge("compiler", END)

    return workflow.compile()