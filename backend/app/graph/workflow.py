from langgraph.graph import StateGraph, END
from app.graph.state import TripState
from app.agents.attraction_agent import recommend_attractions
from app.agents.hotel_agent import recommend_hotels
from app.services.llm_service import llm_service

# ----- 定义各个节点（Agent 执行单元）-----
def attraction_node(state: TripState) -> dict:
    """景点推荐节点"""
    result = recommend_attractions(state["destination"], state["preferences"])
    return {"attractions": result}

def hotel_node(state: TripState) -> dict:
    """酒店推荐节点"""
    result = recommend_hotels(state["destination"], state["preferences"])
    return {"hotels": result}

def compiler_node(state: TripState) -> dict:
    """编译汇总节点：把景点和酒店拼成自然语言行程"""
    attractions_text = ""
    hotels_text = ""
    
    if state.get("attractions") and "recommendations" in state["attractions"]:
        items = state["attractions"]["recommendations"]
        attractions_text = "\n".join([f"- {item['name']}（理由：{item['reason']}）" for item in items])
    
    if state.get("hotels") and "hotels" in state["hotels"]:
        items = state["hotels"]["hotels"]
        hotels_text = "\n".join([f"- {item['name']}（理由：{item['reason']}）" for item in items])
    
    # 让大模型润色成自然语言行程
    prompt = f"""
用户将去「{state['destination']}」旅游，偏好「{state['preferences']}」。
请根据以下推荐信息，生成一段简短的行程安排（100字左右），包含酒店和景点建议：

【推荐景点】
{attractions_text if attractions_text else "暂无"}

【推荐酒店】
{hotels_text if hotels_text else "暂无"}

请用自然、友好的语气输出。
"""
    final_text = llm_service.chat(prompt)
    return {"final_plan": final_text}

# ----- 构建图 -----
def build_trip_graph():
    workflow = StateGraph(TripState)
    
    # 添加节点
    workflow.add_node("attractions", attraction_node)
    workflow.add_node("hotels", hotel_node)
    workflow.add_node("compiler", compiler_node)
    
    # 设置入口
    workflow.set_entry_point("attractions")
    
    # 顺序连接（协作流水线）
    workflow.add_edge("attractions", "hotels")
    workflow.add_edge("hotels", "compiler")
    workflow.add_edge("compiler", END)
    
    return workflow.compile()