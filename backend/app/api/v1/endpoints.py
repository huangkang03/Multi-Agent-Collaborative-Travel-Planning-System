from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_service import llm_service
from app.agents.attraction_agent import recommend_attractions
from app.graph.workflow import build_trip_graph

router = APIRouter()

class RecommendRequest(BaseModel):
    destination: str
    preferences: str = "不限"

@router.get("/test-llm")
def test_llm():
    try:
        result = llm_service.chat("请推荐3个厦门必去的景点，只回复景点名称，用逗号隔开。")
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommend-attractions")
def recommend(data: RecommendRequest):
    try:
        result = recommend_attractions(data.destination, data.preferences)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
def hello():
    return {"msg": "旅游规划师启动成功"}


class PlanTripRequest(BaseModel):
    destination: str
    preferences: str = "不限"
    include_hotel: bool = True  # 新增：默认包含酒店


@router.post("/plan-trip")
def plan_trip(data: PlanTripRequest):
    try:
        graph = build_trip_graph()
        initial_state = {
            "destination": data.destination,
            "preferences": data.preferences,
            "include_hotel": data.include_hotel,  # 传入开关
            "attractions": None,
            "hotels": None,
            "final_plan": None
        }
        final_state = graph.invoke(initial_state)
        return {
            "destination": data.destination,
            "preferences": data.preferences,
            "attractions": final_state.get("attractions"),
            "hotels": final_state.get("hotels"),
            "final_plan": final_state.get("final_plan")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))