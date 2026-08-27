from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_service import llm_service
from app.agents.attraction_agent import recommend_attractions

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