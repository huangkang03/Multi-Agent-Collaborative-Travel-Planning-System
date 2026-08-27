from fastapi import APIRouter, HTTPException
from app.services.llm_service import llm_service

router = APIRouter()

@router.get("/")
def hello():
    return {"msg": "旅游规划师启动成功"}

@router.get("/test-llm")
def test_llm():
    try:
        result = llm_service.chat("请推荐3个厦门必去的景点，只回复景点名称，用逗号隔开。")
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))