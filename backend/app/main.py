from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import endpoints
from app.core.config import settings

app = FastAPI(
    title="多智能体旅游规划系统",
    version="0.1.0",
    description="基于 Multi-Agent 协作的智能旅游规划系统"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(endpoints.router, prefix="/api/v1", tags=["v1"])

@app.get("/")
def root():
    return {"message": "🚀 多智能体旅游规划系统已启动"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )