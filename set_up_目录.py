import os

# 定义需要创建的目录结构（相对路径）
directories = [
    "backend/app/api/v1",
    "backend/app/core",
    "backend/app/services",
    "backend/app/models",          # 稍后用于数据模型
    "backend/app/agents",          # 稍后用于多智能体
    "backend/app/utils",
]

# 定义需要在每个 Python 包中创建的 __init__.py 文件（递归创建）
init_dirs = [
    "backend/app",
    "backend/app/api",
    "backend/app/api/v1",
    "backend/app/core",
    "backend/app/services",
    "backend/app/models",
    "backend/app/agents",
    "backend/app/utils",
]

def create_structure():
    # 创建目录
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")

    # 创建 __init__.py 文件
    for dir_path in init_dirs:
        init_file = os.path.join(dir_path, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w", encoding="utf-8") as f:
                f.write("# 包初始化文件\n")
            print(f"✅ 创建文件: {init_file}")
        else:
            print(f"⏭️ 文件已存在: {init_file}")

    # 额外创建 main.py（如果不存在）
    main_file = "backend/app/main.py"
    if not os.path.exists(main_file):
        with open(main_file, "w", encoding="utf-8") as f:
            f.write('''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.v1 import endpoints
from .core.config import settings

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
''')
        print(f"✅ 创建文件: {main_file}")
    else:
        print(f"⏭️ 文件已存在: {main_file}")

    # 额外创建 config.py（如果不存在）
    config_file = "backend/app/core/config.py"
    if not os.path.exists(config_file):
        with open(config_file, "w", encoding="utf-8") as f:
            f.write('''import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "deepseek-chat")
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()
''')
        print(f"✅ 创建文件: {config_file}")
    else:
        print(f"⏭️ 文件已存在: {config_file}")

    # 额外创建 llm_service.py（如果不存在）
    service_file = "backend/app/services/llm_service.py"
    if not os.path.exists(service_file):
        with open(service_file, "w", encoding="utf-8") as f:
            f.write('''from openai import OpenAI
from ..core.config import settings

class LLMService:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        self.model = settings.MODEL_NAME

    def chat(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

llm_service = LLMService()
''')
        print(f"✅ 创建文件: {service_file}")
    else:
        print(f"⏭️ 文件已存在: {service_file}")

    # 额外创建 endpoints.py（如果不存在）
    endpoints_file = "backend/app/api/v1/endpoints.py"
    if not os.path.exists(endpoints_file):
        with open(endpoints_file, "w", encoding="utf-8") as f:
            f.write('''from fastapi import APIRouter, HTTPException
from ....services.llm_service import llm_service

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
''')
        print(f"✅ 创建文件: {endpoints_file}")
    else:
        print(f"⏭️ 文件已存在: {endpoints_file}")

    print("\n🎉 项目目录结构创建完成！")

if __name__ == "__main__":
    create_structure()