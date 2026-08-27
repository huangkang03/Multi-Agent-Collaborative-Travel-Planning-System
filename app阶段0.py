from fastapi import FastAPI, HTTPException
import uvicorn
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# ---------- 你的初始化部分（完全不变） ----------
load_dotenv(verbose=True)
model = init_chat_model(
    model='deepseek-v4-flash',
    model_provider='deepseek',
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY")
)
# ----------------------------------------------

app = FastAPI()

@app.get("/")
def hello():
    return {"msg": "旅游规划师启动成功"}

@app.get("/test-llm")
def test_llm():
    try:
        # LangChain 调用方式（与 OpenAI SDK 不同）
        response = model.invoke([HumanMessage(content="请推荐3个厦门必去的景点，只回复景点名称，用逗号隔开。")])
        return {"result": response.content}
    except Exception as e:
        # 把详细错误返回给浏览器，方便你定位问题
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)