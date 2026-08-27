import json
from app.tools.amap_tools import search_pois
from app.services.llm_service import llm_service

def recommend_attractions(destination: str, preferences: str = "不限") -> dict:
    """
    景点推荐 Agent 核心逻辑：
    1. 调用高德 API 获取真实景点数据
    2. 让大模型基于用户偏好进行筛选和排序
    3. 返回带推荐理由的结果
    """
    # 步骤1：获取真实数据
    raw_pois = search_pois("景点", destination)
    if not raw_pois:
        return {"error": f"未能在 {destination} 找到景点数据，请检查城市名称或网络。"}
    
    # 只取前10个，避免数据量过大
    top_pois = raw_pois[:10]
    
    # 步骤2：构建 Prompt 让大模型思考
    prompt = f"""
你是一个专业的旅游规划师。用户想去「{destination}」旅游，偏好是「{preferences}」。

以下是高德地图提供的该城市真实景点列表（含评分）：
{json.dumps(top_pois, ensure_ascii=False, indent=2)}

请你根据用户的偏好，从上述列表中**精选 3 个最合适的景点**。
请按推荐优先级排序，并为每个景点写一句简洁的推荐理由（20字以内）。

**必须严格按以下 JSON 格式返回（不要包含其他文字）：**
[
  {{"name": "景点名", "reason": "推荐理由"}},
  {{"name": "景点名", "reason": "推荐理由"}},
  {{"name": "景点名", "reason": "推荐理由"}}
]
"""
    
    # 步骤3：调用大模型
    try:
        result_text = llm_service.chat(prompt)
        # 尝试解析 JSON（如果模型返回了markdown标记，尝试清洗）
        cleaned = result_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:-3]  # 去掉 ```json 和 ```
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:-3]
        recommendations = json.loads(cleaned)
        return {
            "destination": destination,
            "preferences": preferences,
            "recommendations": recommendations
        }
    except Exception as e:
        return {"error": f"大模型解析失败: {e}", "raw_response": result_text}