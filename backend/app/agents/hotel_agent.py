import json
from app.tools.amap_tools import search_pois
from app.services.llm_service import llm_service

def recommend_hotels(destination: str, preferences: str = "不限") -> dict:
    """
    酒店推荐 Agent：查高德酒店数据，让大模型筛选推荐
    """
    # 1. 调用高德搜索酒店（类型 060000 为酒店宾馆）
    raw_pois = search_pois("酒店", destination)
    if not raw_pois:
        return {"error": f"未能在 {destination} 找到酒店数据。"}
    
    top_pois = raw_pois[:10]
    
    # 2. 构建 Prompt
    prompt = f"""
你是一个旅游住宿专家。用户要去「{destination}」，偏好是「{preferences}」。

以下是高德地图提供的该城市真实酒店列表（含评分）：
{json.dumps(top_pois, ensure_ascii=False, indent=2)}

请根据用户偏好，从列表中**精选 2 家最合适的酒店**，按推荐度排序，并各写一句推荐理由（15字以内）。

**必须严格按 JSON 格式返回：**
[
  {{"name": "酒店名", "reason": "推荐理由"}},
  {{"name": "酒店名", "reason": "推荐理由"}}
]
"""
    # 3. 调用大模型
    try:
        result_text = llm_service.chat(prompt)
        cleaned = result_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:-3]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:-3]
        recommendations = json.loads(cleaned)
        return {
            "destination": destination,
            "preferences": preferences,
            "hotels": recommendations
        }
    except Exception as e:
        return {"error": f"大模型解析失败: {e}"}