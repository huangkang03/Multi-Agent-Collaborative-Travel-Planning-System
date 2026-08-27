import os
import requests
from app.core.config import settings

def search_pois(keyword: str, city: str, types: str = "110000") -> list:
    """
    调用高德 POI 搜索接口，获取景点列表
    types: 110000 代表风景名胜（含公园、古镇等），也可用 050000 餐饮
    """
    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "key": settings.AMAP_API_KEY,  # 从环境变量读取
        "keywords": keyword,
        "city": city,
        "types": types,
        "output": "json",
        "page_size": 20,  # 取前20条
        "extensions": "all"  # 获取详细信息（含评分、图片等）
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if data.get("status") == "1":
            pois = data.get("pois", [])
            results = []
            for p in pois:
                results.append({
                    "name": p.get("name"),
                    "address": p.get("address"),
                    "location": p.get("location"),  # 经度,纬度
                    "tel": p.get("tel"),
                    "rating": p.get("biz_ext", {}).get("rating", "暂无评分"),
                })
            return results
        else:
            print(f"高德API错误: {data.get('info')}")
            return []
    except Exception as e:
        print(f"请求高德API异常: {e}")
        return []