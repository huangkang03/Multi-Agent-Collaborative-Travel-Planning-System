import os
import requests
from app.core.config import settings

def search_pois(keyword: str, city: str, types: str = "") -> list:
    """
    调用高德 POI 搜索接口，获取景点列表
    types: 110000 代表风景名胜（含公园、古镇等）
    """
    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "key": settings.AMAP_API_KEY,
        "keywords": keyword,
        "city": city,
        "types": types,
        "output": "json",
        "page_size": 20,
        "extensions": "all"
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
                    "location": p.get("location"),
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