"""
planner agent 使用的 tool：封装高德地图 Web 服务 API 的真实 HTTP 调用。

覆盖行程规划所需的核心接口：
- amap_geocode        地理编码（v3）：地址/城市名 -> 坐标 + adcode
- amap_place_text     关键字搜索（v5）：按关键词搜 POI（含评分/营业时间）
- amap_place_around   周边搜索（v5）：以坐标为中心搜周边（含距离）
- amap_place_detail   ID 查询（v5）：查单个 POI 详情
- amap_route_walking  步行路线（v5）：距离 + 耗时
- amap_route_driving  驾车路线（v5）：距离 + 耗时（打车/自驾参考）

天气查询 get_weather 在 tools/analyse.py，不在本文件。

工具结果做了压缩（POI 只留 id/名称/地址/坐标/类型/评分/营业时间，
路线只留距离/耗时），控制回传给模型的上下文体积。
纯工具逻辑，不实例化模型。PATH_TOOL_MAP 把 api_plan.json 里的
OpenAPI path 映射到工具名，planner 据此只暴露计划选中的接口。
"""

import os
from typing import Optional

import httpx

AMAP_API_KEY_ENV = "AMAP_API_KEY"

# 各接口地址
GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
PLACE_TEXT_URL = "https://restapi.amap.com/v5/place/text"
PLACE_AROUND_URL = "https://restapi.amap.com/v5/place/around"
PLACE_DETAIL_URL = "https://restapi.amap.com/v5/place/detail"
WALKING_URL = "https://restapi.amap.com/v5/direction/walking"
DRIVING_URL = "https://restapi.amap.com/v5/direction/driving"


def _amap_get(url: str, params: dict) -> dict:
    """
    请求高德 Web 服务接口并校验 envelope，返回解析后的 JSON dict。

    与 tools/analyse.py 里的同名助手同款（各自持有一份，保持模块独立）。
    出错时返回 {"success": False, "error": ...}，与其它工具的错误格式一致。
    """
    api_key = os.getenv(AMAP_API_KEY_ENV)
    if not api_key:
        return {"success": False, "error": "未配置 AMAP_API_KEY 环境变量"}
    try:
        response = httpx.get(
            url, params={"key": api_key, "output": "JSON", **params}, timeout=10
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        return {"success": False, "error": f"请求高德接口失败：{e}"}
    data = response.json()
    # v3 的 status/infocode 是字符串，v5 可能是数字，统一转字符串比较
    if str(data.get("status")) != "1" or str(data.get("infocode")) != "10000":
        return {
            "success": False,
            "error": (
                f"高德接口返回错误：{data.get('info')}"
                f"（infocode={data.get('infocode')}）"
            ),
        }
    return data


# ---------- POI 搜索类工具 ----------


def _compress_poi(poi: dict) -> dict:
    """压缩 POI：只保留行程规划需要的字段。"""
    business = poi.get("business") or {}
    distance = poi.get("distance")
    return {
        "id": poi.get("id"),
        "name": poi.get("name"),
        "address": poi.get("address"),
        "location": poi.get("location"),
        "type": poi.get("type"),
        "rating": business.get("rating"),
        "opentime": business.get("opentime_today") or business.get("opentime_week"),
        # 周边搜索按中心点返回距离（米）；关键字搜索没有该字段
        "distance": int(distance) if str(distance).strip().isdigit() else None,
    }


def _search_pois(url: str, params: dict, keywords: str) -> dict:
    """v5 POI 搜索接口的公共封装。"""
    data = _amap_get(url, params)
    if "success" in data:
        return data
    pois = data.get("pois") or []
    result = {
        "success": True,
        "count": len(pois),
        "pois": [_compress_poi(p) for p in pois],
    }
    if keywords:
        result["keywords"] = keywords
    return result


def amap_geocode(address: str) -> dict:
    """地理编码：地址/城市名 -> 经纬度坐标（「lng,lat」）与行政区划信息。"""
    if not isinstance(address, str) or not address.strip():
        return {"success": False, "error": "address 参数不能为空"}
    address = address.strip()
    data = _amap_get(GEOCODE_URL, {"address": address})
    if "success" in data:
        return data
    geocodes = data.get("geocodes") or []
    if not geocodes:
        return {"success": False, "error": f"地理编码找不到「{address}」"}
    g = geocodes[0]
    return {
        "success": True,
        "address": address,
        "location": g.get("location"),
        "adcode": g.get("adcode"),
        "province": g.get("province"),
        "city": g.get("city"),
        "level": g.get("level"),
    }


def amap_place_text(
    keywords: str, region: str, page_size: int = 5, types: Optional[str] = None
) -> dict:
    """关键字搜索（v5）：在 region 城市内按关键词搜 POI。"""
    if not isinstance(keywords, str) or not keywords.strip():
        return {"success": False, "error": "keywords 参数不能为空"}
    keywords = keywords.strip()
    params = {
        "keywords": keywords,
        "page_size": max(1, min(int(page_size or 5), 10)),
        "show_fields": "business",
    }
    if region:
        params["region"] = region
        params["city_limit"] = "true"
    if types:
        params["types"] = types
    return _search_pois(PLACE_TEXT_URL, params, keywords)


def amap_place_around(
    center: str,
    keywords: str = "",
    radius: int = 1000,
    region: str = "",
    page_size: int = 5,
) -> dict:
    """周边搜索（v5）：以 center（「lng,lat」）为中心搜 radius 米内的 POI。"""
    if not isinstance(center, str) or "," not in (center or ""):
        return {"success": False, "error": "center 参数必须是「lng,lat」格式的坐标"}
    params = {
        "location": center,
        "radius": max(100, min(int(radius or 1000), 50000)),
        "page_size": max(1, min(int(page_size or 5), 10)),
        "show_fields": "business",
    }
    if keywords:
        params["keywords"] = keywords
    if region:
        params["region"] = region
    return _search_pois(PLACE_AROUND_URL, params, keywords)


def amap_place_detail(poi_id: str) -> dict:
    """ID 查询（v5）：按搜索结果里的 id 查单个 POI 的详细信息。"""
    if not isinstance(poi_id, str) or not poi_id.strip():
        return {"success": False, "error": "id 参数不能为空"}
    data = _amap_get(
        PLACE_DETAIL_URL, {"id": poi_id.strip(), "show_fields": "business"}
    )
    if "success" in data:
        return data
    pois = data.get("pois") or []
    if not pois:
        return {"success": False, "error": f"查不到 id 为 {poi_id} 的地点"}
    return {"success": True, "poi": _compress_poi(pois[0])}


# ---------- 路线规划类工具 ----------


def _format_duration(seconds) -> Optional[str]:
    """秒数 -> 「约N分钟 / 约X小时Y分钟」。"""
    try:
        minutes = round(int(seconds) / 60)
    except (TypeError, ValueError):
        return None
    if minutes < 60:
        return f"约{minutes}分钟"
    hours, rest = divmod(minutes, 60)
    return f"约{hours}小时" + (f"{rest}分钟" if rest else "")


def _route_result(mode: str, data: dict) -> dict:
    """v5 路线接口的公共封装：取第一条路线的距离与耗时。"""
    paths = (data.get("route") or {}).get("paths") or []
    if not paths:
        return {"success": False, "error": "高德未返回路线"}
    path = paths[0]
    cost = path.get("cost") or {}
    distance = int(path.get("distance") or 0)
    result = {
        "success": True,
        "mode": mode,
        "distance_m": distance,
        "distance_km": round(distance / 1000, 1),
        "duration_text": _format_duration(cost.get("duration")),
    }
    if cost.get("traffic_lights") is not None:
        result["traffic_lights"] = int(cost["traffic_lights"])
    return result


def amap_route_walking(origin: str, destination: str) -> dict:
    """步行路线（v5）：两点间步行距离与耗时。origin/destination 为「lng,lat」。"""
    data = _amap_get(WALKING_URL, {"origin": origin, "destination": destination})
    if "success" in data:
        return data
    return _route_result("步行", data)


def amap_route_driving(origin: str, destination: str) -> dict:
    """驾车路线（v5）：两点间车行距离与耗时（自驾/打车预估用）。"""
    data = _amap_get(
        DRIVING_URL,
        {"origin": origin, "destination": destination, "show_fields": "cost"},
    )
    if "success" in data:
        return data
    return _route_result("驾车", data)


# ---------- function calling 的工具定义 ----------

AMAP_GEOCODE_TOOL = {
    "type": "function",
    "function": {
        "name": "amap_geocode",
        "description": (
            "地理编码：把地址或城市名转成经纬度坐标（location，「lng,lat」格式）"
            "与行政区划代码。规划的第一步，先拿城市中心坐标。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "地址或城市名，如：厦门",
                },
            },
            "required": ["address"],
        },
    },
}

AMAP_PLACE_TEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "amap_place_text",
        "description": (
            "关键字搜索 POI（v5）：在指定城市按关键词搜景点/美食/店铺等，"
            "返回名称、地址、坐标、类型、评分、营业时间。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "搜索关键词，如：园林、咖啡馆、本地菜",
                },
                "region": {
                    "type": "string",
                    "description": "限定搜索的城市名，如：厦门",
                },
                "page_size": {
                    "type": "integer",
                    "description": "返回条数，默认 5，最多 10",
                },
                "types": {
                    "type": "string",
                    "description": "POI 类型编码（可选），如 110000 风景名胜",
                },
            },
            "required": ["keywords", "region"],
        },
    },
}

AMAP_PLACE_AROUND_TOOL = {
    "type": "function",
    "function": {
        "name": "amap_place_around",
        "description": (
            "周边搜索 POI（v5）：以某个坐标为中心，搜索半径内的相关地点"
            "（餐饮、咖啡、景点等），返回名称/地址/坐标/评分/距离。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "center": {
                    "type": "string",
                    "description": "中心点坐标「lng,lat」（用搜索结果的 location）",
                },
                "keywords": {
                    "type": "string",
                    "description": "关键词，如：餐厅、咖啡、小吃；留空则返回中心附近的各类地点",
                },
                "radius": {
                    "type": "integer",
                    "description": "搜索半径（米），默认 1000",
                },
                "page_size": {
                    "type": "integer",
                    "description": "返回条数，默认 5，最多 10",
                },
            },
            "required": ["center"],
        },
    },
}

AMAP_PLACE_DETAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "amap_place_detail",
        "description": "按 POI ID 查询单个地点的详细信息（用搜索结果里的 id）。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "搜索结果里的 POI id"},
            },
            "required": ["id"],
        },
    },
}

AMAP_ROUTE_WALKING_TOOL = {
    "type": "function",
    "function": {
        "name": "amap_route_walking",
        "description": "步行路线规划（v5）：计算两点间步行的距离与耗时。",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "起点坐标「lng,lat」（用 POI 的 location）",
                },
                "destination": {
                    "type": "string",
                    "description": "终点坐标「lng,lat」",
                },
            },
            "required": ["origin", "destination"],
        },
    },
}

AMAP_ROUTE_DRIVING_TOOL = {
    "type": "function",
    "function": {
        "name": "amap_route_driving",
        "description": "驾车路线规划（v5）：计算两点间车行距离与耗时，自驾/打车出行参考。",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "起点坐标「lng,lat」（用 POI 的 location）",
                },
                "destination": {
                    "type": "string",
                    "description": "终点坐标「lng,lat」",
                },
            },
            "required": ["origin", "destination"],
        },
    },
}

AMAP_TOOLS = [
    AMAP_GEOCODE_TOOL,
    AMAP_PLACE_TEXT_TOOL,
    AMAP_PLACE_AROUND_TOOL,
    AMAP_PLACE_DETAIL_TOOL,
    AMAP_ROUTE_WALKING_TOOL,
    AMAP_ROUTE_DRIVING_TOOL,
]

# 工具名 -> 工具定义（planner 按名字挑工具用）
SCHEMA_BY_NAME = {t["function"]["name"]: t for t in AMAP_TOOLS}

# 工具名 -> 执行函数
TOOL_FUNCTIONS = {
    "amap_geocode": amap_geocode,
    "amap_place_text": amap_place_text,
    "amap_place_around": amap_place_around,
    "amap_place_detail": amap_place_detail,
    "amap_route_walking": amap_route_walking,
    "amap_route_driving": amap_route_driving,
}

# OpenAPI path -> 工具名；planner 用它把 api_plan.json 选中的接口映射到本地封装。
# 天气接口映射到 tools/analyse.py 的 get_weather（由 planner 单独处理）。
PATH_TOOL_MAP = {
    "/v3/geocode/geo": "amap_geocode",
    "/v5/geocode/geo": "amap_geocode",
    "/v3/place/text": "amap_place_text",
    "/v5/place/text": "amap_place_text",
    "/v3/place/around": "amap_place_around",
    "/v5/place/around": "amap_place_around",
    "/v3/place/detail": "amap_place_detail",
    "/v5/place/detail": "amap_place_detail",
    "/v3/direction/walking": "amap_route_walking",
    "/v5/direction/walking": "amap_route_walking",
    "/v3/direction/driving": "amap_route_driving",
    "/v5/direction/driving": "amap_route_driving",
    "/v3/weather/weatherInfo": "get_weather",
}


def run_tool(name: str, arguments: dict) -> dict:
    """按工具名执行（planner 的 _execute_tool 委托到这里）。"""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"success": False, "error": f"未知工具：{name}"}
    try:
        return func(**arguments)
    except TypeError as e:
        return {"success": False, "error": f"工具参数不匹配：{e}"}


if __name__ == "__main__":
    # 联网自测：每个工具调一次真实接口
    import json

    import dotenv

    dotenv.load_dotenv()

    print(json.dumps(amap_geocode("厦门"), ensure_ascii=False, indent=2))
    print()
    print(json.dumps(amap_place_text("公园", "厦门", page_size=2), ensure_ascii=False, indent=2))
    print()
    print(json.dumps(amap_place_around("118.087,24.479", "咖啡", radius=500, page_size=2), ensure_ascii=False, indent=2))
    print()
    print(json.dumps(amap_route_walking("118.087,24.479", "118.091,24.436"), ensure_ascii=False, indent=2))
    print()
    print(json.dumps(amap_route_driving("118.087,24.479", "118.091,24.436"), ensure_ascii=False, indent=2))
