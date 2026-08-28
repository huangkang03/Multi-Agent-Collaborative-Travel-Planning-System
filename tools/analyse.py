"""
analyst agent 使用的 tool：从用户输入中分析出旅游倾向信息，查询目的地天气。

抽取的结构化字段：
- city        目标城市（如：杭州）
- days        旅游天数
- start_date  出发日期（格式 YYYY-MM-DD；用户说「明天/后天/周末」时由模型
              换算成具体日期，工具负责校验格式与合理性）
- travel_mode 市内出行方式（漫步 / 驾车 / 打车）
- atmosphere  氛围偏好（热闹 / 清静）

天气工具 get_weather(city, forecast)：城市名经地理编码转 adcode 后
调用高德天气接口，返回实况或未来数天预报。

本文件只包含纯工具逻辑（校验、归一化、HTTP 调用），不实例化模型客户端；
模型配置统一放在 llm.py，由 agent 层使用。接入方式：analyst 在自己的
对话循环里把 ANALYSE_TOOLS + WEATHER_TOOLS 传给模型做 function calling，
收到工具调用后执行 analyse_travel_preference(...) / get_weather(...)。
"""

import json
import os
import re
from datetime import date, datetime
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError


# ---------- 1. 结构化结果模型 ----------


class TravelPreference(BaseModel):
    """用户旅游倾向的结构化描述，供后续 agent（行程规划等）使用。"""

    city: Optional[str] = Field(None, description="目标城市，如：杭州")
    days: Optional[int] = Field(None, ge=1, description="旅游天数")
    start_date: Optional[str] = Field(
        None,
        description="出发日期，格式 YYYY-MM-DD（如 2026-08-30），须不早于今天",
    )
    travel_mode: Optional[Literal["漫步", "驾车", "打车"]] = Field(
        None, description="市内出行方式"
    )
    atmosphere: Optional[Literal["热闹", "清静"]] = Field(
        None, description="氛围偏好：喜欢热闹还是清静"
    )


# ---------- 2. function calling 的工具定义 ----------

ANALYSE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyse_travel_preference",
            "description": (
                "分析用户输入的旅游倾向，提取目标城市、旅游天数、出发日期、"
                "市内出行方式、氛围偏好（热闹/清静）。"
                "用户没有明确提到的字段一律填 JSON 的 null"
                "（空值，不要填字符串 \"null\"）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "用户想去的旅游城市，如：北京、杭州",
                    },
                    "days": {
                        "type": "integer",
                        "description": "旅游天数，如：3",
                    },
                    "start_date": {
                        "type": "string",
                        "description": (
                            "出发日期，格式 YYYY-MM-DD。用户说「明天」「后天」"
                            "「下周五」「8月30号」等时，先换算成今天的具体日期"
                            "再填；今天出发就填今天的日期；用户没提日期则填 null"
                        ),
                    },
                    "travel_mode": {
                        "type": "string",
                        "enum": ["漫步", "驾车", "打车"],
                        "description": (
                            "市内出行方式偏好：漫步（步行/citywalk/散步）、"
                            "驾车（自驾/开车）、打车（出租车/网约车）"
                        ),
                    },
                    "atmosphere": {
                        "type": "string",
                        "enum": ["热闹", "清静"],
                        "description": (
                            "氛围偏好：热闹（喜欢人多繁华、烟火气、"
                            "热门景点）或清静（喜欢安静人少、小众清幽）。"
                            "必须是用户明确表达的偏好（如「喜欢热闹」、"
                            "「找个安静的地方」），不要仅凭出行方式、"
                            "场景或活动推断（「在西湖边漫步」不代表清静）"
                        ),
                    },
                },
                "required": ["city", "days", "start_date", "travel_mode", "atmosphere"],
            },
        },
    }
]

TOOL_NAME = ANALYSE_TOOLS[0]["function"]["name"]

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "查询指定城市的天气。用户想知道目的地天气、或出行前确认"
                "天气是否合适时调用。默认返回实时天气；用户想了解未来几天"
                "的天气时把 forecast 设为 true。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，如：杭州",
                    },
                    "forecast": {
                        "type": "boolean",
                        "description": "true 查未来数天预报，false（默认）查实况天气",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

WEATHER_TOOL_NAME = WEATHER_TOOLS[0]["function"]["name"]


# ---------- 3. 需求抽取工具实现 ----------

# 出行方式的常见同义说法，统一归一化到三个枚举值
_TRAVEL_MODE_ALIASES = {
    # 漫步
    "漫步": "漫步",
    "步行": "漫步",
    "走路": "漫步",
    "散步": "漫步",
    "citywalk": "漫步",
    # 驾车
    "驾车": "驾车",
    "自驾": "驾车",
    "开车": "驾车",
    # 打车
    "打车": "打车",
    "出租车": "打车",
    "网约车": "打车",
    "的士": "打车",
    "taxi": "打车",
}

# 氛围偏好的常见同义说法，统一归一化到两个枚举值
_ATMOSPHERE_ALIASES = {
    # 热闹
    "热闹": "热闹",
    "繁华": "热闹",
    "喧闹": "热闹",
    "嘈杂": "热闹",
    "人多": "热闹",
    "人山人海": "热闹",
    "烟火气": "热闹",
    "人气旺": "热闹",
    # 清静
    "清静": "清静",
    "清净": "清静",
    "安静": "清静",
    "宁静": "清静",
    "幽静": "清静",
    "静谧": "清静",
    "人少": "清静",
    "清幽": "清静",
    "僻静": "清静",
}


def _to_none_if_empty(value):
    """把模型常见的“空值”写法（字符串 'null'、'无'、空白串）归一化成 None。"""
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none", "无"):
        return None
    return value


# 日期的宽松格式：YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD / YYYY年M月D日
_DATE_PATTERNS = [
    (re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"), None),
    (re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日?"), None),
]
# 没带年份的写法：8月30号 / 8-30（补当前年份）
_DATE_NO_YEAR = [
    re.compile(r"^(\d{1,2})[-/月](\d{1,2})日?$"),
]


def _normalize_date(value, field_errors: list) -> Optional[str]:
    """
    归一化模型给的出发日期，返回 YYYY-MM-DD 字符串。

    宽松接受 YYYY-MM-DD / YYYY/M/D / 8月30号 等写法；没年份的补今年。
    无效日期（如 8月32号）或早于今天的日期：记入 field_errors 供返回错误，
    由模型下一轮改填；模型没填（None）则原样返回 None（视为待收集）。
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
    if not isinstance(value, str) or not value:
        return None

    parsed: Optional[date] = None
    for pattern, _ in _DATE_PATTERNS:
        m = pattern.match(value)
        if m:
            year, month, day = (int(x) for x in m.groups())
            try:
                parsed = date(year, month, day)
            except ValueError:
                field_errors.append(f"start_date「{value}」不是有效日期")
                return None
            break
    if parsed is None:
        for pattern in _DATE_NO_YEAR:
            m = pattern.match(value)
            if m:
                month, day = (int(x) for x in m.groups())
                try:
                    parsed = date(date.today().year, month, day)
                except ValueError:
                    field_errors.append(f"start_date「{value}」不是有效日期")
                    return None
                break
    if parsed is None:
        # 已经是合法 YYYY-MM-DD 时 datetime.strptime 兜底；否则报无效
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            field_errors.append(
                f"start_date「{value}」格式无法识别，请换成 YYYY-MM-DD"
            )
            return None

    if parsed < date.today():
        field_errors.append(
            f"start_date「{value}」早于今天（{date.today().isoformat()}），"
            "出发日期不能在过去"
        )
        return None
    return parsed.isoformat()


def analyse_travel_preference(
    city: Optional[str] = None,
    days: Optional[int] = None,
    start_date: Optional[str] = None,
    travel_mode: Optional[str] = None,
    atmosphere: Optional[str] = None,
) -> dict:
    """
    analyst 的 tool 本体：校验并封装抽取到的旅游倾向信息。

    参数通常来自模型 function calling 的解析结果，
    也可以由调用方自行解析用户输入后直接传入。
    返回 dict，方便 JSON 序列化后在多个 agent 之间传递。
    """
    field_errors: list = []
    # 归一化：缺失字段的字符串 'null'/'无' 写法、出行方式/氛围偏好的
    # 同义说法、日期的多种写法、去掉首尾空白、空字符串视为未提供
    city = _to_none_if_empty(city)
    days = _to_none_if_empty(days)
    travel_mode = _to_none_if_empty(travel_mode)
    atmosphere = _to_none_if_empty(atmosphere)
    start_date = _normalize_date(_to_none_if_empty(start_date), field_errors)
    if field_errors:
        return {"success": False, "error": "；".join(field_errors)}
    if isinstance(travel_mode, str):
        travel_mode = _TRAVEL_MODE_ALIASES.get(
            travel_mode.strip().lower(), travel_mode.strip()
        )
    if isinstance(atmosphere, str):
        atmosphere = _ATMOSPHERE_ALIASES.get(
            atmosphere.strip().lower(), atmosphere.strip()
        )
    if isinstance(city, str):
        city = city.strip() or None

    try:
        preference = TravelPreference(
            city=city,
            days=days,
            travel_mode=travel_mode,
            atmosphere=atmosphere,
            start_date=start_date,
        )
    except ValidationError as e:
        return {"success": False, "error": f"参数校验失败：{e}"}

    # 标出缺失的信息，方便 analyst 决定是否追问用户
    missing_fields = [
        field
        for field in ("city", "days", "start_date", "travel_mode", "atmosphere")
        if getattr(preference, field) is None
    ]
    return {
        "success": True,
        **preference.model_dump(),
        "missing_fields": missing_fields,
    }


# ---------- 4. 天气查询工具实现 ----------

# 高德 Web 服务的 key 环境变量名与接口地址
AMAP_API_KEY_ENV = "AMAP_API_KEY"
AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"


def _amap_get(url: str, params: dict) -> dict:
    """
    请求高德 Web 服务接口并校验业务状态，返回解析后的 JSON dict。

    出错时返回统一格式 {"success": False, "error": ...}，
    与 analyse_travel_preference 的错误格式保持一致，方便模型阅读。
    """
    api_key = os.getenv(AMAP_API_KEY_ENV)
    if not api_key:
        return {
            "success": False,
            "error": "未配置 AMAP_API_KEY 环境变量，请在 .env 中写入高德 Web 服务 key",
        }

    try:
        response = httpx.get(
            url, params={"key": api_key, "output": "JSON", **params}, timeout=10
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        return {"success": False, "error": f"请求高德接口失败：{e}"}

    data = response.json()
    if data.get("status") != "1" or data.get("infocode") != "10000":
        return {
            "success": False,
            "error": (
                f"高德接口返回错误：{data.get('info')}"
                f"（infocode={data.get('infocode')}）"
            ),
        }
    return data


def get_weather(city: str, forecast: bool = False) -> dict:
    """
    查询指定城市的天气（高德 Web 服务 API）。

    city 传城市名（如「杭州」），内部先经地理编码接口转成 adcode
    （天气接口只认行政区划代码，不认城市名）。
    forecast=False 返回实况天气，True 返回未来数天预报。
    返回 dict，作为 tool 结果回传给模型。
    """
    if not isinstance(city, str) or not city.strip():
        return {"success": False, "error": "city 参数不能为空"}
    city = city.strip()

    geo_data = _amap_get(AMAP_GEOCODE_URL, {"address": city})
    if not geo_data.get("geocodes"):
        if "success" in geo_data:
            return geo_data  # _amap_get 已返回的错误信息
        return {
            "success": False,
            "error": f"高德地理编码找不到城市「{city}」，请确认城市名",
        }
    geocode = geo_data["geocodes"][0]
    adcode = geocode["adcode"]

    weather_data = _amap_get(
        AMAP_WEATHER_URL,
        {"city": adcode, "extensions": "all" if forecast else "base"},
    )
    if "success" in weather_data:
        return weather_data

    if forecast:
        forecasts = weather_data.get("forecasts") or []
        if not forecasts:
            return {"success": False, "error": "高德未返回预报数据"}
        casts = [
            {
                "date": cast.get("date"),
                "week": cast.get("week"),
                "dayweather": cast.get("dayweather"),
                "nightweather": cast.get("nightweather"),
                "daytemp": cast.get("daytemp"),
                "nighttemp": cast.get("nighttemp"),
            }
            for cast in forecasts[0].get("casts", [])
        ]
        return {
            "success": True,
            "city": geocode.get("city") or city,
            "adcode": adcode,
            "type": "预报",
            "casts": casts,
        }

    lives = weather_data.get("lives") or []
    if not lives:
        return {"success": False, "error": "高德未返回实况数据"}
    live = lives[0]
    return {
        "success": True,
        "city": geocode.get("city") or city,
        "adcode": adcode,
        "type": "实况",
        "weather": live.get("weather"),
        "temperature": live.get("temperature"),
        "winddirection": live.get("winddirection"),
        "windpower": live.get("windpower"),
        "humidity": live.get("humidity"),
        "reporttime": live.get("reporttime"),
    }


if __name__ == "__main__":
    # 自测：需求抽取是纯函数直接测；天气工具联网调用高德真实接口
    import dotenv

    dotenv.load_dotenv()

    print("=" * 20 + " analyse_travel_preference 自测 " + "=" * 20)
    tomorrow = (date.today() + __import__("datetime").timedelta(days=1)).isoformat()
    cases = [
        {"city": "杭州", "days": 3, "travel_mode": "citywalk", "atmosphere": "安静"},
        {"city": " 北京 ", "days": 2, "travel_mode": "自驾", "atmosphere": None},
        {"city": None, "days": None, "travel_mode": None, "atmosphere": None},
        {"city": "成都", "days": 0, "travel_mode": "漫步", "atmosphere": "热闹"},
        {"city": "null", "days": 5, "travel_mode": "taxi", "atmosphere": "人少"},
        # 日期归一化：标准 / 斜杠 / 带年月日 / 无年份补今年 / 过去 / 无效
        {"city": "厦门", "days": 3, "start_date": "2026-08-30",
         "travel_mode": "漫步", "atmosphere": "清静"},
        {"city": "厦门", "days": 3, "start_date": "2026/9/2",
         "travel_mode": "漫步", "atmosphere": "清静"},
        {"city": "厦门", "days": 3, "start_date": "2026年9月5日",
         "travel_mode": "漫步", "atmosphere": "清静"},
        {"city": "厦门", "days": 3, "start_date": tomorrow,
         "travel_mode": "漫步", "atmosphere": "清静"},
        {"city": "厦门", "days": 3, "start_date": "2020-01-01",
         "travel_mode": "漫步", "atmosphere": "清静"},
        {"city": "厦门", "days": 3, "start_date": "不知道",
         "travel_mode": "漫步", "atmosphere": "清静"},
    ]
    for kwargs in cases:
        print("-" * 20 + f" {kwargs} " + "-" * 20)
        print(
            json.dumps(
                analyse_travel_preference(**kwargs), ensure_ascii=False, indent=2
            )
        )
        print()

    print("=" * 20 + " get_weather 自测 " + "=" * 20)
    print("实况：")
    print(json.dumps(get_weather("杭州"), ensure_ascii=False, indent=2))
    print("\n预报：")
    print(json.dumps(get_weather("北京", forecast=True), ensure_ascii=False, indent=2))
