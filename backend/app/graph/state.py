from typing import List, Dict, Optional
from typing_extensions import TypedDict

class TripState(TypedDict):
    # 用户输入
    destination: str
    preferences: str
    include_hotel: bool  # 新增：是否包含酒店推荐（默认 True）

    # Agent 中间结果
    attractions: Optional[Dict]
    hotels: Optional[Dict]

    # 最终输出
    final_plan: Optional[str]