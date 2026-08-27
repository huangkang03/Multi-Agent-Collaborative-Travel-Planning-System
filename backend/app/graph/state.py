from typing import List, Dict, Optional
from typing_extensions import TypedDict

class TripState(TypedDict):
    # 用户输入
    destination: str
    preferences: str
    
    # Agent 中间结果
    attractions: Optional[Dict]   # 景点推荐结果
    hotels: Optional[Dict]        # 酒店推荐结果
    
    # 最终输出
    final_plan: Optional[str]     # 编译后的完整行程文本