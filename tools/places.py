# tools/places.py
import os
import json
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool

# 内部处理类 (Inline Handler)
class _InlineGoogleMapHandler:
    def __init__(self):
        self._api_key = os.getenv("GOOGLEMAP_API_KEY") or os.getenv("GMAPS_API_KEY")

    def get_distance(self, origin, destination, mode="driving"):
        if not self._api_key: return {"error": "Missing API Key"}
        try:
            params = {
                "origins": origin, "destinations": destination,
                "mode": mode, "key": self._api_key, "units": "metric"
            }
            r = requests.get("https://maps.googleapis.com/maps/api/distancematrix/json", params=params, timeout=10)
            data = r.json()
            # ... (保留原有的解析逻辑) ...
            rows = data.get("rows", [])
            if not rows: return {"error": "No data"}
            elements = rows[0].get("elements", [{}])[0]
            return {
                "origin": origin, "destination": destination,
                "distance": elements.get("distance", {}).get("text"),
                "duration": elements.get("duration", {}).get("text")
            }
        except Exception as e:
            return {"error": str(e)}

# 定义 Tool Schema
class GetDistance(BaseModel):
    origin: str = Field(description="Starting location")
    destination: str = Field(description="Ending location")
    mode: Literal["driving", "walking", "transit", "bicycling"] = "driving"

def _distance_run(origin: str, destination: str, mode: str = "driving"):
    return _InlineGoogleMapHandler().get_distance(origin, destination, mode)

get_distance_tool = StructuredTool.from_function(
    name="get_distance",
    func=_distance_run,
    description="Calculate distance and duration between two places.",
    args_schema=GetDistance
)

@tool
def get_detail_place(place_name: str) -> str:
    """Get place details: address, rating, hours, etc. using Google Places."""
    # ... (此处粘贴原 main.py 中 get_detail_place 的完整逻辑) ...
    return json.dumps({"status": "Implemented in tools/places.py (placeholder)"})

@tool
def is_open_now(place_name: str) -> str:
    """Check if a place is currently open."""
    # ... (此处粘贴原 main.py 中 is_open_now 的完整逻辑) ...
    return json.dumps({"status": "Implemented"})

@tool
def get_nearby_place(place_name: str, radius: int = 1000, place_type: str = "tourist_attraction") -> str:
    """Search nearby places."""
    # ... (此处粘贴原 main.py 中 get_nearby_place 的完整逻辑) ...
    return json.dumps({"status": "Implemented"})