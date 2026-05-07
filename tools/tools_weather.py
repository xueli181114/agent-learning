from langchain.tools import tool
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}天气cloudy，25°C"