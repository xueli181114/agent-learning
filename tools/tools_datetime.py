from datetime import datetime
from langchain.tools import tool

@tool
def today() ->str:
    """返回今天的日期，格式：YYYY年MM月DD日"""
    return datetime.now().strftime("%Y年%m月%d日")