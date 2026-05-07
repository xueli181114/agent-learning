from langchain.tools import tool
from utils.blackboard import blackboard
# 工具：写入黑板
@tool
def write_blackboard(key: str, value: str) -> str:
    """将结果存入黑板，后续其他 Agent 可以读取"""
    blackboard.data[key] = value
    return f"已写入黑板：{key} = {value[:50]}..."

# 工具：读取黑板
@tool
def read_blackboard(key: str) -> str:
    """从黑板读取指定 key 的值"""
    return blackboard.data.get(key, "未找到数据")