from langchain.tools import tool
# 工具：获取财务数据（模拟）

@tool
def fetch_financial_data(company: str) -> str:
    """获取公司财务数据（模拟）"""
    return f"{company} 营收100亿，利润20亿，增长15%"

