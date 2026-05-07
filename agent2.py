from langchain.agents import create_agent
from langchain.tools import tool
from typing import Any
from agent import llm  # 假设你的 llm 实例已定义
from agent import write_note
from utils.plan_execute import PlanAndExecuteAgent
from utils.blackboard import blackboard
from tools.tools_financial_data import *
from tools.tools_note import *
from tools.tools_blackboard import *




# 子 Agent 1：数据分析师（负责获取数据并写入黑板）
analyst = create_agent(
    model=llm,
    tools=[fetch_financial_data, write_blackboard],
    system_prompt="""你是一名财务分析师。任务：
1. 调用 fetch_financial_data 获取公司数据。
2. 提取关键指标（营收、利润、增长率）。
3. 调用 write_blackboard，key 使用 "financial_summary"，value 是你整理的摘要。
不要输出其他内容。"""
)

@tool
def call_analyst(company: str) -> str:
    """调用财务分析师子 Agent，分析指定公司的财务数据，结果自动写入黑板"""
    result = analyst.invoke({"messages": [("user", f"分析{company}")]})
    return f"分析师已完成，结果写入黑板 financial_summary"



# 子 Agent 2：报告撰写员
writer = create_agent(
    model=llm,
    tools=[read_blackboard],
    system_prompt="""你是一名报告撰写员。
1. 调用 read_from_blackboard，key 为 "financial_summary"，获取财务摘要。
2. 根据获取的内容，写一份简洁的中文报告（3-5句话）。
不要讨论，直接输出报告。"""
)

@tool
def call_writer() -> str:
    """调用报告撰写员子 Agent，基于黑板上已有的财务摘要生成报告"""
    result = writer.invoke({"messages": [("user", "写报告")]})
    return result["messages"][-1].content


# 运行
if __name__ == "__main__":
    base_agent = create_agent(
        model = llm,
        tools=[call_analyst, call_writer, write_blackboard, read_blackboard, write_note]
    )
    executor = PlanAndExecuteAgent(
        llm=llm, 
        agent=base_agent,
    )
    executor.run("分析腾讯的财务数据，然后生成一份报告，最后保存报告到笔记,并且告知我笔记文件路径")
  