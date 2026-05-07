# ✅ 新版本正确写法（LangChain 1.2+）
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
import os
import subprocess
from typing import Dict, Any
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from tools.tools_note import *
from utils.plan_execute import PlanAndExecuteAgent







@tool
def search_web(query:str) ->str:
    """搜索网络获取实时信息，返回搜索结果摘要"""
    return f"关于「{query}」的搜索结果：示例内容..."



@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except:
        return "计算错误"

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=api_key,
    temperature=0
)

class Blackboard:
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.history: list = []
blackboard = Blackboard()

if __name__ == "main":

    # ✅ 使用新的 create_agent API
    agent = create_agent(
        model=llm,
        tools=[get_weather, calculate,write_note, read_note, exec_command, today],
        system_prompt="你是一个有帮助的助手，可以使用工具来回答问题。",
        debug=True
    )

    plan_execute = PlanAndExecuteAgent(llm=llm, agent=agent)
    plan_execute.run("从我的笔记中查找天气最舒服的城市")

