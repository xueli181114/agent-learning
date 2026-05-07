from agents import master_agent
from llms import deepseek_llm
from utils.plan_execute import PlanAndExecuteAgent
if __name__ == "__main__":
    executor = PlanAndExecuteAgent(
        agent=master_agent,
        llm=deepseek_llm
    )
    executor.run("帮我查看明天北京的天气并且记录笔记")
    executor.run("我明天开会，明天提醒我并且保存笔记")