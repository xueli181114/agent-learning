from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import List, Dict, Any

def observe_agent(agent, user_input: str):
    """独立的观察函数（你已有的）"""
    print(f"\n📝 用户：{user_input}")
    result = agent.invoke({"messages": [("user", user_input)]})
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"🔧 调用了工具：{msg.tool_calls[0]['name']}")
            print(f"   参数：{msg.tool_calls[0]['args']}")
    print(f"💬 最终回答：{result['messages'][-1].content[:200]}")
    return result

class PlanAndExecuteAgent:
    def __init__(self, agent, llm):
        self.agent = agent   # 这是你的 create_agent 返回的可执行对象
        self.llm = llm       # 用于规划的 LLM

    def plan(self, task: str) -> List[str]:
        plan_prompt = f"""将以下任务分解为最多3个简单逻辑步骤，每步一行。不要模拟“打开应用”这类UI操作，只描述信息获取、计算、保存等动作。
任务：{task}
步骤："""
        response = self.llm.invoke(plan_prompt)
        steps = [step.strip() for step in response.content.split('\n') if step.strip()]
        # 强制限制步骤数量
        return steps[:3]

    def run(self, task: str) -> Dict[str, Any]:
        steps = self.plan(task)
        print(f"📋 计划：{steps}")

        context = {}           # 共享上下文，存储中间结果
        previous_answer = ""   # 上一步的最终回答（可能包含工具结果）

        for i, step in enumerate(steps, 1):
            # 构建增强的步骤提示：如果上一步有结果，注入进去
            enhanced_step = step
            if previous_answer:
                enhanced_step += f"\n【上一步的结果参考】{previous_answer}"

            print(f"\n▶️ 执行步骤 {i}: {enhanced_step}")

            # ----- 这里嵌入 observe_agent 的核心逻辑 -----
            result = self.agent.invoke({"messages": [("user", enhanced_step)]})
            # 打印工具调用详情
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"   🔧 调用了工具：{tc['name']}")
                        print(f"      参数：{tc['args']}")
                # 如果是工具返回的消息，可选打印返回值
                if isinstance(msg, ToolMessage):
                    print(f"   📦 工具返回：{msg.content[:100]}")

            final_answer = result["messages"][-1].content
            print(f"   💬 步骤回答：{final_answer[:200]}")

            # 保存结果
            context[f"step_{i}"] = {
                "step": step,
                "full_messages": result["messages"],
                "answer": final_answer
            }
            previous_answer = final_answer   # 传递给下一步

        return context