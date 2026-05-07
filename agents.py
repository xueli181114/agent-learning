# agents.py
import time
import re
from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from tools import tools_note
from tools.tools_financial_data import fetch_financial_data
from tools.tools_blackboard import read_blackboard, write_blackboard
from tools.tools_weather import get_weather
from tools.tools_datetime import today
from llms import deepseek_llm
from langgraph.store.postgres import PostgresStore

_store = None
_store_conn = None  # 保存连接引用，防止被 GC 关闭
DB_URI = "postgresql://xueli@localhost:5432/agentdb"
def get_store():
    global _store, _store_conn
    if _store is None:
        # from_conn_string 是 context manager，不能直接赋值给全局变量
        # 所以我们手动创建连接和 store 实例
        import psycopg
        from psycopg.rows import dict_row

        _store_conn = psycopg.connect(
            DB_URI,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
        _store = PostgresStore(conn=_store_conn)
        _store.setup()
        print("[get_store] PostgreSQL store 初始化成功")
    return _store

def make_memory_aware_agent(tools, prompt_prefix: str = "", name: str = "memory_agent"):
    react_agent = create_agent(
        model=deepseek_llm,
        tools=tools,
        system_prompt=prompt_prefix,
        name=name
    )

    def memory_aware_node(state, config: RunnableConfig):
        print("[personal_agent] 开始执行")
        store = get_store()
        user_id = config.get("configurable", {}).get("user_id", "anonymous")
        namespace = ("user_memories", user_id)
        memory_context = ""
        if store:
            try:
                memories = store.search(namespace, limit=20)
                # 按 updated_at 降序排列，取最新的记忆
                memories_sorted = sorted(memories, key=lambda m: m.updated_at, reverse=True)
                memory_context = "\n".join([m.value["content"] for m in memories_sorted[:5]])
                print(f"[personal_agent] 读取到 {len(memories)} 条记忆（取最新 5 条）")
            except Exception as e:
                print(f"[personal_agent] 读取记忆出错: {e}")
        else:
            print("[personal_agent] 警告：store 为 None，无法读写长期记忆")

        # 注入记忆到用户消息
        for i, msg in enumerate(state["messages"]):
            if isinstance(msg, HumanMessage):
                if memory_context:
                    enhanced = f"[历史记忆：{memory_context}]\n用户问题：{msg.content}"
                    state["messages"][i] = HumanMessage(content=enhanced)
                break

        # 调用底层 agent
        try:
            result = react_agent.invoke(state, config)
        except Exception as e:
            print(f"[personal_agent] react_agent 调用失败: {e}")
            result = {"messages": [AIMessage(content="个人助理暂时无法处理，请稍后再试。")]}

        # 确保结果格式
        if not isinstance(result, dict) or "messages" not in result:
            result = {"messages": [AIMessage(content="个人助理没有生成有效回复。")]}
        elif not result["messages"]:
            result["messages"] = [AIMessage(content="个人助理处理完毕。")]
        elif not isinstance(result["messages"][-1], AIMessage):
            result["messages"].append(AIMessage(content="处理完成。"))

        # 注意：personal_agent 不再负责写入长期记忆
        # 所有个人信息提取由 multi_agent.py 中的 memory_node 统一用 LLM 处理

        return result

    return RunnableLambda(memory_aware_node, name=name)



personal_agent = make_memory_aware_agent(
    tools=[],
    prompt_prefix="你是用户的个人助理，负责记住用户的个人信息（如名字、喜好）并回答相关问题。如果不知道就如实说。",
    name="personal_assistant"
)

def wrap_agent(agent):
    """包装普通 agent，使其符合 (state, config) 签名"""
    def wrapped(state, config: RunnableConfig):
        return agent.invoke(state, config)
    return RunnableLambda(wrapped, name=agent.name if hasattr(agent, "name") else "agent")

def make_note_agent():
    """创建 note_agent，注入 store 记忆读取能力"""
    react_agent = create_agent(
        model=deepseek_llm,
        name="note_agent",
        tools=[tools_note.write_note, tools_note.delete_note, tools_note.get_note_path,
               tools_note.read_note, tools_note.list_notes, today],
        system_prompt="""你是笔记专家，负责记录、整理和查询信息。

当用户询问"有什么笔记"、"有几个会议"、"会议主题是什么"等查询问题时，你必须使用 list_notes 工具列出所有笔记文件，然后用 read_note 工具读取相关文件来获取内容。
不要凭空回答，一定要先调用工具查询实际数据。

当用户提到"会议冲突"、"另一个会议"、"被安排了另一个会议"等情况时，说明有新的会议信息需要记录。
例如用户说"5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议"：
1. 先用 write_note 记录新会议（金融会议）的信息
2. 再执行取消操作（删除或更新研发研讨会议的笔记）
不要只执行取消操作而遗漏了新会议的信息。"""
    )

    def note_node(state, config: RunnableConfig):
        # 读取 store 中的记忆，注入到用户消息中
        store = get_store()
        user_id = config.get("configurable", {}).get("user_id", "anonymous")
        namespace = ("user_memories", user_id)
        memory_context = ""
        if store:
            try:
                memories = store.search(namespace, limit=20)
                memories_sorted = sorted(memories, key=lambda m: m.updated_at, reverse=True)
                memory_context = "\n".join([m.value["content"] for m in memories_sorted[:5]])
            except Exception as e:
                print(f"[note_agent] 读取记忆出错: {e}")

        # 注入记忆到用户消息
        for i, msg in enumerate(state["messages"]):
            if isinstance(msg, HumanMessage):
                if memory_context:
                    enhanced = f"[长期记忆：{memory_context}]\n用户问题：{msg.content}"
                    state["messages"][i] = HumanMessage(content=enhanced)
                break

        result = react_agent.invoke(state, config)
        if not isinstance(result, dict) or "messages" not in result:
            result = {"messages": [AIMessage(content="处理完成。")]}
        return result

    return RunnableLambda(note_node, name="note_agent")

note_agent = make_note_agent()

fina_agent_raw = create_agent(
    model=deepseek_llm,
    name="fina_agent",
    tools=[fetch_financial_data, read_blackboard, write_blackboard],
    system_prompt="你是财务专家，负责预算计算和财务分析。"
)
fina_agent = wrap_agent(fina_agent_raw)

weather_agent_raw = create_agent(
    model=deepseek_llm,
    name="weather_agent",
    tools=[get_weather],
    system_prompt="你是天气专家，负责处理所有与天气相关的问题。"
)
weather_agent = wrap_agent(weather_agent_raw)

__all__ = ["personal_agent", "note_agent", "fina_agent", "weather_agent"]