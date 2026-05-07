# multi_agent.py
import time
from typing import Annotated, TypedDict, Optional, List
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command, Send
import re

from agents import personal_agent, note_agent, fina_agent, weather_agent
from llms import deepseek_llm

# ─── LangSmith 追踪 ──────────────────────────────────────
from langsmith import traceable

# 导入 LangSmith 配置（设置环境变量）
try:
    import langsmith_config
except ImportError:
    pass

# ─── 状态定义 ─────────────────────────────────────────────

class MultiAgentState(TypedDict):
    """全局状态"""
    messages: Annotated[list, add_messages]
    next: str
    user_approved: Optional[bool]
    pending_batches: List[List[str]]   # 每个 batch 是一组可并行执行的任务文本列表
    current_batch_index: int


# ─── Planner：拆解任务 + 分组为并行批次 ─────────────────

@traceable(
    name="planner_node",
    run_type="chain",
    metadata={"node": "planner", "description": "将用户请求拆解为子任务并分组为并行批次"},
)
def planer_node(state: MultiAgentState, config: RunnableConfig):
    """
    1. 将用户请求拆解为多个子任务
    2. 用 LLM 判断哪些任务可以并行执行，分组为 batches
    """
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, HumanMessage):
        return {"pending_batches": [], "current_batch_index": 0}

    user_text = last_msg.content

    # Step 1: 拆解为子任务
    # 注意：每个子任务必须包含完整的操作信息（文件名、内容、城市等），不能丢失上下文
    decompose_prompt = f"""请将以下用户请求拆解为多个独立的子任务，每个子任务只包含一个动作，用数字列表输出，不要解释，不要添加额外文字。

重要规则：
- 不要擅自添加"记录笔记"前缀！只有当用户明确说"记录笔记"、"记笔记"、"写笔记"时才使用"记录笔记"
- 如果用户只是说"我叫XXX"、"我是XXX"、"我喜欢XXX"等自我介绍，直接输出原句，不要包装成"记录笔记"
- 每个子任务必须包含完整的操作信息（如文件名、笔记内容、城市名等）
- 如果用户说"记录笔记"但提到了具体内容，请把内容包含在任务描述中
- 如果用户说"提醒我X"且同时说"记录笔记"，则"记录笔记"任务应包含提醒内容
- 不要过度拆解！如果一句话描述的是一个完整场景（如"我的5点会议和另一个会议冲突了，需要取消"），应该作为一个整体任务，不要拆成碎片
- 不要拆分问句！如果用户问"我明天有几个会议，是什么主题的？"，这是一个完整的查询问题，不要拆成"我明天有几个会议"和"是什么主题的"

例如：
用户：删除笔记 test.md，并查看天气。
输出：
1. 删除笔记 test.md
2. 查看天气

用户：提醒我明天下午四点开会并记录笔记
输出：
1. 提醒我明天下午四点开会
2. 记录笔记：明天下午四点开会

用户：我叫大头儿子
输出：
1. 我叫大头儿子

用户：我的5点钟被schedule了另一个会议来讨论金融问题，这两个会议冲突了，我需要把5点钟的研发研讨会议取消掉
输出：
1. 5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议

用户：我明天有几个会议，是什么主题的？
输出：
1. 我明天有几个会议，是什么主题的？

用户：{user_text}
输出："""
    response = deepseek_llm.invoke([HumanMessage(content=decompose_prompt)])
    tasks = []
    for line in response.content.strip("\n").split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            task = re.sub(r'^[\d\-\.\s]+', '', line).strip()
            if task:
                tasks.append(task)
    if not tasks:
        tasks = [user_text]

    print(f"[Planner] 拆解为 {len(tasks)} 个任务: {tasks}")

    # Step 2: 用 LLM 判断任务依赖关系，分组为并行批次
    tasks_formatted = "\n".join(f"- {t}" for t in tasks)
    grouping_prompt = f"""分析以下任务之间的依赖关系，将可以并行执行的任务分到同一组（batch）。

规则：
- 如果任务 A 的结果是任务 B 的输入，则 A 和 B 不能并行，必须分到不同 batch
- 如果任务之间没有依赖关系，可以放在同一 batch 并行执行
- 每个 batch 内的任务可以同时执行
- 不同 batch 按顺序执行（batch1 → batch2 → ...）
- 重要：不要拆分单个任务！每个任务必须保持原样，不能拆分成更小的片段。
- 重要：不要拆分问句！如果某个任务是一个完整的问句（如"我明天有几个会议，是什么会议？"），它必须作为一个整体放在一个 batch 中，不能拆成"我明天有几个会议"和"是什么会议？"。

输出格式要求（非常重要）：
- 每行代表一个 batch
- 每行只包含该 batch 中的任务原文，多个任务之间用中文逗号（，）分隔
- 不要添加任何序号、前缀、后缀、解释文字
- 如果所有任务都可以并行，只输出一行

示例：
输入：
- 删除笔记 test.md
- 查看天气

输出：
删除笔记 test.md，查看天气

示例2（有依赖关系）：
输入：
- 获取腾讯财务数据
- 根据数据生成报告

输出：
获取腾讯财务数据
根据数据生成报告

示例3（单个完整场景）：
输入：
- 5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议

输出：
5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议

示例4（完整问句）：
输入：
- 我明天有几个会议，是什么会议？

输出：
我明天有几个会议，是什么会议？

任务列表：
{tasks_formatted}

输出："""
    response = deepseek_llm.invoke([HumanMessage(content=grouping_prompt)])
    batches_raw = response.content.strip("\n").split('\n')

    batches = []
    for line in batches_raw:
        line = line.strip()
        if not line:
            continue
        batch_tasks = [t.strip() for t in re.split(r'[，,]', line) if t.strip()]
        if batch_tasks:
            batches.append(batch_tasks)

    if not batches:
        batches = [[t] for t in tasks]

    print(f"[Planner] 分组为 {len(batches)} 个并行批次:")
    for i, batch in enumerate(batches):
        print(f"  Batch {i+1}: {batch}")

    return {
        "pending_batches": batches,
        "current_batch_index": 0,
        "messages": [AIMessage(content=f"我将分解为以下步骤：\n" +
                                "\n".join(f"Batch {i+1}: {' + '.join(b)}" for i, b in enumerate(batches)))]
    }


# ─── Supervisor：处理当前 batch ──────────────────────────
#     先检查是否有删除操作需要用户确认，再并行分发非删除任务

@traceable(
    name="supervisor_node",
    run_type="chain",
    metadata={"node": "supervisor", "description": "检查删除操作、并行分发任务到对应 Agent"},
)
def supervisor_node(state: MultiAgentState, config: RunnableConfig):
    """
    1. 检查当前 batch 中是否有删除操作
    2. 如果有，先通过 interrupt() 等待用户确认
    3. 然后将非删除任务并行分发到对应 Agent
    """
    pending_batches = state.get("pending_batches", [])
    current_index = state.get("current_batch_index", 0)

    if current_index >= len(pending_batches):
        print("[Supervisor] 所有批次已完成")
        return Command(goto=END)

    current_batch = pending_batches[current_index]
    print(f"[Supervisor] 正在处理 Batch {current_index+1}/{len(pending_batches)}: {current_batch}")

    # ── 检查是否有删除操作，需要用户确认 ──
    delete_tasks = [t for t in current_batch if re.search(r'删除|delete', t, re.IGNORECASE)]
    non_delete_tasks = [t for t in current_batch if not re.search(r'删除|delete', t, re.IGNORECASE)]
    delete_confirmed = False

    # 如果有删除操作，先展示完整的执行计划，再中断等待用户确认
    if delete_tasks:
        print("\n" + "=" * 50)
        print("📋 即将执行以下操作：")
        for t in non_delete_tasks:
            print(f"   ✅ {t}")
        for t in delete_tasks:
            print(f"   ⚠️  {t}（需要确认）")
        print("=" * 50)

        delete_desc = "；".join(delete_tasks)
        user_response = interrupt({
            "question": f"你确定要执行以下操作吗？\n{delete_desc}\n请回复 '是' 或 '否'。"
        })
        if user_response.strip() == "是":
            print(f"[Supervisor] 用户确认了删除操作")
            delete_confirmed = True
        else:
            print(f"[Supervisor] 用户取消了删除操作，跳过删除任务")

    # ── 并行分发任务 ──
    agent_names = {
        "personal_assistant": "personal_assistant",
        "note_agent": "note_agent",
        "fina_agent": "fina_agent",
        "weather_agent": "weather_agent",
    }

    system_prompt = """你是任务调度主管。根据用户的问题，选择最合适的专家来处理。
只输出一个单词（agent 名称），不要输出其他内容。
- personal_assistant: 处理所有关于用户个人信息的问题（如名字、偏好、自我介绍）。
- note_agent: 处理笔记相关的操作（记录、读取、删除笔记），以及会议安排、日程管理、提醒事项。注意：即使会议主题涉及"金融"、"财务"等词，只要是会议安排就归 note_agent。
- fina_agent: 处理财务分析、预算、支出数据、股票查询。注意：只处理真正的财务数据分析，不处理"金融会议"这类会议安排。
- weather_agent: 处理天气查询。"""

    sends = []

    # 先分发非删除任务
    for task_text in non_delete_tasks:
        prompt = f"{system_prompt}\n\n用户问题：{task_text}"
        response = deepseek_llm.invoke([HumanMessage(content=prompt)])
        agent_name = response.content.strip().lower()

        if agent_name not in agent_names:
            agent_name = "finish"

        print(f"  [Supervisor] 任务「{task_text}」→ {agent_name}")
        sends.append(Send(
            agent_name,
            {
                "messages": [HumanMessage(content=task_text)],
            }
        ))

    # 如果用户确认了删除，把删除任务也加入并行分发（走 note_agent）
    if delete_confirmed:
        for task_text in delete_tasks:
            print(f"  [Supervisor] 任务「{task_text}」→ note_agent（用户已确认）")
            sends.append(Send(
                "note_agent",
                {
                    "messages": [HumanMessage(content=task_text)],
                }
            ))

    return Command(
        goto=sends,
        update={"current_batch_index": current_index + 1}
    )


# ─── Agent 执行节点 ──────────────────────────────────────

def make_agent_node(agent):
    def node(state: dict, config: RunnableConfig):
        msgs = state.get("messages", [])
        result = agent.invoke({"messages": msgs}, config)
        if "messages" not in result:
            result = {"messages": [AIMessage(content="处理完成。")]}
        return result
    return node


# ─── 记忆节点：从对话中提取重要信息并存入长期记忆 ─────

@traceable(
    name="memory_node",
    run_type="chain",
    metadata={"node": "memory", "description": "从对话中提取重要信息并存入长期记忆"},
)
def memory_node(state: MultiAgentState, config: RunnableConfig):
    """
    在所有 Agent 执行完毕后运行，从对话中提取重要信息
    （如日程安排、提醒事项、用户偏好等）并存入 PostgreSQL store。
    """
    from agents import get_store

    store = get_store()
    if not store:
        print("[Memory] 跳过：store 不可用")
        return {}

    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    namespace = ("user_memories", user_id)

    # 从整个对话历史中提取记忆，但只提取新的信息
    # 使用全部消息确保不会遗漏较早的重要信息（如金融会议）
    all_msgs = state.get("messages", [])

    conversation_text = ""
    for msg in all_msgs:
        if isinstance(msg, HumanMessage):
            conversation_text += f"用户：{msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation_text += f"助手：{msg.content}\n"

    if not conversation_text.strip():
        print("[Memory] 对话为空，跳过")
        return {}

    print(f"[Memory] 正在分析全部对话（共 {len(all_msgs)} 条消息）...")

    # 读取已有记忆（用于去重）
    try:
        existing_memories = store.search(namespace, limit=20)
        existing_facts = [m.value["content"] for m in existing_memories]
        print(f"[Memory] 已有 {len(existing_facts)} 条记忆")
    except Exception as e:
        print(f"[Memory] 读取已有记忆出错: {e}")
        existing_facts = []

    # 用 LLM 提取需要记住的事实
    memory_extract_prompt = f"""从以下对话中提取需要长期记住的重要信息。
包括但不限于：
- 用户的名字、称呼
- 用户的喜好、偏好
- 日程安排、提醒事项
- 重要的约定或承诺

如果没有任何需要记住的信息，请输出"无"。
每个事实一行，不要编号，不要解释。

对话：
{conversation_text}

需要记住的事实："""
    try:
        memory_response = deepseek_llm.invoke([HumanMessage(content=memory_extract_prompt)])
        facts_text = memory_response.content.strip()
        print(f"[Memory] LLM 提取结果: {facts_text}")
    except Exception as e:
        print(f"[Memory] LLM 提取记忆出错: {e}")
        return {}

    # 过滤掉"无信息"类的垃圾数据
    no_info_patterns = [
        r'^无$',
        r'^无。$',
        r'^没有',
        r'^用户没有提供',
        r'^没有任何',
        r'^不需要记住',
        r'^没有需要记住',
        r'^没有新信息',
        r'^暂无',
    ]
    is_noise = False
    for pattern in no_info_patterns:
        if re.search(pattern, facts_text.strip()):
            is_noise = True
            break
    if is_noise:
        print(f"[Memory] ⏭️ 跳过无信息输出: {facts_text}")
        return {}

    if facts_text:
        # 本轮已处理的事实（用于避免同一轮中重复添加）
        processed_facts = set()

        for fact_line in facts_text.split('\n'):
            fact_line = fact_line.strip()
            if not fact_line:
                continue

            # 检查是否与本轮已处理的事实重复（避免同一轮中多次添加相同内容）
            if fact_line in processed_facts:
                print(f"[Memory] ⏭️ 跳过重复（本轮已处理）: {fact_line}")
                continue
            processed_facts.add(fact_line)

            # 语义去重：用一次 LLM 调用判断新事实是否与某条已有记忆重复
            # 注意：即使语义相同但细节不同（如时间从4点改成5点），也应该更新旧记录
            matched_memory = None
            if existing_facts:
                existing_list = "\n".join(f"{i+1}. {f}" for i, f in enumerate(existing_facts))
                dedup_prompt = f"""判断新信息是否与已有记忆中的某一条描述的是同一件事（语义相同或相关）。
注意：即使细节不同（如时间、地点等具体信息变了），只要说的是同一件事，就认为是相同。

例如：
- 已有记忆："明天下午4点开会"
- 新信息："明天下午5点开会"
→ 同一件事（会议安排），只是时间改了 → 输出对应序号

- 已有记忆："用户的名字叫张三"
- 新信息："用户的名字叫李四"
→ 同一件事（用户名字），只是内容变了 → 输出对应序号

- 已有记忆："明天下午4点开会"
- 新信息："用户喜欢喝咖啡"
→ 不同的事 → 输出"无"

已有记忆：
{existing_list}

新信息：{fact_line}

输出（只输出序号或"无"）："""
                try:
                    dedup_response = deepseek_llm.invoke([HumanMessage(content=dedup_prompt)])
                    dedup_result = dedup_response.content.strip()
                    if dedup_result.isdigit():
                        idx = int(dedup_result) - 1
                        if 0 <= idx < len(existing_memories):
                            matched_memory = existing_memories[idx]
                except Exception:
                    pass

            if matched_memory is not None:
                # 更新已有记忆（保留原 key，更新 value）
                try:
                    store.put(namespace, key=matched_memory.key, value={
                        "content": fact_line,
                        "user_id": user_id
                    })
                    print(f"[Memory] 🔄 更新了记忆: {fact_line}")
                except Exception as e:
                    print(f"[Memory] 更新记忆出错: {e}")
            else:
                # 没有重复，新增记忆
                try:
                    store.put(namespace, key=f"fact_{int(time.time())}", value={
                        "content": fact_line,
                        "user_id": user_id
                    })
                    print(f"[Memory] 💾 记住了: {fact_line}")
                except Exception as e:
                    print(f"[Memory] 写入记忆出错: {e}")
    else:
        print("[Memory] 没有需要记住的新信息")

    return {}


# ─── 构建图 ──────────────────────────────────────────────

builder = StateGraph(MultiAgentState)

# 添加节点
builder.add_node("planner", planer_node)
builder.add_node("supervisor", supervisor_node)
builder.add_node("memory", memory_node)
builder.add_node("personal_assistant", make_agent_node(personal_agent))
builder.add_node("note_agent", make_agent_node(note_agent))
builder.add_node("fina_agent", make_agent_node(fina_agent))
builder.add_node("weather_agent", make_agent_node(weather_agent))

# planner → supervisor
builder.add_edge("planner", "supervisor")

# 所有 Agent 完成后 → memory（提取并保存记忆）→ supervisor（处理下一个 batch 或结束）
for agent_name in ["personal_assistant", "note_agent", "fina_agent", "weather_agent"]:
    builder.add_edge(agent_name, "memory")

builder.add_edge("memory", "supervisor")

builder.set_entry_point("planner")

# 导出未编译的图，让调用方自行编译（可注入 checkpointer、store 等）
workflow = builder
