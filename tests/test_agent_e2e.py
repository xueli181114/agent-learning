"""
Agent-E2E 测试套件

安装: pip install agent-e2e
运行: pytest tests/test_agent_e2e.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

# ─── 辅助函数 ─────────────────────────────────────────────

def create_mock_store():
    """创建一个模拟的 PostgreSQL Store"""
    class MockMemory:
        def __init__(self, key, value, updated_at=None):
            self.key = key
            self.value = value
            self.updated_at = updated_at or 0

    class MockStore:
        def __init__(self):
            self.data = {}  # namespace -> {key: value}

        def search(self, namespace, limit=20):
            items = self.data.get(namespace, {})
            return [MockMemory(k, v) for k, v in items.items()][:limit]

        def put(self, namespace, key, value):
            if namespace not in self.data:
                self.data[namespace] = {}
            self.data[namespace][key] = value

        def get(self, namespace, key):
            return self.data.get(namespace, {}).get(key)

    return MockStore()


def create_mock_llm(responses):
    """
    创建一个模拟的 LLM，按顺序返回预设响应。
    
    用法:
        mock_llm = create_mock_llm([
            "1. 删除笔记 test.md\n2. 查看天气",  # planner decompose
            "删除笔记 test.md，查看天气",          # planner grouping
            "note_agent",                          # supervisor routing
        ])
    """
    response_iter = iter(responses)

    def mock_invoke(messages):
        try:
            content = next(response_iter)
        except StopIteration:
            content = "处理完成。"
        return AIMessage(content=content)

    return mock_invoke


# ─── Unit Tests: Planner ─────────────────────────────────

class TestPlannerDecompose:
    """Planner 任务拆解测试"""

    def test_decompose_multi_task(self):
        """多任务拆解：删除笔记 + 查看天气"""
        from multi_agent import planer_node

        mock_llm = create_mock_llm([
            "1. 删除笔记 test.md\n2. 查看天气",
            "删除笔记 test.md，查看天气",
        ])

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(side_effect=mock_llm)
            state = {
                "messages": [HumanMessage(content="删除笔记 test.md，并查看天气。")],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = planer_node(state, config)

            assert len(result["pending_batches"]) == 1
            assert "删除笔记 test.md" in result["pending_batches"][0]
            assert "查看天气" in result["pending_batches"][0]

    def test_decompose_self_intro_no_prefix(self):
        """自我介绍不应添加'记录笔记'前缀"""
        from multi_agent import planer_node

        mock_llm = create_mock_llm([
            "1. 我叫大头儿子",
            "我叫大头儿子",
        ])

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(side_effect=mock_llm)
            state = {
                "messages": [HumanMessage(content="我叫大头儿子")],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = planer_node(state, config)

            tasks_text = str(result["pending_batches"])
            assert "记录笔记" not in tasks_text
            assert "大头儿子" in tasks_text

    def test_decompose_question_not_split(self):
        """完整问句不应拆分"""
        from multi_agent import planer_node

        mock_llm = create_mock_llm([
            "1. 我明天有几个会议，是什么会议？",
            "我明天有几个会议，是什么会议？",
        ])

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(side_effect=mock_llm)
            state = {
                "messages": [HumanMessage(content="我明天有几个会议，是什么会议？")],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = planer_node(state, config)

            assert len(result["pending_batches"]) == 1
            assert len(result["pending_batches"][0]) == 1  # 应该只有一个任务
            assert "我明天有几个会议" in result["pending_batches"][0][0]

    def test_decompose_meeting_conflict_not_split(self):
        """会议冲突场景不应拆分"""
        from multi_agent import planer_node

        mock_llm = create_mock_llm([
            "1. 5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议",
            "5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议",
        ])

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(side_effect=mock_llm)
            state = {
                "messages": [HumanMessage(content="我的5点钟被schedule了另一个会议来讨论金融问题，这两个会议冲突了，我需要把5点钟的研发研讨会议取消掉")],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = planer_node(state, config)

            assert len(result["pending_batches"]) == 1
            assert len(result["pending_batches"][0]) == 1  # 应该只有一个任务
            assert "金融会议" in result["pending_batches"][0][0]
            assert "取消" in result["pending_batches"][0][0]


class TestPlannerGrouping:
    """Planner 并行分组测试"""

    def test_grouping_independent_tasks(self):
        """无依赖任务应放在同一 batch"""
        from multi_agent import planer_node

        mock_llm = create_mock_llm([
            "1. 删除笔记 test.md\n2. 查看天气",
            "删除笔记 test.md，查看天气",
        ])

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(side_effect=mock_llm)
            state = {
                "messages": [HumanMessage(content="删除笔记 test.md，查看天气")],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = planer_node(state, config)

            assert len(result["pending_batches"]) == 1  # 一个 batch
            assert len(result["pending_batches"][0]) == 2  # 两个任务

    def test_grouping_dependent_tasks(self):
        """有依赖任务应分到不同 batch"""
        from multi_agent import planer_node

        mock_llm = create_mock_llm([
            "1. 获取腾讯财务数据\n2. 根据数据生成报告",
            "获取腾讯财务数据\n根据数据生成报告",
        ])

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(side_effect=mock_llm)
            state = {
                "messages": [HumanMessage(content="获取腾讯财务数据，根据数据生成报告")],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = planer_node(state, config)

            assert len(result["pending_batches"]) == 2  # 两个 batch


# ─── Unit Tests: Supervisor ──────────────────────────────

class TestSupervisorRouting:
    """Supervisor 路由测试"""

    def test_route_self_intro_to_personal(self):
        """自我介绍 → personal_assistant"""
        from multi_agent import supervisor_node

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(return_value=AIMessage(content="personal_assistant"))
            state = {
                "pending_batches": [["我叫Alice"]],
                "current_batch_index": 0,
                "messages": [],
            }
            config = {"configurable": {"user_id": "test"}}
            result = supervisor_node(state, config)

            assert isinstance(result, Command)
            assert len(result.goto) == 1
            assert result.goto[0].node == "personal_assistant"

    def test_route_meeting_to_note(self):
        """会议安排 → note_agent"""
        from multi_agent import supervisor_node

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(return_value=AIMessage(content="note_agent"))
            state = {
                "pending_batches": [["提醒我明天下午四点开会"]],
                "current_batch_index": 0,
                "messages": [],
            }
            config = {"configurable": {"user_id": "test"}}
            result = supervisor_node(state, config)

            assert isinstance(result, Command)
            assert result.goto[0].node == "note_agent"

    def test_route_finance_meeting_to_note_not_fina(self):
        """金融会议 → note_agent（不是 fina_agent）"""
        from multi_agent import supervisor_node

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(return_value=AIMessage(content="note_agent"))
            state = {
                "pending_batches": [["5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议"]],
                "current_batch_index": 0,
                "messages": [],
            }
            config = {"configurable": {"user_id": "test"}}
            result = supervisor_node(state, config)

            assert isinstance(result, Command)
            assert result.goto[0].node == "note_agent"

    def test_route_weather_to_weather(self):
        """天气查询 → weather_agent"""
        from multi_agent import supervisor_node

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(return_value=AIMessage(content="weather_agent"))
            state = {
                "pending_batches": [["今天北京天气怎么样？"]],
                "current_batch_index": 0,
                "messages": [],
            }
            config = {"configurable": {"user_id": "test"}}
            result = supervisor_node(state, config)

            assert isinstance(result, Command)
            assert result.goto[0].node == "weather_agent"

    def test_route_finance_to_fina(self):
        """财务分析 → fina_agent"""
        from multi_agent import supervisor_node

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(return_value=AIMessage(content="fina_agent"))
            state = {
                "pending_batches": [["帮我分析腾讯的财务数据"]],
                "current_batch_index": 0,
                "messages": [],
            }
            config = {"configurable": {"user_id": "test"}}
            result = supervisor_node(state, config)

            assert isinstance(result, Command)
            assert result.goto[0].node == "fina_agent"

    def test_parallel_routing(self):
        """并行路由：多个任务同时分发"""
        from multi_agent import supervisor_node

        mock_responses = iter(["personal_assistant", "weather_agent"])

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(side_effect=lambda msgs: AIMessage(content=next(mock_responses)))
            state = {
                "pending_batches": [["我叫Alice", "查看天气"]],
                "current_batch_index": 0,
                "messages": [],
            }
            config = {"configurable": {"user_id": "test"}}
            result = supervisor_node(state, config)

            assert isinstance(result, Command)
            assert len(result.goto) == 2  # 两个并行任务
            assert result.goto[0].node == "personal_assistant"
            assert result.goto[1].node == "weather_agent"


# ─── Unit Tests: Memory ──────────────────────────────────

class TestMemoryNode:
    """Memory 节点测试"""

    def test_memory_extract_name(self):
        """提取用户名字"""
        from multi_agent import memory_node
        from agents import get_store

        mock_store = create_mock_store()

        with patch('multi_agent.deepseek_llm') as mock_llm, \
             patch('agents.get_store', return_value=mock_store):
            mock_llm.invoke = MagicMock(return_value=AIMessage(content="用户的名字是Alice"))
            state = {
                "messages": [
                    HumanMessage(content="My name is Alice"),
                    AIMessage(content="Hello Alice!"),
                ],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = memory_node(state, config)

            # 验证 store 中有了记忆
            memories = mock_store.search(("user_memories", "test"))
            assert len(memories) > 0
            assert "Alice" in memories[0].value["content"]

    def test_memory_skip_no_info(self):
        """无信息时应跳过"""
        from multi_agent import memory_node

        mock_store = create_mock_store()

        with patch('multi_agent.deepseek_llm') as mock_llm, \
             patch('agents.get_store', return_value=mock_store):
            mock_llm.invoke = MagicMock(return_value=AIMessage(content="无"))
            state = {
                "messages": [
                    HumanMessage(content="今天天气怎么样？"),
                    AIMessage(content="今天北京天气cloudy，25°C"),
                ],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = memory_node(state, config)

            # 验证 store 中没有记忆
            memories = mock_store.search(("user_memories", "test"))
            assert len(memories) == 0

    def test_memory_extract_preferences(self):
        """提取用户喜好"""
        from multi_agent import memory_node

        mock_store = create_mock_store()

        with patch('multi_agent.deepseek_llm') as mock_llm, \
             patch('agents.get_store', return_value=mock_store):
            mock_llm.invoke = MagicMock(return_value=AIMessage(content="用户喜欢游泳"))
            state = {
                "messages": [
                    HumanMessage(content="我喜欢游泳"),
                    AIMessage(content="游泳是很好的运动！"),
                ],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = memory_node(state, config)

            memories = mock_store.search(("user_memories", "test"))
            assert len(memories) > 0
            assert "游泳" in memories[0].value["content"]

    def test_memory_extract_meeting(self):
        """提取会议安排"""
        from multi_agent import memory_node

        mock_store = create_mock_store()

        with patch('multi_agent.deepseek_llm') as mock_llm, \
             patch('agents.get_store', return_value=mock_store):
            mock_llm.invoke = MagicMock(return_value=AIMessage(content="明天下午四点开研发研讨会"))
            state = {
                "messages": [
                    HumanMessage(content="提醒我明天下午四点开研发研讨会"),
                    AIMessage(content="好的，已记录"),
                ],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = memory_node(state, config)

            memories = mock_store.search(("user_memories", "test"))
            assert len(memories) > 0
            assert "研发研讨会" in memories[0].value["content"]


# ─── Unit Tests: Tools ───────────────────────────────────

class TestNoteTools:
    """笔记工具函数测试"""

    def test_write_and_read_note(self, tmp_path):
        """写入并读取笔记"""
        import sys
        import os
        # 临时修改 notes 目录
        original_notes = os.path.join(os.path.dirname(__file__), "..", "notes")
        test_notes = tmp_path / "notes"
        test_notes.mkdir()

        # 使用 monkeypatch 模拟 os.makedirs 和文件操作
        from tools import tools_note

        # 直接测试文件操作
        result = tools_note.write_note.invoke({"filename": "test.md", "content": "测试内容"})
        assert "笔记已保存" in result

        result = tools_note.read_note.invoke({"filename": "test.md"})
        assert "测试内容" in result

    def test_list_notes_empty(self):
        """空笔记列表"""
        from tools import tools_note
        result = tools_note.list_notes.invoke({})
        assert "暂无笔记" in result or "笔记列表" in result

    def test_delete_note(self):
        """删除笔记"""
        from tools import tools_note
        # 先创建
        tools_note.write_note.invoke({"filename": "delete_test.md", "content": "待删除"})
        # 再删除
        result = tools_note.delete_note.invoke({"filename": "delete_test.md"})
        assert "已删除" in result


# ─── Integration Tests ───────────────────────────────────

class TestIntegration:
    """集成测试：组件协作"""

    def test_planner_to_supervisor_flow(self):
        """Planner → Supervisor 完整链路"""
        from multi_agent import planer_node, supervisor_node

        mock_llm = create_mock_llm([
            "1. 查看天气",
            "查看天气",
            "weather_agent",
        ])

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(side_effect=mock_llm)
            state = {
                "messages": [HumanMessage(content="查看天气")],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}

            # Planner
            plan_result = planer_node(state, config)
            assert len(plan_result["pending_batches"]) == 1

            # Supervisor
            sup_state = {**state, **plan_result}
            sup_result = supervisor_node(sup_state, config)
            assert isinstance(sup_result, Command)
            assert sup_result.goto[0].node == "weather_agent"

    def test_parallel_batch_execution(self):
        """并行 batch 执行"""
        from multi_agent import planer_node, supervisor_node

        mock_llm = create_mock_llm([
            "1. 查看天气\n2. 我叫Alice",
            "查看天气，我叫Alice",
            "weather_agent",
            "personal_assistant",
        ])

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(side_effect=mock_llm)
            state = {
                "messages": [HumanMessage(content="查看天气，我叫Alice")],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}

            # Planner
            plan_result = planer_node(state, config)
            assert len(plan_result["pending_batches"]) == 1
            assert len(plan_result["pending_batches"][0]) == 2

            # Supervisor - 应该分发两个并行任务
            sup_state = {**state, **plan_result}
            sup_result = supervisor_node(sup_state, config)
            assert isinstance(sup_result, Command)
            assert len(sup_result.goto) == 2


# ─── E2E Tests ───────────────────────────────────────────

class TestE2E:
    """端到端测试（需要真实 LLM 和数据库）"""

    @pytest.mark.skip(reason="需要真实 LLM 和 PostgreSQL，手动运行")
    def test_full_conversation(self):
        """完整对话流程"""
        from agent_memory_db import graph
        config = {"configurable": {"thread_id": "e2e_test", "user_id": "e2e_user"}}

        # Step 1: 自我介绍
        result = graph.invoke(
            {"messages": [HumanMessage(content="My name is Alice")]},
            config=config
        )
        assert "Alice" in result["messages"][-1].content

        # Step 2: 安排会议
        result = graph.invoke(
            {"messages": [HumanMessage(content="提醒我明天下午四点开研发研讨会")]},
            config=config
        )

        # Step 3: 查询会议
        result = graph.invoke(
            {"messages": [HumanMessage(content="我明天有几个会议，是什么会议？")]},
            config=config
        )
        assert "研发研讨会" in result["messages"][-1].content

    @pytest.mark.skip(reason="需要真实 LLM 和 PostgreSQL，手动运行")
    def test_meeting_conflict(self):
        """会议冲突场景"""
        from agent_memory_db import graph
        config = {"configurable": {"thread_id": "e2e_conflict", "user_id": "e2e_user"}}

        # Step 1: 安排研发研讨会
        graph.invoke(
            {"messages": [HumanMessage(content="提醒我明天下午四点开研发研讨会")]},
            config=config
        )

        # Step 2: 会议冲突
        result = graph.invoke(
            {"messages": [HumanMessage(content="我的5点钟被schedule了另一个会议来讨论金融问题，这两个会议冲突了，我需要把5点钟的研发研讨会议取消掉")]},
            config=config
        )

        # Step 3: 查询剩余会议
        result = graph.invoke(
            {"messages": [HumanMessage(content="我明天有几个会议，是什么会议？")]},
            config=config
        )
        # 应该提到金融会议，研发研讨会应该已被取消
        assert "金融" in result["messages"][-1].content

    @pytest.mark.skip(reason="需要真实 LLM 和 PostgreSQL，手动运行")
    def test_memory_persistence(self):
        """记忆持久化：跨会话记住用户信息"""
        from agent_memory_db import graph

        # 会话1：告诉名字
        config1 = {"configurable": {"thread_id": "session1", "user_id": "persist_user"}}
        graph.invoke(
            {"messages": [HumanMessage(content="My name is Bob")]},
            config=config1
        )

        # 会话2：新 thread_id，同一个 user_id，问名字
        config2 = {"configurable": {"thread_id": "session2", "user_id": "persist_user"}}
        result = graph.invoke(
            {"messages": [HumanMessage(content="What is my name?")]},
            config=config2
        )
        assert "Bob" in result["messages"][-1].content
