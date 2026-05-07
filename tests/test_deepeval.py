"""
DeepEval 测试套件

安装: pip install deepeval
运行: deepeval test run tests/test_deepeval.py
  或: pytest tests/test_deepeval.py -v

DeepEval 指标说明：
- 结果导向（Result-Oriented）：验证 Agent 的最终输出是否正确
  - GEval: 用 LLM 评判输出质量
  - AnswerRelevancy: 输出是否与输入相关
  - Faithfulness: 输出是否基于上下文，没有幻觉
  - Hallucination: 输出是否包含幻觉内容

- 过程导向（Process-Oriented）：验证 Agent 的执行过程是否正确
  - ToolCallMetric: 验证工具调用是否正确
  - TaskCompletionMetric: 验证任务是否完成
  - ConversationalMetric: 验证多轮对话的连贯性
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command


# ═══════════════════════════════════════════════════════════
# 结果导向（Result-Oriented）测试
# ═══════════════════════════════════════════════════════════

class TestResultOriented:
    """结果导向测试：验证 Agent 的最终输出是否正确"""

    # ── Planner 结果测试 ──

    def test_planner_decompose_correctness(self):
        """
        结果导向：Planner 拆解结果是否正确
        
        验证：输入"删除笔记 test.md，查看天气"时，
        Planner 应该输出 2 个子任务
        """
        from multi_agent import planer_node

        mock_llm = MagicMock(side_effect=[
            AIMessage(content="1. 删除笔记 test.md\n2. 查看天气"),
            AIMessage(content="删除笔记 test.md，查看天气"),
        ])

        with patch('multi_agent.deepseek_llm', invoke=mock_llm):
            state = {
                "messages": [HumanMessage(content="删除笔记 test.md，并查看天气。")],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = planer_node(state, config)

            # 用 GEval 评判拆解结果
            test_case = LLMTestCase(
                input="删除笔记 test.md，并查看天气。",
                actual_output=str(result["pending_batches"]),
                expected_output="['删除笔记 test.md', '查看天气']",
            )
            metric = GEval(
                name="PlannerDecomposeCorrectness",
                criteria="判断 Planner 的任务拆解是否正确",
                evaluation_steps=[
                    "输出应该包含 2 个子任务",
                    "第一个子任务应该是'删除笔记 test.md'",
                    "第二个子任务应该是'查看天气'",
                ],
                model="gpt-4o",  # 用 GPT-4 作为评判模型
            )
            assert_test(test_case, [metric])

    def test_planner_keeps_question_intact(self):
        """
        结果导向：Planner 不应拆分完整问句
        
        验证：输入"我明天有几个会议，是什么会议？"时，
        Planner 应该保持为 1 个任务
        """
        from multi_agent import planer_node

        mock_llm = MagicMock(side_effect=[
            AIMessage(content="1. 我明天有几个会议，是什么会议？"),
            AIMessage(content="我明天有几个会议，是什么会议？"),
        ])

        with patch('multi_agent.deepseek_llm', invoke=mock_llm):
            state = {
                "messages": [HumanMessage(content="我明天有几个会议，是什么会议？")],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = planer_node(state, config)

            test_case = LLMTestCase(
                input="我明天有几个会议，是什么会议？",
                actual_output=str(result["pending_batches"]),
                expected_output="1 个任务，包含完整问句",
            )
            metric = GEval(
                name="PlannerNoSplitQuestion",
                criteria="判断 Planner 是否将完整问句保持为单个任务",
                evaluation_steps=[
                    "输出应该只有 1 个 batch",
                    "该 batch 中应该只有 1 个任务",
                    "任务内容应该是完整的'我明天有几个会议，是什么会议？'",
                    "不应该被拆分成'我明天有几个会议'和'是什么会议？'两个独立任务",
                ],
                model="gpt-4o",
            )
            assert_test(test_case, [metric])

    def test_planner_meeting_conflict_as_whole(self):
        """
        结果导向：会议冲突场景应保持为整体
        
        验证：输入会议冲突请求时，Planner 不应拆分成碎片
        """
        from multi_agent import planer_node

        mock_llm = MagicMock(side_effect=[
            AIMessage(content="1. 5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议"),
            AIMessage(content="5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议"),
        ])

        with patch('multi_agent.deepseek_llm', invoke=mock_llm):
            state = {
                "messages": [HumanMessage(content="我的5点钟被schedule了另一个会议来讨论金融问题，这两个会议冲突了，我需要把5点钟的研发研讨会议取消掉")],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = planer_node(state, config)

            test_case = LLMTestCase(
                input="会议冲突请求",
                actual_output=str(result["pending_batches"]),
            )
            metric = GEval(
                name="PlannerMeetingConflictAsWhole",
                criteria="判断 Planner 是否将会议冲突场景保持为整体",
                evaluation_steps=[
                    "输出应该只有 1 个 batch",
                    "该 batch 中应该只有 1 个任务",
                    "任务应该同时包含'金融会议'和'取消'和'研发研讨会议'等关键词",
                    "不应该被拆分成多个碎片",
                ],
                model="gpt-4o",
            )
            assert_test(test_case, [metric])

    # ── Supervisor 路由结果测试 ──

    def test_supervisor_routes_finance_meeting_to_note(self):
        """
        结果导向：金融会议应路由到 note_agent
        
        验证：Supervisor 正确区分"金融会议"（日程）和"金融分析"（财务）
        """
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

            test_case = LLMTestCase(
                input="5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议",
                actual_output=result.goto[0].node if hasattr(result, 'goto') else str(result),
                expected_output="note_agent",
            )
            metric = GEval(
                name="SupervisorRoutingFinanceMeeting",
                criteria="判断 Supervisor 是否将金融会议路由到 note_agent",
                evaluation_steps=[
                    "金融会议是日程安排，不是财务分析",
                    "输出应该是 note_agent，不是 fina_agent",
                ],
                model="gpt-4o",
            )
            assert_test(test_case, [metric])

    def test_supervisor_routes_finance_analysis_to_fina(self):
        """
        结果导向：财务分析应路由到 fina_agent
        
        验证：Supervisor 正确路由财务分析请求
        """
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

            test_case = LLMTestCase(
                input="帮我分析腾讯的财务数据",
                actual_output=result.goto[0].node if hasattr(result, 'goto') else str(result),
                expected_output="fina_agent",
            )
            metric = GEval(
                name="SupervisorRoutingFinanceAnalysis",
                criteria="判断 Supervisor 是否将财务分析路由到 fina_agent",
                evaluation_steps=["输出应该是 fina_agent"],
                model="gpt-4o",
            )
            assert_test(test_case, [metric])

    # ── Memory 提取结果测试 ──

    def test_memory_extract_relevancy(self):
        """
        结果导向：Memory 提取的内容应与对话相关
        
        验证：从"我叫Alice"中提取的记忆应该包含名字信息
        """
        from multi_agent import memory_node

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
            memory_node(state, config)

            memories = mock_store.search(("user_memories", "test"))
            stored_content = memories[0].value["content"] if memories else ""

            test_case = LLMTestCase(
                input="My name is Alice",
                actual_output=stored_content,
                retrieval_context=["用户说自己的名字是Alice"],
            )
            metric = AnswerRelevancyMetric(
                threshold=0.7,
                model="gpt-4o",
            )
            assert_test(test_case, [metric])

    def test_memory_extract_faithfulness(self):
        """
        结果导向：Memory 提取的内容应忠实于对话
        
        验证：提取的记忆不应包含对话中没有的信息（无幻觉）
        """
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
            memory_node(state, config)

            memories = mock_store.search(("user_memories", "test"))
            stored_content = memories[0].value["content"] if memories else ""

            test_case = LLMTestCase(
                input="我喜欢游泳",
                actual_output=stored_content,
                retrieval_context=["用户说喜欢游泳"],
            )
            metric = FaithfulnessMetric(
                threshold=0.7,
                model="gpt-4o",
            )
            assert_test(test_case, [metric])

    def test_memory_no_hallucination(self):
        """
        结果导向：Memory 不应产生幻觉
        
        验证：当对话中没有个人信息时，不应提取出虚假信息
        """
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
            memory_node(state, config)

            memories = mock_store.search(("user_memories", "test"))

            test_case = LLMTestCase(
                input="今天天气怎么样？",
                actual_output=str([m.value["content"] for m in memories]),
                retrieval_context=["对话中没有个人信息"],
            )
            metric = HallucinationMetric(
                threshold=0.5,
                model="gpt-4o",
            )
            assert_test(test_case, [metric])


# ═══════════════════════════════════════════════════════════
# 过程导向（Process-Oriented）测试
# ═══════════════════════════════════════════════════════════

class TestProcessOriented:
    """过程导向测试：验证 Agent 的执行过程是否正确"""

    # ── Planner 过程测试 ──

    def test_planner_process_correct_steps(self):
        """
        过程导向：Planner 的执行步骤是否正确
        
        验证：Planner 先拆解任务，再分组，步骤顺序正确
        """
        from multi_agent import planer_node

        mock_llm = MagicMock(side_effect=[
            AIMessage(content="1. 删除笔记 test.md\n2. 查看天气"),
            AIMessage(content="删除笔记 test.md，查看天气"),
        ])

        with patch('multi_agent.deepseek_llm', invoke=mock_llm):
            state = {
                "messages": [HumanMessage(content="删除笔记 test.md，并查看天气。")],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = planer_node(state, config)

            # 验证过程：拆解 → 分组
            test_case = LLMTestCase(
                input="删除笔记 test.md，并查看天气。",
                actual_output=str(result),
            )
            metric = GEval(
                name="PlannerProcessCorrectness",
                criteria="判断 Planner 的执行过程是否正确",
                evaluation_steps=[
                    "输出应该包含 pending_batches 字段",
                    "pending_batches 应该是一个列表",
                    "pending_batches 中的每个元素应该是一个任务列表",
                    "current_batch_index 应该被重置为 0",
                ],
                model="gpt-4o",
            )
            assert_test(test_case, [metric])

    def test_planner_process_no_extra_prefix(self):
        """
        过程导向：Planner 不应添加额外前缀
        
        验证：自我介绍不应被包装成"记录笔记"
        """
        from multi_agent import planer_node

        mock_llm = MagicMock(side_effect=[
            AIMessage(content="1. 我叫大头儿子"),
            AIMessage(content="我叫大头儿子"),
        ])

        with patch('multi_agent.deepseek_llm', invoke=mock_llm):
            state = {
                "messages": [HumanMessage(content="我叫大头儿子")],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = planer_node(state, config)

            test_case = LLMTestCase(
                input="我叫大头儿子",
                actual_output=str(result["pending_batches"]),
            )
            metric = GEval(
                name="PlannerProcessNoExtraPrefix",
                criteria="判断 Planner 是否添加了额外前缀",
                evaluation_steps=[
                    "输出中不应该包含'记录笔记'",
                    "输出应该直接包含'我叫大头儿子'",
                ],
                model="gpt-4o",
            )
            assert_test(test_case, [metric])

    # ── Supervisor 过程测试 ──

    def test_supervisor_process_parallel_dispatch(self):
        """
        过程导向：Supervisor 应并行分发独立任务
        
        验证：多个独立任务应通过 Send() 并行分发
        """
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

            test_case = LLMTestCase(
                input="我叫Alice，查看天气",
                actual_output=f"并行分发 {len(result.goto)} 个任务: {[g.node for g in result.goto]}",
            )
            metric = GEval(
                name="SupervisorProcessParallelDispatch",
                criteria="判断 Supervisor 是否正确并行分发任务",
                evaluation_steps=[
                    "应该通过 Send() 分发任务",
                    "分发数量应该等于任务数量",
                    "每个任务应该路由到正确的 Agent",
                ],
                model="gpt-4o",
            )
            assert_test(test_case, [metric])

    def test_supervisor_process_batch_index_update(self):
        """
        过程导向：Supervisor 应正确更新 batch index
        
        验证：处理完一个 batch 后，current_batch_index 应递增
        """
        from multi_agent import supervisor_node

        with patch('multi_agent.deepseek_llm') as mock:
            mock.invoke = MagicMock(return_value=AIMessage(content="note_agent"))
            state = {
                "pending_batches": [["提醒我明天下午四点开会"], ["查看天气"]],
                "current_batch_index": 0,
                "messages": [],
            }
            config = {"configurable": {"user_id": "test"}}
            result = supervisor_node(state, config)

            assert result.update["current_batch_index"] == 1

    # ── Memory 过程测试 ──

    def test_memory_process_dedup(self):
        """
        过程导向：Memory 应进行语义去重
        
        验证：相同语义的信息不应重复存储
        """
        from multi_agent import memory_node

        mock_store = create_mock_store()
        # 先存入一条记忆
        mock_store.put(("user_memories", "test"), "fact_1", {
            "content": "明天下午四点开会",
            "user_id": "test"
        })

        # 模拟 LLM 返回相同语义但不同表述的信息
        with patch('multi_agent.deepseek_llm') as mock_llm, \
             patch('agents.get_store', return_value=mock_store):
            # 第一次调用：提取记忆
            # 第二次调用：语义去重（判断是否与已有记忆重复）
            mock_llm.invoke = MagicMock(side_effect=[
                AIMessage(content="明天下午4点开会"),
                AIMessage(content="1"),  # 与第1条已有记忆重复
            ])
            state = {
                "messages": [
                    HumanMessage(content="会议改到下午4点"),
                    AIMessage(content="好的，已更新"),
                ],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            memory_node(state, config)

            # 验证：仍然只有 1 条记忆（更新了，没有新增）
            memories = mock_store.search(("user_memories", "test"))
            assert len(memories) == 1

    def test_memory_process_noise_filter(self):
        """
        过程导向：Memory 应过滤垃圾数据
        
        验证："无"、"没有信息"等输出不应存入 store
        """
        from multi_agent import memory_node

        mock_store = create_mock_store()

        with patch('multi_agent.deepseek_llm') as mock_llm, \
             patch('agents.get_store', return_value=mock_store):
            mock_llm.invoke = MagicMock(return_value=AIMessage(content="无"))
            state = {
                "messages": [
                    HumanMessage(content="你好"),
                    AIMessage(content="你好！"),
                ],
                "pending_batches": [],
                "current_batch_index": 0,
            }
            config = {"configurable": {"user_id": "test"}}
            result = memory_node(state, config)

            memories = mock_store.search(("user_memories", "test"))
            assert len(memories) == 0  # 不应存入任何记忆

    # ── 工具调用过程测试 ──

    def test_note_agent_tool_call_process(self):
        """
        过程导向：note_agent 应正确调用工具
        
        验证：写笔记时调用 write_note，读笔记时调用 read_note
        """
        from tools import tools_note

        # 验证 write_note 工具存在且可调用
        assert hasattr(tools_note.write_note, 'invoke')
        result = tools_note.write_note.invoke({"filename": "process_test.md", "content": "过程测试"})
        assert "笔记已保存" in result

        # 验证 read_note 工具存在且可调用
        result = tools_note.read_note.invoke({"filename": "process_test.md"})
        assert "过程测试" in result

        # 验证 list_notes 工具存在且可调用
        result = tools_note.list_notes.invoke({})
        assert "process_test.md" in result

        # 清理
        tools_note.delete_note.invoke({"filename": "process_test.md"})

    def test_weather_agent_tool_call_process(self):
        """
        过程导向：weather_agent 应正确调用天气工具
        
        验证：get_weather 工具返回正确的格式
        """
        from tools.tools_weather import get_weather

        result = get_weather.invoke({"city": "北京"})
        assert "北京" in result
        assert "cloudy" in result or "°C" in result

    def test_finance_agent_tool_call_process(self):
        """
        过程导向：fina_agent 应正确调用财务工具
        
        验证：fetch_financial_data 工具返回正确的格式
        """
        from tools.tools_financial_data import fetch_financial_data

        result = fetch_financial_data.invoke({"company": "腾讯"})
        assert "腾讯" in result
        assert "营收" in result


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def create_mock_store():
    """创建一个模拟的 PostgreSQL Store"""
    class MockMemory:
        def __init__(self, key, value, updated_at=None):
            self.key = key
            self.value = value
            self.updated_at = updated_at or 0

    class MockStore:
        def __init__(self):
            self.data = {}

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
