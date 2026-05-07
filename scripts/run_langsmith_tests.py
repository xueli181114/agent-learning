"""
LangSmith 回归测试运行器

在 LangSmith 数据集上运行回归测试，追踪每次运行的结果。

用法:
    python scripts/run_langsmith_tests.py

前置条件:
    export LANGCHAIN_API_KEY="ls_..."
    export LANGCHAIN_TRACING_V2="true"
    python scripts/seed_langsmith_datasets.py  # 先初始化数据集
"""

import sys
import os

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langsmith import Client, evaluate
from langchain_core.messages import HumanMessage
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage


# ─── 预测函数 ────────────────────────────────────────────

def predict_planner_decompose(inputs: dict) -> dict:
    """运行 Planner 拆解并返回结果"""
    from multi_agent import planer_node

    # 模拟 LLM 响应（实际运行时会被真实 LLM 覆盖）
    user_input = inputs.get("user_input", "")

    state = {
        "messages": [HumanMessage(content=user_input)],
        "pending_batches": [],
        "current_batch_index": 0,
    }
    config = {"configurable": {"user_id": "langsmith_test"}}

    result = planer_node(state, config)
    return {"tasks": result.get("pending_batches", [])}


def predict_supervisor_routing(inputs: dict) -> dict:
    """运行 Supervisor 路由并返回结果"""
    from multi_agent import supervisor_node

    task = inputs.get("task", "")

    state = {
        "pending_batches": [[task]],
        "current_batch_index": 0,
        "messages": [],
    }
    config = {"configurable": {"user_id": "langsmith_test"}}

    result = supervisor_node(state, config)
    if hasattr(result, 'goto') and result.goto:
        agent = result.goto[0].node
    else:
        agent = "unknown"
    return {"agent": agent}


# ─── 运行回归测试 ────────────────────────────────────────

def run_planner_decompose_tests():
    """在 planner-decompose-tests 数据集上运行回归测试"""
    print("\n" + "=" * 50)
    print("运行 Planner 拆解回归测试")
    print("=" * 50)

    client = Client()

    try:
        results = evaluate(
            predict_planner_decompose,
            data="planner-decompose-tests",
            clients=[client],
            experiment_prefix="planner-decompose-regression",
            metadata={
                "test_type": "regression",
                "component": "planner",
                "sub_component": "decompose",
            },
        )
        print(f"✅ Planner 拆解回归测试完成")
        return results
    except Exception as e:
        print(f"❌ Planner 拆解回归测试失败: {e}")
        return None


def run_planner_grouping_tests():
    """在 planner-grouping-tests 数据集上运行回归测试"""
    print("\n" + "=" * 50)
    print("运行 Planner 分组回归测试")
    print("=" * 50)

    client = Client()

    def predict(inputs: dict) -> dict:
        from multi_agent import planer_node

        tasks = inputs.get("tasks", [])
        user_input = "，".join(tasks)

        state = {
            "messages": [HumanMessage(content=user_input)],
            "pending_batches": [],
            "current_batch_index": 0,
        }
        config = {"configurable": {"user_id": "langsmith_test"}}

        result = planer_node(state, config)
        return {"batches": result.get("pending_batches", [])}

    try:
        results = evaluate(
            predict,
            data="planner-grouping-tests",
            clients=[client],
            experiment_prefix="planner-grouping-regression",
            metadata={
                "test_type": "regression",
                "component": "planner",
                "sub_component": "grouping",
            },
        )
        print(f"✅ Planner 分组回归测试完成")
        return results
    except Exception as e:
        print(f"❌ Planner 分组回归测试失败: {e}")
        return None


def run_supervisor_routing_tests():
    """在 supervisor-routing-tests 数据集上运行回归测试"""
    print("\n" + "=" * 50)
    print("运行 Supervisor 路由回归测试")
    print("=" * 50)

    client = Client()

    try:
        results = evaluate(
            predict_supervisor_routing,
            data="supervisor-routing-tests",
            clients=[client],
            experiment_prefix="supervisor-routing-regression",
            metadata={
                "test_type": "regression",
                "component": "supervisor",
                "sub_component": "routing",
            },
        )
        print(f"✅ Supervisor 路由回归测试完成")
        return results
    except Exception as e:
        print(f"❌ Supervisor 路由回归测试失败: {e}")
        return None


def run_memory_extract_tests():
    """在 memory-extract-tests 数据集上运行回归测试"""
    print("\n" + "=" * 50)
    print("运行 Memory 提取回归测试")
    print("=" * 50)

    client = Client()

    def predict(inputs: dict) -> dict:
        from multi_agent import memory_node

        conversation = inputs.get("conversation", "")
        # 解析对话
        lines = conversation.strip().split("\n")
        messages = []
        for line in lines:
            if line.startswith("用户："):
                messages.append(HumanMessage(content=line[3:]))
            elif line.startswith("助手："):
                messages.append(AIMessage(content=line[3:]))

        state = {
            "messages": messages,
            "pending_batches": [],
            "current_batch_index": 0,
        }
        config = {"configurable": {"user_id": "langsmith_test"}}

        # 模拟 store
        mock_store = MagicMock()
        mock_store.search.return_value = []

        with patch('agents.get_store', return_value=mock_store):
            result = memory_node(state, config)

        return {"facts": result.get("facts", [])}

    try:
        results = evaluate(
            predict,
            data="memory-extract-tests",
            clients=[client],
            experiment_prefix="memory-extract-regression",
            metadata={
                "test_type": "regression",
                "component": "memory",
                "sub_component": "extract",
            },
        )
        print(f"✅ Memory 提取回归测试完成")
        return results
    except Exception as e:
        print(f"❌ Memory 提取回归测试失败: {e}")
        return None


# ─── 运行所有回归测试 ────────────────────────────────────

def run_all():
    print("=" * 50)
    print("LangSmith 回归测试套件")
    print("=" * 50)

    results = {}

    results["planner_decompose"] = run_planner_decompose_tests()
    results["planner_grouping"] = run_planner_grouping_tests()
    results["supervisor_routing"] = run_supervisor_routing_tests()
    results["memory_extract"] = run_memory_extract_tests()

    print("\n" + "=" * 50)
    print("回归测试汇总")
    print("=" * 50)
    for name, result in results.items():
        status = "✅" if result is not None else "❌"
        print(f"  {status} {name}")

    print("\n在 LangSmith UI 中查看详细结果: https://smith.langchain.com")


if __name__ == "__main__":
    run_all()
