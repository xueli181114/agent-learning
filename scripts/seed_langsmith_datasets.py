"""
LangSmith 数据集初始化脚本

将测试用例导入 LangSmith 数据集，用于回归测试。

用法:
    python scripts/seed_langsmith_datasets.py

前置条件:
    export LANGCHAIN_API_KEY="ls_..."
    export LANGCHAIN_TRACING_V2="true"
"""

import sys
import os

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langsmith_config import create_dataset, add_examples


def seed_planner_decompose_dataset():
    """Planner 拆解测试数据集"""
    dataset_name = "planner-decompose-tests"
    create_dataset(dataset_name, "Planner 任务拆解测试用例")

    examples = [
        # 多任务拆解
        {
            "inputs": {"user_input": "删除笔记 test.md，并查看天气。"},
            "outputs": {"tasks": ["删除笔记 test.md", "查看天气"]},
        },
        {
            "inputs": {"user_input": "提醒我明天下午四点开会并记录笔记"},
            "outputs": {"tasks": ["提醒我明天下午四点开会", "记录笔记：明天下午四点开会"]},
        },
        # 自我介绍 - 不应加"记录笔记"前缀
        {
            "inputs": {"user_input": "我叫大头儿子"},
            "outputs": {"tasks": ["我叫大头儿子"]},
        },
        {
            "inputs": {"user_input": "我喜欢游泳"},
            "outputs": {"tasks": ["我喜欢游泳"]},
        },
        # 完整场景 - 不应拆分
        {
            "inputs": {"user_input": "我的5点钟被schedule了另一个会议来讨论金融问题，这两个会议冲突了，我需要把5点钟的研发研讨会议取消掉"},
            "outputs": {"tasks": ["5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议"]},
        },
        # 问句 - 不应拆分
        {
            "inputs": {"user_input": "我明天有几个会议，是什么主题的？"},
            "outputs": {"tasks": ["我明天有几个会议，是什么主题的？"]},
        },
        {
            "inputs": {"user_input": "我明天有几个会议，是什么会议？"},
            "outputs": {"tasks": ["我明天有几个会议，是什么会议？"]},
        },
        # 英文输入
        {
            "inputs": {"user_input": "My name is Alice"},
            "outputs": {"tasks": ["My name is Alice"]},
        },
        {
            "inputs": {"user_input": "I like banana"},
            "outputs": {"tasks": ["I like banana"]},
        },
        # 单任务
        {
            "inputs": {"user_input": "今天北京天气怎么样？"},
            "outputs": {"tasks": ["今天北京天气怎么样？"]},
        },
    ]

    add_examples(dataset_name, examples)
    print(f"✅ 数据集 '{dataset_name}' 已初始化，共 {len(examples)} 个用例")


def seed_planner_grouping_dataset():
    """Planner 分组测试数据集"""
    dataset_name = "planner-grouping-tests"
    create_dataset(dataset_name, "Planner 并行分组测试用例")

    examples = [
        # 无依赖 - 可并行
        {
            "inputs": {"tasks": ["删除笔记 test.md", "查看天气"]},
            "outputs": {"batches": [["删除笔记 test.md", "查看天气"]]},
        },
        {
            "inputs": {"tasks": ["查看天气", "分析腾讯财务数据"]},
            "outputs": {"batches": [["查看天气", "分析腾讯财务数据"]]},
        },
        # 有依赖 - 必须串行
        {
            "inputs": {"tasks": ["获取腾讯财务数据", "根据数据生成报告"]},
            "outputs": {"batches": [["获取腾讯财务数据"], ["根据数据生成报告"]]},
        },
        # 单个完整场景
        {
            "inputs": {"tasks": ["5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议"]},
            "outputs": {"batches": [["5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议"]]},
        },
        # 完整问句
        {
            "inputs": {"tasks": ["我明天有几个会议，是什么会议？"]},
            "outputs": {"batches": [["我明天有几个会议，是什么会议？"]]},
        },
        # 混合
        {
            "inputs": {"tasks": ["我叫Alice", "查看天气"]},
            "outputs": {"batches": [["我叫Alice", "查看天气"]]},
        },
    ]

    add_examples(dataset_name, examples)
    print(f"✅ 数据集 '{dataset_name}' 已初始化，共 {len(examples)} 个用例")


def seed_supervisor_routing_dataset():
    """Supervisor 路由测试数据集"""
    dataset_name = "supervisor-routing-tests"
    create_dataset(dataset_name, "Supervisor Agent 路由测试用例")

    examples = [
        # personal_assistant
        {"inputs": {"task": "我叫Alice"}, "outputs": {"agent": "personal_assistant"}},
        {"inputs": {"task": "我喜欢游泳"}, "outputs": {"agent": "personal_assistant"}},
        {"inputs": {"task": "My name is Alice"}, "outputs": {"agent": "personal_assistant"}},
        # note_agent
        {"inputs": {"task": "帮我记一条笔记：明天下午三点开会"}, "outputs": {"agent": "note_agent"}},
        {"inputs": {"task": "我有哪些笔记？"}, "outputs": {"agent": "note_agent"}},
        {"inputs": {"task": "删除笔记 test.md"}, "outputs": {"agent": "note_agent"}},
        {"inputs": {"task": "5点钟的研发研讨会议和金融会议冲突，需要取消研发研讨会议"}, "outputs": {"agent": "note_agent"}},
        {"inputs": {"task": "我明天有几个会议，是什么会议？"}, "outputs": {"agent": "note_agent"}},
        {"inputs": {"task": "提醒我明天下午四点开会"}, "outputs": {"agent": "note_agent"}},
        # fina_agent
        {"inputs": {"task": "帮我分析腾讯的财务数据"}, "outputs": {"agent": "fina_agent"}},
        {"inputs": {"task": "我这个月预算还剩多少？"}, "outputs": {"agent": "fina_agent"}},
        # weather_agent
        {"inputs": {"task": "今天北京天气怎么样？"}, "outputs": {"agent": "weather_agent"}},
        {"inputs": {"task": "北京和上海哪个更冷？"}, "outputs": {"agent": "weather_agent"}},
    ]

    add_examples(dataset_name, examples)
    print(f"✅ 数据集 '{dataset_name}' 已初始化，共 {len(examples)} 个用例")


def seed_memory_extract_dataset():
    """Memory 提取测试数据集"""
    dataset_name = "memory-extract-tests"
    create_dataset(dataset_name, "Memory 记忆提取测试用例")

    examples = [
        {
            "inputs": {"conversation": "用户：我叫Alice\n助手：你好Alice！"},
            "outputs": {"facts": ["用户的名字是Alice"]},
        },
        {
            "inputs": {"conversation": "用户：我喜欢游泳\n助手：游泳很好！"},
            "outputs": {"facts": ["用户喜欢游泳"]},
        },
        {
            "inputs": {"conversation": "用户：提醒我明天下午四点开研发研讨会\n助手：好的，已记录"},
            "outputs": {"facts": ["明天下午四点开研发研讨会"]},
        },
        {
            "inputs": {"conversation": "用户：今天天气怎么样？\n助手：今天北京天气cloudy，25°C"},
            "outputs": {"facts": []},
        },
    ]

    add_examples(dataset_name, examples)
    print(f"✅ 数据集 '{dataset_name}' 已初始化，共 {len(examples)} 个用例")


if __name__ == "__main__":
    print("=" * 50)
    print("LangSmith 数据集初始化")
    print("=" * 50)

    seed_planner_decompose_dataset()
    seed_planner_grouping_dataset()
    seed_supervisor_routing_dataset()
    seed_memory_extract_dataset()

    print("\n" + "=" * 50)
    print("所有数据集初始化完成！")
    print("=" * 50)
    print("\n在 LangSmith UI 中查看: https://smith.langchain.com")
