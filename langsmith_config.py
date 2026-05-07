"""
LangSmith 配置

使用方式：
  1. 设置环境变量:
     export LANGCHAIN_API_KEY="ls_..."
     export LANGCHAIN_TRACING_V2="true"
     export LANGCHAIN_PROJECT="agent-test"
  
  2. 或者创建 .env 文件:
     LANGCHAIN_API_KEY=ls_...
     LANGCHAIN_TRACING_V2=true
     LANGCHAIN_PROJECT=agent-test

功能：
  - Trace: 追踪每次 Agent 调用的完整链路
  - Dataset: 创建测试数据集做回归测试
  - Monitor: 监控生产环境的延迟、token 消耗、错误率
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

# LangSmith 配置
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "agent-test")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")

# 设置环境变量（确保 langchain 能识别）
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# 元数据标签，用于在 LangSmith 中过滤和分组
TRACE_TAGS = {
    "project": "agent-test",
    "framework": "langgraph",
    "llm": "deepseek-chat",
}

def get_langsmith_client():
    """获取 LangSmith 客户端"""
    from langsmith import Client
    return Client()

def create_dataset(name: str, description: str = ""):
    """创建测试数据集"""
    client = get_langsmith_client()
    try:
        dataset = client.create_dataset(
            dataset_name=name,
            description=description,
        )
        print(f"[LangSmith] 创建数据集: {name} (id={dataset.id})")
        return dataset
    except Exception as e:
        print(f"[LangSmith] 创建数据集失败: {e}")
        return None

def add_examples(dataset_name: str, examples: list):
    """向数据集添加测试用例
    
    examples: [{"inputs": {...}, "outputs": {...}}, ...]
    """
    client = get_langsmith_client()
    try:
        client.create_examples(
            inputs=[ex["inputs"] for ex in examples],
            outputs=[ex["outputs"] for ex in examples],
            dataset_name=dataset_name,
        )
        print(f"[LangSmith] 向数据集 {dataset_name} 添加了 {len(examples)} 个示例")
    except Exception as e:
        print(f"[LangSmith] 添加示例失败: {e}")

def run_test_on_dataset(dataset_name: str, target_function, llm_client=None):
    """在数据集上运行回归测试
    
    用法:
        run_test_on_dataset("planner-decompose-tests", planer_node)
    """
    from langsmith import Client, evaluate
    
    client = get_langsmith_client()
    
    def predict(inputs: dict) -> dict:
        """运行目标函数并返回结果"""
        result = target_function(inputs)
        return {"result": str(result)}
    
    try:
        results = evaluate(
            predict,
            data=dataset_name,
            clients=[client],
            experiment_prefix=f"test-{dataset_name}",
        )
        print(f"[LangSmith] 回归测试完成: {dataset_name}")
        return results
    except Exception as e:
        print(f"[LangSmith] 回归测试失败: {e}")
        return None
