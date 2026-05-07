import os
from langchain_deepseek import ChatDeepSeek

# 加载 LangSmith 配置（设置 LANGCHAIN_TRACING_V2 等环境变量）
try:
    import langsmith_config
except ImportError:
    pass

api_key = os.environ.get("DEEPSEEK_API_KEY")
deepseek_llm = ChatDeepSeek(
    name= "DeepSeekChat",
    model= "deepseek-chat",
    verbose=True,
    api_key=api_key,
    temperature=0,
)
