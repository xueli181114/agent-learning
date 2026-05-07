import subprocess
from langchain.tools import tool
@tool
def exec_command(command: str) ->str:
    """执行终端命令并返回结果（仅限安全的只读命令，如 ls、pwd、cat）"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout or result.stderr or "Command running finished"
    except Exception as e:
        return f"Command line running failure: {e}"  