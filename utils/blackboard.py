from typing import Any
# 共享黑板（全局状态）
class Blackboard:
    def __init__(self):
        self.data: dict[str, Any] = {}
        self.history: list = []

blackboard = Blackboard()