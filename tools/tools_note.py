
from langchain.tools import tool
import os
from datetime import datetime

@tool
def write_note(filename: str, content: str) -> str:
    """保存笔记到文件，文件名建议用 .txt 或 .md 结尾，文件名不需要添加文件夹层级，只需要包含文件名"""

    filename =  os.path.basename(filename) # 只传递文件名
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_content = f"[记录时间]：{current_time}\n{content}"
    os.makedirs("notes", exist_ok=True)  # 自动创建目录
    with open(f"notes/{filename}", "w") as f:
        f.write(full_content)
    return f"✅ 笔记已保存：{filename}"

@tool
def read_note(filename: str) -> str:
    """读取之前保存的笔记内容"""
    filepath = f"notes/{filename}"
    if not os.path.exists(filepath):
        return f"❌ 笔记不存在：{filename}"
    with open(filepath, "r") as f:
        content = f.read()
    return f"📝 {filename} 的内容：\n{content}"

@tool
def list_notes() -> str:
    """列出所有保存的笔记"""
    os.makedirs("notes", exist_ok=True)
    notes = os.listdir("notes")
    if not notes:
        return "暂无笔记"
    return "📁 笔记列表：\n" + "\n".join(f"  - {note}" for note in notes)

@tool
def delete_note(filename: str) -> str:
    """删除指定的笔记文件"""
    filepath = f"notes/{filename}"
    if not os.path.exists(filepath):
        return f"❌ 笔记不存在：{filename}"
    os.remove(filepath)
    return f"🗑️ 已删除笔记：{filename}"

@tool
def list_note() ->str:
    """查看所有笔记文件列表"""
    notes_path = "notes"
    notes = os.listdir(notes_path)
    return ",".join(notes)

@tool
def get_note_path(filename: str) -> str:
    """获取笔记文件路径"""
    return os.path.join("notes", filename)
