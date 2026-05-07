# agent_memory_db.py
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain_core.messages import HumanMessage
from multi_agent import workflow, Command

DB_URI = "postgresql://xueli@localhost:5432/agentdb"

if __name__ == "__main__":
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer, \
         PostgresStore.from_conn_string(DB_URI) as store:
        checkpointer.setup()
        store.setup()
        
        # 编译图时注入 checkpointer 和 store
        graph = workflow.compile(checkpointer=checkpointer, 
                                 store=store)
        
        # 第一次会话：告诉名字 (thread_id = session1)
        config = {"configurable": {"thread_id": "delete_test2", "user_id": "alice3"}}
        print("\n=== 会话1: 删除笔记 ===")
        res1 = graph.invoke(
            {"messages": [HumanMessage(content="删除所有笔记，并且查看北京的天气，并且提醒我明天下午四点开研发研讨会并记录新的笔记")]},
            config=config
        )
        print(f"回答: {res1['messages'][-1].content}")
        
        # 第二次会话：新 thread_id，同一个 user_id，问名字
        # config2 = {"configurable": {"thread_id": "session2", "user_id": "alice"}}
        # print("\n=== 会话2: 问名字 ===")
        # res2 = graph.invoke(
        #     {"messages": [HumanMessage(content="我叫什么名字？")]},
        #     config=config2
        # )
        # print(f"回答: {res2['messages'][-1].content}")
        user_decision = input("输入 '是' 或 '否': ")
        result = graph.invoke(Command(resume=user_decision), config=config)
        print("最终结果:", result["messages"][-1].content)
        
        
        recording_messages = [
            "My name is Alice",
            "I like banana",
            "I love swimming",
            "My meeting changed to 5:00 pm",
            "I will be late.",
            "I am going to the meeting",
            "Will my meeting start at 4:00pm?",
            "我的5点钟被邀请了另一个会议来讨论金融问题，这两个会议冲突了，我需要把5点钟的研发研讨会议取消掉",
            "我明天有几个会议，是什么会议？"
        ]
        for message in recording_messages:
            
            result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
            print("Got response:", result["messages"][-1].content)