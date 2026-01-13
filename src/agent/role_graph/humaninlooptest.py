from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import Annotated, TypedDict


class State(TypedDict):
    messages: Annotated[list, "add"]  # 对话历史
    current_task: str
    exception_info: str | None
    agent_suggestion: str | None
    user_decision: str | None


def risky_operation_node(state: State):
    """高风险操作，可能抛异常"""
    try:
        # 模拟异常场景
        if "delete_all" in state["current_task"]:
            raise ValueError("检测到危险操作: delete_all")
        return {
            "messages": [AIMessage(content=f"任务 {state['current_task']} 执行成功")]
        }
    except Exception as e:
        # 异常时中断，携带详情
        return {
            "exception_info": str(e),
            "messages": interrupt(
                {
                    "task": state["current_task"],
                    "error": str(e),
                    "state_summary": f"历史任务: {state['current_task']}",
                }
            ),
        }


def agent_suggestion_node(state: State):
    """resume后，agent分析异常+历史，给决策建议"""
    # 基于异常和历史生成建议
    suggestion = f"""
    异常: {state["exception_info"]}
    建议选项:
    1. 重试 (retry)
    2. 修改任务为 '{state["current_task"].replace("delete_all", "delete_safe")}' (modify_safe)
    3. 跳过 (skip)
    4. 人工指定新任务 (custom: your_instruction)
    """
    return {
        "agent_suggestion": suggestion,
        "messages": [AIMessage(content=suggestion)],  # 注入对话供LLM参考
    }


def human_decision_node(state: State):
    """用户决策节点，可选最终确认"""
    user_decision = state["user_decision"]
    if "custom" in user_decision:
        # 复杂决策再中断确认
        return interrupt(
            {"suggested_decision": user_decision, "needs_final_confirm": True}
        )
    return {"messages": [AIMessage(content=f"执行用户决策: {user_decision}")]}


# 路由：异常->中断->agent建议->用户决策->执行
def router(state: State):
    if state.get("exception_info"):
        return "agent_suggestion"  # resume后进建议
    if state.get("user_decision"):
        return "human_decision"
    return END


# 构建图
workflow = StateGraph(State)
workflow.add_node("risky", risky_operation_node)
workflow.add_node("suggestion", agent_suggestion_node)
workflow.add_node("decision", human_decision_node)

workflow.add_conditional_edges(START, lambda s: "risky", {"risky": "risky"})
workflow.add_conditional_edges(
    "risky",
    router,
    {"agent_suggestion": "suggestion", "human_decision": "decision", END: END},
)
workflow.add_conditional_edges(
    "suggestion",
    lambda s: "interrupt_for_user",
    {"interrupt_for_user": "decision"},  # 显示建议后中断等用户
)
workflow.add_edge("decision", END)

graph = workflow.compile(checkpointer=InMemorySaver())
