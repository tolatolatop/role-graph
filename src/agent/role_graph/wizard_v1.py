"""Wizard for the agent."""

from langchain.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import Annotated, List, Literal, TypedDict

from agent.agent import create_custom_agent


class CompileCheckState(BaseModel):
    status: Literal["pass", "fail"] = Field(default="pass", description="检查结果")
    error_messages: List[str] = Field(default=[], description="错误信息")


class LintCheckState(TypedDict):
    status: Literal["pass", "fail"]
    error_messages: Annotated[list[AnyMessage], add_messages]


class QualityCheckState(TypedDict):
    status: Literal["pass", "fail"]
    error_messages: Annotated[list[AnyMessage], add_messages]


class WizardState(MessagesState):
    context_asset: Annotated[list[AnyMessage], add_messages]
    draft: Annotated[list[AnyMessage], add_messages]
    verify_content: Annotated[list[AnyMessage], add_messages]
    published_content: Annotated[list[AnyMessage], add_messages]
    compile_check: CompileCheckState
    lint_check: LintCheckState
    quality_check: QualityCheckState
    current_stage: Literal[
        "router", "research", "generator", "compiler", "lint", "quality"
    ]


ALL_STAGE = Literal["research", "generator", "compiler", "lint", "quality", END]

system_prompt_template = """Role: {role}
Profile: {profile}

{background}

{constraints}

{workflow}

{standard_output}

{examples}
"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_prompt_template), MessagesPlaceholder(variable_name="messages")]
)


def router_condition(state: WizardState) -> Command[ALL_STAGE]:
    """Router node."""
    if state.get("current_stage") is None:
        return Command(goto="research", update={"current_stage": "research"})
    elif state["current_stage"] == "router":
        return Command(goto="research")
    elif state["current_stage"] == "research":
        return Command(goto="generator")
    elif state["current_stage"] == "generator":
        return Command(goto="compiler")
    elif state["current_stage"] == "compiler":
        return Command(goto="lint")
    elif state["current_stage"] == "lint":
        return Command(goto="quality")
    elif state["current_stage"] == "quality":
        return Command(goto=END)
    else:
        return Command(goto=END)


def research_node(state: WizardState):
    """Research node."""
    return {"context_asset": [state.get("messages", [])[-1]]}


def generator_node(state: WizardState):
    """Generate code node."""
    clear_check_state = {
        "compile_check": None,
        "lint_check": None,
        "quality_check": None,
    }
    llm = create_custom_agent(use_tools=False)
    context_asset = state["context_asset"]
    prompt = prompt_template.invoke(
        {
            "role": "小说创作助手",
            "profile": "你是一个小说创作助手，你的任务是根据用户的需求创作小说。",
            "background": "\n".join(
                ["## Background"] + [f"{msg.content}" for msg in context_asset]
            ),
            "constraints": "## Constraints:\n- 你只能输出小说内容，不能输出任何其他内容。",
            "workflow": "",
            "standard_output": "",
            "examples": "",
            "messages": state.get("published_content", []),
            "pre_filled_output": "",
        }
    )
    msg = llm.invoke(prompt.to_messages())
    return {
        "draft": [msg],
        **clear_check_state,
    }


def compiler_node(state: WizardState) -> Command[Literal["lint", "router"]]:
    """Compiler node."""
    draft = state.get("draft", [])
    if len(draft) == 0:
        return Command(goto="router", update={"compile_check": {"status": "fail"}})

    context_asset = state["context_asset"]
    draft = state.get("draft", [])
    latest_draft = draft[-1]
    messages = state.get("published_content", []) + [latest_draft]
    llm = create_custom_agent(use_tools=False, output_model=CompileCheckState)
    constraints = """## Constraints:
- 按格式要求输出检查结果。
- 检查结果必须包含：
  - 检查结果
  - 检查理由
"""
    prompt = prompt_template.invoke(
        {
            "role": "小说创作助手",
            "profile": "负责检查小说情节是否符合逻辑，是否符合小说创作的规范。",
            "background": "\n".join(
                ["## Background"] + [f"{msg.content}" for msg in context_asset]
            ),
            "constraints": constraints,
            "workflow": "",
            "standard_output": "",
            "examples": "",
            "messages": messages,
            "pre_filled_output": "",
        }
    )
    cc_state: CompileCheckState = llm.invoke(prompt.to_messages())
    if cc_state.status == "fail":
        return Command(goto="generator", update={"compile_check": cc_state})
    return Command(goto="lint", update={"verify_content": [latest_draft]})


def lint_node(state: WizardState):
    """Lint node."""
    verify_content = state.get("verify_content", [])
    msg = verify_content[-1]
    return {
        "published_content": [msg],
        "lint_check": {"status": "pass"},
    }


def quality_check_node(state: WizardState):
    """Quailty check node."""
    published_content = state.get("published_content", [])
    if len(published_content) == 0:
        return Command(goto="router", update={"quality_check": {"status": "fail"}})

    msg = published_content[-1]
    return Command(
        goto="router",
        update={"quality_check": {"status": "pass"}, "messages": [msg]},
    )


wizard_builder = StateGraph(WizardState)
wizard_builder.add_node("router", router_condition)
wizard_builder.add_node("research", research_node)
wizard_builder.add_node("generator", generator_node)
wizard_builder.add_node("compiler", compiler_node)
wizard_builder.add_node("lint", lint_node)
wizard_builder.add_node("quality", quality_check_node)

wizard_builder.add_edge(START, "router")
wizard_builder.add_edge("research", "generator")
wizard_builder.add_edge("generator", "compiler")
wizard_builder.add_edge("compiler", "lint")
wizard_builder.add_edge("lint", "quality")
wizard_builder.add_edge("quality", "router")

graph = wizard_builder.compile()
