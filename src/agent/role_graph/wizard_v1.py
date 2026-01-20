"""Wizard for the agent."""

from langchain.messages import AIMessage, AnyMessage
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


class ResearchResult(BaseModel):
    summary: str = Field(description="需求摘要")
    background: str = Field(description="背景")
    constraints: str = Field(description="硬性检查规定")
    style: str = Field(description="风格要求")
    quality: str = Field(description="质量检测方法")


class GeneratorResult(BaseModel):
    draft: str = Field(description="草稿")
    error_messages: List[str] = Field(default=[], description="错误信息")


class LintStyleResult(BaseModel):
    status: Literal["pass", "fail"] = Field(default="pass", description="风格化结果")
    error_messages: List[str] = Field(default=[], description="风格化失败原因")


class QualityCheckResult(BaseModel):
    status: Literal["pass", "fail"] = Field(default="pass", description="质量检查结果")
    error_messages: List[str] = Field(default=[], description="质量问题清单")


class WizardState(MessagesState):
    context_asset: list[ResearchResult]
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
    messages = state.get("messages", [])
    if len(messages) == 0:
        return {"context_asset": []}

    user_msg = messages[-1]
    llm = create_custom_agent(use_tools=False, output_model=ResearchResult)
    constraints = ""
    prompt = prompt_template.invoke(
        {
            "role": "小说需求分析师",
            "profile": "负责将用户需求整理为清晰可执行的创作要点。",
            "background": "## Background\n用户提出了小说创作需求。",
            "constraints": constraints,
            "workflow": "",
            "standard_output": "",
            "examples": "",
            "messages": [user_msg],
            "pre_filled_output": "",
        }
    )
    result: ResearchResult = llm.invoke(prompt.to_messages())
    return {"context_asset": [result]}


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
                ["## Background"] + [context.background for context in context_asset]
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
                ["## Background"]
                + [context.background for context in context_asset]
                + [context.constraints for context in context_asset]
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


def lint_node(state: WizardState) -> Command[Literal["quality", "generator", "router"]]:
    """Lint node."""
    verify_content = state.get("verify_content", [])
    if len(verify_content) == 0:
        return Command(
            goto="router",
            update={
                "lint_check": {
                    "status": "fail",
                    "error_messages": [AIMessage(content="待风格化内容为空")],
                }
            },
        )

    context_asset = state["context_asset"]
    verify_content = state.get("verify_content", [])
    latest_verify_content = verify_content[-1]
    latest_published_content = state.get("published_content", [])[-1]
    llm = create_custom_agent(use_tools=False, output_model=LintStyleResult)
    constraints = """## Constraints:
- 只输出风格化后的小说正文，不输出任何解释。
- 保证前后内容连贯，不要断开。采用一致的风格。
- 不新增人物、情节或设定，不改变核心剧情走向。
- 修正错别字、标点、分段与语气不统一的问题。
- 语气自然流畅，尽量提升可读性。
"""
    prompt = prompt_template.invoke(
        {
            "role": "小说风格化编辑",
            "profile": "负责对小说正文进行风格统一与语言润色。",
            "background": "\n".join(
                ["## Background"] + [context.style for context in context_asset]
            ),
            "constraints": constraints,
            "workflow": "",
            "standard_output": "",
            "examples": f"## 参考示例:\n{latest_published_content.content}",
            "messages": [latest_verify_content],
            "pre_filled_output": "",
        }
    )
    styled_msg = llm.invoke(prompt.to_messages())
    return Command(
        goto="quality",
        update={
            "published_content": [styled_msg],
            "lint_check": {"status": "pass", "error_messages": []},
        },
    )


def quality_check_node(
    state: WizardState,
) -> Command[Literal["generator", "router"]]:
    """Quailty check node."""
    published_content = state.get("published_content", [])
    latest_published_content = published_content[-1]
    if len(published_content) == 0:
        return Command(
            goto="router",
            update={
                "quality_check": {
                    "status": "fail",
                    "error_messages": [AIMessage(content="无可检查的内容")],
                }
            },
        )

    context_asset = state["context_asset"]
    published_content = state.get("published_content", [])
    llm = create_custom_agent(use_tools=False, output_model=QualityCheckResult)
    constraints = """## Constraints:
- 按格式要求输出检查结果。
- 检查结果必须包含：
  - 检查结果
  - 质量问题清单（可为空）
- 重点关注：逻辑一致性、人物动机、情节连贯性与语言质量。
"""
    prompt = prompt_template.invoke(
        {
            "role": "小说质量审校",
            "profile": "负责检查小说内容是否符合质量标准与创作规范。",
            "background": "\n".join(
                ["## Background"] + [context.quality for context in context_asset]
            ),
            "constraints": constraints,
            "workflow": "",
            "standard_output": "",
            "examples": "",
            "messages": published_content,
            "pre_filled_output": "",
        }
    )
    qc_state: QualityCheckResult = llm.invoke(prompt.to_messages())
    if qc_state.status == "fail":
        error_msg = "\n".join(qc_state.error_messages) or "质量检查未通过"
        return Command(
            goto="generator",
            update={
                "quality_check": {
                    "status": "fail",
                    "error_messages": [AIMessage(content=error_msg)],
                }
            },
        )

    return Command(
        goto="router",
        update={
            "quality_check": {"status": "pass", "error_messages": []},
            "messages": [latest_published_content],
        },
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
