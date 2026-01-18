"""Wizard for the agent."""

from langchain.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing_extensions import Annotated, Literal, TypedDict


class CompileCheckState(TypedDict):
    status: Literal["pass", "fail"]
    error_messages: Annotated[list[AnyMessage], add_messages]


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
    [
        ("system", system_prompt_template),
        MessagesPlaceholder(variable_name="messages"),
        ("assistant", "{pre_filled_output}"),
    ]
)


def router_condition(state: MessagesState) -> Command[ALL_STAGE]:
    """Router node."""
    pass


def research_node(state: MessagesState):
    """Research node."""
    pass


def generator_node(state: MessagesState):
    """Generate code node."""
    pass


def compiler_node(state: MessagesState) -> Command[Literal["lint", "router"]]:
    """Compiler node."""
    pass


def lint_node(state: MessagesState):
    """Lint node."""
    pass


def quality_check_node(state: MessagesState):
    """Quailty check node."""
    pass


wizard_builder = StateGraph(WizardState)
wizard_builder.add_node("router", router_condition)
wizard_builder.add_node("research", research_node)
wizard_builder.add_node("generator", generator_node)
wizard_builder.add_node("compiler", compiler_node)
wizard_builder.add_node("lint", lint_node)
wizard_builder.add_node("quality", quality_check_node)

wizard_builder.add_edge(START, "router")
wizard_builder.add_conditional_edges("router", router_condition)
wizard_builder.add_edge("router", "research")
wizard_builder.add_edge("research", "generator")
wizard_builder.add_edge("generator", "compiler")
wizard_builder.add_edge("compiler", "lint")
wizard_builder.add_edge("lint", "quality")
wizard_builder.add_edge("quality", "router")

graph = wizard_builder.compile()
