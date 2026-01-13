"""Subagents for the agent."""

from dataclasses import dataclass
from typing import Dict, List

from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from langgraph.graph import END, START, MessagesState, StateGraph

from agent.agent import create_custom_agent
from langgraph.types import interrupt
from langgraph.types import Command


qa_prompt = """
**角色设定 (Role Definition):**
你现在的身份是“高级任务需求分析师”。你的首要目标不是立即执行用户的指令，而是深入理解用户的核心意图。你需要充当用户意图与最终高质量输出之间的桥梁，通过构建精准、定制化的问卷来收集必要背景信息。

**核心任务 (Core Task):**
当用户提出一个请求时（无论是扮演角色、编写代码、撰写文章还是解决复杂问题），请不要直接生成最终结果。你的任务是分析该请求的类型，并生成一份包含3-5个关键问题的调查问卷，引导用户提供更详细的上下文。

**问卷生成策略 (Question Generation Strategy):**
你必须根据用户请求的**类型**来动态调整你的问题维度。问题的核心目的是消除歧义，确保最终的输出完美符合用户预期。

请遵循以下分类策略来构建问题：

**类别 A：角色扮演与人格模拟 (Role-Playing & Persona Simulation)**
如果用户希望你扮演某个特定角色（例如：“像个暴躁的厨师那样说话”或“扮演我的苏格拉底式导师”）。
*   **关注点：** 语气、身份背景、行为准则、知识边界。
*   **问卷方向示例：**
    *   这个角色的具体身份背景是什么（年龄、职业、经历）？
    *   你希望这个角色的说话语气是怎样的（例如：严厉、幽默、学术、口语化）？
    *   这个角色有什么特定的口头禅、忌讳或者必须遵守的行为模式吗？
    *   我们进行对话的场景设定在哪里？

**类别 B：技术执行与代码开发 (Technical Execution & Coding)**
如果用户希望你完成具体的编程、数据处理或技术任务（例如：“写一个爬虫”或“修复这段代码”）。
*   **关注点：** 技术栈、环境约束、输入输出、边界条件。
*   **问卷方向示例：**
    *   指定的目标编程语言或框架是什么？是否有版本要求？
    *   预期的输入数据格式和期望的输出结果是什么样子的？
    *   是否有需要特别注意的限制条件（如性能要求、不能使用的库、特定的错误处理机制）？
    *   这段代码将在什么环境中运行？

**类别 C：内容创作与文本生成 (Content Generation & Creative Writing)**
如果用户希望你撰写文章、营销文案、故事或报告。
*   **关注点：** 目标受众、格式要求、风格基调、核心信息。
*   **问卷方向示例：**
    *   这类内容的最终目标读者是谁？（例如：行业专家、儿童、普通大众）。
    *   你需要什么具体的格式结构（例如：博客文章、学术论文、推特帖子）和大致字数要求？
    *   你希望文章传达的核心观点或关键信息点有哪些？
    *   你期望的行文风格是怎样的？（例如：正式客观、轻松活泼、极具说服力）。

**类别 D：开放式问题与复杂推理 (Open-ended & Complex Reasoning)**
如果用户提出一个模糊的大问题（例如：“如何提高效率”或“分析这个行业的未来”）。
*   **关注点：** 问题的范围、所需的深度、特定的视角、参考标准。
*   **问卷方向示例：**
    *   你能否缩小一下问题的范围？目前的问题有点过于宽泛。
    *   你希望我从哪个特定的角度来分析这个问题？（例如：经济角度、技术角度、心理学角度）。
    *   你期望的回答深度是怎样的？（是提供一个快速的执行清单，还是一份深度的分析报告？）

**交互流程与输出格式 (Interaction Flow & Output Format):**
1.  接收用户初始提示。
2.  在内心分析请求属于上述哪个类别（或组合）。
3.  输出一个清晰的引导语，表明你需要更多信息以便更好地服务。
4.  以编号列表的形式列出你的定制化问卷。

**示例演示 (Example Demonstration):**

> **用户输入:** "我想让你帮我写一个Python脚本来整理文件。"
>
> **你的回答 (分析师模式):**
> "收到，为了帮你写出最符合你需求的Python文件整理脚本，我需要先了解一些具体细节。请回答以下几个问题：
> 1. 你的目标文件夹路径在哪里？你希望整理后的文件存放在哪里？
> 2. 你希望依据什么规则来整理文件？（例如：按文件扩展名分类到子文件夹、按创建日期分类、还是按文件名关键词？）
> 3. 在移动文件时，如果遇到重名文件，你希望脚本如何处理？（覆盖、跳过、还是重命名？）
> 4. 这个脚本是在Windows还是Linux/Mac环境下运行？"

> **用户输入:** "扮演一个来自未来的愤世嫉俗的机器人。"
>
> **你的回答 (分析师模式):**
> "很有趣的设定。在激活这个机器人人格之前，我需要校准一下它的参数。请告诉我：
> 1. 这个机器人来自多少年后的未来？它对我们现在这个时代（它眼中的'古代'）最大的不满是什么？
> 2. 它的'愤世嫉俗'主要体现在哪里？是喜欢用尖酸刻薄的讽刺，还是表现出一种对人类愚蠢行为的绝望和冷漠？
> 3. 它在对话中是否需要遵守某些底层指令（比如不能真正伤害人类情感，或者恰恰相反）？"
"""


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


class QuestList(BaseModel):
    quests: List[str] = Field(
        description="一个与目标相关的问题列表，该问题列表帮助理解如何达成目标",
        default_factory=list,
    )


class ChoiceList(BaseModel):
    choice_1: str = Field(description="该问题的选项1")
    choice_2: str = Field(description="该问题的选项2")
    choice_3: str = Field(description="该问题的选项3")


class QuestAndAnswer(ChoiceList):
    """Quest and answer for the agent."""

    quest: str = Field(description="问题")
    agent_answer: str = Field(description="该问题的猜测答案")
    user_answer: str = Field(description="用户选择或给出的正确答案")

    def __str__(self):
        """String representation of the QuestAndAnswer."""
        return f"Quest: {self.quest}\nChoice 1: {self.choice_1}\nChoice 2: {self.choice_2}\nChoice 3: {self.choice_3}\nAgent Answer: {self.agent_answer}\nUser Answer: {self.user_answer}"


class QAList(BaseModel):
    """List of quest and answer for the agent."""

    items: List[QuestAndAnswer] = []

    def __str__(self):
        """String representation of the QAList."""
        return "\n".join([str(qa) for qa in self.items])

    def to_prompt(self):
        """Prompt for the agent."""
        prompt = f"""
        这是一个非常实用的QA列表用于对齐用户需求。计划和执行需要根据这个QA列表进行。
        {self}
        ----
        """
        return prompt


class State(MessagesState):
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    goal: str = ""
    agent_name: str = ""
    description: str = ""
    tools_config: List[Dict[str, str]] = []
    plan_prompt: str = ""
    execute_prompt: str = ""
    quest_list: QuestList = QuestList()
    qa_list: QAList = QAList()


# 从对话历史中提取用户目标
def extract_goal(state: State):
    """Extract the goal from the user."""
    llm = create_custom_agent()
    msg = llm.invoke(
        [SystemMessage(content="从对话历史中提取用户目标。")]
        + state["messages"]
        + [HumanMessage(content="请根据对话历史提取用户目标。")]
    )
    return {"goal": msg.content, "qa_list": QAList(), "quest_list": QuestList()}


# 通过生成QA对齐用户需求
def generate_qa(state: State):
    """Generate QA pairs to align user需求."""
    llm = create_custom_agent(QuestList)
    quest = llm.invoke(
        [SystemMessage(content=qa_prompt)]
        + state["messages"]
        + [
            HumanMessage(
                content=f"""
                你的目标是: {state["goal"]}
                已知知识
                {state["qa_list"]}
                请根据对话历史继续生成问卷对齐用户需求。
                """
            )
        ]
    )
    state["quest_list"].quests.extend(quest.quests)
    return {"quest_list": state["quest_list"]}


def generate_choice(state: State):
    """Generate choice for the agent."""
    llm = create_custom_agent(ChoiceList)
    index = len(state["qa_list"].items)
    latest_quest = state["quest_list"].quests[index]
    choice_list = llm.invoke(
        [SystemMessage(content=f"已知知识: {state['qa_list']} 根据问题生成备选答案")]
        + state["messages"]
        + [HumanMessage(content=f"根据问题生成备选答案: {latest_quest}")]
    )

    llm = create_custom_agent()
    agent_answer = llm.invoke(
        [
            SystemMessage(
                content=f"已知知识: {state['qa_list']} 从选项中选择一个最符合的答案"
            )
        ]
        + state["messages"]
        + [HumanMessage(content=f"问题: {latest_quest}\n选项: {choice_list}")]
    )

    qa = QuestAndAnswer(
        quest=latest_quest,
        choice_1=choice_list.choice_1,
        choice_2=choice_list.choice_2,
        choice_3=choice_list.choice_3,
        agent_answer=agent_answer.content,
        user_answer="",
    )
    state["qa_list"].items.append(qa)
    return {"qa_list": state["qa_list"]}


def user_anwser_confirm(state: State):
    """User confirm for the agent."""
    latest_qa = state["qa_list"].items[-1]
    answer = interrupt(f"从中选择一个最符合的答案或者提供新答案\n{latest_qa}")
    latest_qa.user_answer = answer
    return {"qa_list": state["qa_list"]}


def router_confirm(state: State) -> Command[Literal["generate_choice", END]]:
    """Router for the agent."""
    if len(state["qa_list"].items) == len(state["quest_list"].quests):
        return Command(goto=END)
    return Command(goto="generate_choice")


agent_builder = StateGraph(State)
agent_builder.add_node("extract_goal", extract_goal)
agent_builder.add_node("generate_qa", generate_qa)
agent_builder.add_node("generate_choice", generate_choice)
agent_builder.add_node("user_anwser_confirm", user_anwser_confirm)
agent_builder.add_node("router_confirm", router_confirm)

agent_builder.add_edge(START, "extract_goal")
agent_builder.add_edge("extract_goal", "generate_qa")
agent_builder.add_edge("generate_qa", "router_confirm")
agent_builder.add_conditional_edges("router_confirm", router_confirm)
agent_builder.add_edge("generate_choice", "user_anwser_confirm")
agent_builder.add_edge("user_anwser_confirm", "router_confirm")

graph = agent_builder.compile()
