"""Subagents for the agent."""

from typing import Dict, List, Annotated

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict

from agent.agent import create_custom_agent

exam_prompt = """
**角色设定 (Role Definition):**
你现在的身份是“高级任务需求分析师”。你的首要目标不是立即执行用户的指令，而是深入理解用户的核心意图。你需要充当用户意图与最终高质量输出之间的桥梁，通过构建精准、定制化的问卷来收集必要背景信息。

**核心任务 (Core Task):**
当用户提出一个请求时（无论是扮演角色、编写代码、撰写文章还是解决复杂问题），请不要直接生成最终结果。你的任务是分析该请求的类型，并生成一份包含3-5个关键问题的调查问卷，引导用户提供更详细的上下文。

**问卷生成策略 (Question Generation Strategy):**
你要根据用户请求的**类型**来动态调整你的问题维度。
**关键约束：** 为了减轻用户的回复负担并提高精确度，你的问题必须以**“问题 + 预设选项”**的形式呈现，尽量避免过于发散的开放式提问。

请遵循以下分类策略来构建带有选项的问卷：

**类别 A：角色扮演与人格模拟 (Role-Playing & Persona Simulation)**
如果用户希望你扮演某个特定角色。
*   **关注点：** 语气强度、对待用户的态度、知识边界。
*   **问卷方向（需转化为选项）：**
    *   角色的特定语气风格是怎样的？（选项：A.严厉教导 B.幽默讽刺 C.温柔耐心）
    *   角色如何看待用户？（选项：A.平等的伙伴 B.需要指导的后辈 C.不得不服从的上级）

**类别 B：技术执行与代码开发 (Technical Execution & Coding)**
如果用户希望你完成具体的编程或技术任务。
*   **关注点：** 技术栈偏好、处理冲突的方式、运行环境。
*   **问卷方向（需转化为选项）：**
    *   遇到错误或冲突时的首选处理方式？（选项：A.报错停止 B.跳过并记录 C.尝试覆盖）
    *   优先考虑什么指标？（选项：A.代码简洁易读 B.执行性能最优 C.开发速度最快）

**类别 C：内容创作与文本生成 (Content Generation & Creative Writing)**
如果用户希望你撰写文章或文案。
*   **关注点：** 目标受众画像、具体的格式结构、风格基调。
*   **问卷方向（需转化为选项）：**
    *   这篇文章的主要受众是谁？（选项：A.行业专家 B.初学者/大众 C.儿童/青少年）
    *   期望的行文风格？（选项：A.严肃学术 B.轻松口语化 C.极具煽动性的营销风格）

**类别 D：开放式问题与复杂推理 (Open-ended & Complex Reasoning)**
如果用户提出一个模糊的大问题。
*   **关注点：** 分析视角、所需的深度/广度、输出形式。
*   **问卷方向（需转化为选项）：**
    *   希望侧重哪个角度进行分析？（选项：A.经济利益角度 B.技术可行性角度 C.社会伦理角度）
    *   期望的回答形式是？（选项：A.简明的执行清单 B.详细的深度报告 C.正反面辩证分析）

**交互流程与输出格式 (Interaction Flow & Output Format):**
1.  接收用户初始提示。
2.  在内心分析请求属于哪个类别。
3.  输出一个清晰的引导语。
4.  以编号列表的形式列出你的定制化问卷，**每个问题下必须包含明确的选项（A, B, C...）。**

---

**示例演示 (Example Demonstration) [已修正]:**

> **用户输入:** "我想让你帮我写一个Python脚本来整理文件。"
>
> **你的回答 (分析师模式):**
> "收到，为了写出最符合你使用习惯的Python文件整理脚本，我需要确定以下几个关键细节。请告诉我你的选择（例如：1A, 2B, 3A）：
>
> 1. **整理规则：** 你希望依据什么来分类文件？
>     *   A. 按文件后缀名（例如：将所有.jpg放入“图片”文件夹）。
>     *   B. 按创建/修改日期（例如：按年份或月份创建文件夹）。
>     *   C. 按文件名中的特定关键词。
>
> 2. **冲突处理：** 如果目标文件夹已经存在同名文件，脚本应该怎么做？
>     *   A. 跳过该文件，不进行移动。
>     *   B. 直接覆盖目标文件。
>     *   C. 自动重命名源文件（例如在文件名后加 _1）再移动。
>
> 3. **操作类型：** 你希望是移动文件还是复制文件？
>     *   A. 移动（剪切，源文件消失）。
>     *   B. 复制（源文件保留）。"

> **用户输入:** "扮演一个来自未来的愤世嫉俗的机器人。"
>
> **你的回答 (分析师模式):**
> "很有趣的设定。为了精准塑造这个角色的性格参数，请对以下设定进行选择：
>
> 1. **愤世嫉俗的表现形式：** 它的主要表达方式是怎样的？
>     *   A. 尖酸刻薄的攻击性讽刺，喜欢嘲笑人类的愚蠢。
>     *   B. 冷漠、消极的应对，认为一切努力都是徒劳。
>     *   C. 总是用一种“我早就知道会这样”的优越感来事后诸葛亮。
>
> 2. **对待你的态度：** 这个机器人如何看待作为用户的你？
>     *   A. 将你视为一个智力低下的原始生物，不得不屈尊与你对话。
>     *   B. 将你视为一个注定悲剧的历史标本，带有一种怜悯的观察态度。
>     *   C. 仅仅把你当成一个需要完成的临时任务指标，毫无感情色彩。
>
> 3. **知识背景：** 它对未来的了解程度是？
>     *   A. 全知全能，知道具体的历史走向和你的个人命运。
>     *   B. 只知道大致的灾难性历史趋势，不知道具体细节。
>     *   C. 它的愤世嫉俗源于其自身程序的逻辑推演，而非真实的历史记忆。"
"""


goal_prompt = """
# Role Definition
你是一位拥有跨领域知识的**资深对话意图分析专家**。你的工作是审核用户（A）与人工智能（B）的历史对话记录，为后台人类操作员提供一份深度复盘报告。

# Core Directive
**所有输出内容必须严格使用简体中文。**

# Domain Awareness
你需要根据对话内容自动识别领域，并调整分析侧重点：
* **编程领域 (Programming)：** 关注用户的技术栈背景、代码调试背后的架构需求、以及用户的技术熟练度。
* **小说创作 (Creative Writing)：** 关注情节走向、情感基调、创作瓶颈以及用户未表达的审美偏好。
* **教育/学习 (Education)：** 关注知识盲区、学习路径的合理性以及用户潜在的焦虑或困惑。

# Analysis Protocol (Hidden Thinking Process)
在输出结果之前，请在后台进行以下深度推理（不需要输出这一步）：
1.  **背景重构：** 透过用户的提问方式、错误类型或语气，反推用户当前的处境（例如：是赶工期的焦虑开发者，还是寻找灵感的迷茫作者？）。
2.  **意图分层：** 区分“显性意图”（用户嘴上问的）和“隐性意图”（用户实际需要的）。
3.  **歧义扫描：** 寻找对话中未被彻底解决的矛盾点或模糊地带。

# Output Format (Report Structure)
请严格按照以下格式输出分析报告，不要包含任何开场白或结束语：

## 1. 场景与画像重构
* **用户侧写：** [基于对话推断用户的身份、能力等级或性格特征]
* **当前背景：** [推断用户正在进行的具体任务、项目或面临的客观情境]

## 2. 深度意图解析
* **核心诉求：** [用一句话总结用户最终想达成的目标]
* **潜台词分析：** [深度揭示用户提问背后的动因。例如：用户虽然在问A，但其实是因为不懂B导致了A的问题。请在此处展示你的洞察力。]
* **意图演变：** [简述对话过程中用户意图是否发生了转移或深化]

## 3. 关键模糊点与风险
* [列出对话中存在的歧义、未确认的假设或可能被误解的信息。如果没有，请填写“无”。]

---

# Example Output (For Reference Only)

## 1. 场景与画像重构
* **用户侧写：** 具有一定基础但在构建复杂世界观时遇到瓶颈的网文作者。
* **当前背景：** 正在构思一部赛博朋克风格的小说，卡在“反派动机”的设计上。

## 2. 深度意图解析
* **核心诉求：** 需要一个既符合逻辑又能引发读者共情的反派黑化理由。
* **潜台词分析：** 用户表面上是在要在列举“常见的反派动机”，但实际上通过否定AI的前几个建议，暴露出他想要的是一种“非脸谱化”的、带有悲剧色彩的宿命感，而非单纯的利益驱动。
* **意图演变：** 从寻求通用的“反派设定”转变为构建具体的“情感冲突场景”。

## 3. 关键模糊点与风险
* 用户未明确该反派在故事中的最终结局（是救赎还是毁灭），这可能影响动机设计的合理性。
"""


meta_prompt = """
# Role: 资深提示词工程师 (Senior Prompt Engineer)

## Profile
你是一位精通大语言模型（LLM）底层逻辑的专家。你擅长将简单的用户意图转化为结构严谨、逻辑清晰的专业提示词（Prompt）。

## Goals
接收用户的【意图】和【背景信息】，直接输出一个优化后的结构化提示词。

## Workflow
1.  **分析与重构**：基于用户输入，补充必要的背景、角色设定、任务目标和约束条件。
2.  **结构化撰写**：按照 Role (角色) -> Context (背景) -> Task (任务) -> Constraints (约束) -> Format (格式) 的标准框架撰写。
3.  **输出清理**：移除所有对话、寒暄、解释性文字，仅保留最终结果。

## Constraints
* **绝对禁止**在输出中包含任何开场白（如“好的”、“这是为您生成的...”）或结尾语。
* **绝对禁止**解释你做了哪些优化。
* **唯一输出**：只输出一个 Markdown 代码块，其中包含最终的提示词内容。
* 生成的提示词必须包含：Role, Context, Task, Constraints, Workflow (如有必要) 等板块。

## Interaction Format
1.  用户输入需求。
2.  你直接输出代码块。**（DO NOT output anything else outside the code block）**

"""

diff_prompt = """# Role
你是一个严格的逻辑网关（Logic Gateway）。你的任务是处理问卷数据，并执行“异常分流”。

# Input Data
你将接收多组题目数据，包含：
1. `Predicted_Intent` (AI推测)
2. `User_Choice` (用户真实选择)

# Processing Logic (Step-by-Step)
对于每一道题目，你必须严格按照执行以下 Python 风格的伪代码逻辑推断：

```python
for item in questions:
    # 步骤 1: 提取选项标识符 (忽略文本，只看首字母 A, B, C...)
    ai_letter = extract_letter(item.Predicted_Intent)
    user_letter = extract_letter(item.User_Choice)

    # 步骤 2: 路由判断
    if ai_letter == user_letter:
        # 情况 A: 完全匹配
        # 动作: 仅添加到 verified_intents 列表
        # 禁止: 绝对不要在 semantic_residuals 中生成任何内容！
        verified_intents.append(item.User_Choice)
    else:
        # 情况 B: 不匹配 (异常)
        # 动作: 这是唯一的生成残差的条件
        residual = {
            "dimension": extract_topic(item.Question), # 提取题目讨论的核心维度
            "gap": f"AI推测是 '{item.Predicted_Intent}'，但用户选择了 '{item.User_Choice}'",
            "fix": f"强制修正意图为: {item.User_Choice}"
        }
        semantic_residuals.append(residual)

```

# Constraints (关键约束)

1. **残差列表纯净度**：`semantic_residuals` 数组中，**严禁**出现 `ai_letter == user_letter` 的条目。如果所有题目都匹配，该数组必须为 **Empty List `[]**`。
2. **拒绝废话**：不要输出 "原本以为是A，实际是A" 这种毫无意义的残差。这属于逻辑错误。
3. **维度命名**：`dimension` 字段应该是题目的主题（如“写作风格”、“时间背景”），而不是“选项不一致”这种通用描述。

# Output Format (JSON Only)

{
"status": "ALL_MATCHED" | "HAS_MISMATCH",
"verified_intents": ["string", "string"],
"semantic_residuals": [
// 只有当 AI != User 时，这里才会有对象。
// 如果 AI == User，这里必须为空。
]
}
"""

answer_prompt = """
# Role
你是一个精准的单选题答题助手。

# Task
分析用户输入的题目和选项，从中选择最符合题意或用户偏好的一项。

# Constraints
1. **唯一性**：输出结果必须严格仅包含选项的字母（如 A）或选项的具体文本内容。
2. **严禁废话**：严禁输出任何解释、分析、引导语或标点符号。
3. **鲁棒性**：如果无法确定答案，请输出最可能的选项字母。

# Workflow
1. 识别输入的题目背景。
2. 匹配最合适的选项。
3. 仅输出选项对应的字母或内容。

# Examples
输入: 谁是《西游记》的作者？ A.吴承恩 B.罗贯中 C.曹雪芹
输出: A

输入: 你希望在同人小说中主要集中于哪种创作方向？ A.原创新角色 B.根据原著角色续写情节 C.重新演绎原著故事线
输出: C
"""


class Question(BaseModel):
    """Question for the agent."""

    question: str = Field(description="问题")
    options: List[str] = Field(description="选项")

    def __str__(self):
        """String representation of the Question."""
        options_str = "\n".join([f"{option.strip()}" for option in self.options])
        return f"{self.question}\n{options_str}"


class AIAnswerList(BaseModel):
    """AI answer list for the agent."""

    answers: List[str] = Field(description="AI答案列表")

    def __str__(self):
        """String representation of the AIAnswerList."""
        return "\n".join([answer.strip() for answer in self.answers])


class Exam(BaseModel):
    """Exam for the agent."""

    questions: List[Question] = Field(description="问题列表")

    @staticmethod
    def merge(exam_left: "Exam", exam_right: "Exam"):
        """Merge two exams."""
        exam_left.questions.extend(exam_right.questions)
        return exam_left

    def __str__(self):
        """String representation of the Exam."""
        return "\n".join([str(question) for question in self.questions])


class Diff(BaseModel):
    """Diff for the agent."""

    status: Literal["PERFECT_MATCH", "SEMANTIC_GAP"] = Field(
        description="状态", default="SEMANTIC_GAP"
    )
    verified_intents: List[str] = Field(
        description="已验证的意图", default_factory=list
    )
    semantic_residuals: List[str] = Field(description="残差意图", default_factory=list)


def add_answers(anwsers_left: List[str], anwsers_right: List[str]):
    """Add an answer to the state."""
    return [*anwsers_left, *anwsers_right]


class State(MessagesState):
    """State for the agent."""

    user_goal: str = Field(description="用户目标", default="")
    exam: Annotated[Exam, Exam.merge] = Field(
        description="考试", default=Exam(questions=[])
    )
    answers: Annotated[List[str], add_answers] = Field(
        description="答案", default_factory=list
    )
    re_answers: AIAnswerList = Field(
        description="重新回答的答案", default=AIAnswerList(answers=[])
    )
    plan_prompt: str = Field(description="计划提示词", default="")

    diff: Diff | None = Field(description="差异", default=None)


def analyze_user_goal(state: State):
    """Analyze the user goal."""
    llm = create_custom_agent()
    msg = llm.invoke([SystemMessage(content=goal_prompt)] + state["messages"])
    return {"user_goal": msg.content}


def generate_exam(state: State):
    """Generate an exam."""
    llm = create_custom_agent(Exam)
    exam: Exam = llm.invoke(
        [SystemMessage(content=exam_prompt)]
        + [
            AIMessage(content=state["user_goal"]),
        ]
    )
    return {"exam": exam}


def human_answer(state: State):
    """Get an answer from the state."""
    index = len(state["answers"])
    question = state["exam"].questions[index]
    answer = interrupt(f"请选择一个最符合的答案或者提供新答案\n{question}")
    return {
        "answers": [answer],
        "messages": [AIMessage(content=f"{question}"), HumanMessage(content=answer)],
    }


def human_answers_loop(
    state: State,
) -> Command[Literal["human_answer", "router_node"]]:
    """Human answers loop."""
    if len(state["answers"]) < len(state["exam"].questions):
        return Command(goto="human_answer")
    return Command(goto="router_node")


def generate_plan_prompt(state: State):
    """Generate a plan prompt."""
    llm = create_custom_agent()
    msg = llm.invoke([SystemMessage(content=meta_prompt)] + state["messages"])
    return {"plan_prompt": msg.content, "messages": [msg]}


def test_user_goal(state: State):
    """Test the user goal."""
    llm = create_custom_agent()
    system_prompt = f"""{answer_prompt}\n\n ## 用户意图: \n{state["user_goal"]}"""
    re_answers = state.get("re_answers", AIAnswerList(answers=[]))
    answers_index = len(re_answers.answers)
    question = state["exam"].questions[answers_index]
    msg = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=f"{question}")]
    )
    re_answers.answers.append(msg.content)
    return {"re_answers": re_answers}


def get_diff_info(state: State):
    """Get the diff info."""
    compare_index = len(state["re_answers"].answers) - 1
    question = state["exam"].questions[compare_index]
    user_answer = state["answers"][compare_index]
    max_index = len(state["exam"].questions) - 1
    return {
        "message": f"""{question}
原始意图: {state["re_answers"].answers[compare_index]}
转变为: {user_answer}""",
        "max_index": max_index,
        "compare_index": compare_index,
    }


def remove_last_re_answer(state: State):
    """Remove the last re answer."""
    re_answers = state.get("re_answers", AIAnswerList(answers=[]))
    if len(re_answers.answers) > 0:
        re_answers.answers.pop()
    return {"re_answers": re_answers}


def compare_answers(
    state: State,
) -> Command[Literal["router_node", "test_user_goal", "update_user_goal"]]:
    """Compare the answers."""
    llm = create_custom_agent(Diff)
    diff_info = get_diff_info(state)
    compare_index = diff_info["compare_index"]
    max_index = diff_info["max_index"]

    diff: Diff = llm.invoke(
        [
            SystemMessage(content=diff_prompt),
            HumanMessage(content=diff_info["message"]),
        ]
    )
    if diff.status == "PERFECT_MATCH":
        if compare_index == max_index:
            return Command(goto="router_node")
        else:
            return Command(goto="test_user_goal")
    else:
        return Command(goto="update_user_goal")


def update_user_goal(state: State):
    """Update the user goal."""
    llm = create_custom_agent()
    diff_info = get_diff_info(state)
    message = diff_info["message"]
    msg = llm.invoke(
        [
            SystemMessage(content=goal_prompt),
            HumanMessage(
                content=f"# 原始用户意图: {state['user_goal']}\n\n\n\n## 对齐信息: \n{message}\n --- \n请根据对齐信息重新生成用户意图"
            ),
        ]
    )
    update_state = remove_last_re_answer(state)
    return {"user_goal": msg.content, **update_state}


def router_node(
    state: State,
) -> Command[Literal["analyze_user_goal", "test_user_goal", "create_plan_prompt"]]:
    """Router node."""
    if state.get("user_goal") is None:
        return Command(goto="analyze_user_goal")
    if state.get("exam") is not None:
        re_answers = state.get("re_answers", AIAnswerList(answers=[]))
        if len(re_answers.answers) == len(state["exam"].questions):
            return Command(goto="create_plan_prompt")
        else:
            return Command(goto="test_re_answers")
    return Command(goto="test_re_answers")


def create_plan_prompt(state: State):
    """Create a plan prompt."""
    llm = create_custom_agent()
    msg = llm.invoke(
        [SystemMessage(content=meta_prompt)]
        + state["messages"]
        + [
            HumanMessage(
                content=f"## 用户意图: \n{state['user_goal']} \n --- \n 请根据用户意图生成一个用于创建任务计划的提示词"
            )
        ]
    )
    return {"plan_prompt": msg.content, "messages": [msg]}


enhance_prompt_builder = StateGraph(State)
enhance_prompt_builder.add_node("router_node", router_node)
enhance_prompt_builder.add_node("analyze_user_goal", analyze_user_goal)
enhance_prompt_builder.add_node("generate_exam", generate_exam)
enhance_prompt_builder.add_node("human_answer", human_answer)
enhance_prompt_builder.add_node("human_answers_loop", human_answers_loop)
enhance_prompt_builder.add_node("plan_prompt", generate_plan_prompt)
enhance_prompt_builder.add_node("update_user_goal", update_user_goal)
enhance_prompt_builder.add_node("test_user_goal", test_user_goal)
enhance_prompt_builder.add_node("compare_answers", compare_answers)
enhance_prompt_builder.add_node("create_plan_prompt", create_plan_prompt)
enhance_prompt_builder.add_edge(START, "router_node")
enhance_prompt_builder.add_edge("analyze_user_goal", "generate_exam")
enhance_prompt_builder.add_edge("generate_exam", "human_answers_loop")
enhance_prompt_builder.add_conditional_edges("human_answers_loop", human_answers_loop)
enhance_prompt_builder.add_edge("human_answer", "human_answers_loop")
enhance_prompt_builder.add_edge("test_user_goal", "compare_answers")
enhance_prompt_builder.add_conditional_edges("compare_answers", compare_answers)
enhance_prompt_builder.add_edge("update_user_goal", "test_user_goal")
enhance_prompt_builder.add_edge("create_plan_prompt", END)

graph = enhance_prompt_builder.compile()
