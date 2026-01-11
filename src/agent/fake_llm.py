#!/usr/bin/env python3
"""测试 LangGraph 的流式输出.

通过 GenericFakeChatModel 模拟 LLM 的流式输出，用于测试 LangGraph 的流式输出。
"""

import asyncio

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolCall

# 预定义响应序列 - 可控输出
fake_responses = iter(
    [
        # 第一步：模拟工具调用
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="search", args={"query": "test query"}, id="call_1")
            ],
        ),
        # 第二步：最终响应
        AIMessage(content="Graph 测试完成，结果如预期。"),
    ]
)

fakellm = GenericFakeChatModel(messages=fake_responses)

# 测试非流式
result = fakellm.invoke([("human", "测试 graph")])
print(result)  # 第一条响应（带工具调用）
print("--------------------------------")


# 测试流式输出
async def test_stream():
    async for chunk in fakellm.astream([("human", "流式测试")]):
        print(chunk.content, end="", flush=True)  # 逐 token 流式输出


asyncio.run(test_stream())
