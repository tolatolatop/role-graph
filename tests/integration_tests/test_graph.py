import pytest

from agent.role_graph.standalone import graph

pytestmark = pytest.mark.anyio


@pytest.mark.skip(reason="Skipping test for now")
@pytest.mark.langsmith
async def test_agent_simple_passthrough() -> None:
    inputs = {"changeme": "some_val"}
    res = await graph.ainvoke(inputs)
    assert res is not None
