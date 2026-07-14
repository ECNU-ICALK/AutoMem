"""Agent-side phase trigger contract for fixed G3 behavior."""

from automem.memory_types import MemoryStatus
from flashoagents.agents import ToolCallingAgent


def test_agent_requests_memory_only_once_after_a_refresh_boundary():
    agent = object.__new__(ToolCallingAgent)
    agent.step_number = 4
    agent._memory_refresh_pending = False
    agent._memory_refresh_used = False
    calls = []

    def get_guidance(status, step_number=0, *, refresh_boundary=False):
        calls.append((status, step_number, refresh_boundary))
        return "new guidance"

    agent._get_memory_guidance = get_guidance

    assert agent._consume_pending_memory_refresh() is None
    assert calls == []

    agent._memory_refresh_pending = True
    assert agent._consume_pending_memory_refresh() == "new guidance"
    assert calls == [(MemoryStatus.IN, 4, True)]
    assert not agent._memory_refresh_pending
    assert agent._memory_refresh_used

    agent._memory_refresh_pending = True
    assert agent._consume_pending_memory_refresh() is None
    assert calls == [(MemoryStatus.IN, 4, True)]

