#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this file are modifications by OPPO PersonalAI Team.
# Licensed under the Apache License, Version 2.0.

from dataclasses import asdict, dataclass
from logging import getLogger
from typing import Any, Dict, List, TypedDict, Union

from .models import MessageRole
from .utils import AgentError, make_json_serializable


logger = getLogger(__name__)


class Message(TypedDict):
    role: MessageRole
    content: str | list[dict]

@dataclass
class ToolCall:
    name: str
    arguments: Any
    id: str

    def dict(self):
        return {
            "name": self.name,
            "arguments": make_json_serializable(self.arguments),
        }

@dataclass
class MemoryStep:
    def dict(self):
        return asdict(self)

    def to_messages(self, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    model_input_messages: List[Message] | None = None
    model_output_messages: List[Message] | None = None
    tool_calls: List[ToolCall] | None = None
    start_time: float | None = None
    end_time: float | None = None
    step_number: int | None = None
    error: AgentError | None = None
    duration: float | None = None
    observations: str | None = None
    observations_images: List[str] | None = None
    action_output: Any = None
    action_think: Any = None
    action_reasoning: Any = None
    score: float = 0.0
    evaluate_thought: str | None = None
    memory_guidance: str | None = None
    
    def dict(self):
        return {
            "model_input_messages": self.model_input_messages,
            "model_output_messages": self.model_output_messages,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "step_number": self.step_number,
            "error": self.error.dict() if self.error else None,
            "duration": self.duration,
            "observations": self.observations,
            "action_think": self.action_think,
            "action_output": make_json_serializable(self.action_output),
            "action_reasoning": self.action_reasoning,
            "score": self.score,
            "evaluate_thought": self.evaluate_thought,
        }

    def to_messages(self, summary_mode: bool = False, show_model_input_messages: bool = False) -> List[Message]:
        messages = []
        
        # Add memory guidance if present
        if self.memory_guidance:
            formatted_memory = (
                "————Memory System Guidance————\n"
                f"{self.memory_guidance}\n"
                "————End Memory————"
            )
            messages.append(
                Message(
                    role=MessageRole.USER, 
                    content=[{"type": "text", "text": formatted_memory}]
                )
            )
        
        if self.model_input_messages is not None and show_model_input_messages:
            messages.append(Message(role=MessageRole.SYSTEM, content=self.model_input_messages))

        if self.tool_calls is not None:
            tool_output = {
                "tools":[tc.dict() for tc in self.tool_calls]
            }
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str(tool_output),
                        }
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Tool calling observation:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                Message(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )
        return messages


# ============================================================
# L1 + L2 instruction helpers (Codex Q3-7 fix, 2026-04-28)
# ------------------------------------------------------------
# Previously the L1 instruction header + L2 "acknowledge" step were only
# emitted by PlanningStep.to_messages (history serialization). The actual
# initial planning model call in agents.py::planning_step never saw them,
# so the plan could not comply. Extracting the templates here lets both
# entry points share identical wording.
# ============================================================

L1_MEMORY_INSTRUCTION_PREFIX = (
    "————Memory System Guidance————\n"
    "The memories below were retrieved from PRIOR tasks. They are HINTS, "
    "not authoritative facts. Most retrieved memories are WEAK MATCHES — "
    "you must actively decide which (if any) apply.\n"
    "\n"
    "★ DEFAULT TO IGNORE (toxic-memory fix 2026-04-30):\n"
    "If a memory's Apply-when does not OBVIOUSLY fit THIS question, IGNORE\n"
    "it and answer from your own knowledge. Following an irrelevant memory\n"
    "is worse than having no memory. In particular:\n"
    "  • [Match: VERY_LOW] memories should be ignored unless Apply-when\n"
    "    is a precise semantic fit (not just keyword overlap).\n"
    "  • Memories whose Source domain differs from THIS task should be\n"
    "    treated as INSPIRATION only — never copy their tool sequences.\n"
    "  • If you can't write a 1-sentence concrete reason a memory helps\n"
    "    THIS specific question, ignore it.\n"
    "\n"
    "Common keyword traps (memory fires on surface keywords but applies\n"
    "to a totally different operation):\n"
    "  • 'first year stock crossed $50' ≠ 'first or earliest publication'\n"
    "  • 'King of Pop fifth single' is NOT a name-format question\n"
    "  • 'museum accession of a portrait' is NOT a taxonomic lookup\n"
    "  • 'COUNTING visual elements in video' ≠ 'IDENTIFYING species in video'\n"
    "If the memory's instruction would not directly improve your answer,\n"
    "discard it explicitly and proceed without it.\n"
    "\n"
    "Field semantics:\n"
    "  • [TYPE] tag — TIP=heuristic, WORKFLOW=tool chain, "
    "TRAJECTORY=past episode,\n"
    "                 INSIGHT=failure lesson, SHORTCUT=parameterized macro\n"
    "  • → Use:     — type-specific guidance on HOW to apply this memory\n"
    "  • Apply when — observable triggers; ALL conditions must match THIS\n"
    "                 question (not just one keyword overlap).\n"
    "  • Avoid when — anti-triggers; if any matches, IGNORE this memory\n"
    "  • Source domain — abstracted prior-task category. Differing domain\n"
    "                    = treat as inspiration only, not a recipe.\n"
    "  • [Match: HIGH/MEDIUM/VERY_LOW] — pre-filtered confidence band.\n"
    "                    VERY_LOW means almost certainly off-topic.\n"
    "  • Judge note — distilled action / rationale (1 sentence)\n"
    "  • [TRAJECTORY ⚠ NEGATIVE EXAMPLE] — DO NOT replay; learn what to AVOID\n"
    "\n"
    "Usage protocol (follow strictly):\n"
    "  1. Scan each memory; check Apply-when / Avoid-when / Source domain /\n"
    "     Match-band. EXPECT to discard most.\n"
    "  2. For the few applicable ones, integrate the principle / workflow\n"
    "     into your plan. NEVER copy specifics from a Trajectory verbatim;\n"
    "     adapt to the current task.\n"
    "  3. If memories conflict or look irrelevant, IGNORE them and answer\n"
    "     using your own knowledge directly.\n"
    "  4. Memories are advisory only — your reasoning over the current\n"
    "     question remains primary, and 'no applicable memory' is a\n"
    "     valid conclusion.\n"
    "\n"
)

L1_MEMORY_INSTRUCTION_SUFFIX = "\n————End Memory————"

L2_ACKNOWLEDGE_INSTRUCTION = (
    "Before planning, briefly state in your reasoning:\n"
    "  - Which 1-2 memories you will APPLY (cite [TYPE] + topic and the "
    "matching Apply-when).\n"
    "  - Which memories you will DISCARD (cite reason: e.g. Source-domain "
    "mismatch, Avoid-when triggered, conflicting advice).\n"
    "Then proceed with the plan."
)


def format_memory_guidance_block(memory_guidance: str) -> str:
    """Return the full L1-wrapped memory block (without L2 acknowledge)."""
    if not memory_guidance:
        return ""
    return (
        L1_MEMORY_INSTRUCTION_PREFIX
        + memory_guidance
        + L1_MEMORY_INSTRUCTION_SUFFIX
    )


@dataclass
class PlanningStep(MemoryStep):
    model_input_messages: List[Message]
    plan: str
    plan_think: str
    plan_reasoning: str
    memory_guidance: str | None = None

    def to_messages(self, summary_mode: bool, **kwargs) -> List[Message]:
        messages = []
        # Add memory guidance if present (Codex Q3-7: shared template).
        if self.memory_guidance:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=[{"type": "text", "text":
                              format_memory_guidance_block(self.memory_guidance)}]
                )
            )
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=[{"type": "text", "text": L2_ACKNOWLEDGE_INSTRUCTION}]
                )
            )

        messages.append(
            Message(
                role=MessageRole.USER, content=[{"type": "text", "text": "Now, begin your planning analysis for this task!"}]
            )
        )
        messages.append(
            Message(
                role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"[PLAN]:\n{self.plan.strip()}"}]
            )
        )
        return messages
    
@dataclass
class SummaryStep(MemoryStep):
    model_input_messages: List[Message]
    summary: str
    summary_reasoning: str

    def to_messages(self, summary_mode: bool, **kwargs) -> List[Message]:
        messages = []
        messages.append(
            Message(
                role=MessageRole.USER, content=[{"type": "text", "text": "Now, summarize and analysis the task completion status and provide recommendations for next steps!"}]
            )
        )
        messages.append(
            Message(
                role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"[SUMMARY]:\n{self.summary.strip()}"}]
            )
        )
        return messages

@dataclass
class TaskStep(MemoryStep):
    task: str
    task_images: List[str] | None = None

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[Message]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]

        return [Message(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[Message]:
        if summary_mode:
            return []
        return [Message(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


class AgentMemory:
    def __init__(self, system_prompt: str):
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: List[Union[TaskStep, ActionStep, PlanningStep, SummaryStep]] = []

    def reset(self):
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        return [step.dict() for step in self.steps]