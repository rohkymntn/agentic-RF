"""
LLM-powered antenna design agent.

The orchestrator manages the conversation with the LLM, executing tools
and maintaining design state across the session.

Supports OpenAI and Anthropic APIs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional
from dataclasses import dataclass, field

from .tools import AGENT_TOOLS, get_tools_for_llm, execute_tool
from .prompts import SYSTEM_PROMPT


@dataclass
class DesignSession:
    """Tracks the state of an antenna design session."""
    # Current design
    template_name: str | None = None
    current_params: dict[str, float] = field(default_factory=dict)

    # Target spec
    target_freq_mhz: float = 403.5
    target_band: tuple[float, float] = (402.0, 405.0)
    max_size_mm: float = 15.0
    application: str = "implant"

    # History
    designs_evaluated: int = 0
    best_s11_db: float = 0.0
    optimization_results: list[dict] = field(default_factory=list)
    measurements_loaded: list[str] = field(default_factory=list)

    # Conversation
    messages: list[dict] = field(default_factory=list)


class AntennaDesignAgent:
    """
    Agentic antenna design system.

    Combines LLM reasoning with engineering tools for
    solver-in-the-loop and measurement-in-the-loop design.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
        max_tool_calls: int = 10,
    ):
        """
        Args:
            provider: "openai" or "anthropic"
            model: Model name (default: gpt-4o for OpenAI, claude-sonnet for Anthropic)
            api_key: API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)
            max_tool_calls: Maximum tool calls per turn
        """
        self.provider = provider
        self.model = model or self._default_model()
        self.api_key = api_key or self._get_api_key()
        self.max_tool_calls = max_tool_calls
        self.session = DesignSession()
        self._client = None

        # Initialize conversation with system prompt
        self.session.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _default_model(self) -> str:
        if self.provider == "openai":
            return "gpt-4o"
        elif self.provider == "anthropic":
            return "claude-sonnet-4-20250514"
        return "gpt-4o"

    def _get_api_key(self) -> str:
        if self.provider == "openai":
            return os.environ.get("OPENAI_API_KEY", "")
        elif self.provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY", "")
        return ""

    def _get_client(self):
        """Lazy-initialize the API client."""
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)

        return self._client

    def chat(self, user_message: str) -> str:
        """
        Send a message to the agent and get a response.

        The agent may call tools multiple times before responding.

        Args:
            user_message: User's design request or question

        Returns:
            Agent's text response
        """
        self.session.messages.append({"role": "user", "content": user_message})

        if self.provider == "openai":
            return self._chat_openai()
        elif self.provider == "anthropic":
            return self._chat_anthropic()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _chat_openai(self) -> str:
        """Run agent loop with OpenAI API."""
        client = self._get_client()
        tools = get_tools_for_llm("openai")

        for _ in range(self.max_tool_calls):
            response = client.chat.completions.create(
                model=self.model,
                messages=self.session.messages,
                tools=tools if tools else None,
                tool_choice="auto",
            )

            msg = response.choices[0].message

            # If no tool calls, return the text response
            if not msg.tool_calls:
                text = msg.content or ""
                self.session.messages.append({"role": "assistant", "content": text})
                return text

            # Process tool calls
            self.session.messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                result = execute_tool(name, args)
                result_str = json.dumps(result, default=str) if not isinstance(result, str) else result

                self.session.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

        # If we hit the tool call limit
        return "I've reached the maximum number of tool calls for this turn. Please continue with your next request."

    def _chat_anthropic(self) -> str:
        """Run agent loop with Anthropic API."""
        client = self._get_client()
        tools = get_tools_for_llm("anthropic")

        # Anthropic uses system as a separate param
        system_msg = self.session.messages[0]["content"]
        messages = [m for m in self.session.messages[1:] if m["role"] != "system"]

        for _ in range(self.max_tool_calls):
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_msg,
                messages=messages,
                tools=tools,
            )

            # Check for tool use
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if not tool_use_blocks:
                text = " ".join(b.text for b in text_blocks)
                self.session.messages.append({"role": "assistant", "content": text})
                return text

            # Build assistant message
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            messages.append({"role": "assistant", "content": assistant_content})
            self.session.messages.append({"role": "assistant", "content": str(assistant_content)})

            # Execute tools and build tool results
            tool_results = []
            for block in tool_use_blocks:
                result = execute_tool(block.name, block.input)
                result_str = json.dumps(result, default=str) if not isinstance(result, str) else result
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})

        return "Maximum tool calls reached. Please continue."

    def chat_offline(self, user_message: str) -> str:
        """
        Offline mode: no LLM API needed. Directly interpret common commands.

        Useful for testing or when no API key is available.
        """
        msg = user_message.lower().strip()

        if "list templates" in msg or "what templates" in msg:
            return execute_tool("list_templates", {})

        if "optimize" in msg:
            # Extract template name if mentioned
            for key in ["mifa", "pifa", "loop", "patch", "helix", "meander", "spiral"]:
                if key in msg:
                    result = execute_tool("optimize", {
                        "template_name": key,
                        "target_freq_mhz": self.session.target_freq_mhz,
                        "max_radius_mm": self.session.max_size_mm,
                        "n_iterations": 30,
                    })
                    return json.dumps(result, indent=2, default=str)

        if "chu limit" in msg or "fundamental limit" in msg:
            result = execute_tool("chu_limit", {
                "freq_mhz": self.session.target_freq_mhz,
                "radius_mm": self.session.max_size_mm,
            })
            return json.dumps(result, indent=2, default=str)

        if "material" in msg:
            if "list" in msg:
                return execute_tool("list_materials", {})
            for word in msg.split():
                try:
                    return execute_tool("material_info", {"name": word})
                except Exception:
                    continue

        return (
            "Available commands (offline mode):\n"
            "  - 'list templates' - show antenna templates\n"
            "  - 'optimize [template]' - run optimization\n"
            "  - 'chu limit' - fundamental size limit\n"
            "  - 'list materials' - show material database\n"
            "\nFor full agentic mode, set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )

    def reset_session(self):
        """Reset the design session."""
        self.session = DesignSession()
        self.session.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
