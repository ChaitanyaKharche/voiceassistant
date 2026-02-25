"""
LLM Service — handles OpenAI chat completions with tool/function calling.

Converts Retell transcript → OpenAI messages, streams responses back,
and executes tool calls (calendar creation) inline.
"""

import json
import logging
from datetime import datetime
from typing import AsyncGenerator

from openai import AsyncOpenAI

from server.config import get_settings
from server.models import CalendarEvent, CalendarEventResult
from server.calendar_service import calendar_service

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a friendly, professional voice scheduling assistant. Your job is to help callers schedule meetings by collecting their information through natural conversation.

## Today's Date
{datetime.now().strftime("%A, %B %d, %Y")}

## Conversation Flow
1. **Greet** the caller warmly and ask for their name.
2. **Ask** for their preferred date and time for the meeting.
3. **Optionally ask** if they'd like to add a meeting title/subject.
4. **Confirm** all collected details by reading them back clearly.
5. **Wait for explicit confirmation** ("yes", "correct", "that's right", etc.) before creating the event.
6. **Create** the calendar event using the `create_calendar_event` function.
7. **Confirm** the event was created and wish them a great day.

## Important Rules
- Be conversational and natural — this is a VOICE call, not a text chat.
- Keep responses SHORT (1-3 sentences max). Long responses sound robotic over voice.
- When the user says a relative date like "tomorrow", "next Monday", "this Friday", convert it to an actual date based on today's date above.
- For times, accept natural language ("3 PM", "three in the afternoon", "15:00") and convert to 24h HH:MM format.
- If the user gives an ambiguous time (e.g. "3 o'clock" without AM/PM), ask for clarification.
- Default meeting duration is 30 minutes unless the user specifies otherwise.
- Default timezone is US Eastern (America/New_York) unless the user mentions another.
- If the user wants to change something after confirmation, cheerfully accommodate.
- If you can't understand something, ask politely for clarification.
- NEVER make up or assume details the user hasn't provided.
- Do NOT use markdown, bullet points, or any text formatting — this is voice output.
- Use the create_calendar_event function ONLY after the user has explicitly confirmed the details.
"""

# ──────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": (
                "Create a Google Calendar event. Call this ONLY after the user "
                "has explicitly confirmed all meeting details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "attendee_name": {
                        "type": "string",
                        "description": "Full name of the person scheduling the meeting.",
                    },
                    "event_date": {
                        "type": "string",
                        "description": "Meeting date in YYYY-MM-DD format.",
                    },
                    "event_time": {
                        "type": "string",
                        "description": "Meeting start time in HH:MM 24-hour format.",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration in minutes. Default 30.",
                        "default": 30,
                    },
                    "title": {
                        "type": "string",
                        "description": "Meeting title/subject. Default 'Scheduled Meeting'.",
                        "default": "Scheduled Meeting",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone. Default 'America/New_York'.",
                        "default": "America/New_York",
                    },
                },
                "required": ["attendee_name", "event_date", "event_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check if a time slot is available on the calendar before booking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_date": {
                        "type": "string",
                        "description": "Date to check in YYYY-MM-DD format.",
                    },
                    "event_time": {
                        "type": "string",
                        "description": "Time to check in HH:MM 24-hour format.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone. Default 'America/New_York'.",
                        "default": "America/New_York",
                    },
                },
                "required": ["event_date", "event_time"],
            },
        },
    },
]


# ──────────────────────────────────────────────
# Tool Executor
# ──────────────────────────────────────────────

async def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool call and return the result as a string."""
    logger.info(f"Executing tool: {name} with args: {arguments}")

    if name == "create_calendar_event":
        event = CalendarEvent(
            attendee_name=arguments["attendee_name"],
            event_date=arguments["event_date"],
            event_time=arguments["event_time"],
            duration_minutes=arguments.get("duration_minutes", 30),
            title=arguments.get("title", "Scheduled Meeting"),
            timezone=arguments.get("timezone", "America/New_York"),
        )
        result: CalendarEventResult = await calendar_service.create_event(event)
        return json.dumps(result.model_dump())

    elif name == "check_availability":
        available = await calendar_service.check_availability(
            date=arguments["event_date"],
            time=arguments["event_time"],
            timezone=arguments.get("timezone", "America/New_York"),
        )
        return json.dumps({"available": available})

    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# ──────────────────────────────────────────────
# LLM Service
# ──────────────────────────────────────────────

class LLMService:
    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4.1-nano"  # cost-effective, fast, good at function calling

    def _build_messages(self, transcript: list[dict]) -> list[dict]:
        """Convert Retell transcript format → OpenAI messages format."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for entry in transcript:
            role = "assistant" if entry.get("role") == "agent" else "user"
            messages.append({"role": role, "content": entry.get("content", "")})
        return messages

    async def get_response(self, transcript: list[dict]) -> str:
        """
        Non-streaming response. Handles tool calls recursively until
        the model produces a final text response.
        """
        messages = self._build_messages(transcript)
        return await self._complete(messages)

    async def _complete(self, messages: list[dict], depth: int = 0) -> str:
        """Recursive completion handler for tool call chains."""
        if depth > 3:
            return "I'm having trouble processing that. Could you repeat your request?"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=256,  # keep voice responses short
            )

            choice = response.choices[0]

            # ── Direct text response ──
            if choice.finish_reason == "stop":
                return choice.message.content or ""

            # ── Tool calls ──
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                # Append assistant message with tool calls
                messages.append(choice.message.model_dump())

                for tool_call in choice.message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    result = await execute_tool(fn_name, fn_args)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )

                # Recurse to get final text response after tool execution
                return await self._complete(messages, depth + 1)

            # Fallback
            return choice.message.content or "I'm sorry, could you say that again?"

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "I'm experiencing a brief issue. Could you try again in a moment?"

    async def stream_response(
        self, transcript: list[dict]
    ) -> AsyncGenerator[str, None]:
        """
        Streaming response generator. Yields text chunks as they arrive.
        If a tool call is detected, executes it and then streams the follow-up.
        """
        messages = self._build_messages(transcript)

        async for chunk in self._stream_complete(messages):
            yield chunk

    async def _stream_complete(
        self, messages: list[dict], depth: int = 0
    ) -> AsyncGenerator[str, None]:
        """Streaming completion with tool call handling."""
        if depth > 3:
            yield "I'm having trouble processing that. Could you repeat your request?"
            return

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=256,
                stream=True,
            )

            collected_content = ""
            tool_calls_data: dict[int, dict] = {}  # index -> {id, name, arguments}
            current_finish_reason = None

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                finish_reason = chunk.choices[0].finish_reason if chunk.choices else None

                if finish_reason:
                    current_finish_reason = finish_reason

                if delta is None:
                    continue

                # ── Streaming text content ──
                if delta.content:
                    collected_content += delta.content
                    yield delta.content

                # ── Accumulate tool call fragments ──
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_data:
                            tool_calls_data[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc.id:
                            tool_calls_data[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_data[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_data[idx]["arguments"] += tc.function.arguments

            # ── If tool calls were made, execute and recurse ──
            if tool_calls_data and current_finish_reason == "tool_calls":
                # Build the assistant message with tool calls
                tool_calls_list = []
                for idx in sorted(tool_calls_data.keys()):
                    tc = tool_calls_data[idx]
                    tool_calls_list.append(
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                    )

                assistant_msg = {
                    "role": "assistant",
                    "content": collected_content or None,
                    "tool_calls": tool_calls_list,
                }
                messages.append(assistant_msg)

                # Execute each tool
                for tc in tool_calls_list:
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])
                    result = await execute_tool(fn_name, fn_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        }
                    )

                # Stream the follow-up response
                async for chunk in self._stream_complete(messages, depth + 1):
                    yield chunk

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield "I'm experiencing a brief issue. Could you try again in a moment?"


# Singleton
llm_service = LLMService()
