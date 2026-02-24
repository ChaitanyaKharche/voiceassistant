"""
Retell AI Custom LLM WebSocket Handler.

Implements the Retell ↔ Custom LLM WebSocket protocol:
  - Receives transcript updates and response requests from Retell
  - Streams LLM responses back in real time
  - Handles ping/pong keepalive
  - Manages response_id sequencing and cancellation

Protocol reference: https://docs.retellai.com/api-references/llm-websocket

Key protocol rules:
  - All outgoing events MUST include a `response_type` field
  - WebSocket endpoint must include /{call_id} path parameter
  - First message from server should be a config event (optional) + response event (begin message)
  - response_id from Retell is auto-incrementing; send it back in your response events
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from server.llm_service import llm_service

logger = logging.getLogger(__name__)


class RetellWebSocketHandler:
    """Handles a single Retell WebSocket connection (one per call)."""

    def __init__(self, websocket: WebSocket, call_id: str):
        self.ws = websocket
        self.call_id = call_id
        self.current_task: Optional[asyncio.Task] = None

    async def handle(self):
        """Main loop — receive messages and dispatch."""
        try:
            await self.ws.accept()
            logger.info(f"Retell WebSocket connected (call_id={self.call_id})")

            # Send optional config event + initial greeting (begin message)
            await self._send_config()
            await self._send_initial_greeting()

            async for raw in self.ws.iter_text():
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {raw[:100]}")
                    continue

                interaction_type = data.get("interaction_type", "")

                if interaction_type == "call_details":
                    logger.info(f"Call details received for {self.call_id}")

                elif interaction_type == "ping_pong":
                    await self._handle_ping(data)

                elif interaction_type == "update_only":
                    # Transcript update, no response needed
                    pass

                elif interaction_type == "response_required":
                    await self._handle_response_required(data)

                elif interaction_type == "reminder_required":
                    await self._handle_reminder(data)

                else:
                    logger.debug(f"Unhandled interaction type: {interaction_type}")

        except WebSocketDisconnect:
            logger.info(f"Retell WebSocket disconnected (call_id={self.call_id})")
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
        finally:
            if self.current_task and not self.current_task.done():
                self.current_task.cancel()

    # ──────────────────────────────────────────
    # Protocol Handlers
    # ──────────────────────────────────────────

    async def _send_config(self):
        """Send optional config event at connection open."""
        await self.ws.send_json({
            "response_type": "config",
            "config": {
                "auto_reconnect": True,
                "call_details": True,
            },
        })

    async def _handle_ping(self, data: dict):
        """Respond to keepalive ping with response_type."""
        await self.ws.send_json({
            "response_type": "ping_pong",
            "timestamp": data.get("timestamp", 0),
        })

    async def _handle_response_required(self, data: dict):
        """User stopped speaking — generate and stream LLM response."""
        # Cancel any in-progress response
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass

        response_id = data.get("response_id", 0)
        transcript = data.get("transcript", [])

        # Launch streaming response as a task (so next interrupt can cancel it)
        self.current_task = asyncio.create_task(
            self._stream_response(transcript, response_id)
        )

    async def _handle_reminder(self, data: dict):
        """Handle silence reminder — nudge the user."""
        response_id = data.get("response_id", 0)
        transcript = data.get("transcript", [])

        # Add a hint for the LLM that the user has been silent
        nudge_transcript = transcript + [
            {
                "role": "user",
                "content": "[The user has been silent for a while. Gently prompt them or ask if they're still there.]",
            }
        ]

        self.current_task = asyncio.create_task(
            self._stream_response(nudge_transcript, response_id)
        )

    # ──────────────────────────────────────────
    # Response Streaming
    # ──────────────────────────────────────────

    async def _send_initial_greeting(self):
        """Send the first agent utterance (begin message) when the call connects."""
        greeting = (
            "Hi there! Welcome to our scheduling assistant. "
            "I can help you book a meeting. What's your name?"
        )
        await self.ws.send_json({
            "response_type": "response",
            "response_id": 0,
            "content": greeting,
            "content_complete": True,
            "end_call": False,
        })

    async def _stream_response(self, transcript: list[dict], response_id: int):
        """Stream LLM response to Retell in chunks."""
        try:
            full_response = ""
            chunk_buffer = ""
            min_chunk_size = 3  # send at least N chars per chunk for smooth TTS

            async for text_chunk in llm_service.stream_response(transcript):
                chunk_buffer += text_chunk
                full_response += text_chunk

                # Send chunks at sentence boundaries or when buffer is large enough
                if (
                    len(chunk_buffer) >= min_chunk_size
                    and any(chunk_buffer.rstrip().endswith(p) for p in [".", "!", "?", ","])
                ) or len(chunk_buffer) > 50:
                    await self.ws.send_json({
                        "response_type": "response",
                        "response_id": response_id,
                        "content": chunk_buffer,
                        "content_complete": False,
                        "end_call": False,
                    })
                    chunk_buffer = ""

            # Send remaining buffer + mark complete
            should_end = self._should_end_call(full_response)
            await self.ws.send_json({
                "response_type": "response",
                "response_id": response_id,
                "content": chunk_buffer,
                "content_complete": True,
                "end_call": should_end,
            })

            logger.info(
                f"Response {response_id} complete ({len(full_response)} chars, end_call={should_end})"
            )

        except asyncio.CancelledError:
            logger.debug(f"Response {response_id} cancelled (user interrupted)")
        except Exception as e:
            logger.error(f"Error streaming response: {e}", exc_info=True)
            # Send error recovery message
            try:
                await self.ws.send_json({
                    "response_type": "response",
                    "response_id": response_id,
                    "content": "I'm sorry, I had a brief hiccup. Could you repeat that?",
                    "content_complete": True,
                    "end_call": False,
                })
            except Exception:
                pass

    def _should_end_call(self, response: str) -> bool:
        """Heuristic: end the call if the agent says goodbye after event creation."""
        farewell_signals = [
            "have a great day",
            "have a wonderful day",
            "goodbye",
            "bye bye",
            "take care",
            "talk to you later",
        ]
        response_lower = response.lower()
        return any(signal in response_lower for signal in farewell_signals)