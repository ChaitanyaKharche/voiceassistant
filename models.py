"""
Data models for conversation state, calendar events, and Retell protocol messages.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Calendar Event
# ──────────────────────────────────────────────

class CalendarEvent(BaseModel):
    attendee_name: str
    event_date: str  # YYYY-MM-DD
    event_time: str  # HH:MM (24h)
    duration_minutes: int = 30
    title: str = "Scheduled Meeting"
    timezone: str = "America/New_York"


class CalendarEventResult(BaseModel):
    success: bool
    event_id: Optional[str] = None
    event_link: Optional[str] = None
    error: Optional[str] = None
    summary: str = ""


# ──────────────────────────────────────────────
# Conversation State
# ──────────────────────────────────────────────

class ConversationPhase(str, Enum):
    GREETING = "greeting"
    COLLECTING_NAME = "collecting_name"
    COLLECTING_DATETIME = "collecting_datetime"
    COLLECTING_TITLE = "collecting_title"
    CONFIRMING = "confirming"
    CREATING_EVENT = "creating_event"
    COMPLETED = "completed"
    ERROR = "error"


class ConversationState(BaseModel):
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phase: ConversationPhase = ConversationPhase.GREETING
    attendee_name: Optional[str] = None
    event_date: Optional[str] = None
    event_time: Optional[str] = None
    duration_minutes: int = 30
    title: Optional[str] = None
    timezone: str = "America/New_York"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    event_result: Optional[CalendarEventResult] = None


# ──────────────────────────────────────────────
# Retell WebSocket Protocol
# ──────────────────────────────────────────────

class RetellTranscriptEntry(BaseModel):
    role: str  # "agent" or "user"
    content: str


class RetellRequest(BaseModel):
    """Incoming message from Retell over WebSocket."""
    interaction_type: str  # call_details | update_only | response_required | ping_pong | tool_call_result
    transcript: list[RetellTranscriptEntry] = []
    call: Optional[dict] = None  # present for call_details
    timestamp: Optional[int] = None  # present for ping_pong


class RetellResponse(BaseModel):
    """Outgoing message to Retell over WebSocket."""
    response_id: int
    content: str
    content_complete: bool = True
    end_call: bool = False
