"""
Google Calendar integration using a service account.

Setup:
1. Create a Google Cloud project
2. Enable Google Calendar API
3. Create a service account and download the JSON key
4. Base64-encode the JSON key: base64 -w 0 credentials.json
5. Set GOOGLE_CREDENTIALS_JSON env var to the encoded string
6. Share your calendar with the service account email
"""

import base64
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from server.models import CalendarEvent, CalendarEventResult
from server.config import get_settings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]


class CalendarService:
    def __init__(self):
        self._service = None

    def _get_service(self):
        """Lazy-init the Google Calendar API client."""
        if self._service is not None:
            return self._service

        settings = get_settings()
        if not settings.google_credentials_json:
            logger.warning("No Google credentials configured — calendar integration disabled")
            return None

        try:
            creds_json = base64.b64decode(settings.google_credentials_json)
            creds_dict = json.loads(creds_json)
            credentials = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
            self._service = build("calendar", "v3", credentials=credentials)
            logger.info("Google Calendar service initialized")
            return self._service
        except Exception as e:
            logger.error(f"Failed to init Google Calendar service: {e}")
            return None

    async def create_event(self, event: CalendarEvent) -> CalendarEventResult:
        """Create a calendar event and return the result."""
        service = self._get_service()

        # ── Fallback: if no credentials, return a mock success for demo ──
        if service is None:
            logger.info("Calendar service unavailable — returning mock event")
            return CalendarEventResult(
                success=True,
                event_id="mock-event-id",
                event_link="https://calendar.google.com",
                summary=(
                    f"[DEMO MODE] '{event.title}' with {event.attendee_name} "
                    f"on {event.event_date} at {event.event_time}"
                ),
            )

        try:
            # Parse date + time into datetime
            start_dt = datetime.strptime(
                f"{event.event_date} {event.event_time}", "%Y-%m-%d %H:%M"
            )
            end_dt = start_dt + timedelta(minutes=event.duration_minutes)

            event_body = {
                "summary": event.title or "Scheduled Meeting",
                "description": f"Meeting scheduled via Voice Assistant with {event.attendee_name}.",
                "start": {
                    "dateTime": start_dt.isoformat(),
                    "timeZone": event.timezone,
                },
                "end": {
                    "dateTime": end_dt.isoformat(),
                    "timeZone": event.timezone,
                },
                "reminders": {
                    "useDefault": False,
                    "overrides": [
                        {"method": "popup", "minutes": 10},
                    ],
                },
            }

            settings = get_settings()
            result = (
                service.events()
                .insert(calendarId=settings.google_calendar_id, body=event_body)
                .execute()
            )

            logger.info(f"Calendar event created: {result.get('id')}")

            return CalendarEventResult(
                success=True,
                event_id=result.get("id"),
                event_link=result.get("htmlLink"),
                summary=(
                    f"'{event.title}' with {event.attendee_name} "
                    f"on {event.event_date} at {event.event_time}"
                ),
            )

        except HttpError as e:
            logger.error(f"Google Calendar API error: {e}")
            return CalendarEventResult(
                success=False,
                error=f"Calendar API error: {e.reason}",
            )
        except Exception as e:
            logger.error(f"Unexpected error creating event: {e}")
            return CalendarEventResult(
                success=False,
                error=str(e),
            )

    async def check_availability(
        self, date: str, time: str, timezone: str = "America/New_York"
    ) -> bool:
        """Check if a time slot is available (optional enhancement)."""
        service = self._get_service()
        if service is None:
            return True  # assume available in demo mode

        try:
            start_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
            end_dt = start_dt + timedelta(minutes=30)

            settings = get_settings()
            events_result = (
                service.events()
                .list(
                    calendarId=settings.google_calendar_id,
                    timeMin=start_dt.isoformat() + "Z",
                    timeMax=end_dt.isoformat() + "Z",
                    singleEvents=True,
                )
                .execute()
            )

            return len(events_result.get("items", [])) == 0

        except Exception as e:
            logger.error(f"Availability check failed: {e}")
            return True  # default to available


# Singleton
calendar_service = CalendarService()
