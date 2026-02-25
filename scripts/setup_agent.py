"""
Setup script — creates a Retell AI agent pointing to your Custom LLM WebSocket.

Usage:
    python -m scripts.setup_agent --server-url https://your-app.onrender.com

This will:
  1. Create an Agent with a Custom LLM websocket URL
  2. Print the RETELL_AGENT_ID to add to your .env

You only need to run this ONCE (or when changing the server URL).
"""

import argparse
import sys

from retell import Retell
from dotenv import dotenv_values, set_key


def main():
    parser = argparse.ArgumentParser(description="Setup Retell AI agent")
    parser.add_argument(
        "--server-url",
        required=True,
        help="Public HTTPS URL of your deployed server (e.g. https://voice-scheduler.onrender.com)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file to update",
    )
    args = parser.parse_args()

    # Load API key from .env
    env = dotenv_values(args.env_file)
    api_key = env.get("RETELL_API_KEY")

    if not api_key:
        print("ERROR: RETELL_API_KEY not found in .env file")
        sys.exit(1)

    # Build WebSocket URL from server URL
    ws_url = args.server_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url.rstrip('/')}/ws/retell"

    print(f"Server URL:    {args.server_url}")
    print(f"WebSocket URL: {ws_url}")
    print()

    client = Retell(api_key=api_key)

    # ── Create Agent with Custom LLM (v5 SDK — single step) ──
    print("Creating Agent with Custom LLM...")
    agent = client.agent.create(
        response_engine={
            "type": "custom-llm",
            "llm_websocket_url": ws_url,
        },
        agent_name="Voice Scheduling Assistant",
        voice_id="11labs-Adrian",
        language="en-US",
        voice_speed=1.0,
        voice_temperature=0.7,
        responsiveness=0.8,
        interruption_sensitivity=0.6,
        enable_backchannel=True,
        backchannel_frequency=0.5,
        reminder_trigger_ms=10000,
        reminder_max_count=2,
        end_call_after_silence_ms=30000,
        max_call_duration_ms=300000,
    )
    print(f"  Agent ID: {agent.agent_id}")

    # ── Update .env ──
    set_key(args.env_file, "RETELL_AGENT_ID", agent.agent_id)
    print()
    print(f"Updated {args.env_file} with RETELL_AGENT_ID={agent.agent_id}")
    print()
    print("Setup complete! Add this RETELL_AGENT_ID to Render env vars and redeploy.")


if __name__ == "__main__":
    main()