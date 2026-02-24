"""
Setup script — creates a Retell AI agent pointing to your Custom LLM WebSocket.

Usage:
    python -m scripts.setup_agent --server-url https://your-app.onrender.com

This will:
  1. Create a Custom LLM on Retell (pointing to wss://your-app/ws/retell)
  2. Create an Agent using that LLM
  3. Print the RETELL_AGENT_ID to add to your .env

You only need to run this ONCE (or when changing the server URL).
"""

import argparse
import sys

import retell
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

    print(f"Server URL: {args.server_url}")
    print(f"WebSocket URL: {ws_url}")
    print()

    client = retell.Retell(api_key=api_key)

    # ── Step 1: Create Custom LLM ──
    print("Creating Custom LLM...")
    llm = client.llm.create(
        model="custom_llm",
        custom_llm_url=ws_url,
    )
    print(f"  LLM ID: {llm.llm_id}")

    # ── Step 2: Create Agent ──
    print("Creating Agent...")
    agent = client.agent.create(
        llm_id=llm.llm_id,
        agent_name="Voice Scheduling Assistant",
        voice_id="11labs-Adrian",  # professional male voice
        language="en-US",
        response_engine={"type": "retell-llm", "llm_id": llm.llm_id},
        # Voice settings
        voice_speed=1.0,
        voice_temperature=0.7,
        responsiveness=0.8,  # balance between speed and accuracy
        interruption_sensitivity=0.6,  # allow some interruption
        enable_backchannel=True,  # "mm-hmm", "I see"
        backchannel_frequency=0.5,
        ambient_sound=None,
        # Conversation behavior
        reminder_trigger_ms=10000,  # remind after 10s silence
        reminder_max_count=2,
        end_call_after_silence_ms=30000,  # end after 30s silence
        max_call_duration_ms=300000,  # 5 min max
    )
    print(f"  Agent ID: {agent.agent_id}")

    # ── Step 3: Update .env ──
    set_key(args.env_file, "RETELL_AGENT_ID", agent.agent_id)
    print()
    print(f"Updated {args.env_file} with RETELL_AGENT_ID={agent.agent_id}")
    print()
    print("Setup complete! Restart your server to pick up the new agent ID.")


if __name__ == "__main__":
    main()
