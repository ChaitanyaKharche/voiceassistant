"""
Main FastAPI application.

Endpoints:
  POST  /api/create-web-call  → Creates a Retell web call, returns access token
  GET   /api/health            → Health check
  WS    /ws/retell             → Custom LLM WebSocket for Retell
  GET   /                      → Serves the frontend UI
"""

import logging
import os
from contextlib import asynccontextmanager

import retell
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.config import get_settings
from server.retell_handler import RetellWebSocketHandler

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# App Lifecycle
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(f"Starting server (env={settings.environment})")
    logger.info(f"Retell Agent ID: {settings.retell_agent_id or 'NOT SET'}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Voice Scheduling Agent",
    version="1.0.0",
    lifespan=lifespan,
)

# ──────────────────────────────────────────────
# CORS
# ──────────────────────────────────────────────

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "agent_id": settings.retell_agent_id or "not configured",
    }


@app.post("/api/create-web-call")
async def create_web_call():
    """
    Create a Retell web call and return the access token.
    The frontend uses this token to initiate the voice session.
    """
    settings = get_settings()

    if not settings.retell_agent_id:
        raise HTTPException(
            status_code=500,
            detail="RETELL_AGENT_ID not configured. Run setup_agent.py first.",
        )

    try:
        client = retell.Retell(api_key=settings.retell_api_key)

        web_call = client.call.create_web_call(
            agent_id=settings.retell_agent_id,
        )

        logger.info(f"Web call created: call_id={web_call.call_id}")

        return {
            "access_token": web_call.access_token,
            "call_id": web_call.call_id,
        }

    except Exception as e:
        logger.error(f"Failed to create web call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Retell Custom LLM WebSocket
# ──────────────────────────────────────────────

@app.websocket("/ws/retell")
async def retell_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for Retell's Custom LLM integration.
    Each active call maintains its own WebSocket connection.
    """
    handler = RetellWebSocketHandler(websocket)
    await handler.handle()


# ──────────────────────────────────────────────
# Frontend Static Files
# ──────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# Mount static assets (CSS, JS, images) if the dir exists
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
