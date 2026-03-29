"""BoxBunny Dashboard — FastAPI application factory and entry point.

Serves a mobile-first SPA with REST API and WebSocket for real-time sync.
Designed for local network access via the BoxBunny WiFi access point.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from boxbunny_dashboard.api.auth import router as auth_router
from boxbunny_dashboard.api.chat import router as chat_router
from boxbunny_dashboard.api.coach import router as coach_router
from boxbunny_dashboard.api.export import router as export_router
from boxbunny_dashboard.api.gamification import router as gamification_router
from boxbunny_dashboard.api.presets import router as presets_router
from boxbunny_dashboard.api.sessions import router as sessions_router
from boxbunny_dashboard.db.manager import DatabaseManager
from boxbunny_dashboard.websocket import ConnectionManager

logger = logging.getLogger("boxbunny.dashboard")

# Resolve paths relative to the package
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _PACKAGE_DIR.parent
_STATIC_DIR = _PROJECT_DIR / "static" / "dist"
_WS_ROOT = _PROJECT_DIR.parents[1]  # boxing_robot_ws/
_DATA_DIR = os.environ.get(
    "BOXBUNNY_DATA_DIR",
    str(_WS_ROOT / "data"),
)


def _configure_logging() -> None:
    """Set up structured logging for the dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown resources."""
    logger.info("Starting BoxBunny Dashboard server")
    logger.info("Data directory: %s", _DATA_DIR)
    app.state.db = DatabaseManager(data_dir=_DATA_DIR)
    app.state.ws_manager = ConnectionManager()
    yield
    logger.info("Shutting down BoxBunny Dashboard server")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="BoxBunny Dashboard",
        version="1.0.0",
        description="Mobile training dashboard for BoxBunny boxing robot",
        lifespan=lifespan,
    )

    # -- CORS (allow local network phones to connect) --
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- API routers --
    app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
    app.include_router(sessions_router, prefix="/api/sessions", tags=["sessions"])
    app.include_router(presets_router, prefix="/api/presets", tags=["presets"])
    app.include_router(
        gamification_router, prefix="/api/gamification", tags=["gamification"],
    )
    app.include_router(coach_router, prefix="/api/coach", tags=["coach"])
    app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
    app.include_router(export_router, prefix="/api/export", tags=["export"])

    # -- WebSocket --
    from boxbunny_dashboard.websocket import websocket_endpoint

    app.add_api_websocket_route("/ws", websocket_endpoint)

    # -- Health check --
    @app.get("/api/health", tags=["system"])
    async def health_check() -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "boxbunny_dashboard"})

    # -- Static file serving --
    _ASSETS_DIR = _STATIC_DIR / "assets"
    if _ASSETS_DIR.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(_ASSETS_DIR)),
            name="static",
        )

    # -- SPA fallback: serve index.html for all non-API routes --
    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(request: Request, full_path: str) -> FileResponse:
        # Serve the actual file if it exists in static
        file_path = _STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        # Fall back to index.html for client-side routing
        index_path = _STATIC_DIR / "index.html"
        if index_path.is_file():
            return FileResponse(str(index_path))
        return JSONResponse(
            status_code=404,
            content={"detail": "Dashboard frontend not built. Run the frontend build first."},
        )

    return app


app = create_app()


def main() -> None:
    """Entry point for the dashboard server."""
    _configure_logging()

    host = os.environ.get("BOXBUNNY_HOST", "0.0.0.0")
    port = int(os.environ.get("BOXBUNNY_PORT", "8080"))

    logger.info("BoxBunny Dashboard listening on %s:%d", host, port)
    uvicorn.run(
        "boxbunny_dashboard.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
