"""Authentication endpoints for BoxBunny Dashboard.

Handles login, signup, pattern-lock verification, guest session claiming,
session validation, and logout. All tokens are Bearer tokens.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from boxbunny_dashboard.db.manager import DatabaseManager

logger = logging.getLogger("boxbunny.dashboard.auth")
router = APIRouter()


# ---- Pydantic models ----

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1)
    device_type: str = Field(default="phone")


class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6)
    display_name: str = Field(..., min_length=1, max_length=128)
    user_type: str = Field(default="individual")
    level: str = Field(default="beginner")


class PatternVerifyRequest(BaseModel):
    user_id: int
    pattern: List[int] = Field(..., min_length=4)


class GuestClaimRequest(BaseModel):
    guest_token: str
    username: str
    password: str
    display_name: str


class TokenResponse(BaseModel):
    token: str
    user_id: Optional[int] = None
    username: Optional[str] = None
    display_name: Optional[str] = None
    user_type: Optional[str] = None


class UserInfoResponse(BaseModel):
    user_id: int
    username: str
    display_name: str
    user_type: str
    level: str


# ---- Dependencies ----

def get_db(request: Request) -> DatabaseManager:
    """Retrieve the DatabaseManager from application state."""
    return request.app.state.db


def get_current_user(request: Request) -> dict:
    """Extract and validate the Bearer token from the Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )
    token = auth_header[len("Bearer "):]
    db: DatabaseManager = request.app.state.db
    session = db.validate_session_token(token)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token",
        )
    return session


# ---- Endpoints ----

@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: DatabaseManager = Depends(get_db)) -> TokenResponse:
    """Authenticate with username and password, returning a session token."""
    user = db.verify_password(body.username, body.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = db.create_auth_session(user["id"], body.device_type)
    logger.info("User logged in: %s", body.username)
    return TokenResponse(
        token=token,
        user_id=user["id"],
        username=user["username"],
        display_name=user["display_name"],
        user_type=user["user_type"],
    )


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(body: SignupRequest, db: DatabaseManager = Depends(get_db)) -> TokenResponse:
    """Create a new user account and return a session token."""
    user_id = db.create_user(
        username=body.username,
        password=body.password,
        display_name=body.display_name,
        user_type=body.user_type,
        level=body.level,
    )
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        )
    token = db.create_auth_session(user_id, "phone")
    logger.info("User created: %s (id=%d)", body.username, user_id)
    return TokenResponse(
        token=token,
        user_id=user_id,
        username=body.username,
        display_name=body.display_name,
        user_type=body.user_type,
    )


@router.post("/pattern-verify", response_model=TokenResponse)
async def pattern_verify(
    body: PatternVerifyRequest, db: DatabaseManager = Depends(get_db),
) -> TokenResponse:
    """Verify a pattern-lock and return a session token."""
    if not db.verify_pattern(body.user_id, body.pattern):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid pattern",
        )
    user = db.get_user(body.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    token = db.create_auth_session(user["id"], "phone")
    logger.info("Pattern verified for user_id=%d", body.user_id)
    return TokenResponse(
        token=token,
        user_id=user["id"],
        username=user["username"],
        display_name=user["display_name"],
        user_type=user["user_type"],
    )


@router.post("/guest-claim", response_model=TokenResponse)
async def guest_claim(
    body: GuestClaimRequest, db: DatabaseManager = Depends(get_db),
) -> TokenResponse:
    """Link a guest session to a newly created account."""
    user_id = db.create_user(
        username=body.username,
        password=body.password,
        display_name=body.display_name,
    )
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        )
    claimed = db.claim_guest_session(body.guest_token, user_id)
    if not claimed:
        logger.warning("Guest token claim failed for token=%s", body.guest_token)
    token = db.create_auth_session(user_id, "phone")
    logger.info("Guest claimed: %s -> user_id=%d", body.guest_token, user_id)
    return TokenResponse(
        token=token,
        user_id=user_id,
        username=body.username,
        display_name=body.display_name,
    )


@router.get("/session", response_model=UserInfoResponse)
async def get_session(user: dict = Depends(get_current_user)) -> UserInfoResponse:
    """Validate the current token and return user info."""
    return UserInfoResponse(
        user_id=user["user_id"],
        username=user["username"],
        display_name=user["display_name"],
        user_type=user["user_type"],
        level=user.get("level", "beginner"),
    )


@router.delete("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request, db: DatabaseManager = Depends(get_db)) -> None:
    """Invalidate the current session token."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):]
        db.invalidate_session(token)
        logger.info("Session invalidated")
