"""JWT authentication helpers."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-changeme-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Demo credentials — replace with a real user store in production
_DEMO_USERS: dict[str, str] = {
    "admin": "voiceguard2026",
    "analyst": "analyst2026",
}


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(UTC) + (expires_delta or timedelta(minutes=15))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> str:
    """Decode and verify JWT. Returns username on success."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exc
        return username
    except JWTError as err:
        raise credentials_exc from err


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    return verify_token(token)


def authenticate_user(username: str, password: str) -> bool:
    stored = _DEMO_USERS.get(username)
    if stored is None:
        return False
    return stored == password
