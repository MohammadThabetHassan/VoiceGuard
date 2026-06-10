"""JWT authentication helpers."""

from __future__ import annotations

import os
import secrets
from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

# Known placeholder SECRET_KEY values. The app refuses to start in production
# (see check_production_security) if SECRET_KEY is left at one of these — keep
# this in sync with the defaults in docker-compose.yml and .env.example.
_DEFAULT_SECRET_KEYS = frozenset(
    {
        "dev-secret-changeme-in-production",
        "changeme-in-production",
        "changeme-run-openssl-rand-hex-32",
        "change-me-openssl-rand-hex-32",
    }
)

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-changeme-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Demo credentials used ONLY in development (VG_ENV != "production"). In
# production these are inert; each account is enabled only if its password env
# var (VG_ADMIN_PASSWORD / VG_ANALYST_PASSWORD) is set.
_DEMO_USERS: dict[str, str] = {
    "admin": "voiceguard2026",
    "analyst": "analyst2026",
}
# Maps each account to the env var holding its production password.
_USER_PASSWORD_ENV: dict[str, str] = {
    "admin": "VG_ADMIN_PASSWORD",
    "analyst": "VG_ANALYST_PASSWORD",
}


def is_dev_mode() -> bool:
    """True unless VG_ENV is explicitly set to production."""
    return os.environ.get("VG_ENV", "development").strip().lower() != "production"


def _user_password(username: str) -> str | None:
    """Resolve a user's password: env var first, demo fallback only in dev."""
    env_var = _USER_PASSWORD_ENV.get(username)
    if env_var:
        password = os.environ.get(env_var)
        if password:
            return password
    if is_dev_mode():
        return _DEMO_USERS.get(username)
    return None


def check_production_security() -> None:
    """Fail fast if the app is misconfigured for production.

    Called at application startup (lifespan). Raises RuntimeError when running
    in production with a default/placeholder SECRET_KEY, so JWTs can't be forged
    with a publicly known key.
    """
    if not is_dev_mode() and SECRET_KEY in _DEFAULT_SECRET_KEYS:
        raise RuntimeError(
            "SECRET_KEY is set to a default/placeholder value while VG_ENV=production. "
            "Set SECRET_KEY to a strong random value (e.g. `openssl rand -hex 32`)."
        )


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
    stored = _user_password(username)
    # Compare as bytes so non-ASCII passwords don't raise (compare_digest rejects
    # non-ASCII str). Encoding-failure falls through to a constant-time mismatch.
    pw_bytes = password.encode("utf-8", "surrogatepass")
    if stored is None:
        # Still spend roughly equal time so unknown users aren't distinguishable
        # by timing from a wrong password.
        secrets.compare_digest(pw_bytes, pw_bytes)
        return False
    return secrets.compare_digest(stored.encode("utf-8", "surrogatepass"), pw_bytes)
