from pydantic import BaseModel
from datetime import timedelta, timezone
from jose import jwt
from datetime import datetime, timedelta

from core.env import SECRET_KEY, ALGORITHM

class Token(BaseModel):
    access_token: str
    token_type: str


def create_access_token(
        data: dict,
        expires_delta: timedelta,
        SECRET_KEY=SECRET_KEY,
        ALGORITHM=ALGORITHM,
) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
