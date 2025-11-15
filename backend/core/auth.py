from datetime import timedelta
from typing import Annotated, Dict
from fastapi import APIRouter, Depends, HTTPException
from starlette import status
from database.database import db_dependancy

from models.user import UserDB
from models.token import Token, create_access_token

from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from core.env import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES


bcrypt_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")




# Helper functions
def hash_password(password: str) -> str:
    return bcrypt_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt_context.verify(plain_password, hashed_password)

async def get_current_user(
        token: Annotated[str, Depends(oauth2_scheme)],
        db: db_dependancy):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print("PAYLOAD:", payload)
        userId: int = payload.get("id") # type: ignore
        print("CHECK POINT 0:", userId)
        userName: str = payload.get("username") # type: ignore
        print("CHECK POINT 1:", userName)
        if userId is None or userName is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Error 1: Could not validate credentials"
            )
        user = db.query(UserDB).filter(UserDB.userId == userId).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Token: User does not exist"
            )
        return user
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Error 2: Could not validate credentials"
        )

user_dependancy = Annotated[UserDB, Depends(get_current_user)]



# User Login and Token Generation
router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)

@router.post("/login")
async def login_for_access_token(db: db_dependancy, form_data: OAuth2PasswordRequestForm = Depends()):
    user = db.query(UserDB).filter(UserDB.userName == form_data.username).first()
    if not user or not verify_password(form_data.password, str(user.hashed_password)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    print(ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"id": user.userId, "username": user.userName}, expires_delta=access_token_expires
    )
    user = {
        "id": user.userId,
        "name": user.userName,
        "email": user.userEmail,
        "createdAt": "2025-11-15T10:00:00Z"
    }
    return { "success":True, "user":user ,"token": access_token, "access_token":access_token, "token_type": "bearer"}


