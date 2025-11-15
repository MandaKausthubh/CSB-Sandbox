from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from starlette import status
from database.database import db_dependancy

from models.user import UserDB, CreateUserRequest
from core.auth import bcrypt_context, get_current_user, user_dependancy
from core.env import ACCESS_TOKEN_EXPIRE_MINUTES
from core.auth import create_access_token
from datetime import timedelta


# Creating a new User
router = APIRouter(prefix="/auth",tags=["Auth"])

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def create_user(db: db_dependancy, create_user_request: CreateUserRequest):
    user = db.query(UserDB).filter(UserDB.userName == create_user_request.userName).first()
    user_email = db.query(UserDB).filter(UserDB.userEmail == create_user_request.userEmail).first()
    if user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
    if user_email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")

    hashed_password = bcrypt_context.hash(create_user_request.password)
    new_user = UserDB(
        userName=create_user_request.userName,
        userEmail=create_user_request.userEmail,
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    print(ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"id": new_user.userId, "username": new_user.userName}, expires_delta=access_token_expires
    )

    return {
        "success": True,
        "user": {
            "id": new_user.userId,
            "name": new_user.userName,
            "email": new_user.userEmail,
            "createdAt": "2025-01-01T00:00:00Z"
        },
        "token": access_token
    }


@router.get("/me")
async def get_authenticated_user(user: user_dependancy):
    return {
        "success": True,
        "user": {
            "id": user.userId,
            "email": user.userEmail,
            "name": user.userName,
            "createdAt": "2025-01-01T00:00:00Z"
        }
    }

@router.post("/logout")
async def logout_user(user: UserDB = Depends(get_current_user)):
    return {"success": True}

