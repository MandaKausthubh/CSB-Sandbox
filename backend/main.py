from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import cast

import models.user as user_models
import models.analysis as analysis_models
from core.auth import user_dependancy

from database.database import engine, get_db
import core.auth as auth
import core.users as users
from fastapi.middleware.cors import CORSMiddleware


#
# app = FastAPI()
# app.include_router(auth.router, prefix="/api")
# app.include_router(users.router, prefix="/api")
# user_models.Base.metadata.create_all(bind=engine)  # Ensure tables are created
#
app = FastAPI()
app.include_router(auth.router, prefix="/api")
app.include_router(users.router, prefix="/api")
user_models.Base.metadata.create_all(bind=engine)  # Ensure tables are created
analysis_models.Base.metadata.create_all(bind=engine)  # Ensure tables are created

@app.get("/dev/admin")
def read_api(db: Session = Depends(get_db)):
    return db.query(user_models.UserDB).all()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "backend running!"}

@app.get("/user", response_model=user_models.UserResponse)
def read_user(current_user: user_dependancy, db: Session = Depends(get_db)):
    user = db.query(user_models.UserDB).filter(user_models.UserDB.userName == current_user.userName).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user_response = user_models.UserResponse(
        userId=cast(int, user.userId),
        userName=cast(str, user.userName),
        userEmail=cast(str, user.userEmail)
    )
    return user_response

from langGraph.api import router as analysis_router
from langGraph.router_options import router as options_router

app.include_router(analysis_router, prefix="/api")
app.include_router(options_router, prefix="/api")
