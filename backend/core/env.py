import os
from typing import cast
from dotenv import load_dotenv

load_dotenv()

DATABSE_URL = str(os.getenv("DATABASE_URL"))
SECRET_KEY = str(os.getenv("SECRET_KEY"))
ALGORITHM = str(os.getenv("ALGORITHM"))
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")) # type: ignore
OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))

print(ACCESS_TOKEN_EXPIRE_MINUTES, type(ACCESS_TOKEN_EXPIRE_MINUTES))
print(ALGORITHM, type(ALGORITHM))
