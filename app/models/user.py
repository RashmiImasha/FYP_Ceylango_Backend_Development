from pydantic import BaseModel, EmailStr, Field
from typing import Literal, Optional

class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    role: Literal["admin", "tourist", "service_provider"] = "tourist"

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserInDB(UserBase):
    uid: str
    disabled: Optional[bool] = False
