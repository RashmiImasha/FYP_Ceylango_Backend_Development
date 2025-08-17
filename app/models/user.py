from pydantic import BaseModel, EmailStr, Field
from typing import Literal, Optional

class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    role: Literal["admin", "tourist", "service_provider"] = "tourist"

class UserCreate(UserBase):    # for normal user signup
    password: str

class ServiceProviderApplication(BaseModel):
    email: EmailStr
    full_name: str
    service_name: str
    district: str
    service_category: str
    phone_number: str
    


class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserInDB(UserBase):
    uid: str
    disabled: Optional[bool] = False
