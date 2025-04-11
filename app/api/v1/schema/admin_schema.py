from pydantic import BaseModel, EmailStr
from typing import Optional
from enum import Enum

class RoleEnum(str, Enum):
    admin = "admin"
    owner = "owner"

class CreateUserAdmin(BaseModel):
    email: EmailStr
    password: str
    role: RoleEnum = RoleEnum.owner

class UpdateUserAdmin(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    role: Optional[RoleEnum] = None
