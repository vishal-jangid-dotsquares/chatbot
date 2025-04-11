from fastapi import Depends, HTTPException, status
from app.models.user import RoleEnum
from app.auth.auth import get_current_user

def get_admin_user(user=Depends(get_current_user)):
    if user.role != RoleEnum.admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
