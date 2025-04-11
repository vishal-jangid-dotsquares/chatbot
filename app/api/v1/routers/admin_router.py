from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_db
from app.models import user as user_model, file_asset as file_model, qna as qna_model, page_content as content_model
from app.auth.dependencies import get_admin_user
from sqlalchemy.future import select
from app.schemas.admin import CreateUserAdmin, UpdateUserAdmin
from app.services import admin_user_service

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.get("/users")
async def get_all_users(db: AsyncSession = Depends(get_db), admin=Depends(get_admin_user)):
    result = await db.execute(select(user_model.User))
    return result.scalars().all()

@router.get("/users/{user_id}/files")
async def get_user_files(user_id: int, db: AsyncSession = Depends(get_db), admin=Depends(get_admin_user)):
    result = await db.execute(select(file_model.FileAsset).where(file_model.FileAsset.user_id == user_id))
    return result.scalars().all()

@router.get("/users/{user_id}/qna")
async def get_user_qna(user_id: int, db: AsyncSession = Depends(get_db), admin=Depends(get_admin_user)):
    result = await db.execute(select(qna_model.QnA).where(qna_model.QnA.user_id == user_id))
    return result.scalars().all()

@router.get("/users/{user_id}/page-content")
async def get_user_content(user_id: int, db: AsyncSession = Depends(get_db), admin=Depends(get_admin_user)):
    result = await db.execute(select(content_model.PageContent).where(content_model.PageContent.user_id == user_id))
    return result.scalars().all()



@router.post("/users", response_model=dict)
async def admin_create_user(payload: CreateUserAdmin, db: AsyncSession = Depends(get_db), admin=Depends(get_admin_user)):
    user = await admin_user_service.create_user(payload.dict(), db)
    return {"id": user.id, "email": user.email, "role": user.role}

@router.put("/users/{user_id}", response_model=dict)
async def admin_update_user(user_id: int, payload: UpdateUserAdmin, db: AsyncSession = Depends(get_db), admin=Depends(get_admin_user)):
    user = await admin_user_service.update_user(user_id, payload.dict(exclude_unset=True), db)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "email": user.email, "role": user.role}

@router.delete("/users/{user_id}")
async def admin_delete_user(user_id: int, db: AsyncSession = Depends(get_db), admin=Depends(get_admin_user)):
    user = await admin_user_service.delete_user(user_id, db)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"detail": "User deleted"}
