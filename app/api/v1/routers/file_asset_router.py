from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.auth.dependencies import get_current_owner
from app.utils.file_saver import save_file
from app.services.file_service import store_file_metadata, get_user_files, delete_user_file
from app.db import get_db
from app.schemas.file_upload import FileOut
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/files", tags=["File Uploads"])

@router.post("/", response_model=FileOut)
async def upload_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_owner),
):
    try:
        file_path = await save_file(user.id, file)
        file_record = await store_file_metadata(user.id, file.filename, file_path, db)
        return file_record
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=list[FileOut])
async def list_user_files(db: AsyncSession = Depends(get_db), user=Depends(get_current_owner)):
    return await get_user_files(user.id, db)

@router.delete("/{file_id}")
async def delete_file(file_id: int, db: AsyncSession = Depends(get_db), user=Depends(get_current_owner)):
    deleted = await delete_user_file(user.id, file_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")
    return {"detail": "File deleted"}
