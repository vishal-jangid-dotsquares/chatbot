from app.models.file_asset import FileAsset
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

async def store_file_metadata(user_id: int, filename: str, filepath: str, db: AsyncSession):
    file_record = FileAsset(user_id=user_id, filename=filename, filepath=filepath)
    db.add(file_record)
    await db.commit()
    await db.refresh(file_record)
    return file_record

async def get_user_files(user_id: int, db: AsyncSession):
    result = await db.execute(select(FileAsset).where(FileAsset.user_id == user_id))
    return result.scalars().all()

async def delete_user_file(user_id: int, file_id: int, db: AsyncSession):
    result = await db.execute(select(FileAsset).where(FileAsset.id == file_id, FileAsset.user_id == user_id))
    file = result.scalar_one_or_none()
    if file:
        import os
        if os.path.exists(file.filepath):
            os.remove(file.filepath)
        await db.delete(file)
        await db.commit()
    return file
