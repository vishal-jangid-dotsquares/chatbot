from app.models.page_content import PageContent
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

async def store_page_contents(user_id: int, contents: list[dict], db: AsyncSession):
    entries = [PageContent(user_id=user_id, url=entry["url"], content=entry["content"]) for entry in contents]
    db.add_all(entries)
    await db.commit()
    return entries

async def get_user_page_contents(user_id: int, db: AsyncSession):
    result = await db.execute(select(PageContent).where(PageContent.user_id == user_id))
    return result.scalars().all()

async def delete_page_content(user_id: int, content_id: int, db: AsyncSession):
    result = await db.execute(select(PageContent).where(PageContent.id == content_id, PageContent.user_id == user_id))
    entry = result.scalar_one_or_none()
    if entry:
        await db.delete(entry)
        await db.commit()
    return entry
