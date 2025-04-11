from app.models.qna import QnA
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

async def add_qna(user_id: int, qna_items: list[dict], db: AsyncSession):
    records = [QnA(user_id=user_id, ques=q["ques"], answer=q["answer"]) for q in qna_items]
    db.add_all(records)
    await db.commit()
    return records

async def get_user_qna(user_id: int, db: AsyncSession):
    result = await db.execute(select(QnA).where(QnA.user_id == user_id))
    return result.scalars().all()

async def delete_qna(user_id: int, qna_id: int, db: AsyncSession):
    result = await db.execute(select(QnA).where(QnA.id == qna_id, QnA.user_id == user_id))
    entry = result.scalar_one_or_none()
    if entry:
        await db.delete(entry)
        await db.commit()
    return entry
