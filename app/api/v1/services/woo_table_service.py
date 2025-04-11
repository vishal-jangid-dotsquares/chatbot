from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.api.v1.models.wc_models import WooTableSelection

async def save_woo_tables(user_id: int, tables: list[str], db: AsyncSession):
    records = [WooTableSelection(user_id=user_id, table_name=table) for table in tables]
    db.add_all(records)
    await db.commit()
    return records

async def get_user_woo_tables(user_id: int, db: AsyncSession):
    result = await db.execute(select(WooTableSelection).where(WooTableSelection.user_id == user_id))
    return result.scalars().all()

async def delete_woo_table(user_id: int, table_id: int, db: AsyncSession):
    result = await db.execute(select(WooTableSelection).where(WooTableSelection.id == table_id, WooTableSelection.user_id == user_id))
    table = result.scalar_one_or_none()
    if table:
        await db.delete(table)
        await db
