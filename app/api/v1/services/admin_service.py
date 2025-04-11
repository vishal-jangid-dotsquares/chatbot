from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.user import User
from app.auth.hashing import hash_password

async def create_user(data: dict, db: AsyncSession):
    user = User(
        email=data["email"],
        password_hash=hash_password(data["password"]),
        role=data["role"]
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

async def update_user(user_id: int, data: dict, db: AsyncSession):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        return None

    if "email" in data and data["email"]:
        user.email = data["email"]
    if "password" in data and data["password"]:
        user.password_hash = hash_password(data["password"])
    if "role" in data and data["role"]:
        user.role = data["role"]

    await db.commit()
    await db.refresh(user)
    return user

async def delete_user(user_id: int, db: AsyncSession):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user:
        await db.delete(user)
        await db.commit()
    return user
