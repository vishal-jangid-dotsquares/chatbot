from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.user import UserCreate, UserLogin, UserOut
from services.auth_service import create_user, authenticate_user
from app.core.db import get_db
from app.auth.jwt_handler import create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserOut)
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    existing = await authenticate_user(user.email, user.password, db)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    return await create_user(user, db)

@router.post("/login")
async def login(user: UserLogin, db: AsyncSession = Depends(get_db)):
    user_db = await authenticate_user(user.email, user.password, db)
    if not user_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(data={"sub": user_db.email})
    return {"access_token": token, "token_type": "bearer"}

from app.schemas.auth import RegisterRequest, RegisterResponse
from app.auth.hashing import hash_password
from app.models.user import RoleEnum

@router.post("/register", response_model=RegisterResponse)
async def register_user(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check if user already exists
    result = await db.execute(select(User).where(User.email == payload.email))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=RoleEnum.owner
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return RegisterResponse(id=user.id, email=user.email, role=user.role)
