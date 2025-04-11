from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.v1.routers import auth_router
from app.core.db import Base, engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown: add any cleanup logic if needed later

app = FastAPI(lifespan=lifespan)

# Include your API routes
app.include_router(auth_router.router)
