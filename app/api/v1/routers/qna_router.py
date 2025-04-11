from fastapi import APIRouter, Depends, HTTPException
from app.api.v1.schema.qna_schema import QnAInput, QnAOut
from app.auth.dependencies import get_current_owner
from app.api.v1.services.qna_service import add_qna, get_user_qna, delete_qna
from app.core.db import get_db
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/qna", tags=["Q&A"])

@router.post("/", response_model=list[QnAOut])
async def submit_qna(
    qna_input: QnAInput,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_owner)
):
    return await add_qna(user.id, [q.model_dump() for q in qna_input.qna_list], db)

@router.get("/", response_model=list[QnAOut])
async def list_qna(db: AsyncSession = Depends(get_db), user=Depends(get_current_owner)):
    return await get_user_qna(user.id, db)

@router.delete("/{qna_id}")
async def delete_qna_item(qna_id: int, db: AsyncSession = Depends(get_db), user=Depends(get_current_owner)):
    deleted = await delete_qna(user.id, qna_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Q&A not found")
    return {"detail": "Deleted"}
