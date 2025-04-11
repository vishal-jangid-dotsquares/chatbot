from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.auth.dependencies import get_current_owner
from app.schemas.page_content import ContentInput, PageContentOut
from app.utils.page_scraper import extract_text_from_url
from app.services.page_content_service import (
    store_page_contents,
    get_user_page_contents,
    delete_page_content,
)
from app.db import get_db
from app.schemas.link_submission import URLInput, URLListOut
from app.utils.link_extractor import extract_links

router = APIRouter(prefix="/web-links")

@router.post("/links", tags=["Page Link Submission"], response_model=URLListOut)
async def get_links(data: URLInput, user=Depends(get_current_owner)):
    links = await extract_links(data.url)
    if not links:
        raise HTTPException(status_code=400, detail="Could not fetch or parse links.")
    return {"links": links}


@router.post("/page-content", tags=["Page Content Extractor"], response_model=list[PageContentOut])
async def scrape_and_store_content(
    data: ContentInput,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_owner),
):
    extracted = []
    for url in data.urls:
        text = await extract_text_from_url(str(url))
        if text:
            extracted.append({"url": str(url), "content": text})
    if not extracted:
        raise HTTPException(status_code=400, detail="No content extracted")
    return await store_page_contents(user.id, extracted, db)

@router.get("/", response_model=list[PageContentOut])
async def get_stored_content(db: AsyncSession = Depends(get_db), user=Depends(get_current_owner)):
    return await get_user_page_contents(user.id, db)

@router.delete("/{content_id}")
async def delete_content(content_id: int, db: AsyncSession = Depends(get_db), user=Depends(get_current_owner)):
    entry = await delete_page_content(user.id, content_id, db)
    if not entry:
        raise HTTPException(status_code=404, detail="Content not found")
    return {"detail": "Deleted"}
