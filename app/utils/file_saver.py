import os
from fastapi import UploadFile
from uuid import uuid4

BASE_UPLOAD_DIR = "app/media/uploads"

def allowed_file(filename: str) -> bool:
    return filename.endswith((".pdf", ".txt", ".json"))

async def save_file(user_id: int, file: UploadFile) -> str:
    if not allowed_file(file.filename):
        raise ValueError("Unsupported file type")

    ext = os.path.splitext(file.filename)[1]
    new_filename = f"{uuid4().hex}{ext}"
    user_folder = os.path.join(BASE_UPLOAD_DIR, str(user_id))
    os.makedirs(user_folder, exist_ok=True)

    file_path = os.path.join(user_folder, new_filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return file_path
