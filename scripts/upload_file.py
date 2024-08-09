import os

from fastapi import UploadFile

FILE_UPLOAD_ROOT_DIR = "../uploads"


def save_uploaded_file(file: UploadFile) -> str:
    uploaded_file_save_path = os.path.join(FILE_UPLOAD_ROOT_DIR, file.filename)
    if not os.path.exists(uploaded_file_save_path):
        with open(uploaded_file_save_path, 'wb+') as file_save:
            file_save.write(file.file.read())
    return uploaded_file_save_path
