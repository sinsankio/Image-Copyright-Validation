import numpy as np
from fastapi import UploadFile

from scripts.db import validate_image
from scripts.embedding import embed_image
from scripts.upload_file import save_uploaded_file


def validate_copyrights(file: UploadFile) -> str:
    uploaded_file_save_path = save_uploaded_file(file)
    embedded_image_result = embed_image(uploaded_file_save_path)
    embeddings = embedded_image_result['embeddings']
    embedding = embeddings[0].__dict__['embedding']
    embedding = np.reshape(embedding, (1, 1024))
    return validate_image(embedding)
