import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from scripts.hash import generate_random_hash

IMAGE_COLLECTION_NAME = "images"
ICV_DB_FILE_PATH = "icv"
DB_CLIENT = None
IMAGE_COLLECTION = None


def init_client(icv_db_file_path: str = ICV_DB_FILE_PATH) -> ClientAPI:
    global DB_CLIENT

    if not DB_CLIENT:
        DB_CLIENT = chromadb.PersistentClient(path=icv_db_file_path)
    return DB_CLIENT


def init_collection(image_collection_name: str = IMAGE_COLLECTION_NAME) -> Collection:
    global IMAGE_COLLECTION

    if not IMAGE_COLLECTION:
        try:
            IMAGE_COLLECTION = init_client().get_collection(name=image_collection_name)
        except ValueError:
            IMAGE_COLLECTION = init_client().create_collection(
                name=image_collection_name,
                metadata={
                    'hnsw:space': 'cosine'
                }
            )
    return IMAGE_COLLECTION


def add_image(embeddings: list[float]) -> str:
    hash_val = generate_random_hash()
    collection = init_collection()
    collection.add(
        embeddings=embeddings,
        ids=[hash_val]
    )
    return hash_val


def validate_image(embeddings: list[float] | np.ndarray, n_results: int = 3, max_accept_distance: float = 0.25) -> str:
    collection = init_collection()
    result = collection.query(
        query_embeddings=embeddings,
        n_results=n_results
    )
    if result['distances'] and result['distances'][0]:
        if result['distances'][0][0] <= max_accept_distance:
            return result['ids'][0][0]
    return add_image(embeddings)
