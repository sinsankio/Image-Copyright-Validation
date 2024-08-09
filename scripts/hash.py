import hashlib

import numpy as np


def generate_hash(text: str, algorithm: str = 'sha256') -> str:
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode('utf-8'))
    return hash_obj.hexdigest()


def generate_random_hash(algorithm: str = 'sha256') -> str:
    rand_num = np.random.randint(1, 100000)
    return generate_hash(text=str(rand_num), algorithm=algorithm)
