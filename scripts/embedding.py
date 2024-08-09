import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
ImageEmbedder = mp.tasks.vision.ImageEmbedder
ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
VisionRunningMode = mp.tasks.vision.RunningMode
MP_MODEL_FILE_PATH = '../resources/mobilenet_v3_small_075_224_embedder.tflite'


def load_embedder_options(model_file_path: str = MP_MODEL_FILE_PATH) -> ImageEmbedderOptions:
    return ImageEmbedderOptions(
        base_options=BaseOptions(model_asset_path=model_file_path),
        quantize=True,
        running_mode=VisionRunningMode.IMAGE
    )


def embed_image(img_file_path: str) -> dict:
    image = mp.Image.create_from_file(img_file_path)
    with ImageEmbedder.create_from_options(load_embedder_options()) as embedder:
        return embedder.embed(image).__dict__
