import uvicorn
from fastapi import FastAPI, UploadFile, status

from scripts.services import validate_copyrights

app = FastAPI()


@app.post(
    path='/validate/',
    response_description='validate copyrights for input images',
    status_code=status.HTTP_200_OK
)
async def validate(file: UploadFile) -> str:
    return validate_copyrights(file)


if __name__ == "__main__":
    uvicorn.run(app, port=5000)
