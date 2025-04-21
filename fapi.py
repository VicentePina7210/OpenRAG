from typing import Union

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
app = FastAPI()


@app.get("/")
def readroot():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# Building the request body for recieving embeddings from UI
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("v1/api/embed/")
async def upload_pdf(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Optional: Call your embedding function here with the saved file
    # embed_pdf(file_location)

    return JSONResponse(content={"message": f"Successfully uploaded {file.filename}"})    
 