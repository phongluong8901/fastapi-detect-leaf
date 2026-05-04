import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from contextlib import asynccontextmanager
import os

# ONNX_MODEL_PATH = "vit-model.onnx"


# Cách này giúp Python xác định chính xác vị trí file .onnx ở đâu 
# thì file .data cũng phải nằm ở đó.
current_dir = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(current_dir, "vit_classification.onnx")

# Khi chạy dòng này, nó sẽ tự động "nhìn" sang file .data bên cạnh.
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)


# ort_session = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ort_session
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
    yield

    ort_session = None

app = FastAPI(lifespan=lifespan)

def preprocess_image(image_bytes):

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized_img = img.resize((224,224))
    np_array = np.asarray(resized_img)

    img_transposed = np.transpose(np_array, (2,0,1))

    img_expanded = np.expand_dims(img_transposed, axis=0)
    img_normalized = img_expanded.astype(np.float32)/255.

    return img_normalized

@app.post("/infer/")
async def infer(file:UploadFile = File(...)):
    if ort_session is None:
        return {"error":"Model is not loaded"}
    image_bytes = await file.read()
    image_preprocessed = preprocess_image(image_bytes)

    session_input = {ort_session.get_inputs()[0].name: image_preprocessed}
    onnx_output = ort_session.run(None, session_input)
    onnx_logits = onnx_output[0]

    pred_idx = int(np.argmax(onnx_logits, axis=1)[0])

    return {"predicted_idx": pred_idx}

@app.get("/")
def test_api():
    return {"message":"the api is live !!!"}