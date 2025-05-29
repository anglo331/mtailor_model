import os, base64, io, numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
from pydantic import BaseModel
import asyncio
from model.model import create_session, predict, preprocess_input


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI to handle startup and shutdown events.
    """

    global session
    ONNX_MODEL_PATH = os.path.join(os.getcwd(), "model/mtailor_model.onnx")
    try:
        session =  await asyncio.to_thread(create_session, ONNX_MODEL_PATH)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        session = None # Indicate failure

    yield
    # Cleanup if needed
    if session is not None:
        del session


app = FastAPI(lifespan=lifespan)
session = None

class PredictionRequest(BaseModel):
    image_base64: str


@app.post("/predict")
async def predict_image(request: PredictionRequest):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Save to temporary file for preprocess_input, then clean up
        temp_image_path = "/tmp/input_image.jpeg"
        image.save(temp_image_path)
        processed_input_list = preprocess_input(temp_image_path)
        os.remove(temp_image_path)

        probabilities = predict(session, processed_input_list)
        
        return {
            "predicted_class_id": np.argmax(probabilities, axis=1)[0].item(),
            "confidence": np.max(probabilities).item(),
         }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")