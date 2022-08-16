from fastapi import FastAPI, File, UploadFile
from predictor import *
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import base64

app = FastAPI()

@app.get("/")
async def hello():
    return {"messenger" : "hello"}

@app.post("/api/predict")
async def predict_image(file : bytes  = File(...)):
    print(type(file))
    image = read_image(file)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    # # make prediction
    # prediction = predict(image)
    dice = {
        # 'prediction': file.filename,
        'base64': img_str,
    }
    print(JSONResponse(dice).body)
    return JSONResponse(dice)
