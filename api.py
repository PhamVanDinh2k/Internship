from fastapi import FastAPI, File, UploadFile
from predictor import *
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


app = FastAPI()

@app.get("/")
async def hello():
    return {"messenger" : "hello"}

@app.post("/api/predict")
async def predict_image(file : bytes  = File(...)):
    # print(file.filename)
    print(type(file))
    image = read_image(file)
    image = preprocess(image)

    # make prediction
    prediction = predict(image)
    dice = {
        'prediction': prediction,
    }
    print(JSONResponse(dice).body)
    return JSONResponse(dice)
