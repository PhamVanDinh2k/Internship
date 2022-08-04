from fastapi import FastAPI, File, UploadFile
from predictor import *
import uvicorn
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
    print(JSONResponse(prediction).body)
    return JSONResponse(jsonable_encoder(prediction))

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host= "127.0.0.1")