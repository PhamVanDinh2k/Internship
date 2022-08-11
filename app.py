
from multiprocessing import context
from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from predictor import *
from fastapi.responses import JSONResponse
app = FastAPI()

templates = Jinja2Templates(directory="./templates")

@app.get('/index/', response_class=HTMLResponse)
def index(request: Request):
   context = {'request' : request}
   return templates.TemplateResponse("index.html", context) 

@app.post("/test")
async def handle_form(file : bytes = File(...)): 
    image = read_image(file)
    image = preprocess(image)
    # make prediction
    prediction = predict(image)
    dice = {
        'prediction': prediction,
    }
    print(JSONResponse(dice).body)
    return JSONResponse(dice)

