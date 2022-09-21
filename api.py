import base64
from multiprocessing import context
from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from test import *
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="./templates")


@app.get("/index/", response_class=HTMLResponse)
def index(request: Request):
    context = {"request": request}
    return templates.TemplateResponse("upload.html", context)


@app.post("/test", response_class=HTMLResponse)
async def handle_form(request: Request, file: bytes = File(...)):
    image = read_image(file)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = base64.b64encode(img_byte).decode()
    image1 = preprocess(image)

    # make prediction
    prediction, max = predict(image1)
    maxx = round((max*100),2)
    accuracy = str(maxx)
    context = {
        "request": request,
        "prediction": prediction,
        "accuracy": accuracy,
        "base64": img_str,
    }
    print(f"request : {request}")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "accuracy": accuracy,
            "base64": img_str,
        },
    )
