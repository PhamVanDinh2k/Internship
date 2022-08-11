import base64
from PIL import Image
from io import BytesIO
import cv2

with open("./datatest/ABBOTTS BABBLER/1.jpg", "rb") as image_file:
    data = base64.b64encode(image_file.read())

im = Image.open(BytesIO(base64.b64decode(data)))
print("begin",data)