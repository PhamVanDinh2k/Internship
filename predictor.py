from PIL import Image
from io import BytesIO 
import numpy as np 
import torch
import torch.nn as nn
from torchvision import models, transforms
import os

test_dir ='./datatest'
classes = os.listdir(test_dir)

def read_image(image_encoded):
    print(type(image_encoded))
    pil_image = Image.open(BytesIO(image_encoded))
    print(pil_image)
    return pil_image

class BaseTransform():
  def __init__(self):
    self.base_transform = transforms.Compose([transforms.Resize((224, 224)), 
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

  def __call__(self, img_input):
    return self.base_transform(img_input)

def preprocess(image : Image.Image):
    transforms = BaseTransform()
    img_tranformed = transforms(image)
    img_tranformed = torch.unsqueeze(torch.tensor(img_tranformed), 0)
    print(img_tranformed.shape)
    return img_tranformed

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

class Pretictor():
    def __init__(self, class_index):
        self.class_index = class_index
        self.classes = classes
        
    def predict_max(self, out):
        print(out.shape)
        max_id = np.argmax(out.detach().numpy())
        max = np.max(out.detach().numpy())  
        predicted_label_name = self.classes[(max_id)]
        return [predicted_label_name, max] 
        

def predict(image):
    net = ResNet()
    net.load_state_dict(torch.load("./models/resnet50plus.pth", map_location=torch.device('cpu')))
    net.eval()
    predict = Pretictor(net)
    out = net(image)
    [result, max] = predict.predict_max(out)
    print("result ==", result)
    print("accuracy ==", max)
    return result, max
