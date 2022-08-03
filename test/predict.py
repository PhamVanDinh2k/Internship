
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchsummary import summary
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
from torchvision.datasets import ImageFolder
from torchvision import models, transforms

test_dir ='./datatest'
classes = os.listdir(test_dir)
print(classes)




transformations = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
     transforms.Normalize(
        (0.485, 0.456, 0.406), 
        (0.229, 0.224, 0.225)
     )
])


test_dataset = ImageFolder(test_dir, transform = transformations)

random_seed = 42
torch.manual_seed(random_seed)
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(test_dataset.classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

net = ResNet()
# net = models.resnet50(pretrained = True)
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, len(test_dataset.classes))
net.load_state_dict(torch.load("./models/resnet50plus.pth", map_location=torch.device('cpu')))
# print(net)
net.eval()

class BaseTransform():
  def __init__(self, resize, mean, std):
    self.base_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

  def __call__(self, img_input):
    return self.base_transform(img_input)


class Pretictor():
	def __init__(self, class_index):
		self.class_index = class_index
		self.classes = test_dataset.classes

	def predict_max(self, out):
		print(out.shape)
		max_id = np.argmax(out.detach().numpy())
		#predicted_label_name = self.class_index[str(max_id)]
    
		predicted_label_name = self.classes[(max_id)]
		return predicted_label_name

# class_index = torch.load("resnet50plus.pth", "cpu")

predict = Pretictor(net)


""" img_path = "./test/AFRICAN CROWNED CRANE/3.jpg"
img = Image.open(img_path)

resize = (224, 224)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transforms = BaseTransform(resize, mean, std)

img_tranformed = transforms(img)

img_tranformed = torch.unsqueeze(torch.tensor(img_tranformed), 0)
print(img_tranformed.shape) """
img, label = test_dataset[19]
img = torch.unsqueeze(torch.tensor(img), 0)
print(img.shape)
out = net(img)
print('Label:', test_dataset.classes[label])
result = predict.predict_max(out)

print("result ==", result)
