import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = NeuralNetwork()
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device("cpu")))
model.eval()

img = cv2.imread("test_digit.png", cv2.IMREAD_GRAYSCALE)
_, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
img = cv2.resize(img, (28, 28))
img = img.astype("float32") / 255.0
img = (img - 0.1307) / 0.3081
img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    digit = torch.argmax(output, dim=1).item()

print(f"Predicted Digit: {digit} ")
