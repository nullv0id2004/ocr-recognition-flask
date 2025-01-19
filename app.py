from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

app = Flask(__name__)

class OCRModel(nn.Module):
    def __init__(self):
        super(OCRModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Output: (16, 28, 28)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Output: (32, 14, 14)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: (64, 7, 7)

        
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 27) 

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = F.relu(F.max_pool2d(self.conv3(x), 2))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = OCRModel()
state_dict = torch.load('model/simple_cnn_ocr.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        encoded_image = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_image), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        preprocessed_image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(preprocessed_image)
            predicted_class = outputs.argmax(dim=1).item()
        return jsonify({'prediction': int(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
