import os
import threading
import pandas as pd
import cv2
import numpy as np
from flask_cors import CORS
from flask import Flask, jsonify, request
import torch
import torch.nn as nn
from torchvision import transforms

# Define the port number
port_number = int(os.getenv("PORT", 5000))

app = Flask(__name__)
CORS(app, supports_credentials=True, expose_headers=["Content-Type"])

# Load the PyTorch model
class GreenGuruModel(torch.nn.Module):
    def __init__(self, num_classes=12):
        super(GreenGuruModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the model architecture
model = GreenGuruModel(num_classes=12)

# Load the state dictionary from the .pth file
state_dict = torch.load('GreenGuruPT.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode

# Define the image preprocessing transform
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# EDSR function
def super_res(image_path):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("EDSR_x3.pb")
    sr.setModel("edsr", 3)
    img = cv2.imread(image_path)
    result = sr.upsample(img)
    return result

# Segmentation function
def segmentation(image):
    lower_green = np.array([25, 50, 50])
    upper_green = np.array([80, 255, 255])
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    superimposed_image = cv2.bitwise_and(image, mask_3c)
    return superimposed_image

# Home route
@app.route('/')
def home():
    return "Welcome to GreenGuru API!"

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running"}), 200

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']

        # Save the file to a temporary location
        temp_path = 'temp_uploaded_image.jpg'
        file.save(temp_path)

        # Read the image using OpenCV
        img = cv2.imread(temp_path)

        if img is not None:
            height, width = img.shape[:2]
            if width < 300 and height < 300:
                result = super_res(temp_path)
            else:
                result = img

        prep_img = segmentation(result)
        prep_img_resized = cv2.resize(prep_img, (128, 128))
        prep_img_resized = cv2.cvtColor(prep_img_resized, cv2.COLOR_BGR2RGB)  # Convert to RGB
        prep_img_tensor = preprocess(prep_img_resized)  # Apply preprocessing
        prep_img_tensor = prep_img_tensor.unsqueeze(0)  # Add batch dimension

        # Make predictions
        with torch.no_grad():
            predictions = model(prep_img_tensor)
            predicted_class = torch.argmax(predictions, dim=1).item()

        class_names = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', "Shepherd's Purse", 'Small-flowered Cranesbill', 'Sugar beet']

        return jsonify({
            "prediction": class_names[predicted_class]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port_number)