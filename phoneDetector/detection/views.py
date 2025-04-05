import os
import numpy as np
import joblib
import cv2
from django.conf import settings
from skimage.feature import hog
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

MODEL_PATH = os.path.join(settings.BASE_DIR, "detection", "svm_model.pkl")
SCALER_PATH = os.path.join(settings.BASE_DIR, "detection", "scaler.pkl")

svm_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

IMAGE_SIZE = (150, 150)

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

    return features

def predict_image(img_path):
    features = extract_features(img_path)
    features = scaler.transform([features])  
    prediction_prob = svm_model.predict_proba(features)[0][1]

    accuracy = round(prediction_prob * 100, 2)
    category = "Phone Detected" if prediction_prob > 0.5 else "No Phone Detected"

    return accuracy, category

def upload_image(request):
    if request.method == "POST" and request.FILES["image"]:
        uploaded_file = request.FILES["image"]
        fs = FileSystemStorage()
        file_path = fs.save("uploads/" + uploaded_file.name, uploaded_file)

        file_url = fs.url(file_path)

        accuracy, category = predict_image(os.path.join(settings.MEDIA_ROOT, file_path))
        print("DEBUG: Image URL =", file_url)
        print("DEBUG: Accuracy =", accuracy)
        print("DEBUG: Category =", category)


        return render(request, "detection/result.html", {"image_url": file_url, "accuracy": accuracy, "category": category})

    return render(request, "detection/upload.html")
