import os
import numpy as np
import cv2
import joblib
from sklearn.svm import SVC
from joblib import Parallel, delayed
from skimage.feature import hog
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import train_test_split
from glob import glob
import time

start_time = time.time()
print(start_time)

DATASET_DIR = "Dataset"
PHONE_DIR = "C:/Users/aayus/Desktop/PhoneSVM/Dataset/phone"
NO_PHONE_DIR = "C:/Users/aayus/Desktop/PhoneSVM/Dataset/nophone"

IMAGE_SIZE = (150, 150)



print("Running 1: ", time.time() - start_time)
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Unable to read image {image_path}")
        return None 
    
    img = cv2.resize(img, IMAGE_SIZE)
    features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

    return features




print("Running: 2: ", time.time() - start_time)
X, y = [], []
phone_images = glob(os.path.join(PHONE_DIR, "*.jpg")) + glob(os.path.join(PHONE_DIR, "*.jpeg")) + glob(os.path.join(PHONE_DIR, "*.png"))
no_phone_images = glob(os.path.join(NO_PHONE_DIR, "*.jpg")) + glob(os.path.join(NO_PHONE_DIR, "*.jpeg")) + glob(os.path.join(NO_PHONE_DIR, "*.png")) + glob(os.path.join(NO_PHONE_DIR, "*.bmp"))

print("Running: 3: ", time.time() - start_time)
if not phone_images or not no_phone_images:
    print("Error: No images found in the dataset directories. Please check the paths.")
    exit()



def process_image(img_paths, label):
    results = Parallel(n_jobs=-1)(delayed(extract_features)(img_path) for img_path in img_paths)
    results = [(features, label) for features in results if features is not None]
    
    return results
    
print("Running: 4: ", time.time() - start_time)


# for img_path in phone_images:
#     features = extract_features(img_path)
#     if features is not None:
#         X.append(features)
#         y.append(1)

# for img_path in no_phone_images:
#     features = extract_features(img_path)
#     if features is not None:
#         X.append(features)
#         y.append(0) 
    
phone_features = process_image(phone_images, 1)
no_phone_features = process_image(no_phone_images, 0)   

print("Running: 5: ", time.time() - start_time)

X, y = zip(*phone_features + no_phone_features)
X, y= np.array(X), np.array(y) 

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

print("Running: 6: ", time.time() - start_time)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", class_weight="balanced")
svm_model.fit(X_train, y_train)

print("Running: 7: ", time.time() - start_time)
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"Test Accuracy: {svm_model.score(X_test, y_test) * 100:.2f}%")
print("total time to run: ", time.time() - start_time)




