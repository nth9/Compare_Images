import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix

# 1. Load the Pre-trained Model
print("Loading the pre-trained model...")
model = load_model("apetizing_model_V3.keras")
print("Model loaded successfully.")

# 2. Load and Process Test Dataset
print("Loading dataset from the CSV file...")

CSV_PATH = "datasets/train/newtrainset.csv"
IMAGE_FOLDER = "datasets/train/"

df = pd.read_csv(CSV_PATH)

# Use a maximum of 200 image pairs for testing (or fewer if not available)
df = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0  # Normalize pixel values to the range [0, 1]
    return img

x1, x2, y_true = [], [], []

for _, row in df.iterrows():
    img1_path = os.path.join(IMAGE_FOLDER, row["Image_1"])
    img2_path = os.path.join(IMAGE_FOLDER, row["Image_2"])
    
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        x1.append(load_and_preprocess_image(img1_path))
        x2.append(load_and_preprocess_image(img2_path))
        y_true.append(row["Winner"] - 1)  # Convert 1 -> 0, 2 -> 1

x1, x2, y_true = np.array(x1), np.array(x2), np.array(y_true)

print(f"Successfully loaded {len(x1)} image pairs for testing.")

# 3. Make Predictions Using the Model
print("Making predictions...")
y_pred_prob = model.predict([x1, x2])  # Generate probability values (0 to 1)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions (0 or 1)

print("Prediction process completed.")

# 4. Generate and Display Confusion Matrix
print("Generating the confusion matrix...")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Image_1", "Image_2"], yticklabels=["Image_1", "Image_2"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("Confusion matrix displayed successfully.")
