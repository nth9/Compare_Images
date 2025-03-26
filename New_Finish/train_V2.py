import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# กำหนด path
DATASET_PATH = "datasets/train"  # 🔴 เปลี่ยนเป็น path ของคุณ
CSV_FILE = "datasets/train/newtrainset.csv"  # 🔴 เปลี่ยนเป็น path ของคุณ

# ฟังก์ชันหาไฟล์ภาพ (รองรับโฟลเดอร์)
def find_image(image_path):
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    base_path = os.path.join(DATASET_PATH, image_path).replace("\\", "/")  # เชื่อม path
    for ext in valid_extensions:
        full_path = base_path if base_path.lower().endswith(ext) else base_path + ext
        if os.path.exists(full_path):
            return full_path
    return None

# ฟังก์ชันโหลดภาพ
def load_image(image_path, target_size=(300, 300)):
    image_path = find_image(image_path)
    if image_path is None:
        print(f"❌ Error: File not found - {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Failed to load image - {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

# โหลด dataset
df = pd.read_csv(CSV_FILE)

# 🔍 Debug: เช็คว่าข้อมูล CSV เป็นยังไง
print("🔍 CSV Sample:")
print(df.head())

X1, X2, y = [], [], []
for _, row in df.iterrows():
    img1 = load_image(row['Image_1'])  # รูปแบบ: Ramen/r21.jpg
    img2 = load_image(row['Image_2'])

    if img1 is None or img2 is None:
        print(f"⚠️ Skipping: {row['Image_1']} or {row['Image_2']}")
        continue  

    X1.append(img1)
    X2.append(img2)
    y.append(row['Winner'] - 1)

X1, X2, y = np.array(X1), np.array(X2), np.array(y)

# 🔍 Debug: เช็คขนาดข้อมูล
print(f"✅ Loaded Data: {len(X1)} samples")

if len(X1) == 0:
    raise ValueError("❌ ไม่มีข้อมูลภาพที่โหลดได้เลย กรุณาตรวจสอบไฟล์ CSV และภาพในโฟลเดอร์")

# แบ่ง train 80% และ validate 20%
X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

# ✅ ถ้ารันผ่าน แสดงว่ามีข้อมูลพร้อม train แล้ว!
print("✅ Data is ready for training!")

# สร้างโมเดลเปรียบเทียบรูป
input_shape = (300, 300, 3)

# ใช้ MobileNetV2 แทน EfficientNetB0
base_model = keras.applications.MobileNetV2(include_top=False, input_shape=input_shape, weights="imagenet")
base_model.trainable = False  # ใช้โมเดลที่เทรนไว้แล้ว

def create_branch():
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    return keras.Model(inputs, x)

branch1 = create_branch()
branch2 = create_branch()

combined = layers.concatenate([branch1.output, branch2.output])
x = layers.Dense(128, activation="relu")(combined)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=[branch1.input, branch2.input], outputs=outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# การตั้งค่าการเรียนรู้ที่ปรับได้
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 10:
        lr = 0.0005
    if epoch > 15:
        lr = 0.0001
    print(f"Learning rate for epoch {epoch + 1} is {lr}")
    return lr

# เทรนโมเดล
history = model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_val, X2_val], y_val),
    epochs=10,
    batch_size=32,
    callbacks=[keras.callbacks.LearningRateScheduler(lr_schedule)]
)

# เซฟโมเดล
model.save("food_comparison_model_mobilenet.keras")
print("✅ Model saved as food_comparison_model_mobilenet.keras")
