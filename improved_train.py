import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split

# Step 1: Load Data from CSV
CSV_PATH = "datasets/train/newtrainset.csv"
IMAGE_FOLDER = "datasets/train/"

df = pd.read_csv(CSV_PATH)

# Step 2: Load and Preprocess Images
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    return img

x1, x2, y = [], [], []

for _, row in df.iterrows():
    img1_path = os.path.join(IMAGE_FOLDER, row["Image_1"])
    img2_path = os.path.join(IMAGE_FOLDER, row["Image_2"])
    
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        x1.append(load_and_preprocess_image(img1_path))
        x2.append(load_and_preprocess_image(img2_path))
        y.append(int(row["Winner"]) - 1)

x1, x2, y = np.array(x1), np.array(x2), np.array(y)

# Step 3: Split Data into Training and Validation Sets
x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1, x2, y, test_size=0.2, random_state=42)

# Step 4: Define the Siamese Network Model
def create_siamese_network():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    def create_branch():
        input_layer = Input(shape=(224, 224, 3))
        x = base_model(input_layer, training=False)
        x = GlobalAveragePooling2D()(x)
        return Model(input_layer, x)

    branch = create_branch()

    input_a = Input(shape=(224, 224, 3))
    input_b = Input(shape=(224, 224, 3))

    vector_a = branch(input_a)
    vector_b = branch(input_b)

    merged = Concatenate()([vector_a, vector_b])
    x = Dense(128, activation="relu")(merged)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[input_a, input_b], outputs=output)
    return model

model = create_siamese_network()
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])

# Step 5: Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

val_datagen = ImageDataGenerator()

def create_tf_dataset(x1, x2, y, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices(((x1, x2), y))
    dataset = dataset.shuffle(buffer_size=len(y))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = create_tf_dataset(x1_train, x2_train, y_train)
val_dataset = create_tf_dataset(x1_val, x2_val, y_val)

# Step 6: Define Callbacks
lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
early_stopper = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Step 7: Train the Model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[lr_reducer, early_stopper]
)

print("Model training completed.")

# Step 8: Save the Model
model.save("compare_V3.keras")
print("Model saved as 'compare_V3.keras'.")
