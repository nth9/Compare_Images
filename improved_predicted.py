import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load and Preprocess Images
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image by resizing it to the target size and normalizing the pixel values.
    """
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img) / 255.0  # Normalize to 0-1
    return img

# 2. Load Pre-trained Model
print("Loading pre-trained model...")
model = tf.keras.models.load_model("New_Finish/food_comparison_model_mobilenet.keras")
print("Model loaded successfully.")

# 3. Prediction Function
def predict_appetizing(img1_path, img2_path):
    """
    Predict the more appetizing image between two.
    """
    img1 = load_and_preprocess_image(img1_path)
    img2 = load_and_preprocess_image(img2_path)

    # Display the images using matplotlib before predicting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img1)
    axes[0].set_title("Image 1")
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].set_title("Image 2")
    axes[1].axis("off")

    plt.show()

    # Make prediction using the model
    prediction = model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])
    
    print(f"Prediction output: {prediction}")

    # If prediction <= 0.5, Image 1 is more appetizing, else Image 2 is more appetizing
    winner = 1 if prediction[0] <= 0.5 else 2  # Output 1 for Image 1, 2 for Image 2

    return winner

# 4. Compare Images Based on CSV File
def compare_images_from_csv(csv_file):
    """
    Read image paths from CSV, compare images and save predictions in the same CSV.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if required columns exist
    if 'Image 1' not in df.columns or 'Image 2' not in df.columns:
        print("CSV must have 'Image 1' and 'Image 2' columns.")
        return
    
    # Add a 'Winner' column for predictions
    df['Winner'] = None

    # Compare each pair of images and predict the winner
    for index, row in df.iterrows():
        img1_path = 'test_images/' + row['Image 1']
        img2_path = 'test_images/' + row['Image 2']
        
        print(f"Comparing {img1_path} and {img2_path}...")
        
        # Ensure that the images in the columns are different
        if row['Image 1'] == row['Image 2']:
            print("Warning: Image 1 and Image 2 are the same. Skipping this row.")
            continue

        winner = predict_appetizing(img1_path, img2_path)
        df.at[index, 'Winner'] = winner  # Store the winner in the 'Winner' column

    # Save the results to the CSV file
    df.to_csv(csv_file, index=False)
    print(f"Comparison results saved to '{csv_file}'.")

# 5. Specify CSV File Path
csv_file = "test.csv"  # Update this path with your CSV file containing image paths
compare_images_from_csv(csv_file)
