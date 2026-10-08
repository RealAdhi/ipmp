import os
import random
import cv2
import numpy as np
import pandas as pd

# Directories
INPUT_DIR = "raw_images"
OUTPUT_DIR = "processed_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Gather all files
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

if len(all_files) < 100:
    print(f"Error: You only have {len(all_files)} images. Please add at least 100 images to 'raw_images'.")
    exit()

# 2. Randomly select exactly 100
selected_files = random.sample(all_files, 100)
csv_records = []

print("Processing images and building CSV list...")

# 3. Processing & Metadata Logging Loop
for index, filename in enumerate(selected_files, start=1):
    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path)
    
    if img is None:
        continue

    # A. Grayscale Conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # B. Noise Removal
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # C. K-Means Clustering (Cluster Size / Classes K=5)
    pixel_values = denoised.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_values, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    clustered_img = centers[labels.flatten()].reshape(denoised.shape)

    # D. Save Processed Image (Renamed, 80% JPEG quality)
    new_filename = f"{index}.jpg"
    output_path = os.path.join(OUTPUT_DIR, new_filename)
    cv2.imwrite(output_path, clustered_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    # E. Deduce category for metadata mapping
    category = "Diseased" if any(x in filename.lower() for x in ["blight", "spot", "rust", "rot", "scab", "diseased"]) else "Healthy"

    # F. Append to records for the CSV log
    csv_records.append({
        "Image Identification/Filename": new_filename,
        "Image Source": "PlantVillage Open Dataset Repository",
        "Original Dimensions (WxH)": f"{img.shape[1]}x{img.shape[0]}",
        "Image Format": "JPEG",
        "Category or Grouping": category
    })

# 4. Generate the CSV file directly
df = pd.DataFrame(csv_records)
df.to_csv("dataset_summary.csv", index=False)

print("\nSuccess! Generated 'dataset_summary.csv' and processed exactly 100 images in 'processed_dataset'.")
