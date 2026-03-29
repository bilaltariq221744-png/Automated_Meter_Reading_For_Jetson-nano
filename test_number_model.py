import os
import cv2
from ultralytics import YOLO


MODEL_PATH = r"C:/Users/PMLS/Desktop/Meter_reading_indonesia/Number_detection_model/weights/best.pt"
IMAGE_FOLDER = r"C:/Users/PMLS/Desktop/Meter_reading_indonesia/cropped_screens"
OUTPUT_FOLDER = r"C:/Users/PMLS/Desktop/Meter_reading_indonesia/test_results"

IMG_SIZE = 320

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

for img_name in os.listdir(IMAGE_FOLDER):

    img_path = os.path.join(IMAGE_FOLDER, img_name)

    # Skip non-image files
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Read image
    img = cv2.imread(img_path)

    # Resize image
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Run inference
    results = model(img_resized)

    # Draw predictions
    annotated_img = results[0].plot()

    # Save result
    save_path = os.path.join(OUTPUT_FOLDER, img_name)
    cv2.imwrite(save_path, annotated_img)

    print(f"Processed: {img_name}")

print(" All images processed and saved.")