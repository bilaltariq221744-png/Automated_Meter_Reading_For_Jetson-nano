# yolov8_train_aug.py
from ultralytics import YOLO

# 1️⃣ Load pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")  # Nano model pre-trained on COCO

# 2️⃣ Train the model with augmentation and advanced settings
model.train(
    data="C:\\Users\\PMLS\\Desktop\\Meter_reading_indonesia\\dataset_numbers\\data.yaml",  # dataset YAML
    epochs=100,          # total epochs
    imgsz=320,           # image size (320x320)
    batch=16,            # batch size (reduce for Jetson Nano)
    device='cpu',        # 'cpu' or GPU ID (0)

    # Learning rate
    lr0=0.01,            # initial learning rate
    lrf=0.1,             # final learning rate factor
    
    # Optimizer
    optimizer='SGD',     # 'SGD' or 'Adam'
    
    # Early stopping
    patience=10,         # stop training if no improvement for 10 epochs
    
    # Data augmentation

    mosaic=True,         # combines 4 images into 1
    mixup=0.5,           # mixup augmentation factor (0=no mixup, 0.5 moderate)
    hsv_h=0.015,         # hue augmentation
    hsv_s=0.7,           # saturation augmentation
    hsv_v=0.4,           # value augmentation
    degrees=10.0,        # image rotation ± degrees
    translate=0.1,       # translation ± fraction of image
    scale=0.5,           # scale ± factor
    shear=2.0,           # shear ± degrees
    
    # Other options
    name="meter_digits_aug",  # save folder
    save_period=10,
    verbose=True
)

# ✅ After training, best weights are in:
# runs/train/meter_digits_aug/weights/best.pt