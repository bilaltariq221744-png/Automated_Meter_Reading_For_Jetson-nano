# ============================================================
#        ELECTRICITY METER - LCD DISPLAY DETECTION
#              YOLOv8n Training Script
# ============================================================

import os
import time
import yaml
from pathlib import Path
from ultralytics import YOLO

# ============================================================
#                  CONFIGURATION
#        Only change things in this section
# ============================================================

# Path to your data.yaml file
DATA_YAML = "C:/Users/PMLS/Desktop/Indonesia/LCD_Screen_Detection.v3-main.yolov8/data.yaml"   # Change this to your full path 

# Training Parameters
EPOCHS      = 100     # Number of training epochs
BATCH_SIZE  = 8       # Reduced for CPU (use 16 if you have GPU)
IMAGE_SIZE  = 640     # Must match your dataset export size
PATIENCE    = 20       # Early stopping patience (stops if no improvement)
WORKERS     = 2      # Number of data loading workers (keep low for CPU)
LR          = 0.01   # Initial learning rate

# Output
PROJECT_NAME = "meter_lcd_detection"   # Folder where results will be saved
RUN_NAME     = "yolov8n_run1"          # Name of this training run

# ============================================================
#                  VERIFY PATHS
# ============================================================

def verify_setup():
    print("=" * 60)
    print("      VERIFYING SETUP BEFORE TRAINING")
    print("=" * 60)

    # Check data.yaml
    if not os.path.exists(DATA_YAML):
        print(f"\n[ERROR] data.yaml not found at: {DATA_YAML}")
        print("Please update the DATA_YAML path in the script.")
        return False

    # Read and display data.yaml content
    with open(DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)

    print(f"\n[OK] data.yaml found!")
    print(f"     Classes    : {data.get('nc', 'N/A')}")
    print(f"     Class Names: {data.get('names', 'N/A')}")
    print(f"     Train Path : {data.get('train', 'N/A')}")
    print(f"     Val Path   : {data.get('val', 'N/A')}")
    print(f"     Test Path  : {data.get('test', 'N/A')}")

    # Check train/val folders exist
    yaml_dir = Path(DATA_YAML).parent

    train_path = yaml_dir / data.get('train', '').replace('../', '')
    val_path   = yaml_dir / data.get('val', '').replace('../', '')

    # Try absolute paths from yaml values
    train_abs = Path(data.get('train', ''))
    val_abs   = Path(data.get('val', ''))

    train_ok = train_path.exists() or train_abs.exists()
    val_ok   = val_path.exists()   or val_abs.exists()

    if train_ok:
        print(f"\n[OK] Training images folder found!")
    else:
        print(f"\n[WARNING] Could not auto-verify train path.")
        print(f"          Make sure your dataset folder structure is correct.")

    if val_ok:
        print(f"[OK] Validation images folder found!")
    else:
        print(f"[WARNING] Could not auto-verify validation path.")

    print("\n[OK] Setup looks good — Starting Training!")
    print("=" * 60)
    return True


# ============================================================
#                  TRAINING
# ============================================================

def train():

    # Verify setup first
    if not verify_setup():
        return

    print("\n[INFO] Loading YOLOv8n model...")
    # Load YOLOv8n pretrained model
    # This will auto-download yolov8n.pt (~6MB) on first run
    model = YOLO('yolov8n.pt')

    print("[INFO] Starting training...\n")
    print("=" * 60)
    print("         TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Model       : YOLOv8n")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Batch Size  : {BATCH_SIZE}")
    print(f"  Image Size  : {IMAGE_SIZE}")
    print(f"  Patience    : {PATIENCE} (early stopping)")
    print(f"  Workers     : {WORKERS}")
    print(f"  Device      : CPU")
    print(f"  Output      : {PROJECT_NAME}/{RUN_NAME}")
    print("=" * 60)
    print("\n[INFO] Training started... This will take time on CPU.")
    print("[INFO] You can monitor progress below.\n")

    start_time = time.time()

    # ---- TRAIN ----
    results = model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        batch     = BATCH_SIZE,
        imgsz     = IMAGE_SIZE,
        patience  = PATIENCE,
        workers   = WORKERS,
        lr0       = LR,
        device    = 'cpu',
        project   = PROJECT_NAME,
        name      = RUN_NAME,
        exist_ok  = True,

        # Augmentation settings
        hsv_h     = 0.015,   # Hue augmentation
        hsv_s     = 0.5,    # Saturation augmentation (handles backlight on/off)
        hsv_v     = 0.4,     # Brightness augmentation
        degrees   = 15.0,   # Rotation (handles camera angle variation)
        translate = 0.1,     # Translation
        scale     = 0.5,   # Scale variation (handles different distances)
        shear     = 2.0,   # Slight shear
        flipud    = 0.0,  # No vertical flip (meter is always upright)
        fliplr    = 0.5,   # Horizontal flip
        mosaic    = 1.0,  # Mosaic augmentation
        mixup     = 0.1,  # Mixup augmentation

        # Optimization
        optimizer = 'SGD',   # Stochastic Gradient Descent is more stable on CPU
        momentum  = 0.937,   # Momentum for SGD
        weight_decay = 0.0005, # Regularization to prevent overfitting

        # Logging
        verbose   = True,
        plots     = True,    # Save training plots
        save      = True,    # Save checkpoints
        save_period = 10,    # Save checkpoint every 10 epochs
    )

    end_time = time.time()
    total_time = (end_time - start_time) / 3600

    # ============================================================
    #                  TRAINING COMPLETE
    # ============================================================

    print("\n" + "=" * 60)
    print("            TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Total Training Time : {total_time:.2f} hours")
    print(f"  Results saved to    : {PROJECT_NAME}/{RUN_NAME}/")
    print("=" * 60)

    # ---- VALIDATE on validation set ----
    print("\n[INFO] Running validation on validation set...")
    val_results = model.val()
    print(f"\n[VALIDATION RESULTS]")
    print(f"  mAP50       : {val_results.box.map50:.4f}")
    print(f"  mAP50-95    : {val_results.box.map:.4f}")
    print(f"  Precision   : {val_results.box.mp:.4f}")
    print(f"  Recall      : {val_results.box.mr:.4f}")

    # ---- TEST on test images ----
    print("\n[INFO] Running prediction on test images...")
    # ensure output directory exists (YOLO predict will create it if exist_ok=True)
    os.makedirs(f"{PROJECT_NAME}/test_predictions", exist_ok=True)
    test_results = model.predict(
        source   = 'test/images',
        save     = True,
        conf     = 0.5,
        project  = PROJECT_NAME,
        name     = 'test_predictions',
        exist_ok = True           # allow overwriting or existing folder
    )

    # ---- EXPORT best model ----
    print("\n[INFO] Exporting best model...")
    best_model_path = f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt"

    if os.path.exists(best_model_path):
        print(f"[OK] Best model saved at: {best_model_path}")
        print("\n[INFO] This is the model you will use for:")
        print("       1. Testing on your video")
        print("       2. Converting to TensorRT for Jetson Nano")
    else:
        print(f"[WARNING] Best model not found at expected path.")
        print(f"          Check {PROJECT_NAME}/{RUN_NAME}/weights/")

    print("\n" + "=" * 60)
    print("   WHAT TO DO NEXT:")
    print("=" * 60)
    print(f"  1. Check training plots in {PROJECT_NAME}/{RUN_NAME}/")
    print(f"  2. Best model is at: {best_model_path}")
    print(f"  3. Share your mAP50 result with me")
    print(f"  4. We will then move to Step 2: Perspective Correction")
    print("=" * 60)


# ============================================================
#                       MAIN
# ============================================================

if __name__ == "__main__":
    train()