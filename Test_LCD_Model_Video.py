import cv2
import time
from pathlib import Path
from ultralytics import YOLO

VIDEO_PATH  = r"C:/Users/PMLS/Desktop/Meter_project/WhatsApp Video 2025-08-14 at 2.56.45 AM.mp4"   
MODEL_PATH  = r"C:/Users/PMLS/Desktop/Automated_Meter_Reading_For_Jetson-nano/LCD_Detection_model/weights/best.pt"

# Detection Settings
CONF_THRESHOLD = 0.5             
SAVE_OUTPUT    = True           
OUTPUT_PATH    = r"C:/Users/PMLS/Desktop/Automated_Meter_Reading_For_Jetson-nano/output_detected.mp4"  
SHOW_WINDOW    = True

PROCESS_EVERY_NTH_FRAME = 1 

BOX_COLOR       = (0, 255, 0)    
TEXT_COLOR      = (0, 255, 0)    
CONF_BAR_COLOR  = (0, 200, 255)  

def test_video():

    if not Path(MODEL_PATH).exists():
        print(f"\n[ERROR] Model not found at: {MODEL_PATH}")
        print("Please update MODEL_PATH in the script.")
        return

    if not Path(VIDEO_PATH).exists():
        print(f"\n[ERROR] Video not found at: {VIDEO_PATH}")
        print("Please update VIDEO_PATH in the script.")
        return

    print("\n" + "="*60)
    print("      LCD DISPLAY DETECTOR — VIDEO TEST")
    print("="*60)
    print(f"\n[INFO] Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"[OK]   Model loaded successfully!")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / fps if fps > 0 else 0

    print(f"\n[VIDEO INFO]")
    print(f"  Path         : {VIDEO_PATH}")
    print(f"  Resolution   : {width} x {height}")
    print(f"  FPS          : {fps:.2f}")
    print(f"  Total Frames : {total_frames}")
    print(f"  Duration     : {duration:.2f} seconds")
    print(f"\n[SETTINGS]")
    print(f"  Confidence   : {CONF_THRESHOLD}")
    print(f"  Every Nth    : {PROCESS_EVERY_NTH_FRAME}")
    print(f"  Save Output  : {SAVE_OUTPUT}")
    print(f"  Show Window  : {SHOW_WINDOW}")
    print("="*60)

    writer = None
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
        print(f"\n[INFO] Output will be saved to: {OUTPUT_PATH}")

    if SHOW_WINDOW:
        print(f"\n[INFO] Showing live window...")
        print(f"       Press 'Q' to quit anytime")
        print(f"       Press 'S' to save a screenshot")
        print(f"       Press 'P' to pause/resume")

    print(f"\n[INFO] Processing video...\n")

    frame_count      = 0
    processed_count  = 0
    detection_count  = 0
    total_conf       = 0
    start_time       = time.time()
    paused           = False
    screenshot_count = 0

    while True:

        # Handle pause
        if paused:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('p') or key == ord('P'):
                paused = False
                print("[INFO] Resumed")
            elif key == ord('q') or key == ord('Q'):
                print("[INFO] Quit by user")
                break
            continue

        ret, frame = cap.read()
        if not ret:
            print("\n[INFO] Video ended.")
            break

        frame_count += 1

        if frame_count % PROCESS_EVERY_NTH_FRAME != 0:
            if writer:
                writer.write(frame)
            continue

        processed_count += 1

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        annotated_frame = frame.copy()
        detections_in_frame = 0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls  = int(box.cls[0])

                detections_in_frame += 1
                detection_count     += 1
                total_conf          += conf

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

                label       = f"LCD {conf:.2f}"
                label_size  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                label_bg_x2 = x1 + label_size[0] + 10
                label_bg_y2 = y1 - 5
                label_bg_y1 = y1 - label_size[1] - 15

                cv2.rectangle(annotated_frame,
                              (x1, label_bg_y1),
                              (label_bg_x2, label_bg_y2),
                              BOX_COLOR, -1)

                cv2.putText(annotated_frame, label,
                            (x1 + 5, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 2)

                bar_width  = x2 - x1
                bar_filled = int(bar_width * conf)
                cv2.rectangle(annotated_frame,
                              (x1, y2 + 2),
                              (x1 + bar_width, y2 + 8),
                              (100, 100, 100), -1)
                cv2.rectangle(annotated_frame,
                              (x1, y2 + 2),
                              (x1 + bar_filled, y2 + 8),
                              CONF_BAR_COLOR, -1)

        elapsed      = time.time() - start_time
        current_fps  = processed_count / elapsed if elapsed > 0 else 0
        progress_pct = (frame_count / total_frames) * 100

        cv2.rectangle(annotated_frame, (0, 0), (320, 130), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (0, 0), (320, 130), (0, 255, 0), 1)

        cv2.putText(annotated_frame, f"LCD Meter Detector",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Frame  : {frame_count}/{total_frames} ({progress_pct:.1f}%)",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"FPS    : {current_fps:.1f}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Detections this frame: {detections_in_frame}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if detections_in_frame > 0 else (0, 0, 255), 1)
        cv2.putText(annotated_frame, f"Total detections: {detection_count}",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Conf   : {CONF_THRESHOLD}  |  Press Q to quit",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        if writer:
            writer.write(annotated_frame)

        if SHOW_WINDOW:
            cv2.imshow("LCD Meter Detector", annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("\n[INFO] Quit by user.")
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_count += 1
                screenshot_name = f"screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_name, annotated_frame)
                print(f"[INFO] Screenshot saved: {screenshot_name}")
            elif key == ord('p') or key == ord('P'):
                paused = True
                print("[INFO] Paused. Press P to resume.")

        if processed_count % 100 == 0:
            print(f"  Progress: {progress_pct:.1f}% | "
                  f"Frame {frame_count}/{total_frames} | "
                  f"FPS: {current_fps:.1f} | "
                  f"Detections: {detection_count}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    total_time  = time.time() - start_time
    avg_conf    = total_conf / detection_count if detection_count > 0 else 0
    avg_fps     = processed_count / total_time if total_time > 0 else 0

    print("\n" + "="*60)
    print("              TESTING COMPLETE!")
    print("="*60)
    print(f"  Total Frames     : {frame_count}")
    print(f"  Frames Processed : {processed_count}")
    print(f"  Total Time       : {total_time:.2f} seconds")
    print(f"  Average FPS      : {avg_fps:.2f}")
    print(f"  Total Detections : {detection_count}")
    print(f"  Avg Confidence   : {avg_conf:.4f} ({avg_conf*100:.1f}%)")
    if SAVE_OUTPUT:
        print(f"  Output Video     : {OUTPUT_PATH}")
    if screenshot_count > 0:
        print(f"  Screenshots      : {screenshot_count} saved")
    print("="*60)

    print("\n[VERDICT]")
    if avg_conf >= 0.85:
        print(f"  Model is performing EXCELLENTLY on your video!")
        print(f"  Average confidence of {avg_conf*100:.1f}% is very high.")
        print(f"  Ready to move to Step 2: Perspective Correction.")
    elif avg_conf >= 0.70:
        print(f"  Model is performing WELL on your video.")
        print(f"  Average confidence of {avg_conf*100:.1f}% is acceptable.")
        print(f"  Can proceed to Step 2 but may need more training data.")
    else:
        print(f"  Model confidence is LOW at {avg_conf*100:.1f}%.")
        print(f"  Consider collecting more training data.")


if __name__ == "__main__":
    test_video()