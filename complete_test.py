import cv2
import numpy as np
from ultralytics import YOLO



VIDEO_PATH = r"C:/Users/PMLS/Desktop/Meter_reading_indonesia/WhatsApp Video 2025-08-14 at 2.56.45 AM.mp4"
LCD_MODEL_PATH = r"C:/Users/PMLS/Desktop/Meter_reading_indonesia/LCD_Detection_model/weights/best.pt"
DIGIT_MODEL_PATH = r"C:/Users/PMLS/Desktop/Meter_reading_indonesia/Number_detection_model/weights/best.pt"

OUTPUT_VIDEO = r"C:/Users/PMLS/Desktop/Meter_reading_indonesia/output_video.mp4"

IMG_SIZE = 320
MAX_DIGITS = 7



OVERLAP = 6
STABLE_COUNT_THRESHOLD = 3

last_reading = None
current_streak = 0

unique_readings = []
FINAL_READING = ""



lcd_model = YOLO(LCD_MODEL_PATH)
digit_model = YOLO(DIGIT_MODEL_PATH)



cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))



def keep_main_row(digits):
    if len(digits) <= 2:
        return digits

    ys = np.array([d[2] for d in digits])
    median_y = np.median(ys)
    spread = max(ys) - min(ys)

    if spread < 10:
        return digits

    filtered = [d for d in digits if abs(d[2] - median_y) < 0.3 * spread]

    if len(filtered) < 3:
        return digits

    return filtered



while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    lcd_results = lcd_model(frame, conf=0.5)

    for box in lcd_results[0].boxes.xyxy:

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)

        lcd_crop = frame[y1:y2, x1:x2]

        if lcd_crop.size == 0:
            continue

        h_crop, w_crop = lcd_crop.shape[:2]
        lcd_crop_resized = cv2.resize(lcd_crop, (IMG_SIZE, IMG_SIZE))

       
        digit_results = digit_model(lcd_crop_resized, conf=0.4)

        raw_digits = []

        boxes = digit_results[0].boxes.xyxy
        classes = digit_results[0].boxes.cls

        for box_d, cls in zip(boxes, classes):

            cls = int(cls)
            if cls == 0:
                continue

            digit = cls - 1

            x1_d, y1_d, x2_d, y2_d = map(int, box_d)

            x_scale = w_crop / IMG_SIZE
            y_scale = h_crop / IMG_SIZE

            x1_d = int(x1_d * x_scale)
            x2_d = int(x2_d * x_scale)
            y1_d = int(y1_d * y_scale)
            y2_d = int(y2_d * y_scale)

            x1_g = x1 + x1_d
            y1_g = y1 + y1_d
            x2_g = x1 + x2_d
            y2_g = y1 + y2_d

            x_center = (x1_g + x2_g) // 2
            y_center = (y1_g + y2_g) // 2

            raw_digits.append((x_center, str(digit), y_center,
                               (x1_g, y1_g, x2_g, y2_g)))


        filtered_digits = keep_main_row(raw_digits)
        filtered_digits = sorted(filtered_digits, key=lambda x: x[0])

        if len(filtered_digits) > MAX_DIGITS:
            filtered_digits = filtered_digits[-MAX_DIGITS:]

 

        reading = ""

        for d in filtered_digits:
            x_center, digit, y_center, box_coords = d
            x1_g, y1_g, x2_g, y2_g = box_coords

            cv2.rectangle(display_frame, (x1_g, y1_g), (x2_g, y2_g), (255,0,0), 2)

            cv2.putText(display_frame,
                        f"{digit}",
                        (x1_g, y1_g - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,0,0),
                        2)

            reading += digit



        if reading != "" and len(reading) >= 5:

            if reading == last_reading:
                current_streak += 1
            else:
                current_streak = 1
                last_reading = reading

            # Accept only stable readings
            if current_streak == STABLE_COUNT_THRESHOLD:

                stable_reading = reading

                if len(unique_readings) == 0 or unique_readings[-1] != stable_reading:

                    if len(unique_readings) == 0:
                        unique_readings.append(stable_reading)
                        FINAL_READING = stable_reading

                    else:
                        prev = unique_readings[-1]
                        curr = stable_reading

                        matched = False

                        # Strict overlap
                        if len(prev) >= OVERLAP and len(curr) >= OVERLAP:
                            if prev[-OVERLAP:] == curr[:OVERLAP]:
                                FINAL_READING += curr[-1]
                                unique_readings.append(curr)
                                matched = True

                        # Partial match fallback
                        if not matched:
                            for k in range(OVERLAP-1, 2, -1):
                                if prev[-k:] == curr[:k]:
                                    FINAL_READING += curr[k:]
                                    unique_readings.append(curr)
                                    matched = True
                                    break

                        if not matched:
                            print(f"⚠️ Skipped: {prev} -> {curr}")

      

        cv2.putText(display_frame,
                    f"Live: {reading}",
                    (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,0),
                    2)

        cv2.putText(display_frame,
                    f"Final: {FINAL_READING}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

    
    cv2.imshow("Meter Reading", display_frame)
    out.write(display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break



cap.release()
out.release()
cv2.destroyAllWindows()

print(" FINAL METER READING:", FINAL_READING)