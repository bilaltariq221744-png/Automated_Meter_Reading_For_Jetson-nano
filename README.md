####  Electricity Meter Reading Extraction from Video

### Overview

This project extracts **exact electricity meter readings** from video streams using **YOLOv8-based object detection** and a **robust digit joining pipeline**.  

The challenge was that:

- The meter displays **only 7 digits at a time**.
- Maximum meter reading can be **up to 20 digits**.
- Numbers move from **left to right**, causing some digits to disappear while new digits appear.  

This project solves these challenges and produces **accurate full readings** from moving digits.

###  Problem Statement

Given a video stream of an electricity meter:

1. Identify the **LCD display region (ROI)** in each frame.
2. Detect and classify **digits** within the LCD region.
3. Reconstruct the **full meter reading** despite digits moving and partially disappearing.
4. Handle noise and false detections effectively.

###  Solution

The pipeline consists of **three main stages**:

###  LCD Detection (ROI Extraction)

- Trained a **YOLOv8n model** to detect the **LCD screen** in the video frames.
- The detected ROI is resized to **320×320** and passed to the next stage.

### Digit Detection

- A **second YOLOv8n model** detects and classifies **individual digits** from the LCD crop.
- Detected digits are filtered using a **row-based clustering method** to remove noise:
  - Only digits belonging to the main horizontal row are kept.
  - Digits are sorted **left to right** to maintain proper order.

### Full Reading Reconstruction

- The real challenge: **digits moving and disappearing**.
- Methodology:
  1. Maintain a **buffer of consecutive frames**.
  2. Identify **stable digits** that appear consistently across multiple frames.
  3. Compare **current and previous stable frames** to detect new digits appearing on the right.
  4. **Append new digits** to the full reading while keeping the disappearing left digits intact.
  5. Use a **voting mechanism** across multiple frames to reduce misdetections.

### Key Features

-  **Accurate ROI detection** using YOLOv8n.  
-  **Digit detection and classification** on resized ROI.  
-  **Robust row filtering** to remove false positives.  
-  **Full reading reconstruction** for moving digits up to 20 digits.  
-  Optional **CSV export** for each frame containing frame number and detected reading.  
-  **Video output with overlay** of detected digits and readings.
### Implementation Notes
YOLOv8n models are trained separately for:
 **LCD detection**
 **Digit detection**
**Row filtering** uses clustering of Y-centers of digits to remove extra false positives.
**Full reading reconstruction uses:**
- Frame buffer (e.g., 9 frames)
- Majority voting (e.g., digit must appear in 5 out of 9 frames)
- Comparison of previous and current stable frames to append new digits.
### Results
- Successfully reconstructs the full **meter reading** despite digits moving.
- Reduces misdetections using frame based stability and **majority voting**.
- Produces CSV suitable for downstream analytics or reporting.

