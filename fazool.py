import cv2
import os
import argparse

def extract_frames(video_path, output_folder, frame_interval=10):
    """
    Extract every Nth frame from a video and save as images.
    
    Args:
        video_path: Path to the input video file
        output_folder: Folder to save extracted frames
        frame_interval: Extract every Nth frame (default: 10)
    """
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration    = total_frames / fps if fps > 0 else 0
    
    print("=" * 50)
    print("         VIDEO FRAME EXTRACTOR")
    print("=" * 50)
    print(f"  Video Path    : {video_path}")
    print(f"  Resolution    : {width} x {height}")
    print(f"  FPS           : {fps:.2f}")
    print(f"  Total Frames  : {total_frames}")
    print(f"  Duration      : {duration:.2f} seconds")
    print(f"  Extract Every : {frame_interval}th frame")
    print(f"  Output Folder : {output_folder}")
    print(f"  Estimated Frames to Extract: ~{total_frames // frame_interval}")
    print("=" * 50)
    
    frame_count   = 0  # total frames read
    saved_count   = 0  # total frames saved
    
    print("\n[INFO] Starting extraction...\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save every Nth frame
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            
            # Progress update every 50 saved frames
            if saved_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  [Progress] {progress:.1f}% | Frames Saved: {saved_count}")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n[DONE] Extraction Complete!")
    print(f"  Total Frames Processed : {frame_count}")
    print(f"  Total Frames Saved     : {saved_count}")
    print(f"  Saved to               : {os.path.abspath(output_folder)}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract every Nth frame from a video")
    
    parser.add_argument(
        "--video",
        type=str,
        required=False,
        default=r"C:\Users\PMLS\Desktop\Meter_project\WhatsApp Video 2025-08-14 at 2.56.45 AM.mp4",
        help="Path to the input video file (e.g. /home/user/meter.mp4). Defaults to the provided Meter project video."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="extracted_frames",
        help="Output folder to save frames (default: extracted_frames)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Extract every Nth frame (default: 10)"
    )
    
    args = parser.parse_args()
    
    extract_frames(
        video_path=args.video,
        output_folder=args.output,
        frame_interval=args.interval
    )