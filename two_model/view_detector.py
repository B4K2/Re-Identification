import cv2
from ultralytics import YOLO
import sys
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = "15sec_input_720p.mp4"
PLAYER_MODEL_PATH = "final_training_runs/detector_strong_generalization/weights/best.pt"
OUTPUT_VIDEO_PATH = str(SCRIPT_DIR / 'player_tracking_output.mp4')

CONF_THRESHOLD = 0.3

def main():
    if not Path(PLAYER_MODEL_PATH).exists(): 
        print(f"ERROR: Player model not found at {PLAYER_MODEL_PATH}"); return
    if not Path(VIDEO_PATH).exists(): 
        print(f"ERROR: Video file not found at {VIDEO_PATH}"); return

    model = YOLO(PLAYER_MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)); 
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("Visualizing Player Tracking with ByteTrack... Press 'q' to quit.")
    tracking_results = model.track(
        source=VIDEO_PATH, 
        conf=CONF_THRESHOLD, 
        tracker="bytetrack.yaml", 
        persist=True, 
        stream=True, 
        verbose=False
    )

    for r in tqdm(tracking_results, total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        frame = r.orig_img
        if r.boxes.id is not None:
            for box in r.boxes:
                track_id = int(box.id[0])
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if class_id == 0: label, color = f"Player: {track_id}", (0, 255, 0)
                elif class_id == 1: label, color = f"Ball: {track_id}", (0, 0, 255)
                elif class_id == 2: label, color = f"Referee: {track_id}", (0, 255, 255)
                elif class_id == 3: label, color = f"GK: {track_id}", (255, 255, 0)
                else: continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)
        cv2.imshow('Player Detector Viewer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release(); out.release(); cv2.destroyAllWindows()
    print(f"\nPlayer tracking visualization complete! Saved to:\n{OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()