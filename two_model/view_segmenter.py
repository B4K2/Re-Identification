import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path
from tqdm import tqdm
import random

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = "15sec_input_720p.mp4"
FIELD_MODEL_PATH = "final_training_run/regularized_model/weights/best.pt"
OUTPUT_VIDEO_PATH = str(SCRIPT_DIR / 'field_segmentation_output.mp4')

CONF_THRESHOLD = 0.4

def main():
    if not Path(FIELD_MODEL_PATH).exists(): 
        print(f"ERROR: Field model not found at {FIELD_MODEL_PATH}"); return
    if not Path(VIDEO_PATH).exists(): 
        print(f"ERROR: Video file not found at {VIDEO_PATH}"); return

    model = YOLO(FIELD_MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)); 
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("Visualizing Field Segmentation... Press 'q' to quit.")
    results_generator = model(source=VIDEO_PATH, conf=CONF_THRESHOLD, stream=True, verbose=False)

    for r in tqdm(results_generator, total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        frame = r.orig_img
        overlay = frame.copy()
        if r.masks is not None:
            for i, mask in enumerate(r.masks):
                class_id = int(r.boxes[i].cls[0])
                class_name = model.names.get(class_id, "Unknown")
                random.seed(class_id)
                color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
                contour = mask.xy[0].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [contour], isClosed=True, color=color, thickness=3)
                cv2.fillPoly(overlay, [contour], color=color)

        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        out.write(frame)
        cv2.imshow('Field Segmenter Viewer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release(); out.release(); cv2.destroyAllWindows()
    print(f"\nField segmentation visualization complete! Saved to:\n{OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()