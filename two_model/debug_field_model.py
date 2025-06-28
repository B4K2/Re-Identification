import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

VIDEO_PATH = "15sec_input_720p.mp4" 
MODEL_PATH = 'training_runs/field_segmenter3/weights/best.pt'

def main():
    print("--- Model Debugger ---")
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print("Model loaded successfully. Class names are:", class_names)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ret, frame = cap.read()
    if not ret:
        print("Could not read a frame from the video.")
        cap.release()
        return

    print("\nRunning detection on the first frame...")
    results = model(frame, conf=0.2, verbose=True)

    print("\n--- DETECTION & SEGMENTATION RESULTS ---")
    found_something = False
    r = results[0]
    if r.boxes is not None:
        for box in r.boxes:
            found_something = True
            class_id = int(box.cls[0])
            class_name = class_names.get(class_id, "Unknown")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"✅ [Box] Found '{class_name}' (ID: {class_id}) with confidence: {conf:.2f}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if r.masks is not None:
        for i, mask in enumerate(r.masks):
            found_something = True
            class_id = int(r.boxes[i].cls[0])
            class_name = class_names.get(class_id, "Unknown")
            conf = float(r.boxes[i].conf[0])
            print(f"✅ [Mask] Found '{class_name}' (ID: {class_id}) with confidence: {conf:.2f}")
            contour = mask.xy[0].astype(int).reshape(-1, 1, 2)
            cv2.polylines(frame, [contour], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(frame, f"{class_name} ({conf:.2f})", (contour[0][0][0], contour[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if not found_something:
        print("\n❌ The model did not detect ANY objects or masks with confidence > 0.2.")
        print("This strongly suggests an issue with the trained model file or a mismatch with the input video.")

    output_image_path = SCRIPT_DIR / "field_model_debug_output.jpg"
    cv2.imwrite(str(output_image_path), frame)
    print(f"\n--- DONE ---")
    print(f"An annotated debug image has been saved to: {output_image_path}")
    cv2.imshow("Field Model Debug Output", frame)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()