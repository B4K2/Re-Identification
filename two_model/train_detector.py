from ultralytics import YOLO
from pathlib import Path
import sys

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_YAML_PATH = SCRIPT_DIR / 'new_box (1)' / 'data.yaml'

    if not DATA_YAML_PATH.exists():
        print(f"--- FATAL ERROR: Dataset configuration not found at '{DATA_YAML_PATH}' ---")
        sys.exit(1)

    model = YOLO('yolov8n.pt') 
    print(f"Using DETECTION dataset: '{DATA_YAML_PATH}'")
    print("--- Starting training with STRONG GENERALIZATION settings ---")
    model.train(
        data=str(DATA_YAML_PATH), 
        epochs=120,
        patience=25,
        batch=8,
        imgsz=640,
        dropout=0.35,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        perspective=0.001,
        flipud=0.2,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=15,
        project="final_training_runs",
        name="detector_strong_generalization"
    )
    print("\n--- Player Detector training complete! ---")
    print("The best model is saved in 'final_training_runs/detector_strong_generalization/weights/best.pt'")
    print("This model should be much more robust and less overfit.")

if __name__ == '__main__':
    main()