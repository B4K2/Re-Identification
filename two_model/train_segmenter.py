from ultralytics import YOLO
from pathlib import Path
import sys

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_YAML_PATH = SCRIPT_DIR / 'new_segg' / 'data.yaml'

    if not DATA_YAML_PATH.exists():
        print(f"--- FATAL ERROR ---")
        print(f"Dataset configuration file not found at: '{DATA_YAML_PATH}'")
        print("Please make sure you have unzipped the segmentation dataset and the folder name is correct.")
        sys.exit(1)

    model = YOLO('yolov8n-seg.pt') 
    print(f"Using SEGMENTATION dataset config: '{DATA_YAML_PATH}'")
    print("--- Starting to train the Field Segmenter Model ---")
    model.train(
        data=str(DATA_YAML_PATH), 
        epochs=50,
        patience=15,
        batch=4,
        imgsz=640,
        dropout=0.25,
        fliplr=0.5,
        flipud=0.1,
        scale=0.2,
        translate=0.1,
        project="final_training_run",
        name="regularized_model"
    )
    print("\n--- Field Segmenter training complete! ---")
    print("The best model is saved in 'training_runs/field_segmenter/weights/best.pt'")
    print("Copy this file to 'models/field_segmenter.pt'")

if __name__ == '__main__':
    main()