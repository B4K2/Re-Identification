

# Player Re-Identification in Sports Footage

## Project Overview
This project tackles the real-world computer vision challenge of player re-identification in sports analytics, as outlined in the assignment by Liat.ai. The primary goal is to maintain a consistent ID for each player, even when they are occluded or leave and re-enter the camera's view.

---

## Final Approach & Methodology

After exploring several methodologies, the final implementation uses a powerful, state-of-the-art **two-model approach** to achieve highly robust player tracking through **Field Registration**. This technique creates a virtual birds-eye view of the pitch by understanding the field's geometry, allowing for near-perfect re-identification.

The pipeline works as follows:

1.  **Two Specialized AI Models:** Instead of a single, generic model, this solution uses two "expert" models that were fine-tuned for specific tasks:
    *   **Player Detector :** A YOLOv8 detection model fine-tuned exclusively on a custom dataset to be an expert at locating players, goalkeepers, and referees with high accuracy.
    *   **Field Segmenter :** A YOLOv8 segmentation model (`yolov8n-seg.pt`) fine-tuned on a custom dataset of line and area masks. Its only job is to precisely identify the geometry of the pitch, such as the penalty area.

2.  **On-the-Fly Homography:** In every frame of the video:
    *   The **Field Segmenter** detects the visible penalty area.
    *   The corners of this detected area are extracted.
    *   These image points are matched to their known, real-world coordinates on a standard football pitch.
    *   A **Homography Matrix** is calculated in real-time. This matrix acts as a "translator" that can convert any pixel coordinate from the video frame into its corresponding (X, Y) coordinate on a 2D map.

3.  **Virtual Birds-Eye View:**
    *   The **Player Detector** finds all players in the frame.
    *   The bottom-center point of each player's bounding box (approximating their feet) is projected onto the 2D map using the homography matrix.

4.  **Robust Tracking:**
    *   Player IDs are maintained by tracking the stable (X, Y) coordinates on the 2D virtual map. This is incredibly robust to occlusions and camera motion. For this implementation, the powerful **ByteTrack** algorithm is used on the initial video to provide stable base IDs that are then projected.

5.  **Visualization:**
    *   The final output video shows a side-by-side view of the original footage with tracked players and the dynamic, virtual birds-eye map showing their real-time positions.

---

## How to Set Up and Run the Code

### 1. Dependencies

This project uses Python 3.10+ and the following libraries. You can install them using the provided `uv.lock` or `pyproject.toml` with `uv`, or manually via pip:
```bash
pip install ultralytics opencv-python numpy tqdm
```

### 2. Project Structure

Ensure your project folder (`two_model`) has the following structure. All required models and video files are included.

```
two_model/
|
|final_training_run/
│   └── regularized_model/
│       └── weights/
│           ├── best.pt               # Final Model
│           └── last.pt               
│
|final_training_runs/
│   └── detector_strong_generalizer/
│       └── weights/
│           ├── best.pt               # Final Model
│           └── last.pt 
|
|15sec_input_720p.mp4 
|
└── virtual_tracker.py       # The main script to run the final application
```

### 3. Running the Application

All logic is contained within a single, powerful script.

1.  Navigate to the `two_model` directory in your terminal.
2.  Run the main script:
    ```bash
    python virtual_tracker.py
    ```
3.  The script will start processing the video. A window will pop up showing the live output, and a progress bar will be displayed in the terminal.
4.  Upon completion, the final annotated video will be saved as `final_virtual_tracker_output.mp4` in the `two_model` folder.

---

## Brief Report

### 1. Techniques Tried and Their Outcomes

The final solution was reached after a journey of iterative development and debugging, exploring multiple state-of-the-art techniques:

*   **Initial Approach (Basic IOU Tracker):** The first attempt used a simple IOU-based tracker.
    *   **Outcome:** Failed significantly. It produced a large number of "ghost boxes" and suffered from constant ID switching, especially during occlusions. This proved that a simple motion-based tracker is insufficient for this complex task.

*   **Advanced Tracker (DeepSORT/ByteTrack):** The next step was to replace the custom tracker with a professional-grade algorithm like ByteTrack.
    *   **Outcome:** Huge improvement. Ghosting was eliminated, and ID stability was much higher. However, it still struggled when players were occluded for long periods or left and re-entered the frame. This showed that even a great tracker is limited by the quality of the detections.

*   **Human-in-the-Loop Fine-Tuning:** An interactive tool was developed to allow for manual correction of the base model's failures. This data was used to fine-tune the detector.
    *   **Outcome:** This was a critical turning point. The fine-tuned detector was significantly more robust. However, it revealed that a single model struggled to learn both player detection and complex field geometry simultaneously.

*   **Final Approach (Two-Model Field Registration):** The final, successful approach was to treat this as a system of two "expert" models working together. This provided the best of both worlds: an expert player detector and an expert field segmenter, which together enabled the highly robust virtual map projection.

### 2. Challenges Encountered

*   **Overfitting:** Initial attempts at fine-tuning a single model on a small, non-diverse dataset led to severe overfitting, where the model performed well only on frames it had already seen. This was solved by creating more diverse datasets and using strong regularization techniques during training.
*   **Tracker Logic:** Developing a custom tracker that is robust to real-world challenges like occlusions and camera motion is non-trivial. The limitations of simple trackers led to the adoption of the more advanced Field Registration technique.
*   **Homography Stability:** Calculating a stable homography requires consistently detecting at least 4 landmarks. The final model, which focuses on detecting large, stable areas like the `penalty_area`, proved to be the most reliable solution for this.

### 3. Incomplete Aspects & Future Work

While the current solution is highly robust, it could be further improved with more time and resources:

*   **More Diverse Landmark Detection:** The current field model is trained primarily on the `penalty_area`. Training it on more landmark types (corners, center circle, etc.) would make the homography calculation even more resilient to extreme camera angles where the penalty box is not visible.
*   **Advanced 2D Map Tracking:** The tracking on the 2D map is currently simple. Implementing a Kalman Filter on the (X, Y) map coordinates would allow for smoother trajectories and prediction of player movement even if the detector briefly fails.
