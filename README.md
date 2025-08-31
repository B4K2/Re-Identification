# Player Re-Identification with Virtual Birds-Eye View

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-0059C1?style=for-the-badge&logo=yolo)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)
![Project Status](https://img.shields.io/badge/Project_Status-Complete-green?style=for-the-badge)

<br>

![Project Demo GIF](https://raw.githubusercontent.com/B4K2/Re-Identification/main/Untitled%20video.gif)

This project tackles the advanced computer vision challenge of player re-identification in sports footage. The system maintains consistent IDs for each player, even through occlusion or when re-entering the frame, by projecting their positions onto a virtual 2D birds-eye view of the field.

---

## 📋 Table of Contents

- [Key Features](#key-features)
- [Final Methodology](#final-methodology)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Development Journey & Report](#development-journey--report)
  - [Techniques Explored](#1-techniques-explored)
  - [Challenges Encountered](#2-challenges-encountered)
  - [Future Work](#3-future-work)

---

## ✨ Key Features

*   **Dual-Model Architecture:** Utilizes two specialized YOLOv8 models—one for high-accuracy player detection and another for field geometry segmentation.
*   **Real-time Field Registration:** Calculates a homography matrix on-the-fly in every frame to map player positions from the 2D video to a virtual 2D overhead map.
*   **Virtual Birds-Eye View:** Generates a dynamic, top-down tactical map showing player positions in real-time.
*   **Robust ID Tracking:** Achieves near-perfect player re-identification by tracking stable (X, Y) coordinates on the virtual map, making it resilient to camera motion and player occlusions.

---

## 🛠️ Final Methodology

The final implementation uses a powerful, state-of-the-art **two-model approach** to achieve highly robust player tracking through **Field Registration**. This technique creates a virtual birds-eye view of the pitch by understanding the field's geometry, allowing for near-perfect re-identification.

The pipeline works as follows:

1.  **Two Specialized AI Models:** Instead of a single, generic model, this solution uses two "expert" models fine-tuned for specific tasks:
    *   **Player Detector:** A YOLOv8 model fine-tuned exclusively to be an expert at locating players, goalkeepers, and referees.
    *   **Field Segmenter:** A YOLOv8 segmentation model fine-tuned to precisely identify the geometry of the pitch, such as the penalty area.

2.  **On-the-Fly Homography:** In every frame of the video:
    *   The **Field Segmenter** detects the visible penalty area and extracts its corners.
    *   These image points are matched to their known, real-world coordinates on a standard football pitch.
    *   A **Homography Matrix** is calculated in real-time. This matrix acts as a "translator" that can convert any pixel coordinate from the video frame into its corresponding (X, Y) coordinate on a 2D map.

3.  **Virtual Birds-Eye View Projection:**
    *   The **Player Detector** finds all players in the frame.
    *   The bottom-center point of each player's bounding box is projected onto the 2D map using the homography matrix.

4.  **Robust Tracking:**
    *   Player IDs are maintained by tracking the stable (X, Y) coordinates on the 2D virtual map. This is incredibly robust to occlusions and camera motion. The powerful **ByteTrack** algorithm is used on the initial video to provide stable base IDs that are then projected.

5.  **Visualization:**
    *   The final output video shows a side-by-side view of the original footage with tracked players and the dynamic, virtual birds-eye map showing their real-time positions.

---

## ⚙️ Setup and Installation

### 1. Dependencies

This project uses Python 3.10+. You can install the required libraries via pip:
```bash
pip install ultralytics opencv-python numpy tqdm
```

### 2. Project Structure

Ensure your project folder (`two_model`) has the following structure. All required models and video files are included.

```
two_model/
|
├── final_training_run/
│   └── regularized_model/
│       └── weights/
│           ├── best.pt               # Field Segmenter Model
│           └── last.pt               
│
├── final_training_runs/
│   └── detector_strong_generalizer/
│       └── weights/
│           ├── best.pt               # Player Detector Model
│           └── last.pt 
|
├── 15sec_input_720p.mp4 
|
└── virtual_tracker.py                # Main script
```


---

## ▶️ How to Run

All logic is contained within a single, powerful script.

1.  Navigate to the `two_model` directory in your terminal.
2.  Run the main script:
    ```bash
    python virtual_tracker.py
    ```
3.  The script will start processing the video. A window will pop up showing the live output, and a progress bar will be displayed in the terminal.
4.  Upon completion, the final annotated video will be saved as `final_virtual_tracker_output.mp4` in the `two_model` folder.

---

## 🔬 Development Journey & Report

### 1. Techniques Explored

The final solution was reached after iterative development, exploring multiple state-of-the-art techniques:

*   **Initial Approach (IOU Tracker):** The first attempt used a simple IOU-based tracker.
    *   **Outcome:** Failed significantly. It produced a large number of "ghost boxes" and suffered from constant ID switching.

*   **Advanced Tracker (ByteTrack):** The next step was to use a professional-grade algorithm like ByteTrack.
    *   **Outcome:** Huge improvement. Ghosting was eliminated, and ID stability was much higher, but it still struggled with long occlusions.

*   **Human-in-the-Loop Fine-Tuning:** An interactive tool was developed to manually correct the base model's failures, and this data was used to fine-tune the detector.
    *   **Outcome:** A critical turning point. The fine-tuned detector was significantly more robust but revealed that a single model struggled to learn both player detection and field geometry simultaneously.

*   **Final Approach (Two-Model Field Registration):** The final, successful approach was to treat this as a system of two "expert" models working together. This provided an expert player detector and an expert field segmenter, enabling the highly robust virtual map projection.

### 2. Challenges Encountered

*   **Overfitting:** Initial fine-tuning on small datasets led to overfitting. This was solved by creating more diverse datasets and using strong regularization during training.
*   **Tracker Logic:** The limitations of simple trackers led to the adoption of the more advanced Field Registration technique.
*   **Homography Stability:** Calculating a stable homography requires consistently detecting at least 4 landmarks. The final model, which focuses on detecting large areas like the `penalty_area`, proved to be the most reliable solution.

### 3. Future Work

*   **More Diverse Landmark Detection:** Train the field model on more landmark types (corners, center circle) to make the homography calculation even more resilient.
*   **Advanced 2D Map Tracking:** Implement a Kalman Filter on the (X, Y) map coordinates to allow for smoother trajectories and prediction of player movement.
