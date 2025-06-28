import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = "15sec_input_720p.mp4"
PLAYER_MODEL_PATH = "final_training_box/regularized_model2/weights/best.pt"
FIELD_MODEL_PATH = "final_training_run/regularized_model/weights/best.pt"
OUTPUT_VIDEO_PATH = SCRIPT_DIR / 'final_virtual_tracker.mp4'

CONF_THRESHOLD = 0.4

PITCH_LENGTH = 105; PITCH_WIDTH = 68
FIELD_CLASS_MAP = {0: "goalpost", 1: "midline", 2: "boundary_line", 3: "penalty_area"}
REAL_WORLD_CORNERS = {
    'left_penalty_box': np.array([
        [0, (PITCH_WIDTH/2) - 20.15], [16.5, (PITCH_WIDTH/2) - 20.15],
        [16.5, (PITCH_WIDTH/2) + 20.15], [0, (PITCH_WIDTH/2) + 20.15]
    ], dtype=np.float32),
    'right_penalty_box': np.array([
        [PITCH_LENGTH - 16.5, (PITCH_WIDTH/2) - 20.15], [PITCH_LENGTH - 16.5, (PITCH_WIDTH/2) + 20.15],
        [PITCH_LENGTH, (PITCH_WIDTH/2) + 20.15], [PITCH_LENGTH, (PITCH_WIDTH/2) - 20.15]
    ], dtype=np.float32)
}

def get_bottom_center(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))

def draw_pitch_map(height=720, width=1155):
    pitch_map = np.full((height, width, 3), (50, 200, 50), dtype=np.uint8)
    px_per_meter_w, px_per_meter_h = width / PITCH_LENGTH, height / PITCH_WIDTH
    cv2.rectangle(pitch_map, (0, 0), (width - 1, height - 1), (255, 255, 255), 2)
    cv2.line(pitch_map, (width // 2, 0), (width // 2, height - 1), (255, 255, 255), 2)
    cv2.circle(pitch_map, (width // 2, height // 2), int(9.15 * px_per_meter_w), (255, 255, 255), 2)
    left_box_tl = (0, int((height/2) - 20.15*px_per_meter_h)); left_box_br = (int(16.5*px_per_meter_w), int((height/2) + 20.15*px_per_meter_h))
    right_box_tl = (width - int(16.5*px_per_meter_w), int((height/2) - 20.15*px_per_meter_h)); right_box_br = (width-1, int((height/2) + 20.15*px_per_meter_h))
    cv2.rectangle(pitch_map, left_box_tl, left_box_br, (255, 255, 255), 2)
    cv2.rectangle(pitch_map, right_box_tl, right_box_br, (255, 255, 255), 2)
    return pitch_map

def find_corners_from_mask(mask):
    if mask is None or mask.size == 0: return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)
    if len(approx_corners) == 4: return np.squeeze(approx_corners)
    return []

def main():
    player_model = YOLO(PLAYER_MODEL_PATH); field_model = YOLO(FIELD_MODEL_PATH)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)); map_h, map_w = 720, 1155
    out = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width + map_w, max(height, map_h)))
    last_known_homography = None

    player_tracking_results_generator = player_model.track(source=VIDEO_PATH, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD, stream=True, verbose=False)

    print("Starting final virtual tracker...")
    for r in tqdm(player_tracking_results_generator, total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        frame = r.orig_img.copy()
        overlay = frame.copy()
        field_results = field_model(frame, verbose=False)
        image_points = []; world_points_key = None
        if field_results[0].masks is not None:
            for i, box in enumerate(field_results[0].boxes):
                if box.conf[0] > 0.6:
                    class_name = field_model.names.get(int(box.cls[0]))
                    if class_name == "penalty_area":
                        mask = field_results[0].masks.data[i].cpu().numpy().astype(np.uint8)
                        corners = find_corners_from_mask(mask)
                        if len(corners) == 4:
                            cv2.fillPoly(overlay, [corners.astype(np.int32)], color=(255, 0, 0))
                            centroid_x = np.mean(corners[:, 0])
                            world_points_key = 'left_penalty_box' if centroid_x < width / 2 else 'right_penalty_box'
                            image_points = corners; break

        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        if len(image_points) == 4 and world_points_key:
            image_points = sorted(image_points, key=lambda p: p[1])
            top_points = sorted(image_points[:2], key=lambda p: p[0])
            bottom_points = sorted(image_points[2:], key=lambda p: p[0])
            sorted_image_points = np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype=np.float32)
            H, _ = cv2.findHomography(sorted_image_points, REAL_WORLD_CORNERS[world_points_key])
            if H is not None: last_known_homography = H

        pitch_map = draw_pitch_map(map_h, map_w)
        if last_known_homography is not None and r.boxes.id is not None:
            for box in r.boxes:
                track_id = int(box.id[0]); player_bbox = box.xyxy[0].cpu().numpy()
                player_point = np.array([get_bottom_center(player_bbox)], dtype=np.float32).reshape(-1, 1, 2)
                map_point = cv2.perspectiveTransform(player_point, last_known_homography)
                if map_point is not None:
                    map_pos = map_point.reshape(-1, 2)[0]
                    map_x = int(map_pos[0] * (map_w / PITCH_LENGTH)); map_y = int(map_pos[1] * (map_h / PITCH_WIDTH))
                    if 0 <= map_x < map_w and 0 <= map_y < map_h:
                        cv2.circle(pitch_map, (map_x, map_y), 10, (0, 0, 255), -1)
                        cv2.putText(pitch_map, str(track_id), (map_x + 15, map_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                x1, y1, x2, y2 = map(int, player_bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        frame_resized = cv2.resize(frame, (width, height)); map_resized = cv2.resize(pitch_map, (map_w, map_h))
        final_h = max(height, map_h)
        frame_resized = cv2.copyMakeBorder(frame_resized, 0, final_h - height, 0, 0, cv2.BORDER_CONSTANT)
        map_resized = cv2.copyMakeBorder(map_resized, 0, final_h - map_h, 0, 0, cv2.BORDER_CONSTANT)
        combined_frame = np.hstack((frame_resized, map_resized))
        out.write(combined_frame)
        cv2.imshow('Virtual Tracker', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release(); out.release(); cv2.destroyAllWindows()
    print(f"\nProcessing complete! Final video saved to:\n{OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()