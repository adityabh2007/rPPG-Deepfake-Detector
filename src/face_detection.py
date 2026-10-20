import sys

import cv2
import mediapipe as mp
import numpy as np

FOREHEAD = [67, 109, 10, 338, 297, 334, 296, 336, 9, 107, 66, 105]
LEFT_CHEEK = [117, 118, 119, 100, 126, 209, 49, 129, 203, 205, 50, 187, 123]
RIGHT_CHEEK = [346, 347, 348, 329, 355, 429, 279, 358, 423, 425, 280, 411, 352]

ROIS = {
    "forehead": FOREHEAD,
    "left_cheek": LEFT_CHEEK,
    "right_cheek": RIGHT_CHEEK,
}


class FaceDetector:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,  
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_landmarks(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) 
        result = self.mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        face = result.multi_face_landmarks[0]
        points = np.array([(lm.x * w, lm.y * h) for lm in face.landmark])
        return points.astype(np.int32)

    def get_roi_masks(self, frame_bgr):
        points = self.get_landmarks(frame_bgr)
        if points is None:
            return None

        h, w = frame_bgr.shape[:2]
        masks = {}
        for name, indices in ROIS.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            region = cv2.convexHull(points[indices])
            cv2.fillConvexPoly(mask, region, 255)
            masks[name] = mask
        return masks

    def close(self):
        self.mesh.close()


def draw_rois(frame_bgr, masks):
    overlay = frame_bgr.copy()
    for mask in masks.values():
        overlay[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0)


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    detector = FaceDetector()
    total, found = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        total += 1
        masks = detector.get_roi_masks(frame)

        if masks is None:
            cv2.putText(frame, "No face", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            shown = frame
        else:
            found += 1
            shown = draw_rois(frame, masks)

        cv2.imshow("Face ROIs (press q to quit)", shown)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    if total:
        print(f"Face found in {found}/{total} frames ({100 * found / total:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/face_detection.py path/to/video.mp4")
    else:
        main(sys.argv[1])
