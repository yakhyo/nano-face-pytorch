import cv2
import numpy as np


def draw_detections(original_image, detections, vis_threshold):
    """
    Draws bounding boxes and landmarks on the image based on multiple detections.

    Args:
        original_image (ndarray): The image on which to draw detections.
        detections (ndarray): Array of detected bounding boxes and landmarks.
        vis_threshold (float): The confidence threshold for displaying detections.
    """

    # Colors for visualization
    LANDMARK_COLORS = [
        (0, 0, 255),    # Right eye (Red)
        (0, 255, 255),  # Left eye (Yellow)
        (255, 0, 255),  # Nose (Magenta)
        (0, 255, 0),    # Right mouth (Green)
        (255, 0, 0)     # Left mouth (Blue)
    ]
    BOX_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)

    # Filter by confidence
    detections = detections[detections[:, 4] >= vis_threshold]

    print(f"#faces: {len(detections)}")

    # Slice arrays efficiently
    boxes = detections[:, 0:4].astype(np.int32)
    scores = detections[:, 4]
    landmarks = detections[:, 5:15].reshape(-1, 5, 2).astype(np.int32)

    for box, score, landmark in zip(boxes, scores, landmarks):
        # Draw bounding box
        cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), BOX_COLOR, 2)

        # Draw confidence score
        text = f"{score:.2f}"
        cx, cy = box[0], box[1] + 12
        cv2.putText(original_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_COLOR)

        # Draw landmarks
        for point, color in zip(landmark, LANDMARK_COLORS):
            cv2.circle(original_image, point, 1, color, 4)
