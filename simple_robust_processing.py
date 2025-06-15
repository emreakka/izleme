import cv2
import numpy as np
from typing import List, Dict, Tuple
from robust_face_detector import RobustFaceDetector

def simple_robust_process_frame(frame: np.ndarray, robust_detector: RobustFaceDetector, 
                               confidence_threshold: float = 0.3) -> Tuple[np.ndarray, List[Dict]]:
    """
    Simple robust frame processing that detects all faces with gaze directions
    """
    try:
        # Use robust face detection
        detections = robust_detector.detect_all_faces(frame)
        
        # Draw detections on frame
        processed_frame = robust_detector.draw_detections(frame, detections)
        
        # Convert to standard format for storage
        results = []
        for detection in detections:
            result = {
                'face_id': detection['face_id'],
                'person_name': detection['person_name'],
                'gaze_direction': detection['gaze_direction'],
                'gaze_pitch': 0.0,
                'gaze_yaw': 0.0,
                'gaze_confidence': detection['gaze_confidence'],
                'head_pose': (0.0, 0.0, 0.0),
                'emotion': detection['emotion'],
                'emotion_confidence': detection['emotion_confidence'],
                'emotion_scores': {},
                'is_known_person': detection['is_known_person'],
                'face_similarity': 0.0,
                'encounter_count': 1,
                'processing_time_ms': detection['processing_time_ms']
            }
            results.append(result)
        
        return processed_frame, results
        
    except Exception as e:
        print(f"Error in simple_robust_process_frame: {e}")
        return frame, []