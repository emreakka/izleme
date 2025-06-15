import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, List, Dict

class GazeDetector:
    """Gaze direction detector using MediaPipe face landmarks"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face mesh model
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Iris landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
    
    def detect_gaze(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect gaze direction from image
        
        Args:
            image: Input image in BGR format
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detection results for each face
        """
        results = []
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        face_results = self.face_mesh.process(rgb_image)
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                gaze_info = self._analyze_gaze(face_landmarks, image.shape)
                
                if gaze_info['confidence'] >= confidence_threshold:
                    results.append(gaze_info)
        
        return results
    
    def _analyze_gaze(self, landmarks, image_shape: Tuple[int, int, int]) -> Dict:
        """Analyze gaze direction from face landmarks"""
        h, w = image_shape[:2]
        
        # Convert landmarks to pixel coordinates
        landmarks_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_points.append([x, y])
        
        landmarks_array = np.array(landmarks_points)
        
        # Get eye regions
        left_eye_points = landmarks_array[self.LEFT_EYE_LANDMARKS]
        right_eye_points = landmarks_array[self.RIGHT_EYE_LANDMARKS]
        
        # Get iris centers if available
        left_iris_center = None
        right_iris_center = None
        
        try:
            if len(landmarks_array) > max(self.LEFT_IRIS):
                left_iris_points = landmarks_array[self.LEFT_IRIS]
                left_iris_center = np.mean(left_iris_points, axis=0)
            
            if len(landmarks_array) > max(self.RIGHT_IRIS):
                right_iris_points = landmarks_array[self.RIGHT_IRIS]
                right_iris_center = np.mean(right_iris_points, axis=0)
        except:
            pass
        
        # Calculate gaze direction
        gaze_direction, gaze_angles, confidence = self._calculate_gaze_direction(
            left_eye_points, right_eye_points, left_iris_center, right_iris_center
        )
        
        return {
            'gaze_direction': gaze_direction,
            'gaze_angles': gaze_angles,
            'confidence': confidence,
            'left_eye_points': left_eye_points,
            'right_eye_points': right_eye_points,
            'left_iris_center': left_iris_center,
            'right_iris_center': right_iris_center,
            'landmarks': landmarks_array
        }
    
    def _calculate_gaze_direction(self, left_eye: np.ndarray, right_eye: np.ndarray, 
                                left_iris: Optional[np.ndarray], right_iris: Optional[np.ndarray]) -> Tuple[str, Tuple[float, float], float]:
        """Calculate gaze direction from eye landmarks"""
        
        # Calculate eye centers
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        
        # Use iris centers if available, otherwise use eye centers
        if left_iris is not None and right_iris is not None:
            # Calculate relative position of iris in eye
            left_eye_bbox = [np.min(left_eye, axis=0), np.max(left_eye, axis=0)]
            right_eye_bbox = [np.min(right_eye, axis=0), np.max(right_eye, axis=0)]
            
            # Normalize iris position relative to eye
            left_rel_x = (left_iris[0] - left_eye_bbox[0][0]) / (left_eye_bbox[1][0] - left_eye_bbox[0][0])
            left_rel_y = (left_iris[1] - left_eye_bbox[0][1]) / (left_eye_bbox[1][1] - left_eye_bbox[0][1])
            
            right_rel_x = (right_iris[0] - right_eye_bbox[0][0]) / (right_eye_bbox[1][0] - right_eye_bbox[0][0])
            right_rel_y = (right_iris[1] - right_eye_bbox[0][1]) / (right_eye_bbox[1][1] - right_eye_bbox[0][1])
            
            # Average the relative positions
            avg_rel_x = (left_rel_x + right_rel_x) / 2
            avg_rel_y = (left_rel_y + right_rel_y) / 2
            
            confidence = 0.8
        else:
            # Fallback: estimate based on eye shape
            left_eye_width = np.max(left_eye[:, 0]) - np.min(left_eye[:, 0])
            left_eye_height = np.max(left_eye[:, 1]) - np.min(left_eye[:, 1])
            
            right_eye_width = np.max(right_eye[:, 0]) - np.min(right_eye[:, 0])
            right_eye_height = np.max(right_eye[:, 1]) - np.min(right_eye[:, 1])
            
            # Simple estimation - this is less accurate
            avg_rel_x = 0.5  # Assume center
            avg_rel_y = 0.5  # Assume center
            confidence = 0.4
        
        # Convert to angles (approximate)
        # Map relative position to angles
        yaw = (avg_rel_x - 0.5) * 60  # -30 to +30 degrees
        pitch = (avg_rel_y - 0.5) * 40  # -20 to +20 degrees
        
        # Determine gaze direction
        direction = self._get_direction_label(yaw, pitch)
        
        return direction, (pitch, yaw), confidence
    
    def _get_direction_label(self, yaw: float, pitch: float) -> str:
        """Convert gaze angles to direction label"""
        # Thresholds for direction classification
        yaw_threshold = 15
        pitch_threshold = 10
        
        # Horizontal direction
        if yaw < -yaw_threshold:
            horizontal = "Left"
        elif yaw > yaw_threshold:
            horizontal = "Right"
        else:
            horizontal = "Center"
        
        # Vertical direction
        if pitch < -pitch_threshold:
            vertical = "Up"
        elif pitch > pitch_threshold:
            vertical = "Down"
        else:
            vertical = "Center"
        
        # Combine directions
        if horizontal == "Center" and vertical == "Center":
            return "Center"
        elif horizontal == "Center":
            return vertical
        elif vertical == "Center":
            return horizontal
        else:
            return f"{vertical}-{horizontal}"
    
    def draw_gaze_overlay(self, image: np.ndarray, gaze_info: Dict, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw gaze direction overlay on image"""
        result_image = image.copy()
        
        # Draw eye landmarks
        if 'left_eye_points' in gaze_info:
            cv2.polylines(result_image, [gaze_info['left_eye_points']], True, color, 1)
        
        if 'right_eye_points' in gaze_info:
            cv2.polylines(result_image, [gaze_info['right_eye_points']], True, color, 1)
        
        # Draw iris centers
        if gaze_info['left_iris_center'] is not None:
            center = tuple(map(int, gaze_info['left_iris_center']))
            cv2.circle(result_image, center, 3, (0, 0, 255), -1)
        
        if gaze_info['right_iris_center'] is not None:
            center = tuple(map(int, gaze_info['right_iris_center']))
            cv2.circle(result_image, center, 3, (0, 0, 255), -1)
        
        # Draw gaze direction arrow
        self._draw_gaze_arrow(result_image, gaze_info, color)
        
        return result_image
    
    def _draw_gaze_arrow(self, image: np.ndarray, gaze_info: Dict, color: Tuple[int, int, int]):
        """Draw arrow indicating gaze direction"""
        if 'gaze_angles' not in gaze_info:
            return
        
        pitch, yaw = gaze_info['gaze_angles']
        
        # Calculate arrow start point (between eyes)
        if gaze_info['left_iris_center'] is not None and gaze_info['right_iris_center'] is not None:
            start_point = ((gaze_info['left_iris_center'] + gaze_info['right_iris_center']) / 2).astype(int)
        else:
            # Use eye centers
            left_center = np.mean(gaze_info['left_eye_points'], axis=0)
            right_center = np.mean(gaze_info['right_eye_points'], axis=0)
            start_point = ((left_center + right_center) / 2).astype(int)
        
        # Calculate arrow end point
        arrow_length = 50
        end_x = start_point[0] + int(arrow_length * np.sin(np.radians(yaw)))
        end_y = start_point[1] + int(arrow_length * np.sin(np.radians(pitch)))
        end_point = (end_x, end_y)
        
        # Draw arrow
        cv2.arrowedLine(image, tuple(start_point), end_point, color, 2, tipLength=0.3)
        
        # Add text label
        text_pos = (start_point[0] - 50, start_point[1] - 20)
        cv2.putText(image, f"Gaze: {gaze_info['gaze_direction']}", text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
