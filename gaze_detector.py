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
        
        # Face mesh model - increased max faces for better multi-face detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Eye landmarks indices
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Iris landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Head pose landmarks for improved gaze accuracy
        self.NOSE_TIP = 1
        self.CHIN = 152
        self.LEFT_CHEEK = 234
        self.RIGHT_CHEEK = 454
        self.FOREHEAD = 9
    
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
    
    def _analyze_gaze(self, landmarks, image_shape) -> Dict:
        """Analyze gaze direction from face landmarks"""
        h, w = image_shape[:2]
        
        # Convert landmarks to pixel coordinates
        landmarks_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_points.append([x, y])
        
        landmarks_array = np.array(landmarks_points)
        
        # Get head pose landmarks for improved accuracy
        nose_tip = landmarks_array[self.NOSE_TIP] if len(landmarks_array) > self.NOSE_TIP else None
        chin = landmarks_array[self.CHIN] if len(landmarks_array) > self.CHIN else None
        left_cheek = landmarks_array[self.LEFT_CHEEK] if len(landmarks_array) > self.LEFT_CHEEK else None
        right_cheek = landmarks_array[self.RIGHT_CHEEK] if len(landmarks_array) > self.RIGHT_CHEEK else None
        
        # Calculate head pose angles
        head_pose_angles = self._calculate_head_pose(nose_tip, chin, left_cheek, right_cheek, (w, h))
        
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
        
        # Calculate gaze direction combining eye tracking and head pose
        gaze_direction, gaze_angles, confidence = self._calculate_combined_gaze_direction(
            left_eye_points, right_eye_points, left_iris_center, right_iris_center, head_pose_angles
        )
        
        return {
            'gaze_direction': gaze_direction,
            'gaze_angles': gaze_angles,
            'head_pose_angles': head_pose_angles,
            'confidence': confidence,
            'left_eye_points': left_eye_points,
            'right_eye_points': right_eye_points,
            'left_iris_center': left_iris_center,
            'right_iris_center': right_iris_center,
            'landmarks': landmarks_array
        }
    
    def _calculate_head_pose(self, nose_tip, chin, left_cheek, right_cheek, image_size) -> Tuple[float, float, float]:
        """Calculate head pose angles (pitch, yaw, roll) from facial landmarks"""
        if nose_tip is None or chin is None or left_cheek is None or right_cheek is None:
            return (0.0, 0.0, 0.0)
        
        w, h = image_size
        
        # Calculate pitch (up/down head rotation)
        nose_chin_distance = np.linalg.norm(nose_tip - chin)
        face_height_estimate = h * 0.3  # Approximate face height as 30% of image
        pitch_factor = (nose_chin_distance - face_height_estimate * 0.7) / (face_height_estimate * 0.3)
        pitch = np.clip(pitch_factor * 30, -45, 45)  # Convert to degrees
        
        # Calculate yaw (left/right head rotation)
        cheek_distance_left = np.linalg.norm(nose_tip - left_cheek)
        cheek_distance_right = np.linalg.norm(nose_tip - right_cheek)
        yaw_factor = (cheek_distance_right - cheek_distance_left) / max(cheek_distance_left, cheek_distance_right)
        yaw = np.clip(yaw_factor * 45, -60, 60)  # Convert to degrees
        
        # Calculate roll (head tilt)
        cheek_vector = right_cheek - left_cheek
        roll = np.degrees(np.arctan2(cheek_vector[1], cheek_vector[0]))
        roll = np.clip(roll, -30, 30)
        
        return (float(pitch), float(yaw), float(roll))
    
    def _calculate_combined_gaze_direction(self, left_eye: np.ndarray, right_eye: np.ndarray, 
                                         left_iris: Optional[np.ndarray], right_iris: Optional[np.ndarray],
                                         head_pose_angles: Tuple[float, float, float]) -> Tuple[str, Tuple[float, float], float]:
        """Calculate combined gaze direction using eye tracking and head pose"""
        head_pitch, head_yaw, head_roll = head_pose_angles
        
        # Calculate eye centers
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        
        # Calculate eye-based gaze angles
        eye_yaw = 0.0
        eye_pitch = 0.0
        eye_confidence = 0.3
        
        # Use iris centers if available for more accurate eye gaze
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
            
            # Convert to eye gaze angles
            eye_yaw = (avg_rel_x - 0.5) * 30  # -15 to +15 degrees from eyes
            eye_pitch = (avg_rel_y - 0.5) * 20  # -10 to +10 degrees from eyes
            eye_confidence = 0.7
        
        # Combine head pose and eye gaze for final gaze direction
        # Head pose has more influence on overall gaze direction
        combined_yaw = head_yaw * 0.7 + eye_yaw * 0.3
        combined_pitch = head_pitch * 0.7 + eye_pitch * 0.3
        
        # Calculate combined confidence
        head_confidence = 0.8 if abs(head_yaw) > 5 or abs(head_pitch) > 5 else 0.6
        combined_confidence = (head_confidence * 0.6 + eye_confidence * 0.4)
        
        # Determine gaze direction using combined angles
        direction = self._get_direction_label(combined_yaw, combined_pitch)
        
        return direction, (float(combined_pitch), float(combined_yaw)), combined_confidence
    
    def _get_direction_label(self, yaw: float, pitch: float) -> str:
        """Convert gaze angles to meaningful location descriptions"""
        # Define thresholds for different regions
        yaw_threshold_narrow = 8   # For center detection
        yaw_threshold_wide = 25    # For far left/right
        pitch_threshold_narrow = 6 # For straight ahead
        pitch_threshold_wide = 20  # For far up/down
        
        # Determine detailed gaze location
        if abs(yaw) < yaw_threshold_narrow and abs(pitch) < pitch_threshold_narrow:
            return "looking at camera"
        
        # Looking down variations
        elif pitch > pitch_threshold_wide:
            if abs(yaw) < yaw_threshold_narrow:
                return "looking at table/floor"
            elif yaw > yaw_threshold_narrow:
                return "looking at something down-right"
            else:
                return "looking at something down-left"
        
        # Looking up variations  
        elif pitch < -pitch_threshold_wide:
            if abs(yaw) < yaw_threshold_narrow:
                return "looking at ceiling/above"
            elif yaw > yaw_threshold_narrow:
                return "looking at something up-right"
            else:
                return "looking at something up-left"
        
        # Horizontal looking
        elif yaw > yaw_threshold_wide:
            if abs(pitch) < pitch_threshold_narrow:
                return "looking at person/object on right"
            elif pitch > 0:
                return "looking at something down-right"
            else:
                return "looking at something up-right"
                
        elif yaw < -yaw_threshold_wide:
            if abs(pitch) < pitch_threshold_narrow:
                return "looking at person/object on left"
            elif pitch > 0:
                return "looking at something down-left"
            else:
                return "looking at something up-left"
        
        # Moderate angles
        elif yaw > yaw_threshold_narrow:
            if pitch > pitch_threshold_narrow:
                return "looking down-right"
            elif pitch < -pitch_threshold_narrow:
                return "looking up-right"
            else:
                return "looking right"
                
        elif yaw < -yaw_threshold_narrow:
            if pitch > pitch_threshold_narrow:
                return "looking down-left"
            elif pitch < -pitch_threshold_narrow:
                return "looking up-left"
            else:
                return "looking left"
        
        # Vertical only
        elif pitch > pitch_threshold_narrow:
            return "looking down"
        elif pitch < -pitch_threshold_narrow:
            return "looking up"
        else:
            return "looking straight ahead"
    
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
        
        # Add text label with head pose info
        text_pos = (start_point[0] - 50, start_point[1] - 20)
        cv2.putText(image, f"Gaze: {gaze_info['gaze_direction']}", text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add head pose angles if available
        if 'head_pose_angles' in gaze_info:
            head_pitch, head_yaw, head_roll = gaze_info['head_pose_angles']
            head_text_pos = (start_point[0] - 50, start_point[1] - 45)
            cv2.putText(image, f"Head: Y{head_yaw:.1f}° P{head_pitch:.1f}°", head_text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
