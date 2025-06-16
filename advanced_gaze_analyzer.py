import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, List, Optional
import math
import logging

class AdvancedGazeAnalyzer:
    """
    Advanced gaze direction analysis using multiple computer vision techniques:
    - MediaPipe face mesh for precise eye tracking
    - Geometric analysis of facial features
    - Eye region analysis and iris detection
    - Head pose estimation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe face mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices from MediaPipe face mesh
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Eye corner landmarks for gaze analysis
        self.left_eye_corners = [33, 133]  # outer, inner
        self.right_eye_corners = [362, 263]  # outer, inner
        
        # Iris tracking points
        self.left_iris_indices = [468, 469, 470, 471, 472]
        self.right_iris_indices = [473, 474, 475, 476, 477]
        
        self.logger.info("Advanced gaze analyzer initialized")
    
    def analyze_gaze(self, image: np.ndarray, face_detection: Dict) -> Dict:
        """
        Analyze gaze direction for a detected face
        
        Args:
            image: Full image in BGR format
            face_detection: Face detection result containing bounding box
            
        Returns:
            Dictionary containing gaze analysis results
        """
        try:
            x, y, w, h = face_detection['x'], face_detection['y'], face_detection['w'], face_detection['h']
            
            # Extract face region with padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return self._get_default_gaze_result()
            
            # Convert to RGB for MediaPipe
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Get face mesh landmarks
            results = self.face_mesh.process(rgb_face)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Convert landmarks to pixel coordinates
                h_face, w_face = face_region.shape[:2]
                landmark_points = []
                for lm in landmarks.landmark:
                    x_lm = int(lm.x * w_face) + x1
                    y_lm = int(lm.y * h_face) + y1
                    landmark_points.append([x_lm, y_lm])
                
                # Analyze gaze using multiple methods
                gaze_result = self._comprehensive_gaze_analysis(
                    image, landmark_points, face_detection
                )
                
                return gaze_result
            else:
                # Fallback to geometric analysis
                return self._geometric_gaze_analysis(image, face_detection)
                
        except Exception as e:
            self.logger.warning(f"Gaze analysis failed: {e}")
            return self._get_default_gaze_result()
    
    def _comprehensive_gaze_analysis(self, image: np.ndarray, landmarks: List[List[int]], 
                                   face_detection: Dict) -> Dict:
        """Comprehensive gaze analysis using facial landmarks"""
        try:
            # Extract eye regions and analyze
            left_eye_analysis = self._analyze_eye_region(
                image, landmarks, self.left_eye_indices, "left"
            )
            right_eye_analysis = self._analyze_eye_region(
                image, landmarks, self.right_eye_indices, "right"
            )
            
            # Head pose analysis
            head_pose = self._estimate_head_pose(landmarks, image.shape)
            
            # Combine eye analyses
            combined_gaze = self._combine_eye_analyses(left_eye_analysis, right_eye_analysis)
            
            # Adjust for head pose
            final_gaze = self._adjust_gaze_for_head_pose(combined_gaze, head_pose)
            
            # Generate descriptive gaze direction
            gaze_description = self._generate_gaze_description(final_gaze, head_pose, face_detection, image.shape)
            
            return {
                'gaze_direction': gaze_description,
                'gaze_vector': final_gaze,
                'head_pose': head_pose,
                'left_eye': left_eye_analysis,
                'right_eye': right_eye_analysis,
                'confidence': self._calculate_gaze_confidence(left_eye_analysis, right_eye_analysis),
                'method': 'mediapipe_comprehensive'
            }
            
        except Exception as e:
            self.logger.warning(f"Comprehensive gaze analysis failed: {e}")
            return self._geometric_gaze_analysis(image, face_detection)
    
    def _analyze_eye_region(self, image: np.ndarray, landmarks: List[List[int]], 
                           eye_indices: List[int], eye_side: str) -> Dict:
        """Analyze individual eye region for gaze direction"""
        try:
            # Get eye landmark points
            eye_points = [landmarks[i] for i in eye_indices]
            eye_array = np.array(eye_points)
            
            # Calculate eye bounding box
            x_min, y_min = np.min(eye_array, axis=0)
            x_max, y_max = np.max(eye_array, axis=0)
            
            # Extract eye region
            eye_region = image[y_min:y_max, x_min:x_max]
            
            if eye_region.size == 0:
                return {'gaze_x': 0, 'gaze_y': 0, 'confidence': 0.0}
            
            # Convert to grayscale for analysis
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Find iris/pupil using circular Hough transform
            iris_center = self._detect_iris_center(gray_eye)
            
            if iris_center is not None:
                # Calculate gaze relative to eye corners
                eye_width = x_max - x_min
                eye_height = y_max - y_min
                
                # Normalize iris position within eye
                iris_x_norm = (iris_center[0] - eye_width/2) / (eye_width/2)
                iris_y_norm = (iris_center[1] - eye_height/2) / (eye_height/2)
                
                return {
                    'gaze_x': iris_x_norm,
                    'gaze_y': iris_y_norm,
                    'confidence': 0.8,
                    'iris_center': [x_min + iris_center[0], y_min + iris_center[1]],
                    'eye_bounds': [x_min, y_min, x_max, y_max]
                }
            else:
                # Fallback to geometric analysis
                return self._geometric_eye_analysis(eye_points, eye_side)
                
        except Exception as e:
            self.logger.warning(f"Eye region analysis failed for {eye_side} eye: {e}")
            return {'gaze_x': 0, 'gaze_y': 0, 'confidence': 0.0}
    
    def _detect_iris_center(self, gray_eye: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect iris center using circular Hough transform and contour analysis"""
        try:
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray_eye, (9, 9), 2)
            
            # Use HoughCircles to detect circular iris
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,
                param1=50,
                param2=30,
                minRadius=5,
                maxRadius=25
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # Return the most central circle
                center_x, center_y = gray_eye.shape[1] // 2, gray_eye.shape[0] // 2
                
                best_circle = None
                min_distance = float('inf')
                
                for (x, y, r) in circles:
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        best_circle = (x, y)
                
                return best_circle
            
            # Fallback: find darkest region (pupil)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
            return min_loc
            
        except Exception:
            return None
    
    def _geometric_eye_analysis(self, eye_points: List[List[int]], eye_side: str) -> Dict:
        """Geometric analysis of eye points for gaze estimation"""
        try:
            eye_array = np.array(eye_points)
            
            # Calculate eye center
            eye_center = np.mean(eye_array, axis=0)
            
            # Calculate eye corners
            x_coords = eye_array[:, 0]
            y_coords = eye_array[:, 1]
            
            left_corner = [np.min(x_coords), np.mean(y_coords)]
            right_corner = [np.max(x_coords), np.mean(y_coords)]
            
            # Estimate gaze based on eye shape asymmetry
            eye_width = right_corner[0] - left_corner[0]
            
            if eye_width > 0:
                # Calculate relative position within eye
                center_offset = (eye_center[0] - (left_corner[0] + right_corner[0])/2) / (eye_width/2)
                
                return {
                    'gaze_x': center_offset,
                    'gaze_y': 0.0,
                    'confidence': 0.5,
                    'method': 'geometric'
                }
            
        except Exception:
            pass
        
        return {'gaze_x': 0, 'gaze_y': 0, 'confidence': 0.0}
    
    def _estimate_head_pose(self, landmarks: List[List[int]], image_shape: Tuple[int, int, int]) -> Dict:
        """Estimate head pose using facial landmarks"""
        try:
            # Key facial landmarks for pose estimation
            nose_tip = landmarks[1]
            chin = landmarks[152]
            left_eye_corner = landmarks[33]
            right_eye_corner = landmarks[263]
            left_mouth_corner = landmarks[61]
            right_mouth_corner = landmarks[291]
            
            # Calculate face center
            face_center_x = (left_eye_corner[0] + right_eye_corner[0]) / 2
            face_center_y = (left_eye_corner[1] + right_eye_corner[1]) / 2
            
            # Calculate yaw (left-right rotation)
            eye_distance = right_eye_corner[0] - left_eye_corner[0]
            nose_offset = nose_tip[0] - face_center_x
            yaw = np.arctan2(nose_offset, eye_distance) * 180 / np.pi
            
            # Calculate pitch (up-down rotation)
            nose_eye_distance = nose_tip[1] - face_center_y
            face_height = chin[1] - face_center_y
            if face_height > 0:
                pitch = np.arctan2(nose_eye_distance, face_height) * 180 / np.pi
            else:
                pitch = 0
            
            # Calculate roll (rotation around nose)
            eye_slope = (right_eye_corner[1] - left_eye_corner[1]) / max(eye_distance, 1)
            roll = np.arctan(eye_slope) * 180 / np.pi
            
            return {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll),
                'confidence': 0.7
            }
            
        except Exception as e:
            self.logger.warning(f"Head pose estimation failed: {e}")
            return {'yaw': 0, 'pitch': 0, 'roll': 0, 'confidence': 0.0}
    
    def _combine_eye_analyses(self, left_eye: Dict, right_eye: Dict) -> Dict:
        """Combine left and right eye analyses"""
        try:
            left_conf = left_eye.get('confidence', 0)
            right_conf = right_eye.get('confidence', 0)
            
            if left_conf == 0 and right_conf == 0:
                return {'gaze_x': 0, 'gaze_y': 0, 'confidence': 0.0}
            
            # Weighted average based on confidence
            total_conf = left_conf + right_conf
            
            if total_conf > 0:
                gaze_x = (left_eye.get('gaze_x', 0) * left_conf + 
                         right_eye.get('gaze_x', 0) * right_conf) / total_conf
                gaze_y = (left_eye.get('gaze_y', 0) * left_conf + 
                         right_eye.get('gaze_y', 0) * right_conf) / total_conf
                
                return {
                    'gaze_x': gaze_x,
                    'gaze_y': gaze_y,
                    'confidence': min(total_conf / 2, 1.0)
                }
            
        except Exception:
            pass
        
        return {'gaze_x': 0, 'gaze_y': 0, 'confidence': 0.0}
    
    def _adjust_gaze_for_head_pose(self, gaze: Dict, head_pose: Dict) -> Dict:
        """Adjust gaze direction based on head pose"""
        try:
            gaze_x = gaze.get('gaze_x', 0)
            gaze_y = gaze.get('gaze_y', 0)
            
            yaw = head_pose.get('yaw', 0)
            pitch = head_pose.get('pitch', 0)
            
            # Adjust gaze based on head rotation
            # Yaw affects horizontal gaze
            adjusted_gaze_x = gaze_x + (yaw / 45.0)  # Normalize yaw influence
            
            # Pitch affects vertical gaze
            adjusted_gaze_y = gaze_y + (pitch / 30.0)  # Normalize pitch influence
            
            # Clamp values
            adjusted_gaze_x = np.clip(adjusted_gaze_x, -1.0, 1.0)
            adjusted_gaze_y = np.clip(adjusted_gaze_y, -1.0, 1.0)
            
            return {
                'gaze_x': adjusted_gaze_x,
                'gaze_y': adjusted_gaze_y,
                'confidence': gaze.get('confidence', 0)
            }
            
        except Exception:
            return gaze
    
    def _generate_gaze_description(self, gaze: Dict, head_pose: Dict, 
                                 face_detection: Dict, image_shape: Tuple[int, int, int]) -> str:
        """Generate descriptive gaze direction"""
        try:
            gaze_x = gaze.get('gaze_x', 0)
            gaze_y = gaze.get('gaze_y', 0)
            yaw = head_pose.get('yaw', 0)
            pitch = head_pose.get('pitch', 0)
            
            # Face position in image
            face_x = face_detection['x'] + face_detection['w'] // 2
            face_y = face_detection['y'] + face_detection['h'] // 2
            img_h, img_w = image_shape[:2]
            
            face_pos_x = face_x / img_w  # 0 to 1
            face_pos_y = face_y / img_h  # 0 to 1
            
            # Determine gaze direction with contextual descriptions
            descriptions = []
            
            # Horizontal gaze
            if abs(gaze_x) > 0.3 or abs(yaw) > 15:
                if gaze_x > 0.3 or yaw > 15:
                    if face_pos_x < 0.3:
                        descriptions.append("looking at person on right")
                    elif face_pos_x > 0.7:
                        descriptions.append("looking away right")
                    else:
                        descriptions.append("looking right")
                elif gaze_x < -0.3 or yaw < -15:
                    if face_pos_x > 0.7:
                        descriptions.append("looking at person on left")
                    elif face_pos_x < 0.3:
                        descriptions.append("looking away left")
                    else:
                        descriptions.append("looking left")
            
            # Vertical gaze
            if abs(gaze_y) > 0.2 or abs(pitch) > 10:
                if gaze_y > 0.2 or pitch > 10:
                    if face_pos_y < 0.4:
                        descriptions.append("looking down at table/game")
                    else:
                        descriptions.append("looking down")
                elif gaze_y < -0.2 or pitch < -10:
                    descriptions.append("looking up/away")
            
            # Camera gaze detection
            if abs(gaze_x) < 0.2 and abs(gaze_y) < 0.2 and abs(yaw) < 10 and abs(pitch) < 10:
                descriptions.append("looking at camera")
            
            # Contextual additions based on position
            if len(descriptions) == 0:
                if face_pos_x < 0.25:
                    descriptions.append("looking slightly right")
                elif face_pos_x > 0.75:
                    descriptions.append("looking slightly left")
                else:
                    descriptions.append("looking forward")
            
            # Join descriptions
            if descriptions:
                return descriptions[0]  # Return primary description
            else:
                return "looking forward"
                
        except Exception:
            return "gaze unknown"
    
    def _calculate_gaze_confidence(self, left_eye: Dict, right_eye: Dict) -> float:
        """Calculate overall confidence in gaze analysis"""
        try:
            left_conf = left_eye.get('confidence', 0)
            right_conf = right_eye.get('confidence', 0)
            
            # Average confidence with bonus for having both eyes
            avg_conf = (left_conf + right_conf) / 2
            
            if left_conf > 0 and right_conf > 0:
                # Bonus for having both eyes detected
                return min(avg_conf * 1.2, 1.0)
            else:
                return avg_conf
                
        except Exception:
            return 0.5
    
    def _geometric_gaze_analysis(self, image: np.ndarray, face_detection: Dict) -> Dict:
        """Fallback geometric gaze analysis without landmarks"""
        try:
            x, y, w, h = face_detection['x'], face_detection['y'], face_detection['w'], face_detection['h']
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            if face_region.size == 0:
                return self._get_default_gaze_result()
            
            # Estimate eye regions
            eye_y = int(y + h * 0.35)
            eye_h = int(h * 0.15)
            
            left_eye_x = int(x + w * 0.2)
            right_eye_x = int(x + w * 0.65)
            eye_w = int(w * 0.15)
            
            # Analyze face position for contextual gaze
            img_h, img_w = image.shape[:2]
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # Generate description based on position
            gaze_desc = self._position_based_gaze(face_center_x, face_center_y, img_w, img_h)
            
            return {
                'gaze_direction': gaze_desc,
                'gaze_vector': {'gaze_x': 0, 'gaze_y': 0, 'confidence': 0.3},
                'head_pose': {'yaw': 0, 'pitch': 0, 'roll': 0, 'confidence': 0.2},
                'confidence': 0.3,
                'method': 'geometric_fallback'
            }
            
        except Exception:
            return self._get_default_gaze_result()
    
    def _position_based_gaze(self, face_x: int, face_y: int, img_w: int, img_h: int) -> str:
        """Generate gaze description based on face position in image"""
        pos_x = face_x / img_w
        pos_y = face_y / img_h
        
        if pos_x < 0.3:
            return "looking at person on right"
        elif pos_x > 0.7:
            return "looking at person on left"
        elif pos_y < 0.4:
            return "looking down at game board"
        elif pos_y > 0.6:
            return "looking up/away"
        else:
            return "looking at camera"
    
    def _get_default_gaze_result(self) -> Dict:
        """Return default gaze result when analysis fails"""
        return {
            'gaze_direction': 'looking forward',
            'gaze_vector': {'gaze_x': 0, 'gaze_y': 0, 'confidence': 0.1},
            'head_pose': {'yaw': 0, 'pitch': 0, 'roll': 0, 'confidence': 0.1},
            'confidence': 0.1,
            'method': 'default'
        }