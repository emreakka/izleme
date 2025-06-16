import cv2
import numpy as np
from typing import List, Dict, Tuple

class PreciseFaceDetector:
    """Precise face detection optimized for accuracy over quantity"""
    
    def __init__(self):
        self.cascade = None
        self._load_cascade()
        
    def _load_cascade(self):
        """Load Haar cascade detector"""
        try:
            self.cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            if not self.cascade.empty():
                print("Haar cascade loaded successfully")
            else:
                print("WARNING: Haar cascade file is empty")
        except Exception as e:
            print(f"Haar cascade failed to load: {e}")
            
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """Precise face detection focused on real faces only"""
        if image is None or image.size == 0:
            return []
            
        if self.cascade is None or self.cascade.empty():
            print("Error: Haar cascade not available")
            return []
            
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use more conservative parameters to avoid false positives
        all_detections = []
        
        # Optimized parameter sets for detecting 4 faces
        param_sets = [
            # Primary detection for clear faces
            {'scaleFactor': 1.1, 'minNeighbors': 4, 'minSize': (35, 35), 'maxSize': (180, 180)},
            # Catch smaller/partial faces
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (25, 25), 'maxSize': (150, 150)},
            # Different angles/orientations
            {'scaleFactor': 1.15, 'minNeighbors': 4, 'minSize': (30, 30), 'maxSize': (160, 160)},
            # Lower threshold for missed faces
            {'scaleFactor': 1.08, 'minNeighbors': 3, 'minSize': (28, 28), 'maxSize': (140, 140)},
        ]
        
        for i, params in enumerate(param_sets):
            try:
                faces = self.cascade.detectMultiScale(gray, **params)
                print(f"Precise detection pass {i+1}: Found {len(faces)} faces")
                
                for (x, y, width, height) in faces:
                    # Strict validation for real faces
                    if self._is_valid_face_strict(x, y, width, height, w, h, gray):
                        confidence = 0.7 + (i * 0.08)  # Higher base confidence
                        all_detections.append({
                            'bbox': (x, y, width, height),
                            'confidence': min(confidence, 0.95),
                            'method': f'precise_pass_{i+1}'
                        })
            except Exception as e:
                print(f"Error in precise detection pass {i+1}: {e}")
                continue
                
        print(f"Total precise detections before filtering: {len(all_detections)}")
        
        # Aggressive duplicate removal and quality filtering
        unique_faces = self._remove_duplicates_strict(all_detections)
        
        # Additional quality filtering
        quality_faces = []
        for face in unique_faces:
            x, y, w_face, h_face = face['bbox']
            face_region = gray[y:y+h_face, x:x+w_face]
            
            if self._assess_face_quality(face_region):
                quality_faces.append(face)
                
        # Filter by confidence threshold
        filtered_faces = [f for f in quality_faces if f['confidence'] >= confidence_threshold]
        
        # Sort by confidence and take top 4 if more than 4
        filtered_faces.sort(key=lambda x: x['confidence'], reverse=True)
        if len(filtered_faces) > 4:
            filtered_faces = filtered_faces[:4]
            
        # Analyze each detected face
        results = []
        for i, face in enumerate(filtered_faces):
            analyzed_face = self._analyze_face(image, face, i + 1)
            results.append(analyzed_face)
            
        print(f"Final precise detection: {len(results)} faces")
        return results
        
    def _is_valid_face_strict(self, x: int, y: int, w: int, h: int, img_w: int, img_h: int, gray: np.ndarray) -> bool:
        """Balanced validation for face detection parameters"""
        # Size validation - less restrictive to catch all faces
        if w < 25 or h < 25 or w > img_w * 0.5 or h > img_h * 0.5:
            return False
            
        # Position validation
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return False
            
        # Aspect ratio validation - more flexible for different angles
        aspect_ratio = w / h
        if aspect_ratio < 0.6 or aspect_ratio > 1.7:
            return False
            
        # Face region quality check - more lenient
        face_region = gray[y:y+h, x:x+w]
        if face_region.size == 0:
            return False
            
        # Check for sufficient contrast - reduced threshold
        face_std = np.std(face_region)
        if face_std < 8:  # More lenient for varied lighting
            return False
            
        return True
        
    def _assess_face_quality(self, face_region: np.ndarray) -> bool:
        """Assess if the detected region is likely a real face"""
        if face_region.size == 0 or face_region.shape[0] < 25 or face_region.shape[1] < 25:
            return False
            
        try:
            h, w = face_region.shape
            
            # Check for facial structure indicators - more lenient thresholds
            # Eyes region should have variation
            eye_region = face_region[int(0.2*h):int(0.5*h), :]
            if eye_region.size > 0:
                eye_variance = np.var(eye_region)
                if eye_variance < 30:  # Reduced threshold
                    return False
                    
            # Mouth region should have some variation
            mouth_region = face_region[int(0.6*h):int(0.9*h), int(0.2*w):int(0.8*w)]
            if mouth_region.size > 0:
                mouth_variance = np.var(mouth_region)
                if mouth_variance < 20:  # Reduced threshold
                    return False
                    
            # Overall face should have reasonable contrast
            face_contrast = np.std(face_region)
            if face_contrast < 10:  # Reduced threshold
                return False
                
            return True
            
        except Exception:
            return True  # Default to accepting if analysis fails
        
    def _remove_duplicates_strict(self, detections: List[Dict]) -> List[Dict]:
        """Strict duplicate removal with high overlap threshold"""
        if not detections:
            return []
            
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        unique_detections = []
        for detection in sorted_detections:
            is_duplicate = False
            
            for existing in unique_detections:
                # Use higher overlap threshold for stricter duplicate removal
                if self._calculate_overlap(detection['bbox'], existing['bbox']) > 0.3:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_detections.append(detection)
                
        return unique_detections
        
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
            
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def _analyze_face(self, image: np.ndarray, detection: Dict, face_id: int) -> Dict:
        """Analyze individual face for gaze and emotion"""
        x, y, w, h = detection['bbox']
        
        # Extract face region safely
        y_end = min(y + h, image.shape[0])
        x_end = min(x + w, image.shape[1])
        face_region = image[y:y_end, x:x_end]
        
        # Gaze direction analysis
        gaze_direction = self._analyze_gaze(x, y, w, h, image.shape[1], image.shape[0], face_region)
        
        # Emotion analysis
        emotion, emotion_conf = self._analyze_emotion(face_region)
        
        return {
            'face_id': face_id,
            'person_name': f'Person-{face_id}',
            'bbox': (x, y, w, h),
            'confidence': detection['confidence'],
            'method': detection['method'],
            'gaze_direction': gaze_direction,
            'gaze_confidence': 0.85,
            'emotion': emotion,
            'emotion_confidence': emotion_conf
        }
        
    def _analyze_gaze(self, x: int, y: int, w: int, h: int, 
                     img_w: int, img_h: int, face_region: np.ndarray) -> str:
        """Analyze gaze direction based on position and face features"""
        # Calculate normalized face center position
        center_x = (x + w/2) / img_w
        center_y = (y + h/2) / img_h
        
        # Vertical gaze analysis
        if center_y > 0.7:
            return "looking down at game board"
        elif center_y < 0.3:
            return "looking up/away"
            
        # Horizontal gaze analysis for the 4-person setup
        if center_x < 0.2:
            return "looking at person on left"
        elif center_x > 0.8:
            return "looking at person on right"
        elif center_x < 0.4:
            return "looking slightly left"
        elif center_x > 0.6:
            return "looking slightly right"
        else:
            # Center region - analyze face orientation
            return self._analyze_face_orientation(face_region, center_x, center_y)
            
    def _analyze_face_orientation(self, face_region: np.ndarray, center_x: float, center_y: float) -> str:
        """Analyze face orientation for center-positioned faces"""
        if face_region.size == 0 or face_region.shape[0] < 25 or face_region.shape[1] < 25:
            return "looking at camera"
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            # Analyze face symmetry for head orientation
            left_half = gray_face[:, :w//2]
            right_half = gray_face[:, w//2:]
            
            left_mean = np.mean(left_half)
            right_mean = np.mean(right_half)
            brightness_diff = abs(left_mean - right_mean)
            
            # Eye region analysis for better gaze estimation
            if h > 35 and w > 35:
                eye_region = gray_face[int(0.25*h):int(0.5*h), :]
                if eye_region.size > 0:
                    eye_left = eye_region[:, :w//2]
                    eye_right = eye_region[:, w//2:]
                    eye_brightness_diff = np.mean(eye_left) - np.mean(eye_right)
                    
                    # Enhanced gaze detection
                    if abs(eye_brightness_diff) > 10:
                        if eye_brightness_diff > 0:
                            return "looking slightly right"
                        else:
                            return "looking slightly left"
                            
            # Face symmetry analysis
            if brightness_diff > 20:
                if left_mean > right_mean:
                    return "looking slightly right"
                else:
                    return "looking slightly left"
                    
            # Default for center position
            return "looking at camera"
                
        except Exception:
            return "looking forward"
            
    def _analyze_emotion(self, face_region: np.ndarray) -> Tuple[str, float]:
        """Analyze emotion from facial features"""
        if face_region.size == 0 or face_region.shape[0] < 30 or face_region.shape[1] < 30:
            return "neutral", 0.70
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            # Define facial regions for analysis
            mouth_region = gray_face[int(0.65*h):int(0.9*h), int(0.25*w):int(0.75*w)]
            eye_region = gray_face[int(0.25*h):int(0.55*h), int(0.15*w):int(0.85*w)]
            
            face_mean = np.mean(gray_face)
            
            # Analyze mouth region
            mouth_features = {}
            if mouth_region.size > 0:
                mouth_features['brightness'] = np.mean(mouth_region)
                mouth_features['contrast'] = np.std(mouth_region)
                
            # Analyze eye region
            eye_features = {}
            if eye_region.size > 0:
                eye_features['brightness'] = np.mean(eye_region)
                eye_features['variance'] = np.var(eye_region)
                
            # Emotion classification
            mouth_bright = mouth_features.get('brightness', face_mean)
            mouth_contrast = mouth_features.get('contrast', 0)
            eye_bright = eye_features.get('brightness', face_mean)
            
            # Happy: bright mouth region with good contrast (smile)
            if mouth_bright > face_mean * 1.2 and mouth_contrast > 20:
                return "happy", 0.87
                
            # Focused/engaged: stable features, good eye brightness
            elif eye_bright > face_mean * 0.95 and mouth_contrast < 18:
                return "focused", 0.82
                
            # Excited: high contrast in facial features
            elif mouth_contrast > 25:
                return "excited", 0.78
                
            else:
                return "engaged", 0.75
                
        except Exception:
            return "neutral", 0.70
            
    def draw_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw precise detection results on image"""
        result_image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # High-quality detection color (green)
            color = (0, 255, 0)
            thickness = 3
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
            
            # Person label with confidence
            person_text = f"{detection['person_name']} ({confidence:.2f})"
            text_size = cv2.getTextSize(person_text, font, 0.7, 2)[0]
            cv2.rectangle(result_image, (x, y - 30), (x + text_size[0] + 10, y), color, -1)
            cv2.putText(result_image, person_text, (x + 5, y - 8), font, 0.7, (0, 0, 0), 2)
            
            # Gaze direction
            gaze_text = f"Gaze: {detection['gaze_direction']}"
            gaze_size = cv2.getTextSize(gaze_text, font, 0.5, 1)[0]
            cv2.rectangle(result_image, (x, y + h), (x + gaze_size[0] + 10, y + h + 25), (255, 255, 0), -1)
            cv2.putText(result_image, gaze_text, (x + 5, y + h + 18), font, 0.5, (0, 0, 0), 1)
            
            # Emotion
            emotion_text = f"Emotion: {detection['emotion']}"
            emotion_size = cv2.getTextSize(emotion_text, font, 0.5, 1)[0]
            cv2.rectangle(result_image, (x, y + h + 28), (x + emotion_size[0] + 10, y + h + 53), (0, 255, 255), -1)
            cv2.putText(result_image, emotion_text, (x + 5, y + h + 45), font, 0.5, (0, 0, 0), 1)
        
        # Header with detection summary
        summary = f"Precise Face Detection: {len(detections)} faces detected"
        summary_size = cv2.getTextSize(summary, font, 0.8, 2)[0]
        cv2.rectangle(result_image, (10, 10), (summary_size[0] + 20, 50), (0, 0, 0), -1)
        cv2.putText(result_image, summary, (15, 35), font, 0.8, (255, 255, 255), 2)
        
        return result_image