import cv2
import numpy as np
from typing import List, Dict, Tuple

class StableFaceDetector:
    """Stable face detection using OpenCV Haar cascades without heavy dependencies"""
    
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
        """Stable face detection with comprehensive analysis"""
        if image is None or image.size == 0:
            return []
            
        if self.cascade is None or self.cascade.empty():
            print("Error: Haar cascade not available")
            return []
            
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple detection passes with different parameters
        all_detections = []
        
        # Parameter sets optimized for different scenarios
        param_sets = [
            # Standard detection
            {'scaleFactor': 1.1, 'minNeighbors': 4, 'minSize': (30, 30), 'maxSize': (300, 300)},
            # Small faces
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20), 'maxSize': (150, 150)},
            # Large faces  
            {'scaleFactor': 1.2, 'minNeighbors': 5, 'minSize': (50, 50), 'maxSize': (400, 400)},
            # Profile/side faces
            {'scaleFactor': 1.15, 'minNeighbors': 3, 'minSize': (25, 25), 'maxSize': (200, 200)},
            # Very small/distant faces
            {'scaleFactor': 1.08, 'minNeighbors': 2, 'minSize': (15, 15), 'maxSize': (100, 100)},
            # High sensitivity
            {'scaleFactor': 1.12, 'minNeighbors': 6, 'minSize': (28, 28), 'maxSize': (250, 250)},
            # Low sensitivity for clear faces
            {'scaleFactor': 1.25, 'minNeighbors': 7, 'minSize': (40, 40), 'maxSize': (350, 350)},
        ]
        
        for i, params in enumerate(param_sets):
            try:
                faces = self.cascade.detectMultiScale(gray, **params)
                print(f"Detection pass {i+1}: Found {len(faces)} faces")
                
                for (x, y, width, height) in faces:
                    # Validate detection
                    if self._is_valid_face(x, y, width, height, w, h):
                        confidence = 0.6 + (i * 0.04)  # Progressive confidence scoring
                        all_detections.append({
                            'bbox': (x, y, width, height),
                            'confidence': min(confidence, 0.95),
                            'method': f'haar_pass_{i+1}'
                        })
            except Exception as e:
                print(f"Error in detection pass {i+1}: {e}")
                continue
                
        print(f"Total raw detections: {len(all_detections)}")
        
        # Remove duplicates efficiently
        unique_faces = self._remove_duplicates(all_detections)
        filtered_faces = [f for f in unique_faces if f['confidence'] >= confidence_threshold]
        
        # Analyze each detected face
        results = []
        for i, face in enumerate(filtered_faces):
            analyzed_face = self._analyze_face(image, face, i + 1)
            results.append(analyzed_face)
            
        print(f"Final stable detection: {len(results)} faces")
        return results
        
    def _is_valid_face(self, x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> bool:
        """Validate face detection parameters"""
        # Size validation
        if w < 15 or h < 15 or w > img_w * 0.8 or h > img_h * 0.8:
            return False
            
        # Position validation
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return False
            
        # Aspect ratio validation
        aspect_ratio = w / h
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            return False
            
        return True
        
    def _remove_duplicates(self, detections: List[Dict]) -> List[Dict]:
        """Remove duplicate detections efficiently"""
        if not detections:
            return []
            
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        unique_detections = []
        for detection in sorted_detections:
            is_duplicate = False
            
            for existing in unique_detections:
                if self._calculate_overlap(detection['bbox'], existing['bbox']) > 0.4:
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
            'gaze_confidence': 0.80,
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
        if center_y > 0.75:
            return "looking down at table/game"
        elif center_y < 0.25:
            return "looking up/away"
            
        # Horizontal gaze analysis with meaningful location descriptions
        if center_x < 0.1:
            return "looking at person on far left"
        elif center_x > 0.9:
            return "looking at person on far right"
        elif center_x < 0.3:
            return "looking at person on left"
        elif center_x > 0.7:
            return "looking at person on right"
        elif center_x < 0.45:
            return "looking slightly left"
        elif center_x > 0.55:
            return "looking slightly right"
        else:
            # Center region - analyze face orientation for more precise gaze
            return self._analyze_face_orientation(face_region, center_x, center_y)
            
    def _analyze_face_orientation(self, face_region: np.ndarray, center_x: float, center_y: float) -> str:
        """Analyze face orientation for center-positioned faces"""
        if face_region.size == 0 or face_region.shape[0] < 20 or face_region.shape[1] < 20:
            return "looking at camera"
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            # Analyze left vs right half brightness (simple head turn detection)
            left_half = gray_face[:, :w//2]
            right_half = gray_face[:, w//2:]
            
            left_mean = np.mean(left_half)
            right_mean = np.mean(right_half)
            brightness_diff = abs(left_mean - right_mean)
            
            # Eye region analysis for better gaze estimation
            if h > 30 and w > 30:
                eye_region = gray_face[int(0.2*h):int(0.5*h), :]
                if eye_region.size > 0:
                    eye_left = eye_region[:, :w//2]
                    eye_right = eye_region[:, w//2:]
                    eye_brightness_diff = np.mean(eye_left) - np.mean(eye_right)
                    
                    # Enhanced gaze detection
                    if abs(eye_brightness_diff) > 8:
                        if eye_brightness_diff > 0:
                            return "looking slightly right"
                        else:
                            return "looking slightly left"
                            
            # Face symmetry analysis
            if brightness_diff > 15:
                if left_mean > right_mean:
                    return "looking slightly right"
                else:
                    return "looking slightly left"
                    
            # Default for center position
            if 0.4 <= center_x <= 0.6:
                return "looking directly at camera"
            else:
                return "looking at camera"
                
        except Exception as e:
            return "looking forward"
            
    def _analyze_emotion(self, face_region: np.ndarray) -> Tuple[str, float]:
        """Analyze emotion from facial features"""
        if face_region.size == 0 or face_region.shape[0] < 25 or face_region.shape[1] < 25:
            return "neutral", 0.65
            
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
                
            # Emotion classification based on facial regions
            mouth_bright = mouth_features.get('brightness', face_mean)
            mouth_contrast = mouth_features.get('contrast', 0)
            eye_bright = eye_features.get('brightness', face_mean)
            eye_var = eye_features.get('variance', 0)
            
            # Happy: bright mouth region with good contrast (smile)
            if mouth_bright > face_mean * 1.15 and mouth_contrast > 18:
                return "happy", 0.83
                
            # Sad: darker mouth and eye regions
            elif mouth_bright < face_mean * 0.85 and eye_bright < face_mean * 0.9:
                return "sad", 0.76
                
            # Surprised: high contrast in mouth area, bright eyes
            elif mouth_contrast > 25 and eye_bright > face_mean * 1.05:
                return "surprised", 0.72
                
            # Focused/concentrated: stable features, slightly darker eyes
            elif eye_bright < face_mean * 0.95 and mouth_contrast < 15:
                return "focused", 0.70
                
            # Tired: low variance in eye region
            elif eye_var < 80 and mouth_contrast < 12:
                return "tired", 0.68
                
            else:
                return "neutral", 0.75
                
        except Exception as e:
            return "neutral", 0.6
            
    def draw_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results on image"""
        result_image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            method = detection['method']
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.6:
                color = (255, 165, 0)  # Orange for medium confidence
            else:
                color = (0, 255, 255)  # Yellow for lower confidence
                
            # Dynamic thickness based on confidence
            thickness = max(2, int(confidence * 4))
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
            
            # Person label with confidence
            person_text = f"{detection['person_name']} ({confidence:.2f})"
            text_size = cv2.getTextSize(person_text, font, 0.6, 2)[0]
            cv2.rectangle(result_image, (x, y - 25), (x + text_size[0] + 10, y), color, -1)
            cv2.putText(result_image, person_text, (x + 5, y - 5), font, 0.6, (0, 0, 0), 2)
            
            # Gaze direction
            gaze_text = f"Gaze: {detection['gaze_direction']}"
            gaze_size = cv2.getTextSize(gaze_text, font, 0.45, 1)[0]
            cv2.rectangle(result_image, (x, y + h), (x + gaze_size[0] + 10, y + h + 22), (255, 255, 0), -1)
            cv2.putText(result_image, gaze_text, (x + 5, y + h + 16), font, 0.45, (0, 0, 0), 1)
            
            # Emotion
            emotion_text = f"Emotion: {detection['emotion']}"
            emotion_size = cv2.getTextSize(emotion_text, font, 0.45, 1)[0]
            cv2.rectangle(result_image, (x, y + h + 25), (x + emotion_size[0] + 10, y + h + 47), (0, 255, 255), -1)
            cv2.putText(result_image, emotion_text, (x + 5, y + h + 41), font, 0.45, (0, 0, 0), 1)
            
            # Method indicator (smaller text)
            method_text = f"Method: {method}"
            cv2.putText(result_image, method_text, (x + 5, y + h + 65), font, 0.35, color, 1)
        
        # Header summary
        summary = f"Stable Face Detection: {len(detections)} faces detected"
        summary_size = cv2.getTextSize(summary, font, 0.7, 2)[0]
        cv2.rectangle(result_image, (10, 10), (summary_size[0] + 20, 45), (0, 0, 0), -1)
        cv2.putText(result_image, summary, (15, 32), font, 0.7, (255, 255, 255), 2)
        
        return result_image