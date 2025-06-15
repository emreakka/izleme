import cv2
import numpy as np
from typing import List, Dict, Tuple

class SimpleDetector:
    """Simple, fast face detection using only OpenCV"""
    
    def __init__(self):
        self.cascade = None
        self._load_cascade()
        
    def _load_cascade(self):
        """Load Haar cascade detector"""
        try:
            self.cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            if self.cascade.empty():
                print("ERROR: Haar cascade file is empty or corrupted")
            else:
                print("Haar cascade loaded successfully")
        except Exception as e:
            print(f"ERROR loading Haar cascade: {e}")
            
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """Detect faces using OpenCV Haar cascades"""
        if self.cascade is None or self.cascade.empty():
            print("No cascade detector available")
            return []
            
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        
        # Multiple detection passes with different parameters
        detection_params = [
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20), 'maxSize': (300, 300)},
            {'scaleFactor': 1.1, 'minNeighbors': 4, 'minSize': (30, 30), 'maxSize': (250, 250)},
            {'scaleFactor': 1.15, 'minNeighbors': 3, 'minSize': (25, 25), 'maxSize': (200, 200)},
            {'scaleFactor': 1.2, 'minNeighbors': 5, 'minSize': (35, 35), 'maxSize': (180, 180)},
            {'scaleFactor': 1.3, 'minNeighbors': 2, 'minSize': (40, 40), 'maxSize': (150, 150)},
        ]
        
        all_faces = []
        for i, params in enumerate(detection_params):
            try:
                faces = self.cascade.detectMultiScale(gray, **params)
                print(f"Pass {i+1}: Found {len(faces)} faces with params {params}")
                
                for (x, y, w, h) in faces:
                    # Validate detection
                    if self._is_valid_face(x, y, w, h, image.shape[1], image.shape[0]):
                        all_faces.append({
                            'bbox': (x, y, w, h),
                            'confidence': 0.7 + (i * 0.05),  # Slightly different confidence per method
                            'method': f'haar_pass_{i+1}'
                        })
            except Exception as e:
                print(f"Error in detection pass {i+1}: {e}")
                continue
        
        print(f"Total raw detections: {len(all_faces)}")
        
        # Remove duplicates
        unique_faces = self._remove_duplicates(all_faces)
        print(f"After duplicate removal: {len(unique_faces)}")
        
        # Filter by confidence
        filtered_faces = [f for f in unique_faces if f['confidence'] >= confidence_threshold]
        print(f"After confidence filter: {len(filtered_faces)}")
        
        # Analyze each face
        results = []
        for i, face in enumerate(filtered_faces):
            analyzed_face = self._analyze_face(image, face, i + 1)
            results.append(analyzed_face)
            
        print(f"Final results: {len(results)} faces")
        return results
        
    def _is_valid_face(self, x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> bool:
        """Validate face detection"""
        # Size checks
        if w < 15 or h < 15:
            return False
        if w > img_w * 0.8 or h > img_h * 0.8:
            return False
            
        # Aspect ratio check
        aspect_ratio = w / h
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            return False
            
        # Boundary check
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return False
            
        return True
        
    def _remove_duplicates(self, detections: List[Dict]) -> List[Dict]:
        """Remove duplicate detections"""
        if not detections:
            return []
            
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        unique = []
        for detection in detections:
            x, y, w, h = detection['bbox']
            is_duplicate = False
            
            for existing in unique:
                ex, ey, ew, eh = existing['bbox']
                
                # Calculate center distance
                center1_x, center1_y = x + w/2, y + h/2
                center2_x, center2_y = ex + ew/2, ey + eh/2
                
                distance = ((center1_x - center2_x)**2 + (center1_y - center2_y)**2)**0.5
                min_size = min(w, h, ew, eh)
                
                # If centers are close, it's likely the same face
                if distance < min_size * 0.6:
                    is_duplicate = True
                    break
                    
                # Also check overlap
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y
                
                area1 = w * h
                area2 = ew * eh
                min_area = min(area1, area2)
                
                if overlap_area > 0.4 * min_area:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(detection)
                
        return unique
        
    def _analyze_face(self, image: np.ndarray, detection: Dict, face_id: int) -> Dict:
        """Analyze individual face"""
        x, y, w, h = detection['bbox']
        
        # Gaze analysis based on position
        img_h, img_w = image.shape[:2]
        center_x = (x + w/2) / img_w
        center_y = (y + h/2) / img_h
        
        if center_y > 0.75:
            gaze = "looking down at table"
        elif center_y < 0.25:
            gaze = "looking up"
        elif center_x < 0.2:
            gaze = "looking at person on far left"
        elif center_x > 0.8:
            gaze = "looking at person on far right"
        elif center_x < 0.4:
            gaze = "looking at person on left"
        elif center_x > 0.6:
            gaze = "looking at person on right"
        else:
            gaze = "looking at camera"
            
        # Simple emotion analysis
        emotion = "neutral"
        emotion_conf = 0.6
        
        try:
            face_region = image[y:y+h, x:x+w]
            if face_region.size > 0:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray_face)
                
                if brightness > 130:
                    emotion = "happy"
                    emotion_conf = 0.7
                elif brightness < 90:
                    emotion = "serious"
                    emotion_conf = 0.6
        except:
            pass
            
        return {
            'face_id': face_id,
            'person_name': f'Person-{face_id}',
            'bbox': (x, y, w, h),
            'confidence': detection['confidence'],
            'method': detection['method'],
            'gaze_direction': gaze,
            'gaze_confidence': 0.8,
            'emotion': emotion,
            'emotion_confidence': emotion_conf
        }
        
    def draw_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results"""
        result_image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
            
            # Person name with background
            person_text = f"{detection['person_name']} ({confidence:.2f})"
            text_size = cv2.getTextSize(person_text, font, 0.7, 2)[0]
            cv2.rectangle(result_image, (x, y - 30), (x + text_size[0] + 10, y), color, -1)
            cv2.putText(result_image, person_text, (x + 5, y - 8), font, 0.7, (0, 0, 0), 2)
            
            # Gaze direction
            gaze_text = f"Gaze: {detection['gaze_direction']}"
            gaze_size = cv2.getTextSize(gaze_text, font, 0.5, 2)[0]
            cv2.rectangle(result_image, (x, y + h), (x + gaze_size[0] + 10, y + h + 25), (255, 255, 0), -1)
            cv2.putText(result_image, gaze_text, (x + 5, y + h + 18), font, 0.5, (0, 0, 0), 2)
            
            # Emotion
            emotion_text = f"Emotion: {detection['emotion']}"
            emotion_size = cv2.getTextSize(emotion_text, font, 0.5, 2)[0]
            cv2.rectangle(result_image, (x, y + h + 30), (x + emotion_size[0] + 10, y + h + 55), (0, 255, 255), -1)
            cv2.putText(result_image, emotion_text, (x + 5, y + h + 48), font, 0.5, (0, 0, 0), 2)
        
        # Summary
        summary = f"Simple Detection: {len(detections)} faces found"
        summary_size = cv2.getTextSize(summary, font, 0.8, 2)[0]
        cv2.rectangle(result_image, (10, 10), (summary_size[0] + 20, 45), (0, 0, 0), -1)
        cv2.putText(result_image, summary, (15, 35), font, 0.8, (255, 255, 255), 2)
        
        return result_image