import cv2
import numpy as np
from typing import List, Dict, Tuple

class FastDetector:
    """Fast and reliable face detection without heavy dependencies"""
    
    def __init__(self):
        self.cascade = None
        self.mp_detectors = []
        self._init_detectors()
        
    def _init_detectors(self):
        """Initialize detectors quickly"""
        # Load Haar cascade
        try:
            self.cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            print(f"Haar cascade loaded: {not self.cascade.empty()}")
        except:
            print("Haar cascade failed to load")
            
        # Load MediaPipe if available
        try:
            import mediapipe as mp
            face_detection = mp.solutions.face_detection
            
            self.mp_detectors = [
                face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.1),
                face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1)
            ]
            print(f"MediaPipe loaded: 2 detectors")
        except:
            print("MediaPipe not available")
            
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """Fast face detection"""
        detections = []
        h, w = image.shape[:2]
        
        # Haar cascade detection
        if self.cascade and not self.cascade.empty():
            haar_faces = self._detect_haar_fast(image)
            detections.extend(haar_faces)
            
        # MediaPipe detection
        for i, detector in enumerate(self.mp_detectors):
            try:
                mp_faces = self._detect_mediapipe_fast(image, detector, f'mp_{i}')
                detections.extend(mp_faces)
            except:
                continue
                
        # Remove duplicates
        unique_faces = self._remove_duplicates_fast(detections)
        
        # Filter by confidence
        filtered_faces = [f for f in unique_faces if f['confidence'] >= confidence_threshold]
        
        # Analyze faces
        results = []
        for i, face in enumerate(filtered_faces):
            analyzed_face = self._analyze_face_fast(image, face, i + 1)
            results.append(analyzed_face)
            
        return results
        
    def _detect_haar_fast(self, image: np.ndarray) -> List[Dict]:
        """Fast Haar cascade detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Quick parameter sets
        param_sets = [
            {'scaleFactor': 1.1, 'minNeighbors': 3, 'minSize': (30, 30)},
            {'scaleFactor': 1.2, 'minNeighbors': 4, 'minSize': (25, 25)},
        ]
        
        detections = []
        for params in param_sets:
            faces = self.cascade.detectMultiScale(gray, **params)
            for (x, y, w, h) in faces:
                if w > 20 and h > 20:
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': 0.7,
                        'method': 'haar'
                    })
                    
        return detections
        
    def _detect_mediapipe_fast(self, image: np.ndarray, detector, method_name: str) -> List[Dict]:
        """Fast MediaPipe detection"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        results = detector.process(rgb_image)
        detections = []
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                fw = int(bbox.width * w)
                fh = int(bbox.height * h)
                
                # Boundary check
                x = max(0, x)
                y = max(0, y)
                fw = min(w - x, fw)
                fh = min(h - y, fh)
                
                if fw > 20 and fh > 20:
                    confidence = detection.score[0] if detection.score else 0.8
                    detections.append({
                        'bbox': (x, y, fw, fh),
                        'confidence': confidence,
                        'method': method_name
                    })
                    
        return detections
        
    def _remove_duplicates_fast(self, detections: List[Dict]) -> List[Dict]:
        """Fast duplicate removal"""
        if not detections:
            return []
            
        unique = []
        for detection in detections:
            x, y, w, h = detection['bbox']
            is_duplicate = False
            
            for existing in unique:
                ex, ey, ew, eh = existing['bbox']
                
                # Simple overlap check
                center_dist = ((x + w/2) - (ex + ew/2))**2 + ((y + h/2) - (ey + eh/2))**2
                min_size = min(w, h, ew, eh)
                
                if center_dist < (min_size * 0.7)**2:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique.append(detection)
                
        return unique
        
    def _analyze_face_fast(self, image: np.ndarray, detection: Dict, face_id: int) -> Dict:
        """Fast face analysis"""
        x, y, w, h = detection['bbox']
        
        # Quick gaze analysis based on position
        img_h, img_w = image.shape[:2]
        center_x = (x + w/2) / img_w
        center_y = (y + h/2) / img_h
        
        if center_y > 0.75:
            gaze = "looking down at table/game"
        elif center_y < 0.25:
            gaze = "looking up/away"
        elif center_x < 0.25:
            gaze = "looking at person on far left"
        elif center_x > 0.75:
            gaze = "looking at person on far right"
        elif center_x < 0.4:
            gaze = "looking at person on left"
        elif center_x > 0.6:
            gaze = "looking at person on right"
        else:
            gaze = "looking at camera"
            
        # Quick emotion analysis
        try:
            face_region = image[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            if gray_face.size > 0:
                brightness = np.mean(gray_face)
                if brightness > 140:
                    emotion = "happy"
                    emotion_conf = 0.7
                elif brightness < 100:
                    emotion = "sad"
                    emotion_conf = 0.6
                else:
                    emotion = "neutral"
                    emotion_conf = 0.8
            else:
                emotion = "neutral"
                emotion_conf = 0.5
        except:
            emotion = "neutral"
            emotion_conf = 0.5
            
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
        """Draw results quickly"""
        result_image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # Color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green
            else:
                color = (0, 255, 255)  # Yellow
                
            # Draw face box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
            
            # Person name
            person_text = f"{detection['person_name']} ({confidence:.2f})"
            cv2.rectangle(result_image, (x, y - 25), (x + len(person_text) * 12, y), color, -1)
            cv2.putText(result_image, person_text, (x + 5, y - 5), font, 0.6, (0, 0, 0), 2)
            
            # Gaze
            gaze_text = f"Gaze: {detection['gaze_direction']}"
            cv2.rectangle(result_image, (x, y + h), (x + len(gaze_text) * 8, y + h + 20), (255, 255, 0), -1)
            cv2.putText(result_image, gaze_text, (x + 5, y + h + 15), font, 0.4, (0, 0, 0), 1)
            
            # Emotion
            emotion_text = f"Emotion: {detection['emotion']}"
            cv2.rectangle(result_image, (x, y + h + 25), (x + len(emotion_text) * 8, y + h + 45), (0, 255, 255), -1)
            cv2.putText(result_image, emotion_text, (x + 5, y + h + 40), font, 0.4, (0, 0, 0), 1)
            
        # Summary
        summary = f"Fast Detection: {len(detections)} faces found"
        cv2.rectangle(result_image, (10, 10), (len(summary) * 10 + 20, 35), (0, 0, 0), -1)
        cv2.putText(result_image, summary, (15, 25), font, 0.6, (255, 255, 255), 2)
        
        return result_image