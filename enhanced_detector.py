import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

class EnhancedDetector:
    """Enhanced face detection without heavy dependencies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detectors = []
        self.init_detectors()
        
    def init_detectors(self):
        """Initialize lightweight detectors"""
        # Haar cascade with multiple configurations
        try:
            cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            if not cascade.empty():
                self.detectors.append(('haar', cascade))
                print("Haar cascade loaded successfully")
        except:
            pass
            
        # MediaPipe with error handling
        try:
            import mediapipe as mp
            mp_face = mp.solutions.face_detection
            
            short_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.1)
            long_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.1)
            
            self.detectors.extend([
                ('mp_short', short_detector),
                ('mp_long', long_detector)
            ])
            print("MediaPipe detectors loaded successfully")
        except Exception as e:
            print(f"MediaPipe not available: {e}")
            
        print(f"Initialized {len(self.detectors)} detectors")
    
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """Detect faces using all available methods"""
        all_detections = []
        h, w = image.shape[:2]
        
        for detector_name, detector in self.detectors:
            try:
                if detector_name == 'haar':
                    detections = self._detect_haar(image, detector)
                else:
                    detections = self._detect_mediapipe(image, detector, detector_name)
                
                for det in detections:
                    det['method'] = detector_name
                all_detections.extend(detections)
                
            except Exception as e:
                print(f"Error with {detector_name}: {e}")
                continue
        
        # Remove duplicates and create final results
        unique_detections = self._remove_duplicates(all_detections)
        filtered_detections = [d for d in unique_detections if d['confidence'] >= confidence_threshold]
        
        # Analyze each face
        results = []
        for i, detection in enumerate(filtered_detections):
            result = self._analyze_face(image, detection, i + 1)
            results.append(result)
            
        print(f"Final detection: {len(results)} faces")
        return results
    
    def _detect_haar(self, image: np.ndarray, cascade) -> List[Dict]:
        """Haar cascade detection with multiple parameter sets"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple parameter combinations for better coverage
        params = [
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20)},
            {'scaleFactor': 1.1, 'minNeighbors': 4, 'minSize': (30, 30)},
            {'scaleFactor': 1.2, 'minNeighbors': 3, 'minSize': (25, 25)},
            {'scaleFactor': 1.15, 'minNeighbors': 5, 'minSize': (35, 35)},
        ]
        
        for param_set in params:
            faces = cascade.detectMultiScale(gray, **param_set)
            for (x, y, w, h) in faces:
                if self._is_valid_face(x, y, w, h, image.shape[1], image.shape[0]):
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': 0.75,
                        'source': f"haar_{param_set['scaleFactor']}"
                    })
        
        return detections
    
    def _detect_mediapipe(self, image: np.ndarray, detector, name: str) -> List[Dict]:
        """MediaPipe face detection"""
        detections = []
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        results = detector.process(rgb_image)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                fw = int(bbox.width * w)
                fh = int(bbox.height * h)
                
                # Ensure coordinates are within bounds
                x = max(0, x)
                y = max(0, y)
                fw = min(w - x, fw)
                fh = min(h - y, fh)
                
                if self._is_valid_face(x, y, fw, fh, w, h):
                    confidence = detection.score[0] if detection.score else 0.8
                    detections.append({
                        'bbox': (x, y, fw, fh),
                        'confidence': confidence,
                        'source': name
                    })
        
        return detections
    
    def _is_valid_face(self, x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> bool:
        """Validate face detection"""
        # Size check
        if w < 20 or h < 20:
            return False
            
        # Aspect ratio check
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
            
        # Boundary check
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return False
            
        return True
    
    def _remove_duplicates(self, detections: List[Dict]) -> List[Dict]:
        """Remove duplicate detections using overlap analysis"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        unique = []
        for detection in detections:
            x, y, w, h = detection['bbox']
            is_duplicate = False
            
            for existing in unique:
                ex, ey, ew, eh = existing['bbox']
                
                # Calculate overlap
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y
                
                area1 = w * h
                area2 = ew * eh
                
                if overlap_area > 0.3 * min(area1, area2):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(detection)
        
        return unique
    
    def _analyze_face(self, image: np.ndarray, detection: Dict, face_id: int) -> Dict:
        """Analyze individual face for gaze and emotion"""
        x, y, w, h = detection['bbox']
        
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        
        # Analyze gaze direction
        gaze_direction = self._analyze_gaze_position(x, y, w, h, image.shape[1], image.shape[0], face_region)
        
        # Analyze emotion
        emotion, emotion_conf = self._analyze_emotion_features(face_region)
        
        return {
            'face_id': face_id,
            'person_name': f'Person-{face_id}',
            'bbox': (x, y, w, h),
            'confidence': detection['confidence'],
            'method': detection['method'],
            'gaze_direction': gaze_direction,
            'gaze_confidence': 0.8,
            'emotion': emotion,
            'emotion_confidence': emotion_conf
        }
    
    def _analyze_gaze_position(self, x: int, y: int, w: int, h: int, 
                              img_w: int, img_h: int, face_region: np.ndarray) -> str:
        """Analyze gaze direction based on position and face orientation"""
        center_x = (x + w/2) / img_w
        center_y = (y + h/2) / img_h
        
        # Position-based gaze analysis
        if center_y > 0.75:
            return "looking down at table/game"
        elif center_y < 0.25:
            return "looking up/away"
        elif center_x < 0.2:
            return "looking at person on far left"
        elif center_x > 0.8:
            return "looking at person on far right"
        elif center_x < 0.4:
            return "looking at person on left"
        elif center_x > 0.6:
            return "looking at person on right"
        else:
            # Center region - analyze face orientation
            return self._analyze_face_direction(face_region)
    
    def _analyze_face_direction(self, face_region: np.ndarray) -> str:
        """Analyze face direction for center positioned faces"""
        if face_region.size == 0:
            return "looking straight ahead"
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            if h < 30 or w < 30:
                return "looking at camera"
            
            # Analyze face symmetry
            left_half = gray_face[:, :w//2]
            right_half = gray_face[:, w//2:]
            
            left_mean = np.mean(left_half)
            right_mean = np.mean(right_half)
            
            diff = abs(left_mean - right_mean)
            
            if diff > 10:
                if left_mean > right_mean:
                    return "looking slightly right"
                else:
                    return "looking slightly left"
            else:
                return "looking directly at camera"
                
        except:
            return "looking forward"
    
    def _analyze_emotion_features(self, face_region: np.ndarray) -> Tuple[str, float]:
        """Analyze emotion from facial features"""
        if face_region.size == 0:
            return "neutral", 0.5
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            if h < 30 or w < 30:
                return "neutral", 0.6
            
            # Analyze different facial regions
            mouth_region = gray_face[2*h//3:, w//4:3*w//4]
            eye_region = gray_face[h//4:h//2, :]
            
            if mouth_region.size > 0 and eye_region.size > 0:
                mouth_brightness = np.mean(mouth_region)
                eye_brightness = np.mean(eye_region)
                face_brightness = np.mean(gray_face)
                
                mouth_ratio = mouth_brightness / face_brightness
                eye_ratio = eye_brightness / face_brightness
                
                # Enhanced emotion detection
                if mouth_ratio > 1.25:
                    return "happy", 0.8
                elif mouth_ratio < 0.75:
                    return "sad", 0.7
                elif eye_ratio < 0.85:
                    return "tired", 0.6
                elif mouth_ratio > 1.15 and abs(mouth_ratio - eye_ratio) > 0.2:
                    return "surprised", 0.65
                else:
                    return "neutral", 0.75
            else:
                return "neutral", 0.5
                
        except:
            return "neutral", 0.5
    
    def draw_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results on image"""
        result_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            method = detection['method']
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw face box
            thickness = max(2, int(confidence * 5))
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
            
            # Draw labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Person name
            person_text = f"{detection['person_name']} ({confidence:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(person_text, font, 0.7, 2)
            cv2.rectangle(result_image, (x, y - text_h - 15), (x + text_w + 10, y), color, -1)
            cv2.putText(result_image, person_text, (x + 5, y - 5), font, 0.7, (0, 0, 0), 2)
            
            # Gaze direction
            gaze_text = f"Gaze: {detection['gaze_direction']}"
            (gaze_w, gaze_h), _ = cv2.getTextSize(gaze_text, font, 0.5, 2)
            cv2.rectangle(result_image, (x, y + h + 5), (x + gaze_w + 10, y + h + gaze_h + 15), (255, 255, 0), -1)
            cv2.putText(result_image, gaze_text, (x + 5, y + h + gaze_h + 10), font, 0.5, (0, 0, 0), 2)
            
            # Emotion
            emotion_text = f"Emotion: {detection['emotion']} ({detection['emotion_confidence']:.2f})"
            (emo_w, emo_h), _ = cv2.getTextSize(emotion_text, font, 0.45, 1)
            cv2.rectangle(result_image, (x, y + h + gaze_h + 20), 
                        (x + emo_w + 10, y + h + gaze_h + emo_h + 30), (0, 255, 255), -1)
            cv2.putText(result_image, emotion_text, (x + 5, y + h + gaze_h + emo_h + 25), 
                      font, 0.45, (0, 0, 0), 1)
            
            # Method info
            method_text = f"Method: {method}"
            cv2.putText(result_image, method_text, (x + 5, y + h + gaze_h + emo_h + 45), 
                      font, 0.4, color, 1)
        
        # Summary
        summary_text = f"Enhanced Detection: {len(detections)} faces found"
        cv2.rectangle(result_image, (10, 10), (len(summary_text) * 12 + 20, 40), (0, 0, 0), -1)
        cv2.putText(result_image, summary_text, (15, 30), font, 0.7, (255, 255, 255), 2)
        
        return result_image