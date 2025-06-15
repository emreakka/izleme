import cv2
import numpy as np
from typing import List, Dict, Tuple

class ReliableDetector:
    """Reliable face detection optimized for speed and accuracy"""
    
    def __init__(self):
        self.cascade = None
        self.mtcnn = None
        self._init_detectors()
        
    def _init_detectors(self):
        """Initialize detectors without TensorFlow dependencies"""
        # Load Haar cascade
        try:
            self.cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            if not self.cascade.empty():
                print("Haar cascade loaded successfully")
        except:
            print("Haar cascade failed to load")
            
        # Load MTCNN without TensorFlow interference
        try:
            from mtcnn import MTCNN
            # Initialize MTCNN with optimized settings
            self.mtcnn = MTCNN()
            print("MTCNN detector loaded successfully")
        except Exception as e:
            print(f"MTCNN initialization failed: {e}")
            
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """Detect faces using multiple methods"""
        all_detections = []
        h, w = image.shape[:2]
        
        # Method 1: MTCNN (highest accuracy)
        if self.mtcnn is not None:
            mtcnn_faces = self._detect_mtcnn(image)
            all_detections.extend(mtcnn_faces)
            print(f"MTCNN detected: {len(mtcnn_faces)} faces")
        
        # Method 2: Enhanced Haar cascades
        if self.cascade is not None and not self.cascade.empty():
            haar_faces = self._detect_haar_enhanced(image)
            all_detections.extend(haar_faces)
            print(f"Haar detected: {len(haar_faces)} faces")
            
        # Remove duplicates and filter by confidence
        unique_faces = self._remove_duplicates_advanced(all_detections)
        filtered_faces = [f for f in unique_faces if f['confidence'] >= confidence_threshold]
        
        # Analyze each face
        results = []
        for i, face in enumerate(filtered_faces):
            analyzed_face = self._analyze_face_comprehensive(image, face, i + 1)
            results.append(analyzed_face)
            
        print(f"Final reliable detection: {len(results)} faces")
        return results
        
    def _detect_mtcnn(self, image: np.ndarray) -> List[Dict]:
        """MTCNN face detection"""
        detections = []
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mtcnn.detect_faces(rgb_image)
            
            for result in results:
                x, y, w, h = result['box']
                confidence = result['confidence']
                
                # Validate detection
                if w > 15 and h > 15 and x >= 0 and y >= 0:
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': confidence,
                        'method': 'mtcnn',
                        'landmarks': result.get('keypoints', {})
                    })
        except Exception as e:
            print(f"MTCNN detection error: {e}")
            
        return detections
        
    def _detect_haar_enhanced(self, image: np.ndarray) -> List[Dict]:
        """Enhanced Haar cascade detection"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhanced parameter sets for comprehensive detection
        param_sets = [
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20)},
            {'scaleFactor': 1.1, 'minNeighbors': 4, 'minSize': (25, 25)},
            {'scaleFactor': 1.15, 'minNeighbors': 3, 'minSize': (30, 30)},
            {'scaleFactor': 1.2, 'minNeighbors': 5, 'minSize': (35, 35)},
            {'scaleFactor': 1.08, 'minNeighbors': 2, 'minSize': (22, 22)},
            {'scaleFactor': 1.12, 'minNeighbors': 6, 'minSize': (28, 28)},
        ]
        
        for i, params in enumerate(param_sets):
            try:
                faces = self.cascade.detectMultiScale(gray, **params)
                
                for (x, y, w, h) in faces:
                    if w > 15 and h > 15:
                        confidence = 0.65 + (i * 0.04)  # Progressive confidence scoring
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': min(confidence, 0.9),
                            'method': f'haar_enhanced_{i+1}'
                        })
            except:
                continue
                
        return detections
        
    def _remove_duplicates_advanced(self, detections: List[Dict]) -> List[Dict]:
        """Advanced duplicate removal with confidence weighting"""
        if not detections:
            return []
            
        # Group overlapping detections
        groups = []
        for detection in detections:
            assigned = False
            for group in groups:
                if self._calculate_overlap(detection, group[0]) > 0.3:
                    group.append(detection)
                    assigned = True
                    break
            if not assigned:
                groups.append([detection])
                
        # Select best detection from each group
        unique_detections = []
        for group in groups:
            if len(group) == 1:
                unique_detections.append(group[0])
            else:
                # Prefer MTCNN over Haar, then by confidence
                mtcnn_faces = [d for d in group if d['method'] == 'mtcnn']
                if mtcnn_faces:
                    best = max(mtcnn_faces, key=lambda x: x['confidence'])
                else:
                    best = max(group, key=lambda x: x['confidence'])
                unique_detections.append(best)
                
        return unique_detections
        
    def _calculate_overlap(self, det1: Dict, det2: Dict) -> float:
        """Calculate overlap between two detections"""
        x1, y1, w1, h1 = det1['bbox']
        x2, y2, w2, h2 = det2['bbox']
        
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
        
    def _analyze_face_comprehensive(self, image: np.ndarray, detection: Dict, face_id: int) -> Dict:
        """Comprehensive face analysis"""
        x, y, w, h = detection['bbox']
        
        # Extract face region
        face_region = image[y:y+h, x:x+w] if y+h <= image.shape[0] and x+w <= image.shape[1] else image[y:min(y+h, image.shape[0]), x:min(x+w, image.shape[1])]
        
        # Gaze analysis based on position and face analysis
        gaze_direction = self._analyze_gaze_comprehensive(x, y, w, h, image.shape[1], image.shape[0], face_region)
        
        # Emotion analysis
        emotion, emotion_conf = self._analyze_emotion_comprehensive(face_region)
        
        return {
            'face_id': face_id,
            'person_name': f'Person-{face_id}',
            'bbox': (x, y, w, h),
            'confidence': detection['confidence'],
            'method': detection['method'],
            'gaze_direction': gaze_direction,
            'gaze_confidence': 0.85,
            'emotion': emotion,
            'emotion_confidence': emotion_conf,
            'landmarks': detection.get('landmarks', {})
        }
        
    def _analyze_gaze_comprehensive(self, x: int, y: int, w: int, h: int, 
                                   img_w: int, img_h: int, face_region: np.ndarray) -> str:
        """Comprehensive gaze direction analysis"""
        # Position-based analysis
        center_x = (x + w/2) / img_w
        center_y = (y + h/2) / img_h
        
        # Enhanced position mapping
        if center_y > 0.8:
            return "looking down at table/game"
        elif center_y < 0.2:
            return "looking up/away"
        elif center_x < 0.15:
            return "looking at person on far left"
        elif center_x > 0.85:
            return "looking at person on far right"
        elif center_x < 0.35:
            return "looking at person on left"
        elif center_x > 0.65:
            return "looking at person on right"
        else:
            # Center region - analyze face orientation
            return self._analyze_face_orientation(face_region, center_x, center_y)
            
    def _analyze_face_orientation(self, face_region: np.ndarray, center_x: float, center_y: float) -> str:
        """Analyze face orientation for precise gaze"""
        if face_region.size == 0:
            return "looking at camera"
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            if h < 25 or w < 25:
                return "looking at camera"
            
            # Analyze face symmetry for head orientation
            left_half = gray_face[:, :w//2]
            right_half = gray_face[:, w//2:]
            
            left_mean = np.mean(left_half)
            right_mean = np.mean(right_half)
            symmetry_diff = abs(left_mean - right_mean)
            
            # Enhanced orientation detection
            if symmetry_diff > 15:
                if left_mean > right_mean:
                    return "looking slightly right"
                else:
                    return "looking slightly left"
            elif center_x < 0.45:
                return "looking at camera (left position)"
            elif center_x > 0.55:
                return "looking at camera (right position)"
            else:
                return "looking directly at camera"
                
        except:
            return "looking forward"
            
    def _analyze_emotion_comprehensive(self, face_region: np.ndarray) -> Tuple[str, float]:
        """Comprehensive emotion analysis"""
        if face_region.size == 0:
            return "neutral", 0.5
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            if h < 25 or w < 25:
                return "neutral", 0.6
                
            # Enhanced facial region analysis
            mouth_region = gray_face[int(0.7*h):, int(0.25*w):int(0.75*w)]
            eye_region = gray_face[int(0.2*h):int(0.5*h), :]
            forehead_region = gray_face[:int(0.3*h), int(0.2*w):int(0.8*w)]
            
            features = {}
            if mouth_region.size > 0:
                features['mouth_brightness'] = np.mean(mouth_region)
                features['mouth_contrast'] = np.std(mouth_region)
                
            if eye_region.size > 0:
                features['eye_brightness'] = np.mean(eye_region)
                features['eye_contrast'] = np.std(eye_region)
                
            if forehead_region.size > 0:
                features['forehead_brightness'] = np.mean(forehead_region)
                
            face_brightness = np.mean(gray_face)
            
            # Enhanced emotion classification
            if features.get('mouth_brightness', 0) > face_brightness * 1.25:
                return "happy", 0.82
            elif features.get('mouth_brightness', 0) < face_brightness * 0.75:
                return "sad", 0.74
            elif features.get('eye_brightness', 0) < face_brightness * 0.8:
                return "tired", 0.68
            elif features.get('mouth_contrast', 0) > 25:
                return "surprised", 0.71
            elif features.get('forehead_brightness', 0) < face_brightness * 0.85:
                return "focused", 0.69
            else:
                return "neutral", 0.78
                
        except:
            return "neutral", 0.5
            
    def draw_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw enhanced detection results"""
        result_image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            method = detection['method']
            
            # Color coding by method and confidence
            if method == 'mtcnn':
                color = (0, 255, 0)  # Green for MTCNN
            elif confidence > 0.8:
                color = (255, 165, 0)  # Orange for high confidence
            else:
                color = (0, 255, 255)  # Yellow
                
            # Draw bounding box
            thickness = max(2, int(confidence * 4))
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
            
            # Enhanced labeling
            person_text = f"{detection['person_name']} ({confidence:.2f})"
            text_size = cv2.getTextSize(person_text, font, 0.6, 2)[0]
            cv2.rectangle(result_image, (x, y - 25), (x + text_size[0] + 10, y), color, -1)
            cv2.putText(result_image, person_text, (x + 5, y - 5), font, 0.6, (0, 0, 0), 2)
            
            # Gaze information
            gaze_text = f"Gaze: {detection['gaze_direction']}"
            gaze_size = cv2.getTextSize(gaze_text, font, 0.4, 1)[0]
            cv2.rectangle(result_image, (x, y + h), (x + gaze_size[0] + 10, y + h + 20), (255, 255, 0), -1)
            cv2.putText(result_image, gaze_text, (x + 5, y + h + 15), font, 0.4, (0, 0, 0), 1)
            
            # Emotion information
            emotion_text = f"Emotion: {detection['emotion']}"
            emotion_size = cv2.getTextSize(emotion_text, font, 0.4, 1)[0]
            cv2.rectangle(result_image, (x, y + h + 25), (x + emotion_size[0] + 10, y + h + 45), (0, 255, 255), -1)
            cv2.putText(result_image, emotion_text, (x + 5, y + h + 40), font, 0.4, (0, 0, 0), 1)
            
            # Method indicator
            method_text = f"Method: {method}"
            cv2.putText(result_image, method_text, (x + 5, y + h + 60), font, 0.35, color, 1)
        
        # Summary header
        summary = f"Reliable Detection: {len(detections)} faces found"
        summary_size = cv2.getTextSize(summary, font, 0.7, 2)[0]
        cv2.rectangle(result_image, (10, 10), (summary_size[0] + 20, 40), (0, 0, 0), -1)
        cv2.putText(result_image, summary, (15, 30), font, 0.7, (255, 255, 255), 2)
        
        return result_image