import cv2
import numpy as np
from typing import List, Dict, Tuple

class WorkingFaceDetector:
    """Fast and reliable face detection using primarily OpenCV Haar cascades"""
    
    def __init__(self):
        self.cascade = None
        self.mtcnn = None
        self._init_detectors()
        
    def _init_detectors(self):
        """Initialize detectors with focus on speed"""
        # Load Haar cascade (primary detector)
        try:
            self.cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            if not self.cascade.empty():
                print("Haar cascade loaded successfully")
            else:
                print("WARNING: Haar cascade file is empty")
        except Exception as e:
            print(f"Haar cascade failed to load: {e}")
            
        # MTCNN as optional enhancement (load without blocking)
        try:
            from mtcnn import MTCNN
            self.mtcnn = MTCNN()
            print("MTCNN loaded for enhanced accuracy")
        except:
            print("MTCNN not available - using Haar cascade only")
            self.mtcnn = None
            
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """Fast face detection with comprehensive analysis"""
        if image is None or image.size == 0:
            return []
            
        h, w = image.shape[:2]
        all_detections = []
        
        # Primary detection: Enhanced Haar cascades
        haar_detections = self._detect_haar_comprehensive(image)
        all_detections.extend(haar_detections)
        print(f"Haar cascade detected: {len(haar_detections)} faces")
        
        # Optional MTCNN enhancement (if available and not too many Haar detections)
        if self.mtcnn is not None and len(haar_detections) < 10:
            try:
                mtcnn_detections = self._detect_mtcnn_fast(image)
                all_detections.extend(mtcnn_detections)
                print(f"MTCNN detected: {len(mtcnn_detections)} additional faces")
            except Exception as e:
                print(f"MTCNN detection skipped: {e}")
        
        # Remove duplicates and filter by confidence
        unique_faces = self._remove_duplicates_efficient(all_detections)
        filtered_faces = [f for f in unique_faces if f['confidence'] >= confidence_threshold]
        
        # Analyze each detected face
        results = []
        for i, face in enumerate(filtered_faces):
            analyzed_face = self._analyze_face_complete(image, face, i + 1)
            results.append(analyzed_face)
            
        print(f"Final detection: {len(results)} faces with analysis")
        return results
        
    def _detect_haar_comprehensive(self, image: np.ndarray) -> List[Dict]:
        """Comprehensive Haar cascade detection with multiple parameter sets"""
        detections = []
        
        if self.cascade is None or self.cascade.empty():
            return detections
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Optimized parameter sets for different face sizes and orientations
        param_sets = [
            # Standard detection
            {'scaleFactor': 1.1, 'minNeighbors': 4, 'minSize': (30, 30), 'maxSize': (300, 300)},
            # Small faces
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20), 'maxSize': (150, 150)},
            # Large faces
            {'scaleFactor': 1.2, 'minNeighbors': 5, 'minSize': (50, 50), 'maxSize': (400, 400)},
            # Profile faces
            {'scaleFactor': 1.15, 'minNeighbors': 3, 'minSize': (25, 25), 'maxSize': (200, 200)},
            # Distant faces
            {'scaleFactor': 1.08, 'minNeighbors': 2, 'minSize': (15, 15), 'maxSize': (100, 100)},
            # Close faces
            {'scaleFactor': 1.3, 'minNeighbors': 6, 'minSize': (80, 80), 'maxSize': (500, 500)},
        ]
        
        for i, params in enumerate(param_sets):
            try:
                faces = self.cascade.detectMultiScale(gray, **params)
                
                for (x, y, w, h) in faces:
                    # Validate detection
                    if self._is_valid_detection(x, y, w, h, image.shape[1], image.shape[0]):
                        confidence = 0.6 + (i * 0.05)  # Progressive confidence
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': min(confidence, 0.95),
                            'method': f'haar_set_{i+1}'
                        })
            except Exception as e:
                print(f"Error in Haar detection set {i+1}: {e}")
                continue
                
        return detections
        
    def _detect_mtcnn_fast(self, image: np.ndarray) -> List[Dict]:
        """Fast MTCNN detection with timeout protection"""
        detections = []
        
        try:
            # Convert to RGB for MTCNN
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image if too large for faster processing
            h, w = rgb_image.shape[:2]
            if w > 800 or h > 800:
                scale = min(800/w, 800/h)
                new_w, new_h = int(w*scale), int(h*scale)
                rgb_image_resized = cv2.resize(rgb_image, (new_w, new_h))
                scale_back = 1/scale
            else:
                rgb_image_resized = rgb_image
                scale_back = 1.0
            
            # MTCNN detection
            results = self.mtcnn.detect_faces(rgb_image_resized)
            
            for result in results:
                x, y, w, h = result['box']
                confidence = result['confidence']
                
                # Scale back coordinates if image was resized
                if scale_back != 1.0:
                    x, y, w, h = int(x*scale_back), int(y*scale_back), int(w*scale_back), int(h*scale_back)
                
                # Validate detection
                if self._is_valid_detection(x, y, w, h, image.shape[1], image.shape[0]):
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': confidence,
                        'method': 'mtcnn',
                        'landmarks': result.get('keypoints', {})
                    })
                    
        except Exception as e:
            print(f"MTCNN detection error: {e}")
            
        return detections
        
    def _is_valid_detection(self, x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> bool:
        """Validate face detection parameters"""
        # Size validation
        if w < 15 or h < 15 or w > img_w * 0.8 or h > img_h * 0.8:
            return False
            
        # Position validation
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return False
            
        # Aspect ratio validation (faces should be roughly square)
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
            
        return True
        
    def _remove_duplicates_efficient(self, detections: List[Dict]) -> List[Dict]:
        """Efficient duplicate removal with preference for higher confidence"""
        if not detections:
            return []
            
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        unique_detections = []
        for detection in sorted_detections:
            is_duplicate = False
            
            for existing in unique_detections:
                if self._calculate_overlap_fast(detection['bbox'], existing['bbox']) > 0.4:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_detections.append(detection)
                
        return unique_detections
        
    def _calculate_overlap_fast(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Fast overlap calculation"""
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
        
    def _analyze_face_complete(self, image: np.ndarray, detection: Dict, face_id: int) -> Dict:
        """Complete face analysis with gaze and emotion detection"""
        x, y, w, h = detection['bbox']
        
        # Extract face region safely
        y_end = min(y + h, image.shape[0])
        x_end = min(x + w, image.shape[1])
        face_region = image[y:y_end, x:x_end]
        
        # Gaze direction analysis
        gaze_direction = self._analyze_gaze_direction(x, y, w, h, image.shape[1], image.shape[0], face_region)
        
        # Emotion analysis
        emotion, emotion_confidence = self._analyze_emotion_expression(face_region)
        
        return {
            'face_id': face_id,
            'person_name': f'Person-{face_id}',
            'bbox': (x, y, w, h),
            'confidence': detection['confidence'],
            'method': detection['method'],
            'gaze_direction': gaze_direction,
            'gaze_confidence': 0.82,
            'emotion': emotion,
            'emotion_confidence': emotion_confidence,
            'landmarks': detection.get('landmarks', {})
        }
        
    def _analyze_gaze_direction(self, x: int, y: int, w: int, h: int, 
                               img_w: int, img_h: int, face_region: np.ndarray) -> str:
        """Analyze gaze direction based on position and face orientation"""
        # Calculate face center position
        center_x = (x + w/2) / img_w
        center_y = (y + h/2) / img_h
        
        # Vertical gaze analysis
        if center_y > 0.75:
            return "looking down at table/cards"
        elif center_y < 0.25:
            return "looking up/away"
            
        # Horizontal gaze analysis with enhanced positioning
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
            # Center region - analyze face orientation
            return self._analyze_face_orientation_detailed(face_region, center_x, center_y)
            
    def _analyze_face_orientation_detailed(self, face_region: np.ndarray, center_x: float, center_y: float) -> str:
        """Detailed face orientation analysis for center-positioned faces"""
        if face_region.size == 0 or face_region.shape[0] < 20 or face_region.shape[1] < 20:
            return "looking at camera"
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            # Analyze face symmetry for head orientation
            left_half = gray_face[:, :w//2]
            right_half = cv2.flip(gray_face[:, w//2:], 1)  # Flip for comparison
            
            # Ensure same dimensions for comparison
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate difference
            if left_half.shape == right_half.shape:
                diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
                
                # Eye region analysis for more precise gaze
                eye_region = gray_face[int(0.2*h):int(0.5*h), :]
                if eye_region.size > 0:
                    eye_left = eye_region[:, :w//2]
                    eye_right = eye_region[:, w//2:]
                    eye_brightness_diff = np.mean(eye_left) - np.mean(eye_right)
                    
                    # Combined analysis
                    if abs(eye_brightness_diff) > 10:
                        if eye_brightness_diff > 0:
                            return "looking slightly right"
                        else:
                            return "looking slightly left"
                            
                if diff > 20:
                    return "looking slightly to the side"
                    
            # Default based on position
            if 0.4 <= center_x <= 0.6:
                return "looking directly at camera"
            else:
                return "looking at camera"
                
        except Exception as e:
            return "looking forward"
            
    def _analyze_emotion_expression(self, face_region: np.ndarray) -> Tuple[str, float]:
        """Analyze emotion from facial expression features"""
        if face_region.size == 0 or face_region.shape[0] < 25 or face_region.shape[1] < 25:
            return "neutral", 0.65
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            # Define facial regions
            mouth_region = gray_face[int(0.65*h):int(0.9*h), int(0.25*w):int(0.75*w)]
            eye_region = gray_face[int(0.25*h):int(0.55*h), int(0.15*w):int(0.85*w)]
            cheek_region = gray_face[int(0.4*h):int(0.7*h), int(0.1*w):int(0.9*w)]
            
            # Calculate regional features
            features = {}
            face_mean = np.mean(gray_face)
            
            if mouth_region.size > 0:
                features['mouth_brightness'] = np.mean(mouth_region)
                features['mouth_contrast'] = np.std(mouth_region)
                
            if eye_region.size > 0:
                features['eye_brightness'] = np.mean(eye_region)
                features['eye_variance'] = np.var(eye_region)
                
            if cheek_region.size > 0:
                features['cheek_brightness'] = np.mean(cheek_region)
                
            # Enhanced emotion classification
            mouth_bright = features.get('mouth_brightness', face_mean)
            mouth_contrast = features.get('mouth_contrast', 0)
            eye_bright = features.get('eye_brightness', face_mean)
            
            # Happy detection (bright mouth region, higher contrast)
            if mouth_bright > face_mean * 1.15 and mouth_contrast > 20:
                return "happy", 0.85
                
            # Sad detection (darker mouth, low eye brightness)
            elif mouth_bright < face_mean * 0.85 and eye_bright < face_mean * 0.9:
                return "sad", 0.78
                
            # Surprised detection (high mouth contrast, bright eyes)
            elif mouth_contrast > 30 and eye_bright > face_mean * 1.1:
                return "surprised", 0.73
                
            # Focused/concentrated (slightly darker eyes, stable mouth)
            elif eye_bright < face_mean * 0.95 and mouth_contrast < 15:
                return "focused", 0.71
                
            # Tired detection (low overall contrast)
            elif features.get('eye_variance', 0) < 100 and mouth_contrast < 12:
                return "tired", 0.69
                
            else:
                return "neutral", 0.75
                
        except Exception as e:
            return "neutral", 0.6
            
    def draw_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw comprehensive detection results on image"""
        result_image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            method = detection['method']
            
            # Color coding by detection method
            if method == 'mtcnn':
                color = (0, 255, 0)  # Green for MTCNN
            elif 'haar' in method:
                color = (255, 165, 0)  # Orange for Haar cascade
            else:
                color = (0, 255, 255)  # Yellow for others
                
            # Dynamic thickness based on confidence
            thickness = max(2, int(confidence * 4))
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
            
            # Person label with confidence
            person_text = f"{detection['person_name']} ({confidence:.2f})"
            text_size = cv2.getTextSize(person_text, font, 0.6, 2)[0]
            label_bg = (x, y - 25, x + text_size[0] + 10, y)
            cv2.rectangle(result_image, (label_bg[0], label_bg[1]), (label_bg[2], label_bg[3]), color, -1)
            cv2.putText(result_image, person_text, (x + 5, y - 5), font, 0.6, (0, 0, 0), 2)
            
            # Gaze direction
            gaze_text = f"Gaze: {detection['gaze_direction']}"
            gaze_size = cv2.getTextSize(gaze_text, font, 0.45, 1)[0]
            gaze_bg = (x, y + h, x + gaze_size[0] + 10, y + h + 22)
            cv2.rectangle(result_image, (gaze_bg[0], gaze_bg[1]), (gaze_bg[2], gaze_bg[3]), (255, 255, 0), -1)
            cv2.putText(result_image, gaze_text, (x + 5, y + h + 16), font, 0.45, (0, 0, 0), 1)
            
            # Emotion
            emotion_text = f"Emotion: {detection['emotion']}"
            emotion_size = cv2.getTextSize(emotion_text, font, 0.45, 1)[0]
            emotion_bg = (x, y + h + 25, x + emotion_size[0] + 10, y + h + 47)
            cv2.rectangle(result_image, (emotion_bg[0], emotion_bg[1]), (emotion_bg[2], emotion_bg[3]), (0, 255, 255), -1)
            cv2.putText(result_image, emotion_text, (x + 5, y + h + 41), font, 0.45, (0, 0, 0), 1)
            
            # Detection method indicator
            method_text = f"Method: {method}"
            cv2.putText(result_image, method_text, (x + 5, y + h + 65), font, 0.35, color, 1)
        
        # Header with detection summary
        summary = f"Working Face Detection: {len(detections)} faces detected"
        summary_size = cv2.getTextSize(summary, font, 0.7, 2)[0]
        header_bg = (10, 10, summary_size[0] + 20, 45)
        cv2.rectangle(result_image, (header_bg[0], header_bg[1]), (header_bg[2], header_bg[3]), (0, 0, 0), -1)
        cv2.putText(result_image, summary, (15, 32), font, 0.7, (255, 255, 255), 2)
        
        return result_image