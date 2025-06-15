import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

class StableFaceDetector:
    """Highly stable and accurate face detection system"""
    
    def __init__(self):
        self.setup_logging()
        self.detection_methods = []
        self.initialize_detectors()
        
    def setup_logging(self):
        """Setup logging for debugging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def initialize_detectors(self):
        """Initialize all available detection methods"""
        # Method 1: OpenCV Haar Cascade - Most stable
        self.init_haar_cascade()
        
        # Method 2: MediaPipe - Good for modern faces
        self.init_mediapipe()
        
        # Method 3: OpenCV DNN (if available)
        self.init_dnn_detector()
        
        self.logger.info(f"Initialized {len(self.detection_methods)} detection methods")
        
    def init_haar_cascade(self):
        """Initialize Haar cascade detector"""
        try:
            cascade_path = 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            
            if not cascade.empty():
                self.detection_methods.append({
                    'name': 'haar_cascade',
                    'detector': cascade,
                    'confidence': 0.8,
                    'process_func': self._detect_haar_faces
                })
                self.logger.info("Haar cascade detector initialized successfully")
            else:
                self.logger.warning("Haar cascade detector failed to load")
        except Exception as e:
            self.logger.error(f"Haar cascade initialization error: {e}")
            
    def init_mediapipe(self):
        """Initialize MediaPipe face detection"""
        try:
            import mediapipe as mp
            
            # Short range detector
            mp_face_detection = mp.solutions.face_detection
            short_detector = mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.3
            )
            
            # Long range detector
            long_detector = mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.3
            )
            
            self.detection_methods.append({
                'name': 'mediapipe_short',
                'detector': short_detector,
                'confidence': 0.9,
                'process_func': self._detect_mediapipe_faces
            })
            
            self.detection_methods.append({
                'name': 'mediapipe_long',
                'detector': long_detector,
                'confidence': 0.85,
                'process_func': self._detect_mediapipe_faces
            })
            
            self.logger.info("MediaPipe detectors initialized successfully")
        except Exception as e:
            self.logger.error(f"MediaPipe initialization error: {e}")
            
    def init_dnn_detector(self):
        """Initialize OpenCV DNN face detector"""
        try:
            # Try to load DNN model (ResNet-based)
            model_file = "opencv_face_detector_uint8.pb"
            config_file = "opencv_face_detector.pbtxt"
            
            # For now, skip DNN as it requires model files
            # In production, you would download these models
            pass
            
        except Exception as e:
            self.logger.error(f"DNN detector initialization error: {e}")
    
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """
        Detect faces using multiple methods with ensemble approach
        """
        if image is None or image.size == 0:
            return []
            
        h, w = image.shape[:2]
        if h < 50 or w < 50:
            return []
            
        all_detections = []
        
        # Run all detection methods
        for method in self.detection_methods:
            try:
                detections = method['process_func'](image, method['detector'])
                for detection in detections:
                    detection['method'] = method['name']
                    detection['base_confidence'] = method['confidence']
                all_detections.extend(detections)
                    
            except Exception as e:
                self.logger.error(f"Detection method {method['name']} failed: {e}")
                continue
        
        # Remove duplicate detections using Non-Maximum Suppression
        unique_detections = self._remove_duplicates(all_detections, w, h)
        
        # Filter by confidence threshold
        filtered_detections = [d for d in unique_detections if d['confidence'] >= confidence_threshold]
        
        # Assign face IDs and analyze each face
        final_results = []
        for i, detection in enumerate(filtered_detections):
            face_data = self._analyze_face(image, detection, i + 1)
            final_results.append(face_data)
            
        self.logger.info(f"Detected {len(final_results)} faces using {len(self.detection_methods)} methods")
        return final_results
    
    def _detect_haar_faces(self, image: np.ndarray, cascade) -> List[Dict]:
        """Detect faces using Haar cascade with multiple parameter sets"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Multiple parameter sets for better detection
        param_sets = [
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (30, 30)},
            {'scaleFactor': 1.1, 'minNeighbors': 4, 'minSize': (40, 40)},
            {'scaleFactor': 1.15, 'minNeighbors': 5, 'minSize': (50, 50)},
            {'scaleFactor': 1.2, 'minNeighbors': 3, 'minSize': (25, 25)},
        ]
        
        for params in param_sets:
            faces = cascade.detectMultiScale(gray, **params)
            for (x, y, fw, fh) in faces:
                # Validate detection
                if self._validate_face_region(x, y, fw, fh, w, h):
                    detections.append({
                        'bbox': (x, y, fw, fh),
                        'confidence': 0.7,
                        'source': f"haar_{params['scaleFactor']}"
                    })
        
        return detections
    
    def _detect_mediapipe_faces(self, image: np.ndarray, detector) -> List[Dict]:
        """Detect faces using MediaPipe"""
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
                
                # Validate and adjust coordinates
                x = max(0, x)
                y = max(0, y)
                fw = min(w - x, fw)
                fh = min(h - y, fh)
                
                if self._validate_face_region(x, y, fw, fh, w, h):
                    confidence = detection.score[0] if detection.score else 0.8
                    detections.append({
                        'bbox': (x, y, fw, fh),
                        'confidence': confidence,
                        'source': 'mediapipe'
                    })
        
        return detections
    
    def _validate_face_region(self, x: int, y: int, fw: int, fh: int, img_w: int, img_h: int) -> bool:
        """Validate if detected region is likely a face"""
        # Size validation
        if fw < 20 or fh < 20:
            return False
            
        # Aspect ratio validation
        aspect_ratio = fw / fh
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
            
        # Position validation
        if x < 0 or y < 0 or x + fw > img_w or y + fh > img_h:
            return False
            
        # Relative size validation
        face_area = fw * fh
        image_area = img_w * img_h
        relative_size = face_area / image_area
        
        if relative_size < 0.001 or relative_size > 0.8:
            return False
            
        return True
    
    def _remove_duplicates(self, detections: List[Dict], img_w: int, img_h: int) -> List[Dict]:
        """Remove duplicate detections using Non-Maximum Suppression"""
        if not detections:
            return []
            
        # Convert to format suitable for NMS
        boxes = []
        scores = []
        indices = []
        
        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            boxes.append([x, y, x + w, y + h])
            scores.append(det['confidence'])
            indices.append(i)
        
        # Apply Non-Maximum Suppression
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # OpenCV NMS
        nms_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.3, 0.4)
        
        if len(nms_indices) > 0:
            if hasattr(nms_indices, 'flatten'):
                nms_indices = nms_indices.flatten()
            return [detections[i] for i in nms_indices]
        else:
            # Fallback: manual duplicate removal
            return self._manual_duplicate_removal(detections)
    
    def _manual_duplicate_removal(self, detections: List[Dict]) -> List[Dict]:
        """Manual duplicate removal when NMS fails"""
        unique_detections = []
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            center_x = x + w // 2
            center_y = y + h // 2
            
            is_duplicate = False
            for existing in unique_detections:
                ex, ey, ew, eh = existing['bbox']
                existing_center_x = ex + ew // 2
                existing_center_y = ey + eh // 2
                
                distance = np.sqrt((center_x - existing_center_x)**2 + (center_y - existing_center_y)**2)
                overlap_threshold = min(w, h, ew, eh) * 0.5
                
                if distance < overlap_threshold:
                    # Keep the one with higher confidence
                    if detection['confidence'] > existing['confidence']:
                        unique_detections.remove(existing)
                        unique_detections.append(detection)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(detection)
        
        return unique_detections
    
    def _analyze_face(self, image: np.ndarray, detection: Dict, face_id: int) -> Dict:
        """Analyze individual face for gaze and emotion"""
        x, y, w, h = detection['bbox']
        
        # Extract face region safely
        face_region = image[max(0, y):min(image.shape[0], y + h), 
                          max(0, x):min(image.shape[1], x + w)]
        
        # Analyze gaze direction
        gaze_direction = self._analyze_gaze_direction(x, y, w, h, image.shape[1], image.shape[0], face_region)
        
        # Analyze emotion
        emotion = self._analyze_emotion(face_region)
        
        return {
            'face_id': face_id,
            'person_name': f'Person-{face_id}',
            'bbox': (x, y, w, h),
            'confidence': detection['confidence'],
            'method': detection['method'],
            'gaze_direction': gaze_direction,
            'gaze_confidence': 0.7,
            'emotion': emotion,
            'emotion_confidence': 0.6,
            'is_stable': True
        }
    
    def _analyze_gaze_direction(self, x: int, y: int, w: int, h: int, 
                              img_w: int, img_h: int, face_region: np.ndarray) -> str:
        """Enhanced gaze direction analysis"""
        # Face center relative position
        center_x = (x + w/2) / img_w
        center_y = (y + h/2) / img_h
        
        # Detailed position-based gaze analysis
        if center_y > 0.75:
            return "looking down at table/game"
        elif center_y < 0.25:
            return "looking up at ceiling/away"
        elif center_x < 0.25:
            return "looking at person on far left"
        elif center_x > 0.75:
            return "looking at person on far right"
        elif center_x < 0.4:
            return "looking at person on left"
        elif center_x > 0.6:
            return "looking at person on right"
        else:
            # For center faces, try to analyze face orientation
            return self._analyze_face_orientation(face_region, center_x, center_y)
    
    def _analyze_face_orientation(self, face_region: np.ndarray, center_x: float, center_y: float) -> str:
        """Analyze face orientation for gaze direction"""
        if face_region.size == 0:
            return "looking straight ahead"
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            if h < 30 or w < 30:
                return "looking at camera"
            
            # Analyze horizontal gradients in eye region
            eye_region = gray_face[h//4:h//2, :]
            
            if eye_region.size > 0:
                # Calculate horizontal gradient
                grad_x = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
                left_grad = np.mean(grad_x[:, :w//2])
                right_grad = np.mean(grad_x[:, w//2:])
                
                gradient_diff = right_grad - left_grad
                
                if gradient_diff > 5:
                    return "looking slightly right"
                elif gradient_diff < -5:
                    return "looking slightly left"
                else:
                    return "looking at camera"
            else:
                return "looking straight ahead"
                
        except Exception as e:
            self.logger.error(f"Face orientation analysis error: {e}")
            return "looking forward"
    
    def _analyze_emotion(self, face_region: np.ndarray) -> str:
        """Enhanced emotion analysis"""
        if face_region.size == 0:
            return "neutral"
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            if h < 30 or w < 30:
                return "neutral"
            
            # Analyze mouth region (lower third of face)
            mouth_y_start = 2 * h // 3
            mouth_region = gray_face[mouth_y_start:, w//4:3*w//4]
            
            # Analyze eye region (upper third of face)
            eye_region = gray_face[h//4:h//2, :]
            
            if mouth_region.size > 0 and eye_region.size > 0:
                mouth_brightness = np.mean(mouth_region)
                eye_brightness = np.mean(eye_region)
                face_brightness = np.mean(gray_face)
                
                # Simple emotion classification
                if mouth_brightness > face_brightness * 1.2:
                    return "happy"
                elif mouth_brightness < face_brightness * 0.8:
                    return "sad"
                elif eye_brightness < face_brightness * 0.9:
                    return "tired"
                else:
                    return "neutral"
            else:
                return "neutral"
                
        except Exception as e:
            self.logger.error(f"Emotion analysis error: {e}")
            return "neutral"
    
    def draw_enhanced_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw enhanced detection results with stability indicators"""
        result_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            
            # Color based on detection method and confidence
            if detection['confidence'] > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif detection['confidence'] > 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 165, 255)  # Orange for low confidence
            
            # Draw face rectangle with thickness based on confidence
            thickness = max(2, int(detection['confidence'] * 5))
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
            
            # Person name with confidence
            person_text = f"{detection['person_name']} ({detection['confidence']:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            text_thickness = 2
            
            # Person name background
            (text_w, text_h), baseline = cv2.getTextSize(person_text, font, font_scale, text_thickness)
            cv2.rectangle(result_image, (x, y - text_h - 15), (x + text_w + 10, y), color, -1)
            cv2.putText(result_image, person_text, (x + 5, y - 8), font, font_scale, (0, 0, 0), text_thickness)
            
            # Gaze direction
            gaze_text = f"Gaze: {detection['gaze_direction']}"
            (gaze_w, gaze_h), _ = cv2.getTextSize(gaze_text, font, 0.6, 2)
            cv2.rectangle(result_image, (x, y + h + 5), (x + gaze_w + 10, y + h + gaze_h + 15), (255, 255, 0), -1)
            cv2.putText(result_image, gaze_text, (x + 5, y + h + gaze_h + 10), font, 0.6, (0, 0, 0), 2)
            
            # Emotion
            emotion_text = f"Emotion: {detection['emotion']}"
            (emo_w, emo_h), _ = cv2.getTextSize(emotion_text, font, 0.5, 1)
            cv2.rectangle(result_image, (x, y + h + gaze_h + 20), 
                        (x + emo_w + 10, y + h + gaze_h + emo_h + 30), (0, 255, 255), -1)
            cv2.putText(result_image, emotion_text, (x + 5, y + h + gaze_h + emo_h + 25), 
                      font, 0.5, (0, 0, 0), 1)
            
            # Detection method
            method_text = f"Method: {detection['method']}"
            cv2.putText(result_image, method_text, (x + 5, y + h + gaze_h + emo_h + 45), 
                      font, 0.4, color, 1)
        
        # Add detection summary
        summary_text = f"Faces detected: {len(detections)} | Methods: {len(self.detection_methods)}"
        summary_font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result_image, summary_text, (10, 30), summary_font, 0.8, (255, 255, 255), 2)
        cv2.putText(result_image, summary_text, (10, 30), summary_font, 0.8, (0, 0, 0), 1)
        
        return result_image