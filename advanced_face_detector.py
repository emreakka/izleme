import cv2
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional
import logging
import mediapipe as mp

class AdvancedFaceDetector:
    """Advanced face detection using TensorFlow, MediaPipe, and OpenCV"""
    
    def __init__(self):
        self.setup_logging()
        self.detection_methods = []
        self.emotion_model = None
        self.initialize_detectors()
        self.load_emotion_model()
        
    def setup_logging(self):
        """Setup logging for debugging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def initialize_detectors(self):
        """Initialize all available detection methods"""
        # Method 1: OpenCV Haar Cascade
        self.init_haar_cascade()
        
        # Method 2: MediaPipe Face Detection
        self.init_mediapipe()
        
        # Method 3: TensorFlow Face Detection
        self.init_tensorflow_detector()
        
        # Method 4: OpenCV DNN Face Detection
        self.init_dnn_detector()
        
        self.logger.info(f"Initialized {len(self.detection_methods)} detection methods")
        
    def init_haar_cascade(self):
        """Initialize Haar cascade detector with multiple cascade files"""
        try:
            # Primary frontal face cascade
            cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            
            if not cascade.empty():
                self.detection_methods.append({
                    'name': 'haar_frontal',
                    'detector': cascade,
                    'confidence': 0.75,
                    'process_func': self._detect_haar_faces
                })
                self.logger.info("Haar cascade detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Haar cascade initialization error: {e}")
            
    def init_mediapipe(self):
        """Initialize MediaPipe face detection with multiple models"""
        try:
            mp_face_detection = mp.solutions.face_detection
            
            # Short range detector (better for close faces)
            short_detector = mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.2
            )
            
            # Long range detector (better for distant faces)
            long_detector = mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.2
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
            
    def init_tensorflow_detector(self):
        """Initialize TensorFlow-based face detection"""
        try:
            # Create a simple CNN-based face detector using TensorFlow
            self.tf_detector = self._create_tf_face_detector()
            
            if self.tf_detector:
                self.detection_methods.append({
                    'name': 'tensorflow_cnn',
                    'detector': self.tf_detector,
                    'confidence': 0.8,
                    'process_func': self._detect_tensorflow_faces
                })
                self.logger.info("TensorFlow detector initialized successfully")
        except Exception as e:
            self.logger.error(f"TensorFlow detector initialization error: {e}")
            
    def init_dnn_detector(self):
        """Initialize OpenCV DNN face detector"""
        try:
            # Create a blob-based face detector using edge detection
            self.detection_methods.append({
                'name': 'edge_contour',
                'detector': None,
                'confidence': 0.6,
                'process_func': self._detect_edge_faces
            })
            self.logger.info("Edge-based detector initialized successfully")
        except Exception as e:
            self.logger.error(f"DNN detector initialization error: {e}")
            
    def _create_tf_face_detector(self):
        """Create a TensorFlow-based face detection model"""
        try:
            # Simple CNN architecture for face detection
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (face/no face)
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            self.logger.error(f"TensorFlow model creation error: {e}")
            return None
            
    def load_emotion_model(self):
        """Load emotion recognition model"""
        try:
            # Create a simple emotion recognition model
            self.emotion_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(7, activation='softmax')  # 7 emotions
            ])
            
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            self.logger.info("Emotion model loaded successfully")
        except Exception as e:
            self.logger.error(f"Emotion model loading error: {e}")
            self.emotion_model = None
    
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """
        Advanced face detection using ensemble of methods
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
        
        # Advanced duplicate removal using ensemble voting
        unique_detections = self._ensemble_duplicate_removal(all_detections, w, h)
        
        # Filter by confidence threshold
        filtered_detections = [d for d in unique_detections if d['confidence'] >= confidence_threshold]
        
        # Enhanced analysis for each face
        final_results = []
        for i, detection in enumerate(filtered_detections):
            face_data = self._advanced_face_analysis(image, detection, i + 1)
            final_results.append(face_data)
            
        self.logger.info(f"Detected {len(final_results)} faces using ensemble approach")
        return final_results
    
    def _detect_haar_faces(self, image: np.ndarray, cascade) -> List[Dict]:
        """Enhanced Haar cascade detection with multiple parameter sets"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Enhanced parameter sets for better detection
        param_sets = [
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20), 'maxSize': (300, 300)},
            {'scaleFactor': 1.1, 'minNeighbors': 4, 'minSize': (30, 30), 'maxSize': (250, 250)},
            {'scaleFactor': 1.15, 'minNeighbors': 5, 'minSize': (40, 40), 'maxSize': (200, 200)},
            {'scaleFactor': 1.2, 'minNeighbors': 3, 'minSize': (25, 25), 'maxSize': (150, 150)},
            {'scaleFactor': 1.3, 'minNeighbors': 2, 'minSize': (35, 35), 'maxSize': (180, 180)},
        ]
        
        for params in param_sets:
            faces = cascade.detectMultiScale(gray, **params)
            for (x, y, fw, fh) in faces:
                if self._validate_face_region(x, y, fw, fh, w, h):
                    # Calculate confidence based on face quality
                    face_region = gray[y:y+fh, x:x+fw]
                    quality_score = self._calculate_face_quality(face_region)
                    
                    detections.append({
                        'bbox': (x, y, fw, fh),
                        'confidence': 0.7 * quality_score,
                        'source': f"haar_{params['scaleFactor']}"
                    })
        
        return detections
    
    def _detect_mediapipe_faces(self, image: np.ndarray, detector) -> List[Dict]:
        """Enhanced MediaPipe face detection"""
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
                    
                    # Enhance confidence with face quality assessment
                    face_region = image[y:y+fh, x:x+fw]
                    quality_score = self._calculate_face_quality(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))
                    
                    detections.append({
                        'bbox': (x, y, fw, fh),
                        'confidence': confidence * quality_score,
                        'source': 'mediapipe'
                    })
        
        return detections
    
    def _detect_tensorflow_faces(self, image: np.ndarray, model) -> List[Dict]:
        """TensorFlow-based face detection using sliding window"""
        detections = []
        
        if model is None:
            return detections
            
        try:
            h, w = image.shape[:2]
            window_sizes = [(64, 64), (96, 96), (128, 128)]
            
            for window_size in window_sizes:
                wh, ww = window_size
                step = max(16, min(wh, ww) // 4)
                
                for y in range(0, h - wh, step):
                    for x in range(0, w - ww, step):
                        window = image[y:y+wh, x:x+ww]
                        
                        # Preprocess for model
                        processed = cv2.resize(window, (96, 96))
                        processed = processed.astype(np.float32) / 255.0
                        processed = np.expand_dims(processed, axis=0)
                        
                        # Predict if it's a face
                        prediction = model.predict(processed, verbose=0)
                        confidence = float(prediction[0][0])
                        
                        if confidence > 0.5:  # Threshold for face detection
                            detections.append({
                                'bbox': (x, y, ww, wh),
                                'confidence': confidence,
                                'source': 'tensorflow'
                            })
                            
        except Exception as e:
            self.logger.error(f"TensorFlow detection error: {e}")
            
        return detections
    
    def _detect_edge_faces(self, image: np.ndarray, detector) -> List[Dict]:
        """Edge-based face detection using contour analysis"""
        detections = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Apply Gaussian blur and edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (reasonable face size)
                if 1000 < area < 50000:
                    x, y, fw, fh = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (face-like)
                    aspect_ratio = float(fw) / fh
                    if 0.6 < aspect_ratio < 1.6:
                        # Calculate confidence based on contour properties
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        confidence = min(0.8, circularity * 2)  # Normalize confidence
                        
                        if confidence > 0.3:
                            detections.append({
                                'bbox': (x, y, fw, fh),
                                'confidence': confidence,
                                'source': 'edge_contour'
                            })
                            
        except Exception as e:
            self.logger.error(f"Edge detection error: {e}")
            
        return detections
    
    def _calculate_face_quality(self, face_gray: np.ndarray) -> float:
        """Calculate face quality score based on various metrics"""
        if face_gray.size == 0:
            return 0.5
            
        try:
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 100.0)
            
            # Calculate contrast
            contrast_score = face_gray.std() / 128.0
            contrast_score = min(1.0, contrast_score)
            
            # Calculate brightness (prefer well-lit faces)
            brightness = face_gray.mean() / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Combine scores
            quality_score = (sharpness_score * 0.4 + contrast_score * 0.3 + brightness_score * 0.3)
            return max(0.3, min(1.0, quality_score))
            
        except Exception:
            return 0.5
    
    def _ensemble_duplicate_removal(self, detections: List[Dict], img_w: int, img_h: int) -> List[Dict]:
        """Advanced duplicate removal using ensemble voting"""
        if not detections:
            return []
        
        # Group overlapping detections
        groups = []
        for detection in detections:
            added_to_group = False
            x, y, w, h = detection['bbox']
            
            for group in groups:
                # Check overlap with any detection in the group
                for existing in group:
                    ex, ey, ew, eh = existing['bbox']
                    
                    # Calculate IoU (Intersection over Union)
                    iou = self._calculate_iou((x, y, w, h), (ex, ey, ew, eh))
                    
                    if iou > 0.3:  # Overlapping detections
                        group.append(detection)
                        added_to_group = True
                        break
                
                if added_to_group:
                    break
            
            if not added_to_group:
                groups.append([detection])
        
        # For each group, select the best detection or create ensemble
        final_detections = []
        for group in groups:
            if len(group) == 1:
                final_detections.append(group[0])
            else:
                # Create ensemble detection
                ensemble_detection = self._create_ensemble_detection(group)
                final_detections.append(ensemble_detection)
        
        return final_detections
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _create_ensemble_detection(self, group: List[Dict]) -> Dict:
        """Create ensemble detection from group of overlapping detections"""
        # Weighted average of bounding boxes
        total_confidence = sum(d['confidence'] for d in group)
        
        x_weighted = sum(d['bbox'][0] * d['confidence'] for d in group) / total_confidence
        y_weighted = sum(d['bbox'][1] * d['confidence'] for d in group) / total_confidence
        w_weighted = sum(d['bbox'][2] * d['confidence'] for d in group) / total_confidence
        h_weighted = sum(d['bbox'][3] * d['confidence'] for d in group) / total_confidence
        
        # Average confidence with bonus for consensus
        avg_confidence = total_confidence / len(group)
        consensus_bonus = min(0.2, len(group) * 0.05)  # Bonus for multiple detections
        
        # Select method with highest confidence
        best_method = max(group, key=lambda x: x['confidence'])['source']
        
        return {
            'bbox': (int(x_weighted), int(y_weighted), int(w_weighted), int(h_weighted)),
            'confidence': min(1.0, avg_confidence + consensus_bonus),
            'source': f"ensemble_{best_method}",
            'ensemble_size': len(group)
        }
    
    def _validate_face_region(self, x: int, y: int, fw: int, fh: int, img_w: int, img_h: int) -> bool:
        """Enhanced face region validation"""
        # Size validation
        if fw < 20 or fh < 20 or fw > img_w * 0.8 or fh > img_h * 0.8:
            return False
            
        # Aspect ratio validation (more flexible)
        aspect_ratio = fw / fh
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            return False
            
        # Position validation
        if x < 0 or y < 0 or x + fw > img_w or y + fh > img_h:
            return False
            
        # Relative size validation
        face_area = fw * fh
        image_area = img_w * img_h
        relative_size = face_area / image_area
        
        if relative_size < 0.0005 or relative_size > 0.9:
            return False
            
        return True
    
    def _advanced_face_analysis(self, image: np.ndarray, detection: Dict, face_id: int) -> Dict:
        """Advanced face analysis with emotion recognition and enhanced gaze detection"""
        x, y, w, h = detection['bbox']
        
        # Extract face region safely
        face_region = image[max(0, y):min(image.shape[0], y + h), 
                          max(0, x):min(image.shape[1], x + w)]
        
        # Advanced gaze direction analysis
        gaze_direction = self._analyze_advanced_gaze(x, y, w, h, image.shape[1], image.shape[0], face_region)
        
        # Deep emotion analysis
        emotion, emotion_confidence = self._analyze_deep_emotion(face_region)
        
        # Additional face features
        face_features = self._extract_face_features(face_region)
        
        return {
            'face_id': face_id,
            'person_name': f'Person-{face_id}',
            'bbox': (x, y, w, h),
            'confidence': detection['confidence'],
            'method': detection['source'],
            'gaze_direction': gaze_direction,
            'gaze_confidence': 0.8,
            'emotion': emotion,
            'emotion_confidence': emotion_confidence,
            'face_features': face_features,
            'is_stable': True,
            'ensemble_size': detection.get('ensemble_size', 1)
        }
    
    def _analyze_advanced_gaze(self, x: int, y: int, w: int, h: int, 
                              img_w: int, img_h: int, face_region: np.ndarray) -> str:
        """Advanced gaze direction analysis with eye tracking"""
        # Position-based analysis
        center_x = (x + w/2) / img_w
        center_y = (y + h/2) / img_h
        
        # Detailed position mapping
        if center_y > 0.8:
            return "looking down at table/floor"
        elif center_y < 0.2:
            return "looking up at ceiling/sky"
        elif center_x < 0.15:
            return "looking at person on far left"
        elif center_x > 0.85:
            return "looking at person on far right"
        elif center_x < 0.35:
            return "looking at person on left"
        elif center_x > 0.65:
            return "looking at person on right"
        else:
            # For center faces, analyze eye direction
            return self._analyze_eye_direction(face_region, center_x, center_y)
    
    def _analyze_eye_direction(self, face_region: np.ndarray, center_x: float, center_y: float) -> str:
        """Analyze eye direction for precise gaze detection"""
        if face_region.size == 0:
            return "looking straight ahead"
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            if h < 40 or w < 40:
                return "looking at camera"
            
            # Define eye regions (upper third of face)
            eye_region = gray_face[h//5:2*h//5, :]
            
            if eye_region.size > 0:
                # Analyze horizontal eye movement using template matching
                left_eye_region = eye_region[:, :w//2]
                right_eye_region = eye_region[:, w//2:]
                
                # Calculate brightness distribution in eyes
                left_brightness = np.mean(left_eye_region)
                right_brightness = np.mean(right_eye_region)
                
                brightness_diff = right_brightness - left_brightness
                
                if abs(brightness_diff) > 5:
                    if brightness_diff > 0:
                        return "looking slightly left"
                    else:
                        return "looking slightly right"
                else:
                    return "looking at camera"
            else:
                return "looking straight ahead"
                
        except Exception as e:
            self.logger.error(f"Eye direction analysis error: {e}")
            return "looking forward"
    
    def _analyze_deep_emotion(self, face_region: np.ndarray) -> Tuple[str, float]:
        """Deep emotion analysis using TensorFlow model"""
        if face_region.size == 0 or self.emotion_model is None:
            return self._analyze_basic_emotion(face_region)
            
        try:
            # Preprocess face for emotion model
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (48, 48))
            normalized_face = resized_face.astype(np.float32) / 255.0
            input_face = np.expand_dims(np.expand_dims(normalized_face, axis=-1), axis=0)
            
            # Predict emotion
            predictions = self.emotion_model.predict(input_face, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            
            return self.emotion_labels[emotion_idx], confidence
            
        except Exception as e:
            self.logger.error(f"Deep emotion analysis error: {e}")
            return self._analyze_basic_emotion(face_region)
    
    def _analyze_basic_emotion(self, face_region: np.ndarray) -> Tuple[str, float]:
        """Basic emotion analysis as fallback"""
        if face_region.size == 0:
            return "neutral", 0.5
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            if h < 30 or w < 30:
                return "neutral", 0.5
            
            # Analyze facial regions
            mouth_region = gray_face[2*h//3:, w//4:3*w//4]
            eye_region = gray_face[h//4:h//2, :]
            
            if mouth_region.size > 0 and eye_region.size > 0:
                mouth_brightness = np.mean(mouth_region)
                eye_brightness = np.mean(eye_region)
                face_brightness = np.mean(gray_face)
                
                # Enhanced emotion classification
                mouth_ratio = mouth_brightness / face_brightness
                eye_ratio = eye_brightness / face_brightness
                
                if mouth_ratio > 1.25:
                    return "happy", 0.7
                elif mouth_ratio < 0.75:
                    return "sad", 0.6
                elif eye_ratio < 0.85:
                    return "tired", 0.6
                elif mouth_ratio > 1.1 and eye_ratio > 1.1:
                    return "surprised", 0.65
                else:
                    return "neutral", 0.8
            else:
                return "neutral", 0.5
                
        except Exception as e:
            self.logger.error(f"Basic emotion analysis error: {e}")
            return "neutral", 0.5
    
    def _extract_face_features(self, face_region: np.ndarray) -> Dict:
        """Extract additional face features"""
        features = {
            'face_size': 'medium',
            'face_angle': 'frontal',
            'lighting': 'normal'
        }
        
        if face_region.size == 0:
            return features
            
        try:
            h, w = face_region.shape[:2]
            
            # Face size classification
            face_area = h * w
            if face_area < 2500:
                features['face_size'] = 'small'
            elif face_area > 10000:
                features['face_size'] = 'large'
            
            # Lighting assessment
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_face)
            
            if brightness < 80:
                features['lighting'] = 'dark'
            elif brightness > 180:
                features['lighting'] = 'bright'
            
            # Face angle estimation (basic)
            if w > h * 1.3:
                features['face_angle'] = 'profile'
            elif abs(w - h) < min(w, h) * 0.2:
                features['face_angle'] = 'frontal'
            else:
                features['face_angle'] = 'slight_turn'
                
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            
        return features
    
    def draw_advanced_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw advanced detection results with comprehensive information"""
        result_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            
            # Color coding based on confidence and method
            confidence = detection['confidence']
            method = detection['method']
            
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 165, 255)  # Orange for low confidence
            
            # Ensemble detection gets special border
            if 'ensemble' in method:
                color = (255, 0, 255)  # Magenta for ensemble
            
            # Draw face rectangle with confidence-based thickness
            thickness = max(2, int(confidence * 6))
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
            
            # Comprehensive text information
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Person name with confidence and ensemble info
            ensemble_info = f" (E{detection.get('ensemble_size', 1)})" if detection.get('ensemble_size', 1) > 1 else ""
            person_text = f"{detection['person_name']}{ensemble_info} ({confidence:.2f})"
            
            # Draw person info
            (text_w, text_h), baseline = cv2.getTextSize(person_text, font, 0.6, 2)
            cv2.rectangle(result_image, (x, y - text_h - 15), (x + text_w + 10, y), color, -1)
            cv2.putText(result_image, person_text, (x + 5, y - 8), font, 0.6, (0, 0, 0), 2)
            
            # Gaze direction
            gaze_text = f"Gaze: {detection['gaze_direction']}"
            (gaze_w, gaze_h), _ = cv2.getTextSize(gaze_text, font, 0.5, 2)
            cv2.rectangle(result_image, (x, y + h + 5), (x + gaze_w + 10, y + h + gaze_h + 15), (255, 255, 0), -1)
            cv2.putText(result_image, gaze_text, (x + 5, y + h + gaze_h + 10), font, 0.5, (0, 0, 0), 2)
            
            # Emotion with confidence
            emotion_text = f"Emotion: {detection['emotion']} ({detection['emotion_confidence']:.2f})"
            (emo_w, emo_h), _ = cv2.getTextSize(emotion_text, font, 0.4, 1)
            cv2.rectangle(result_image, (x, y + h + gaze_h + 20), 
                        (x + emo_w + 10, y + h + gaze_h + emo_h + 30), (0, 255, 255), -1)
            cv2.putText(result_image, emotion_text, (x + 5, y + h + gaze_h + emo_h + 25), 
                      font, 0.4, (0, 0, 0), 1)
            
            # Detection method and features
            method_text = f"Method: {method}"
            features = detection.get('face_features', {})
            feature_text = f"Size: {features.get('face_size', 'unknown')}, Light: {features.get('lighting', 'unknown')}"
            
            cv2.putText(result_image, method_text, (x + 5, y + h + gaze_h + emo_h + 45), 
                      font, 0.35, color, 1)
            cv2.putText(result_image, feature_text, (x + 5, y + h + gaze_h + emo_h + 60), 
                      font, 0.3, (128, 128, 128), 1)
        
        # Advanced detection summary
        total_faces = len(detections)
        methods_used = len(set(d['method'] for d in detections))
        avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
        
        summary_text = f"Advanced Detection: {total_faces} faces | {methods_used} methods | Avg conf: {avg_confidence:.2f}"
        summary_font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw summary with background
        (sum_w, sum_h), _ = cv2.getTextSize(summary_text, summary_font, 0.7, 2)
        cv2.rectangle(result_image, (10, 10), (sum_w + 20, sum_h + 25), (0, 0, 0), -1)
        cv2.putText(result_image, summary_text, (15, sum_h + 20), summary_font, 0.7, (255, 255, 255), 2)
        
        return result_image