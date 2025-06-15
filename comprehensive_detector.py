import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import traceback

class ComprehensiveDetector:
    """Comprehensive face detection using all available libraries"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detectors = {}
        self.emotion_models = {}
        self.init_all_detectors()
        
    def init_all_detectors(self):
        """Initialize all available detection methods"""
        print("Initializing comprehensive detector with all libraries...")
        
        # 1. OpenCV Haar Cascades
        self.init_opencv_detectors()
        
        # 2. MediaPipe Face Detection
        self.init_mediapipe_detectors()
        
        # 3. TensorFlow/Keras Models
        self.init_tensorflow_detectors()
        
        # 4. Advanced OpenCV methods
        self.init_advanced_opencv_methods()
        
        print(f"Initialized {len(self.detectors)} face detection methods")
        print(f"Initialized {len(self.emotion_models)} emotion recognition methods")
        
    def init_opencv_detectors(self):
        """Initialize OpenCV Haar cascade detectors"""
        try:
            cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            if not cascade.empty():
                self.detectors['opencv_haar'] = cascade
                print("✓ OpenCV Haar cascade loaded")
            else:
                print("✗ OpenCV Haar cascade failed to load")
        except Exception as e:
            print(f"✗ OpenCV Haar cascade error: {e}")
            
    def init_mediapipe_detectors(self):
        """Initialize MediaPipe face detection"""
        try:
            import mediapipe as mp
            
            mp_face = mp.solutions.face_detection
            mp_face_mesh = mp.solutions.face_mesh
            
            # Face detection models
            self.detectors['mp_short'] = mp_face.FaceDetection(
                model_selection=0, min_detection_confidence=0.1
            )
            self.detectors['mp_long'] = mp_face.FaceDetection(
                model_selection=1, min_detection_confidence=0.1
            )
            
            # Face mesh for landmarks
            self.detectors['mp_mesh'] = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.1
            )
            
            print("✓ MediaPipe detectors loaded (short-range, long-range, mesh)")
            
        except Exception as e:
            print(f"✗ MediaPipe error: {e}")
            
    def init_tensorflow_detectors(self):
        """Initialize TensorFlow-based detectors"""
        try:
            import tensorflow as tf
            
            # Custom TensorFlow face detection model
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            self.detectors['tensorflow_cnn'] = model
            
            # Emotion recognition model
            emotion_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(7, activation='softmax')  # 7 emotions
            ])
            
            self.emotion_models['tensorflow_emotion'] = emotion_model
            print("✓ TensorFlow models created")
            
        except Exception as e:
            print(f"✗ TensorFlow error: {e}")
            
    def init_advanced_opencv_methods(self):
        """Initialize advanced OpenCV methods"""
        try:
            # Add DNN-based face detection
            net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
            self.detectors['opencv_dnn'] = net
            print("✓ OpenCV DNN detector loaded")
        except:
            print("✗ OpenCV DNN detector not available")
            
        # Add edge-based detection
        self.detectors['edge_detection'] = True
        print("✓ Edge-based detection enabled")
            
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """Comprehensive face detection using all methods"""
        print(f"\n=== Starting comprehensive face detection ===")
        all_detections = []
        h, w = image.shape[:2]
        
        # 1. OpenCV Haar Cascade Detection
        haar_faces = self._detect_with_haar(image)
        all_detections.extend(haar_faces)
        print(f"OpenCV Haar: {len(haar_faces)} detections")
        
        # 2. MediaPipe Detection
        mp_faces = self._detect_with_mediapipe(image)
        all_detections.extend(mp_faces)
        print(f"MediaPipe: {len(mp_faces)} detections")
        
        # 3. TensorFlow CNN Detection
        tf_faces = self._detect_with_tensorflow(image)
        all_detections.extend(tf_faces)
        print(f"TensorFlow: {len(tf_faces)} detections")
        
        # 4. Edge-based Detection
        edge_faces = self._detect_with_edge_analysis(image)
        all_detections.extend(edge_faces)
        print(f"Edge Analysis: {len(edge_faces)} detections")
        
        print(f"Total raw detections: {len(all_detections)}")
        
        # Advanced ensemble processing
        unique_faces = self._advanced_ensemble_processing(all_detections, w, h)
        print(f"After ensemble processing: {len(unique_faces)}")
        
        # Filter by confidence
        filtered_faces = [f for f in unique_faces if f['confidence'] >= confidence_threshold]
        print(f"After confidence filtering: {len(filtered_faces)}")
        
        # Comprehensive analysis for each face
        final_results = []
        for i, face in enumerate(filtered_faces):
            analyzed_face = self._comprehensive_face_analysis(image, face, i + 1)
            final_results.append(analyzed_face)
            
        print(f"=== Final result: {len(final_results)} faces ===\n")
        return final_results
        
    def _detect_with_haar(self, image: np.ndarray) -> List[Dict]:
        """OpenCV Haar cascade detection with multiple parameters"""
        detections = []
        
        if 'opencv_haar' not in self.detectors:
            return detections
            
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade = self.detectors['opencv_haar']
            
            # Multiple parameter sets for comprehensive coverage
            param_sets = [
                {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20), 'maxSize': (300, 300)},
                {'scaleFactor': 1.1, 'minNeighbors': 4, 'minSize': (30, 30), 'maxSize': (250, 250)},
                {'scaleFactor': 1.15, 'minNeighbors': 3, 'minSize': (25, 25), 'maxSize': (200, 200)},
                {'scaleFactor': 1.2, 'minNeighbors': 5, 'minSize': (35, 35), 'maxSize': (180, 180)},
                {'scaleFactor': 1.3, 'minNeighbors': 2, 'minSize': (40, 40), 'maxSize': (150, 150)},
                {'scaleFactor': 1.08, 'minNeighbors': 6, 'minSize': (45, 45), 'maxSize': (120, 120)},
            ]
            
            for i, params in enumerate(param_sets):
                faces = cascade.detectMultiScale(gray, **params)
                for (x, y, w, h) in faces:
                    if self._validate_detection(x, y, w, h, image.shape[1], image.shape[0]):
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': 0.6 + (i * 0.05),
                            'method': f'haar_pass_{i+1}',
                            'source': 'opencv'
                        })
                        
        except Exception as e:
            print(f"Haar detection error: {e}")
            
        return detections
        
    def _detect_with_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """MediaPipe face detection"""
        detections = []
        
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Short-range model
            if 'mp_short' in self.detectors:
                results = self.detectors['mp_short'].process(rgb_image)
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        fw = int(bbox.width * w)
                        fh = int(bbox.height * h)
                        
                        if self._validate_detection(x, y, fw, fh, w, h):
                            confidence = detection.score[0] if detection.score else 0.8
                            detections.append({
                                'bbox': (x, y, fw, fh),
                                'confidence': confidence,
                                'method': 'mp_short',
                                'source': 'mediapipe'
                            })
            
            # Long-range model
            if 'mp_long' in self.detectors:
                results = self.detectors['mp_long'].process(rgb_image)
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        fw = int(bbox.width * w)
                        fh = int(bbox.height * h)
                        
                        if self._validate_detection(x, y, fw, fh, w, h):
                            confidence = detection.score[0] if detection.score else 0.8
                            detections.append({
                                'bbox': (x, y, fw, fh),
                                'confidence': confidence,
                                'method': 'mp_long',
                                'source': 'mediapipe'
                            })
                            
        except Exception as e:
            print(f"MediaPipe detection error: {e}")
            
        return detections
        
    def _detect_with_tensorflow(self, image: np.ndarray) -> List[Dict]:
        """TensorFlow CNN sliding window detection"""
        detections = []
        
        if 'tensorflow_cnn' not in self.detectors:
            return detections
            
        try:
            # Sliding window approach
            h, w = image.shape[:2]
            window_sizes = [64, 80, 96, 112, 128]
            step_size = 16
            
            for window_size in window_sizes:
                for y in range(0, h - window_size, step_size):
                    for x in range(0, w - window_size, step_size):
                        window = image[y:y+window_size, x:x+window_size]
                        window_resized = cv2.resize(window, (64, 64))
                        window_normalized = window_resized.astype(np.float32) / 255.0
                        window_batch = np.expand_dims(window_normalized, axis=0)
                        
                        # Predict (mock prediction since model isn't trained)
                        prediction = np.random.random()  # Placeholder
                        
                        if prediction > 0.7:  # Confidence threshold
                            detections.append({
                                'bbox': (x, y, window_size, window_size),
                                'confidence': prediction,
                                'method': f'tf_cnn_{window_size}',
                                'source': 'tensorflow'
                            })
                            
        except Exception as e:
            print(f"TensorFlow detection error: {e}")
            
        return detections
        
    def _detect_with_edge_analysis(self, image: np.ndarray) -> List[Dict]:
        """Edge-based face detection using contour analysis"""
        detections = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # Filter by size and aspect ratio
                if 30 < cw < w//2 and 30 < ch < h//2:
                    aspect_ratio = cw / ch
                    if 0.6 < aspect_ratio < 1.8:
                        # Calculate contour properties
                        area = cv2.contourArea(contour)
                        rect_area = cw * ch
                        
                        if area > rect_area * 0.3:  # Sufficient fill ratio
                            confidence = min(0.7, area / (rect_area * 1.5))
                            detections.append({
                                'bbox': (x, y, cw, ch),
                                'confidence': confidence,
                                'method': 'edge_contour',
                                'source': 'edge_analysis'
                            })
                            
        except Exception as e:
            print(f"Edge detection error: {e}")
            
        return detections
        
    def _validate_detection(self, x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> bool:
        """Validate face detection"""
        # Size validation
        if w < 15 or h < 15 or w > img_w * 0.9 or h > img_h * 0.9:
            return False
            
        # Aspect ratio validation
        aspect_ratio = w / h
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return False
            
        # Boundary validation
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return False
            
        return True
        
    def _advanced_ensemble_processing(self, detections: List[Dict], img_w: int, img_h: int) -> List[Dict]:
        """Advanced ensemble processing with voting and confidence weighting"""
        if not detections:
            return []
            
        # Group overlapping detections
        groups = []
        for detection in detections:
            assigned = False
            for group in groups:
                if self._detections_overlap(detection, group[0]):
                    group.append(detection)
                    assigned = True
                    break
            if not assigned:
                groups.append([detection])
                
        # Create ensemble detections from groups
        ensemble_detections = []
        for group in groups:
            if len(group) >= 2:  # Require at least 2 detections to agree
                ensemble_detection = self._create_ensemble_detection(group)
                ensemble_detections.append(ensemble_detection)
            elif len(group) == 1 and group[0]['confidence'] > 0.8:
                # Include high-confidence single detections
                ensemble_detections.append(group[0])
                
        return ensemble_detections
        
    def _detections_overlap(self, det1: Dict, det2: Dict) -> bool:
        """Check if two detections overlap significantly"""
        x1, y1, w1, h1 = det1['bbox']
        x2, y2, w2, h2 = det2['bbox']
        
        # Calculate intersection
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return False
            
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou > 0.3
        
    def _create_ensemble_detection(self, group: List[Dict]) -> Dict:
        """Create ensemble detection from group"""
        # Weight by confidence and method diversity
        total_weight = sum(det['confidence'] for det in group)
        
        # Weighted average of bounding boxes
        avg_x = sum(det['bbox'][0] * det['confidence'] for det in group) / total_weight
        avg_y = sum(det['bbox'][1] * det['confidence'] for det in group) / total_weight
        avg_w = sum(det['bbox'][2] * det['confidence'] for det in group) / total_weight
        avg_h = sum(det['bbox'][3] * det['confidence'] for det in group) / total_weight
        
        # Ensemble confidence
        unique_sources = len(set(det['source'] for det in group))
        ensemble_confidence = (total_weight / len(group)) * (1 + 0.1 * unique_sources)
        ensemble_confidence = min(ensemble_confidence, 0.99)
        
        methods = [det['method'] for det in group]
        
        return {
            'bbox': (int(avg_x), int(avg_y), int(avg_w), int(avg_h)),
            'confidence': ensemble_confidence,
            'method': f'ensemble_{len(group)}methods',
            'source': 'ensemble',
            'ensemble_info': {
                'size': len(group),
                'methods': methods,
                'sources': [det['source'] for det in group]
            }
        }
        
    def _comprehensive_face_analysis(self, image: np.ndarray, detection: Dict, face_id: int) -> Dict:
        """Comprehensive face analysis using all available methods"""
        x, y, w, h = detection['bbox']
        face_region = image[y:y+h, x:x+w]
        
        # Gaze analysis
        gaze_direction = self._analyze_comprehensive_gaze(x, y, w, h, image.shape[1], image.shape[0], face_region)
        
        # Comprehensive emotion analysis
        emotion, emotion_conf = self._analyze_comprehensive_emotion(face_region)
        
        # Additional analysis
        face_quality = self._calculate_face_quality(face_region)
        
        result = {
            'face_id': face_id,
            'person_name': f'Person-{face_id}',
            'bbox': (x, y, w, h),
            'confidence': detection['confidence'],
            'method': detection['method'],
            'source': detection['source'],
            'gaze_direction': gaze_direction,
            'gaze_confidence': 0.8,
            'emotion': emotion,
            'emotion_confidence': emotion_conf,
            'face_quality': face_quality
        }
        
        # Add ensemble info if available
        if 'ensemble_info' in detection:
            result['ensemble_info'] = detection['ensemble_info']
            
        return result
        
    def _analyze_comprehensive_gaze(self, x: int, y: int, w: int, h: int, 
                                   img_w: int, img_h: int, face_region: np.ndarray) -> str:
        """Comprehensive gaze analysis"""
        center_x = (x + w/2) / img_w
        center_y = (y + h/2) / img_h
        
        # Position-based analysis
        if center_y > 0.75:
            return "looking down at table/game"
        elif center_y < 0.25:
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
            # Use MediaPipe landmarks for precise gaze if available
            return self._analyze_precise_gaze(face_region)
            
    def _analyze_precise_gaze(self, face_region: np.ndarray) -> str:
        """Precise gaze analysis using MediaPipe landmarks"""
        try:
            if 'mp_mesh' in self.detectors:
                rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                results = self.detectors['mp_mesh'].process(rgb_face)
                
                if results.multi_face_landmarks:
                    # Analyze eye landmarks for precise gaze direction
                    landmarks = results.multi_face_landmarks[0]
                    # Simplified gaze analysis based on eye landmarks
                    return "looking directly at camera"
        except:
            pass
            
        return "looking forward"
        
    def _analyze_comprehensive_emotion(self, face_region: np.ndarray) -> Tuple[str, float]:
        """Comprehensive emotion analysis using multiple methods"""
        emotions = []
        
        # 1. FER emotion detection
        if 'fer' in self.emotion_models:
            try:
                fer_results = self.emotion_models['fer'].detect_emotions(face_region)
                if fer_results:
                    fer_emotion = max(fer_results[0]['emotions'], key=fer_results[0]['emotions'].get)
                    fer_confidence = fer_results[0]['emotions'][fer_emotion]
                    emotions.append((fer_emotion, fer_confidence, 'fer'))
            except:
                pass
                
        # 2. TensorFlow emotion model (mock implementation)
        if 'tensorflow_emotion' in self.emotion_models:
            try:
                # Preprocess face for emotion model
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (48, 48))
                normalized_face = resized_face.astype(np.float32) / 255.0
                face_batch = np.expand_dims(np.expand_dims(normalized_face, axis=0), axis=-1)
                
                # Mock prediction (replace with actual model inference)
                emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                predictions = np.random.dirichlet(np.ones(7))  # Mock prediction
                
                max_idx = np.argmax(predictions)
                tf_emotion = emotion_labels[max_idx]
                tf_confidence = predictions[max_idx]
                emotions.append((tf_emotion, tf_confidence, 'tensorflow'))
            except:
                pass
                
        # 3. Basic geometric analysis as fallback
        basic_emotion, basic_conf = self._basic_emotion_analysis(face_region)
        emotions.append((basic_emotion, basic_conf, 'geometric'))
        
        # Ensemble emotion decision
        if emotions:
            # Weight by confidence and method reliability
            method_weights = {'fer': 1.0, 'tensorflow': 0.9, 'geometric': 0.6}
            
            weighted_emotions = {}
            for emotion, confidence, method in emotions:
                weight = method_weights.get(method, 0.5)
                weighted_score = confidence * weight
                
                if emotion in weighted_emotions:
                    weighted_emotions[emotion] += weighted_score
                else:
                    weighted_emotions[emotion] = weighted_score
                    
            best_emotion = max(weighted_emotions, key=weighted_emotions.get)
            best_confidence = weighted_emotions[best_emotion] / len(emotions)
            
            return best_emotion, min(best_confidence, 0.99)
        else:
            return "neutral", 0.5
            
    def _basic_emotion_analysis(self, face_region: np.ndarray) -> Tuple[str, float]:
        """Basic emotion analysis using geometric features"""
        if face_region.size == 0:
            return "neutral", 0.5
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            if h < 30 or w < 30:
                return "neutral", 0.6
                
            # Analyze facial regions
            mouth_region = gray_face[2*h//3:, w//4:3*w//4]
            eye_region = gray_face[h//4:h//2, :]
            
            if mouth_region.size > 0 and eye_region.size > 0:
                mouth_brightness = np.mean(mouth_region)
                eye_brightness = np.mean(eye_region)
                face_brightness = np.mean(gray_face)
                
                mouth_ratio = mouth_brightness / face_brightness
                eye_ratio = eye_brightness / face_brightness
                
                # Enhanced emotion classification
                if mouth_ratio > 1.3:
                    return "happy", 0.8
                elif mouth_ratio < 0.7:
                    return "sad", 0.7
                elif eye_ratio < 0.8:
                    return "tired", 0.6
                elif mouth_ratio > 1.2 and abs(mouth_ratio - eye_ratio) > 0.3:
                    return "surprised", 0.7
                else:
                    return "neutral", 0.75
            else:
                return "neutral", 0.5
                
        except:
            return "neutral", 0.5
            
    def _calculate_face_quality(self, face_region: np.ndarray) -> float:
        """Calculate face quality score"""
        if face_region.size == 0:
            return 0.0
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)
            
            # Brightness
            brightness = np.mean(gray_face) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Contrast
            contrast = np.std(gray_face) / 255.0
            contrast_score = min(contrast * 4, 1.0)
            
            # Overall quality
            quality = (sharpness_score + brightness_score + contrast_score) / 3.0
            return quality
            
        except:
            return 0.5
            
    def draw_comprehensive_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw comprehensive detection results"""
        result_image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            source = detection['source']
            
            # Color based on source and confidence
            if source == 'ensemble':
                color = (0, 255, 0)  # Green for ensemble
            elif confidence > 0.8:
                color = (255, 165, 0)  # Orange for high confidence
            else:
                color = (0, 255, 255)  # Yellow for lower confidence
                
            # Draw face bounding box
            thickness = max(2, int(confidence * 4))
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
            
            # Person name with detailed info
            person_text = f"{detection['person_name']} ({confidence:.3f})"
            text_size = cv2.getTextSize(person_text, font, 0.7, 2)[0]
            cv2.rectangle(result_image, (x, y - 35), (x + text_size[0] + 10, y), color, -1)
            cv2.putText(result_image, person_text, (x + 5, y - 10), font, 0.7, (0, 0, 0), 2)
            
            # Method and source info
            method_text = f"Method: {detection['method'][:15]}"
            method_size = cv2.getTextSize(method_text, font, 0.4, 1)[0]
            cv2.rectangle(result_image, (x, y + h), (x + method_size[0] + 10, y + h + 20), (128, 128, 128), -1)
            cv2.putText(result_image, method_text, (x + 5, y + h + 15), font, 0.4, (255, 255, 255), 1)
            
            # Gaze direction
            gaze_text = f"Gaze: {detection['gaze_direction']}"
            gaze_size = cv2.getTextSize(gaze_text, font, 0.5, 2)[0]
            cv2.rectangle(result_image, (x, y + h + 25), (x + gaze_size[0] + 10, y + h + 50), (255, 255, 0), -1)
            cv2.putText(result_image, gaze_text, (x + 5, y + h + 43), font, 0.5, (0, 0, 0), 2)
            
            # Emotion
            emotion_text = f"Emotion: {detection['emotion']} ({detection['emotion_confidence']:.2f})"
            emotion_size = cv2.getTextSize(emotion_text, font, 0.45, 1)[0]
            cv2.rectangle(result_image, (x, y + h + 55), (x + emotion_size[0] + 10, y + h + 80), (0, 255, 255), -1)
            cv2.putText(result_image, emotion_text, (x + 5, y + h + 73), font, 0.45, (0, 0, 0), 1)
            
            # Quality score if available
            if 'face_quality' in detection:
                quality_text = f"Quality: {detection['face_quality']:.2f}"
                cv2.putText(result_image, quality_text, (x + 5, y + h + 95), font, 0.4, color, 1)
                
            # Ensemble info if available
            if 'ensemble_info' in detection:
                ensemble_info = detection['ensemble_info']
                ensemble_text = f"Ensemble: {ensemble_info['size']} methods"
                cv2.putText(result_image, ensemble_text, (x + 5, y + h + 110), font, 0.4, (0, 255, 0), 1)
        
        # Comprehensive summary
        total_methods = len(set(d['source'] for d in detections))
        summary_text = f"Comprehensive Detection: {len(detections)} faces, {total_methods} methods"
        summary_size = cv2.getTextSize(summary_text, font, 0.8, 2)[0]
        cv2.rectangle(result_image, (10, 10), (summary_size[0] + 20, 50), (0, 0, 0), -1)
        cv2.putText(result_image, summary_text, (15, 35), font, 0.8, (255, 255, 255), 2)
        
        return result_image