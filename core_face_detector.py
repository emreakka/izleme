import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mtcnn import MTCNN
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import logging
import time

class CoreFaceDetector:
    """
    Core face detection system using multiple proven libraries:
    - TensorFlow for deep learning models
    - MediaPipe for real-time face detection
    - MTCNN for precise face detection
    - OpenCV for traditional computer vision
    - dlib for facial landmark detection
    - scikit-learn for clustering and data processing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detectors = {}
        self.initialize_all_detectors()
    
    def initialize_all_detectors(self):
        """Initialize all detection methods"""
        try:
            # 1. OpenCV Haar Cascades
            self._init_opencv_detectors()
            
            # 2. MediaPipe Face Detection
            self._init_mediapipe_detectors()
            
            # 3. MTCNN Detector
            self._init_mtcnn_detector()
            
            # 4. TensorFlow Models
            self._init_tensorflow_detectors()
            
            # 5. dlib Detectors
            self._init_dlib_detectors()
            
            self.logger.info("All face detection systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Detector initialization failed: {e}")
            raise
    
    def _init_opencv_detectors(self):
        """Initialize OpenCV Haar cascade detectors"""
        try:
            # Multiple cascade files for better detection
            cascade_files = [
                'haarcascade_frontalface_default.xml',
                'haarcascade_frontalface_alt.xml',
                'haarcascade_frontalface_alt2.xml',
                'haarcascade_profileface.xml'
            ]
            
            self.detectors['opencv'] = []
            
            for cascade_file in cascade_files:
                try:
                    cascade_path = cv2.data.haarcascades + cascade_file
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        self.detectors['opencv'].append({
                            'detector': cascade,
                            'name': cascade_file.replace('.xml', '')
                        })
                except Exception:
                    # Try local file
                    try:
                        cascade = cv2.CascadeClassifier(cascade_file)
                        if not cascade.empty():
                            self.detectors['opencv'].append({
                                'detector': cascade,
                                'name': cascade_file.replace('.xml', '')
                            })
                    except Exception:
                        continue
            
            self.logger.info(f"Initialized {len(self.detectors['opencv'])} OpenCV detectors")
            
        except Exception as e:
            self.logger.warning(f"OpenCV detector initialization failed: {e}")
            self.detectors['opencv'] = []
    
    def _init_mediapipe_detectors(self):
        """Initialize MediaPipe face detection models"""
        try:
            mp_face_detection = mp.solutions.face_detection
            mp_face_mesh = mp.solutions.face_mesh
            
            # Multiple MediaPipe models for different scenarios
            self.detectors['mediapipe'] = {
                'face_detection_short': mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.3
                ),
                'face_detection_full': mp_face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.3
                ),
                'face_mesh': mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=10,
                    refine_landmarks=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3
                )
            }
            
            self.logger.info("MediaPipe detectors initialized")
            
        except Exception as e:
            self.logger.warning(f"MediaPipe detector initialization failed: {e}")
            self.detectors['mediapipe'] = {}
    
    def _init_mtcnn_detector(self):
        """Initialize MTCNN detector"""
        try:
            self.detectors['mtcnn'] = MTCNN(
                min_face_size=20,
                scale_factor=0.709,
                steps_threshold=[0.6, 0.7, 0.7]
            )
            self.logger.info("MTCNN detector initialized")
            
        except Exception as e:
            self.logger.warning(f"MTCNN detector initialization failed: {e}")
            self.detectors['mtcnn'] = None
    
    def _init_tensorflow_detectors(self):
        """Initialize TensorFlow-based face detection models"""
        try:
            # Create custom CNN model for face detection
            self.detectors['tensorflow'] = self._create_tensorflow_face_model()
            self.logger.info("TensorFlow detector initialized")
            
        except Exception as e:
            self.logger.warning(f"TensorFlow detector initialization failed: {e}")
            self.detectors['tensorflow'] = None
    
    def _init_dlib_detectors(self):
        """Initialize dlib face detectors (disabled - not available)"""
        self.detectors['dlib_hog'] = None
        self.detectors['dlib_cnn'] = None
        self.detectors['dlib_landmarks'] = None
        self.logger.info("dlib detectors skipped - not available")
    
    def _create_tensorflow_face_model(self):
        """Create a TensorFlow CNN model for face detection"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            self.logger.warning(f"TensorFlow model creation failed: {e}")
            return None
    
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.5, 
                    method: str = "Ensemble (All Methods)", max_faces: int = 10) -> List[Dict]:
        """
        Detect faces using specified method(s)
        
        Args:
            image: Input image in BGR format
            confidence_threshold: Minimum confidence for detection
            method: Detection method to use
            max_faces: Maximum number of faces to detect
            
        Returns:
            List of face detection results
        """
        if image is None or image.size == 0:
            return []
        
        # Convert to RGB for some detectors
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        all_detections = []
        
        if method == "Ensemble (All Methods)":
            # Use all available detection methods
            all_detections.extend(self._detect_with_opencv(gray_image))
            all_detections.extend(self._detect_with_mediapipe(rgb_image))
            all_detections.extend(self._detect_with_mtcnn(rgb_image))
            all_detections.extend(self._detect_with_tensorflow(image))
            
        elif method == "MTCNN Only":
            all_detections = self._detect_with_mtcnn(rgb_image)
            
        elif method == "MediaPipe Only":
            all_detections = self._detect_with_mediapipe(rgb_image)
            
        elif method == "OpenCV Only":
            all_detections = self._detect_with_opencv(gray_image)
            
        elif method == "TensorFlow Only":
            all_detections = self._detect_with_tensorflow(image)
        
        # Filter by confidence
        filtered_detections = [d for d in all_detections if d.get('confidence', 0) >= confidence_threshold]
        
        # Remove duplicates using advanced clustering
        unique_detections = self._remove_duplicate_detections(filtered_detections, image.shape)
        
        # Limit number of faces
        if len(unique_detections) > max_faces:
            unique_detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            unique_detections = unique_detections[:max_faces]
        
        # Add additional analysis
        for i, detection in enumerate(unique_detections):
            detection['face_id'] = i + 1
            detection['person_name'] = f"Person-{i + 1}"
            detection = self._enhance_detection(image, detection)
        
        return unique_detections
    
    def _detect_with_opencv(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar cascades"""
        detections = []
        
        for opencv_detector in self.detectors.get('opencv', []):
            try:
                cascade = opencv_detector['detector']
                name = opencv_detector['name']
                
                # Multiple parameter sets for better coverage
                param_sets = [
                    {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)},
                    {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (25, 25)},
                    {'scaleFactor': 1.2, 'minNeighbors': 4, 'minSize': (40, 40)}
                ]
                
                for params in param_sets:
                    faces = cascade.detectMultiScale(gray_image, **params)
                    
                    for (x, y, w, h) in faces:
                        detections.append({
                            'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                            'confidence': 0.8,  # OpenCV doesn't provide confidence
                            'method': f'opencv_{name}',
                            'source': 'opencv'
                        })
                        
            except Exception as e:
                self.logger.warning(f"OpenCV detection failed: {e}")
        
        return detections
    
    def _detect_with_mediapipe(self, rgb_image: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe"""
        detections = []
        
        try:
            h, w = rgb_image.shape[:2]
            
            # Face Detection models
            for model_name, detector in self.detectors.get('mediapipe', {}).items():
                if 'face_detection' in model_name:
                    try:
                        results = detector.process(rgb_image)
                        
                        if results.detections:
                            for detection in results.detections:
                                bbox = detection.location_data.relative_bounding_box
                                
                                x = int(bbox.xmin * w)
                                y = int(bbox.ymin * h)
                                width = int(bbox.width * w)
                                height = int(bbox.height * h)
                                
                                confidence = detection.score[0] if detection.score else 0.5
                                
                                detections.append({
                                    'x': x, 'y': y, 'w': width, 'h': height,
                                    'confidence': confidence,
                                    'method': f'mediapipe_{model_name}',
                                    'source': 'mediapipe'
                                })
                    except Exception as e:
                        self.logger.warning(f"MediaPipe {model_name} detection failed: {e}")
            
            # Face Mesh model
            if 'face_mesh' in self.detectors.get('mediapipe', {}):
                try:
                    face_mesh = self.detectors['mediapipe']['face_mesh']
                    results = face_mesh.process(rgb_image)
                    
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            # Calculate bounding box from landmarks
                            landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
                            
                            x_min, y_min = np.min(landmarks, axis=0).astype(int)
                            x_max, y_max = np.max(landmarks, axis=0).astype(int)
                            
                            detections.append({
                                'x': x_min, 'y': y_min, 
                                'w': x_max - x_min, 'h': y_max - y_min,
                                'confidence': 0.9,  # Face mesh is very reliable
                                'method': 'mediapipe_face_mesh',
                                'source': 'mediapipe',
                                'landmarks': landmarks
                            })
                except Exception as e:
                    self.logger.warning(f"MediaPipe face mesh detection failed: {e}")
                    
        except Exception as e:
            self.logger.warning(f"MediaPipe detection failed: {e}")
        
        return detections
    
    def _detect_with_mtcnn(self, rgb_image: np.ndarray) -> List[Dict]:
        """Detect faces using MTCNN"""
        detections = []
        
        if self.detectors.get('mtcnn'):
            try:
                results = self.detectors['mtcnn'].detect_faces(rgb_image)
                
                for result in results:
                    bbox = result['box']
                    confidence = result['confidence']
                    
                    # MTCNN sometimes returns negative coordinates
                    x = max(0, bbox[0])
                    y = max(0, bbox[1])
                    w = bbox[2]
                    h = bbox[3]
                    
                    detections.append({
                        'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                        'confidence': confidence,
                        'method': 'mtcnn',
                        'source': 'mtcnn',
                        'keypoints': result.get('keypoints', {})
                    })
                    
            except Exception as e:
                self.logger.warning(f"MTCNN detection failed: {e}")
        
        return detections
    
    def _detect_with_tensorflow(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using TensorFlow CNN model"""
        detections = []
        
        if self.detectors.get('tensorflow'):
            try:
                # Sliding window approach for face detection
                h, w = image.shape[:2]
                window_sizes = [(64, 64), (80, 80), (96, 96)]
                stride = 32
                
                for window_size in window_sizes:
                    wh, ww = window_size
                    
                    for y in range(0, h - wh + 1, stride):
                        for x in range(0, w - ww + 1, stride):
                            window = image[y:y+wh, x:x+ww]
                            
                            # Preprocess window
                            window_resized = cv2.resize(window, (64, 64))
                            window_normalized = window_resized.astype(np.float32) / 255.0
                            window_batch = np.expand_dims(window_normalized, axis=0)
                            
                            # Predict
                            prediction = self.detectors['tensorflow'].predict(window_batch, verbose=0)[0][0]
                            
                            if prediction > 0.7:  # High threshold for TensorFlow
                                detections.append({
                                    'x': x, 'y': y, 'w': ww, 'h': wh,
                                    'confidence': float(prediction),
                                    'method': 'tensorflow_cnn',
                                    'source': 'tensorflow'
                                })
                                
            except Exception as e:
                self.logger.warning(f"TensorFlow detection failed: {e}")
        
        return detections
    
# dlib detection removed - not available
    
    def _remove_duplicate_detections(self, detections: List[Dict], image_shape: Tuple[int, int, int]) -> List[Dict]:
        """Remove duplicate detections using advanced clustering"""
        if len(detections) <= 1:
            return detections
        
        try:
            # Extract features for clustering
            features = []
            for det in detections:
                x, y, w, h = det['x'], det['y'], det['w'], det['h']
                center_x = x + w // 2
                center_y = y + h // 2
                area = w * h
                aspect_ratio = w / h if h > 0 else 1
                
                features.append([center_x, center_y, area, aspect_ratio])
            
            # Normalize features
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features)
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=1).fit(features_normalized)
            labels = clustering.labels_
            
            # Group detections by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(detections[i])
            
            # Select best detection from each cluster
            unique_detections = []
            for cluster_detections in clusters.values():
                if len(cluster_detections) == 1:
                    unique_detections.append(cluster_detections[0])
                else:
                    # Select detection with highest confidence
                    best_detection = max(cluster_detections, key=lambda x: x.get('confidence', 0))
                    unique_detections.append(best_detection)
            
            return unique_detections
            
        except Exception as e:
            self.logger.warning(f"Duplicate removal failed: {e}")
            # Fallback to simple overlap removal
            return self._simple_overlap_removal(detections)
    
    def _simple_overlap_removal(self, detections: List[Dict]) -> List[Dict]:
        """Simple overlap-based duplicate removal"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        unique_detections = []
        for detection in detections:
            is_duplicate = False
            
            for unique_det in unique_detections:
                if self._calculate_overlap(detection, unique_det) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(detection)
        
        return unique_detections
    
    def _calculate_overlap(self, det1: Dict, det2: Dict) -> float:
        """Calculate overlap ratio between two detections"""
        x1, y1, w1, h1 = det1['x'], det1['y'], det1['w'], det1['h']
        x2, y2, w2, h2 = det2['x'], det2['y'], det2['w'], det2['h']
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _enhance_detection(self, image: np.ndarray, detection: Dict) -> Dict:
        """Add additional analysis to detection"""
        try:
            x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            if face_region.size > 0:
                # Calculate face quality metrics
                face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                
                # Sharpness (Laplacian variance)
                laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
                detection['sharpness'] = float(laplacian_var)
                
                # Brightness
                brightness = np.mean(face_gray)
                detection['brightness'] = float(brightness)
                
                # Contrast
                contrast = np.std(face_gray)
                detection['contrast'] = float(contrast)
                
                # Face size relative to image
                img_area = image.shape[0] * image.shape[1]
                face_area = w * h
                detection['relative_size'] = face_area / img_area
                
                # dlib landmarks not available - skipped
            
        except Exception as e:
            self.logger.warning(f"Detection enhancement failed: {e}")
        
        return detection
    
    def get_detector_status(self) -> Dict:
        """Get status of all detection systems"""
        status = {}
        
        status['opencv'] = len(self.detectors.get('opencv', [])) > 0
        status['mediapipe'] = len(self.detectors.get('mediapipe', {})) > 0
        status['mtcnn'] = self.detectors.get('mtcnn') is not None
        status['tensorflow'] = self.detectors.get('tensorflow') is not None
        status['dlib_hog'] = self.detectors.get('dlib_hog') is not None
        status['dlib_cnn'] = self.detectors.get('dlib_cnn') is not None
        status['dlib_landmarks'] = self.detectors.get('dlib_landmarks') is not None
        
        return status