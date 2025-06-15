import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import mediapipe as mp

class OptimizedFaceDetector:
    """Optimized face detection using multiple lightweight methods"""
    
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
        # Method 1: Enhanced Haar Cascade
        self.init_haar_cascade()
        
        # Method 2: MediaPipe (multiple models)
        self.init_mediapipe()
        
        # Method 3: Edge-based detection
        self.init_edge_detector()
        
        # Method 4: Color-space based detection
        self.init_color_detector()
        
        self.logger.info(f"Initialized {len(self.detection_methods)} detection methods")
        
    def init_haar_cascade(self):
        """Initialize enhanced Haar cascade detector"""
        try:
            cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            
            if not cascade.empty():
                self.detection_methods.append({
                    'name': 'haar_enhanced',
                    'detector': cascade,
                    'confidence': 0.8,
                    'process_func': self._detect_haar_enhanced
                })
                self.logger.info("Enhanced Haar cascade detector initialized")
        except Exception as e:
            self.logger.error(f"Haar cascade initialization error: {e}")
            
    def init_mediapipe(self):
        """Initialize MediaPipe face detection with optimized settings"""
        try:
            mp_face_detection = mp.solutions.face_detection
            
            # Ultra-sensitive short range detector
            ultra_detector = mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.1
            )
            
            # Ultra-sensitive long range detector
            long_detector = mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.1
            )
            
            self.detection_methods.extend([
                {
                    'name': 'mediapipe_ultra',
                    'detector': ultra_detector,
                    'confidence': 0.95,
                    'process_func': self._detect_mediapipe_enhanced
                },
                {
                    'name': 'mediapipe_long_ultra',
                    'detector': long_detector,
                    'confidence': 0.9,
                    'process_func': self._detect_mediapipe_enhanced
                }
            ])
            
            self.logger.info("Optimized MediaPipe detectors initialized")
        except Exception as e:
            self.logger.error(f"MediaPipe initialization error: {e}")
            
    def init_edge_detector(self):
        """Initialize edge-based face detector"""
        try:
            self.detection_methods.append({
                'name': 'edge_enhanced',
                'detector': None,
                'confidence': 0.7,
                'process_func': self._detect_edge_enhanced
            })
            self.logger.info("Enhanced edge detector initialized")
        except Exception as e:
            self.logger.error(f"Edge detector initialization error: {e}")
            
    def init_color_detector(self):
        """Initialize color-space based face detector"""
        try:
            self.detection_methods.append({
                'name': 'skin_color',
                'detector': None,
                'confidence': 0.6,
                'process_func': self._detect_skin_color
            })
            self.logger.info("Skin color detector initialized")
        except Exception as e:
            self.logger.error(f"Color detector initialization error: {e}")
    
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """
        Optimized face detection using ensemble of lightweight methods
        """
        if image is None or image.size == 0:
            return []
            
        h, w = image.shape[:2]
        if h < 50 or w < 50:
            return []
            
        all_detections = []
        
        # Run all detection methods in parallel concept
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
        
        # Advanced ensemble processing
        unique_detections = self._ensemble_processing(all_detections, w, h)
        
        # Filter by confidence threshold
        filtered_detections = [d for d in unique_detections if d['confidence'] >= confidence_threshold]
        
        # Comprehensive face analysis
        final_results = []
        for i, detection in enumerate(filtered_detections):
            face_data = self._comprehensive_face_analysis(image, detection, i + 1)
            final_results.append(face_data)
            
        self.logger.info(f"Detected {len(final_results)} faces using optimized ensemble")
        return final_results
    
    def _detect_haar_enhanced(self, image: np.ndarray, cascade) -> List[Dict]:
        """Enhanced Haar cascade detection with advanced parameters"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Comprehensive parameter grid for maximum detection
        param_combinations = [
            {'scaleFactor': 1.03, 'minNeighbors': 2, 'minSize': (15, 15), 'maxSize': (400, 400)},
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20), 'maxSize': (350, 350)},
            {'scaleFactor': 1.08, 'minNeighbors': 4, 'minSize': (25, 25), 'maxSize': (300, 300)},
            {'scaleFactor': 1.1, 'minNeighbors': 3, 'minSize': (30, 30), 'maxSize': (250, 250)},
            {'scaleFactor': 1.15, 'minNeighbors': 5, 'minSize': (35, 35), 'maxSize': (200, 200)},
            {'scaleFactor': 1.2, 'minNeighbors': 2, 'minSize': (40, 40), 'maxSize': (180, 180)},
            {'scaleFactor': 1.25, 'minNeighbors': 4, 'minSize': (45, 45), 'maxSize': (160, 160)},
        ]
        
        for params in param_combinations:
            try:
                faces = cascade.detectMultiScale(gray, **params)
                for (x, y, fw, fh) in faces:
                    if self._validate_face_region(x, y, fw, fh, w, h):
                        # Enhanced quality assessment
                        face_region = gray[y:y+fh, x:x+fw]
                        quality_score = self._advanced_quality_assessment(face_region)
                        
                        detections.append({
                            'bbox': (x, y, fw, fh),
                            'confidence': 0.75 * quality_score,
                            'source': f"haar_{params['scaleFactor']:.2f}",
                            'quality': quality_score
                        })
            except Exception as e:
                continue
        
        return detections
    
    def _detect_mediapipe_enhanced(self, image: np.ndarray, detector) -> List[Dict]:
        """Enhanced MediaPipe face detection with quality assessment"""
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
                    
                    # Quality enhancement
                    face_region = image[y:y+fh, x:x+fw]
                    quality_score = self._advanced_quality_assessment(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))
                    
                    detections.append({
                        'bbox': (x, y, fw, fh),
                        'confidence': confidence * quality_score,
                        'source': 'mediapipe_enhanced',
                        'quality': quality_score
                    })
        
        return detections
    
    def _detect_edge_enhanced(self, image: np.ndarray, detector) -> List[Dict]:
        """Enhanced edge-based face detection"""
        detections = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Multiple edge detection approaches
            edge_methods = [
                {'name': 'canny', 'params': (30, 100)},
                {'name': 'canny', 'params': (50, 150)},
                {'name': 'canny', 'params': (70, 200)},
            ]
            
            for method in edge_methods:
                # Apply edge detection
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                edges = cv2.Canny(blurred, method['params'][0], method['params'][1])
                
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # Face size filtering
                    if 800 < area < 80000:
                        x, y, fw, fh = cv2.boundingRect(contour)
                        
                        # Face-like aspect ratio
                        aspect_ratio = float(fw) / fh
                        if 0.5 < aspect_ratio < 2.0:
                            # Enhanced shape analysis
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            solidity = float(area) / hull_area if hull_area > 0 else 0
                            
                            if solidity > 0.6:  # Face-like solidity
                                confidence = min(0.8, solidity * 1.2)
                                
                                detections.append({
                                    'bbox': (x, y, fw, fh),
                                    'confidence': confidence,
                                    'source': f'edge_{method["name"]}',
                                    'solidity': solidity
                                })
                                
        except Exception as e:
            self.logger.error(f"Edge detection error: {e}")
            
        return detections
    
    def _detect_skin_color(self, image: np.ndarray, detector) -> List[Dict]:
        """Skin color based face detection"""
        detections = []
        
        try:
            # Convert to different color spaces for skin detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            # Define skin color ranges in HSV
            lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
            upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
            
            # Define skin color ranges in YCrCb
            lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
            upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
            
            # Create skin masks
            mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
            mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
            
            # Combine masks
            skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in skin regions
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if 1500 < area < 100000:  # Reasonable face area
                    x, y, fw, fh = cv2.boundingRect(contour)
                    
                    # Check aspect ratio
                    aspect_ratio = float(fw) / fh
                    if 0.6 < aspect_ratio < 1.8:
                        # Calculate skin percentage in bounding box
                        roi_mask = skin_mask[y:y+fh, x:x+fw]
                        skin_percentage = np.sum(roi_mask > 0) / (fw * fh)
                        
                        if skin_percentage > 0.3:  # At least 30% skin
                            confidence = min(0.7, skin_percentage * 1.5)
                            
                            detections.append({
                                'bbox': (x, y, fw, fh),
                                'confidence': confidence,
                                'source': 'skin_color',
                                'skin_percentage': skin_percentage
                            })
                            
        except Exception as e:
            self.logger.error(f"Skin color detection error: {e}")
            
        return detections
    
    def _advanced_quality_assessment(self, face_gray: np.ndarray) -> float:
        """Advanced face quality assessment"""
        if face_gray.size == 0:
            return 0.5
            
        try:
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 200.0)
            
            # Contrast (standard deviation)
            contrast_score = min(1.0, face_gray.std() / 100.0)
            
            # Brightness optimization
            brightness = face_gray.mean()
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            
            # Edge density (measure of facial features)
            edges = cv2.Canny(face_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(1.0, edge_density * 10)
            
            # Combine all quality metrics
            quality_score = (
                sharpness_score * 0.3 +
                contrast_score * 0.25 +
                brightness_score * 0.25 +
                edge_score * 0.2
            )
            
            return max(0.3, min(1.0, quality_score))
            
        except Exception:
            return 0.5
    
    def _ensemble_processing(self, detections: List[Dict], img_w: int, img_h: int) -> List[Dict]:
        """Advanced ensemble processing for duplicate removal and confidence boosting"""
        if not detections:
            return []
        
        # Group overlapping detections
        groups = []
        for detection in detections:
            added_to_group = False
            x, y, w, h = detection['bbox']
            
            for group in groups:
                for existing in group:
                    ex, ey, ew, eh = existing['bbox']
                    iou = self._calculate_iou((x, y, w, h), (ex, ey, ew, eh))
                    
                    if iou > 0.25:  # Lower threshold for more grouping
                        group.append(detection)
                        added_to_group = True
                        break
                
                if added_to_group:
                    break
            
            if not added_to_group:
                groups.append([detection])
        
        # Process each group
        final_detections = []
        for group in groups:
            if len(group) == 1:
                final_detections.append(group[0])
            else:
                # Create sophisticated ensemble
                ensemble_detection = self._create_sophisticated_ensemble(group)
                final_detections.append(ensemble_detection)
        
        return final_detections
    
    def _create_sophisticated_ensemble(self, group: List[Dict]) -> Dict:
        """Create sophisticated ensemble detection"""
        # Calculate weighted averages
        weights = [d['confidence'] for d in group]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return group[0]
        
        # Weighted bounding box
        x_weighted = sum(d['bbox'][0] * d['confidence'] for d in group) / total_weight
        y_weighted = sum(d['bbox'][1] * d['confidence'] for d in group) / total_weight
        w_weighted = sum(d['bbox'][2] * d['confidence'] for d in group) / total_weight
        h_weighted = sum(d['bbox'][3] * d['confidence'] for d in group) / total_weight
        
        # Ensemble confidence with method diversity bonus
        unique_methods = len(set(d['source'] for d in group))
        diversity_bonus = min(0.3, unique_methods * 0.1)
        
        avg_confidence = total_weight / len(group)
        final_confidence = min(1.0, avg_confidence + diversity_bonus)
        
        # Select best method
        best_detection = max(group, key=lambda x: x['confidence'])
        
        return {
            'bbox': (int(x_weighted), int(y_weighted), int(w_weighted), int(h_weighted)),
            'confidence': final_confidence,
            'source': f"ensemble_{best_detection['source']}",
            'ensemble_size': len(group),
            'methods_used': unique_methods,
            'diversity_score': diversity_bonus
        }
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _validate_face_region(self, x: int, y: int, fw: int, fh: int, img_w: int, img_h: int) -> bool:
        """Enhanced face region validation"""
        # Size constraints
        if fw < 15 or fh < 15 or fw > img_w * 0.9 or fh > img_h * 0.9:
            return False
            
        # Aspect ratio (more permissive)
        aspect_ratio = fw / fh
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return False
            
        # Boundary check
        if x < 0 or y < 0 or x + fw > img_w or y + fh > img_h:
            return False
            
        # Relative size check
        face_area = fw * fh
        image_area = img_w * img_h
        relative_size = face_area / image_area
        
        if relative_size < 0.0003 or relative_size > 0.95:
            return False
            
        return True
    
    def _comprehensive_face_analysis(self, image: np.ndarray, detection: Dict, face_id: int) -> Dict:
        """Comprehensive face analysis"""
        x, y, w, h = detection['bbox']
        
        # Extract face region
        face_region = image[max(0, y):min(image.shape[0], y + h), 
                          max(0, x):min(image.shape[1], x + w)]
        
        # Advanced gaze analysis
        gaze_direction = self._analyze_precise_gaze(x, y, w, h, image.shape[1], image.shape[0], face_region)
        
        # Advanced emotion analysis
        emotion, emotion_confidence = self._analyze_advanced_emotion(face_region)
        
        # Face characteristics
        characteristics = self._analyze_face_characteristics(face_region)
        
        return {
            'face_id': face_id,
            'person_name': f'Person-{face_id}',
            'bbox': (x, y, w, h),
            'confidence': detection['confidence'],
            'method': detection['source'],
            'gaze_direction': gaze_direction,
            'gaze_confidence': 0.85,
            'emotion': emotion,
            'emotion_confidence': emotion_confidence,
            'characteristics': characteristics,
            'ensemble_info': {
                'size': detection.get('ensemble_size', 1),
                'methods': detection.get('methods_used', 1),
                'diversity': detection.get('diversity_score', 0.0)
            }
        }
    
    def _analyze_precise_gaze(self, x: int, y: int, w: int, h: int, 
                             img_w: int, img_h: int, face_region: np.ndarray) -> str:
        """Precise gaze direction analysis"""
        center_x = (x + w/2) / img_w
        center_y = (y + h/2) / img_h
        
        # Detailed spatial mapping
        if center_y > 0.85:
            return "looking down at table/game board"
        elif center_y < 0.15:
            return "looking up at ceiling/away"
        elif center_x < 0.1:
            return "looking at person on far left side"
        elif center_x > 0.9:
            return "looking at person on far right side"
        elif center_x < 0.3:
            return "looking at person on left"
        elif center_x > 0.7:
            return "looking at person on right"
        else:
            # Center region - analyze face orientation
            return self._analyze_detailed_face_orientation(face_region)
    
    def _analyze_detailed_face_orientation(self, face_region: np.ndarray) -> str:
        """Detailed face orientation analysis"""
        if face_region.size == 0:
            return "looking straight ahead"
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            if h < 30 or w < 30:
                return "looking at camera"
            
            # Analyze facial symmetry
            left_half = gray_face[:, :w//2]
            right_half = gray_face[:, w//2:]
            
            left_intensity = np.mean(left_half)
            right_intensity = np.mean(right_half)
            
            intensity_diff = right_intensity - left_intensity
            
            # Enhanced eye region analysis
            eye_region = gray_face[h//4:h//2, :]
            if eye_region.size > 0:
                # Horizontal gradient analysis
                grad_x = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
                left_grad = np.mean(grad_x[:, :w//3])
                right_grad = np.mean(grad_x[:, 2*w//3:])
                
                gradient_diff = right_grad - left_grad
                
                # Combined analysis
                if intensity_diff > 8 and gradient_diff > 3:
                    return "looking slightly right"
                elif intensity_diff < -8 and gradient_diff < -3:
                    return "looking slightly left"
                elif abs(intensity_diff) < 3 and abs(gradient_diff) < 2:
                    return "looking directly at camera"
                else:
                    return "looking straight ahead"
            else:
                return "looking forward"
                
        except Exception:
            return "looking ahead"
    
    def _analyze_advanced_emotion(self, face_region: np.ndarray) -> Tuple[str, float]:
        """Advanced emotion analysis using multiple features"""
        if face_region.size == 0:
            return "neutral", 0.5
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            if h < 40 or w < 40:
                return "neutral", 0.6
            
            # Define facial regions
            forehead_region = gray_face[:h//4, :]
            eye_region = gray_face[h//4:h//2, :]
            nose_region = gray_face[h//3:2*h//3, w//3:2*w//3]
            mouth_region = gray_face[2*h//3:, w//4:3*w//4]
            
            # Calculate regional statistics
            regions = {
                'forehead': forehead_region,
                'eyes': eye_region,
                'nose': nose_region,
                'mouth': mouth_region
            }
            
            region_stats = {}
            for name, region in regions.items():
                if region.size > 0:
                    region_stats[name] = {
                        'brightness': np.mean(region),
                        'contrast': np.std(region),
                        'edges': np.sum(cv2.Canny(region, 50, 150) > 0) / region.size
                    }
            
            # Advanced emotion classification
            if 'mouth' in region_stats and 'eyes' in region_stats:
                mouth_brightness = region_stats['mouth']['brightness']
                eye_brightness = region_stats['eyes']['brightness']
                face_brightness = np.mean(gray_face)
                
                mouth_ratio = mouth_brightness / face_brightness
                eye_ratio = eye_brightness / face_brightness
                mouth_contrast = region_stats['mouth']['contrast']
                
                # Sophisticated emotion detection
                if mouth_ratio > 1.3 and mouth_contrast > 20:
                    return "happy", 0.85
                elif mouth_ratio < 0.7 and mouth_contrast < 15:
                    return "sad", 0.8
                elif eye_ratio < 0.8 and mouth_ratio < 0.9:
                    return "tired", 0.75
                elif mouth_contrast > 25 and abs(mouth_ratio - 1.0) > 0.2:
                    return "surprised", 0.7
                elif eye_ratio > 1.1 and mouth_ratio < 1.1:
                    return "focused", 0.7
                else:
                    return "neutral", 0.8
            else:
                return "neutral", 0.6
                
        except Exception:
            return "neutral", 0.5
    
    def _analyze_face_characteristics(self, face_region: np.ndarray) -> Dict:
        """Analyze detailed face characteristics"""
        characteristics = {
            'size_category': 'medium',
            'lighting_quality': 'normal',
            'face_angle': 'frontal',
            'sharpness': 'good',
            'visibility': 'clear'
        }
        
        if face_region.size == 0:
            return characteristics
            
        try:
            h, w = face_region.shape[:2]
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Size classification
            face_area = h * w
            if face_area < 1600:
                characteristics['size_category'] = 'small'
            elif face_area > 12000:
                characteristics['size_category'] = 'large'
            
            # Lighting assessment
            brightness = np.mean(gray_face)
            if brightness < 60:
                characteristics['lighting_quality'] = 'dark'
            elif brightness > 200:
                characteristics['lighting_quality'] = 'bright'
            elif 80 <= brightness <= 180:
                characteristics['lighting_quality'] = 'excellent'
            
            # Sharpness assessment
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            if laplacian_var < 50:
                characteristics['sharpness'] = 'blurry'
            elif laplacian_var > 200:
                characteristics['sharpness'] = 'excellent'
            
            # Face angle assessment
            if w > h * 1.4:
                characteristics['face_angle'] = 'profile'
            elif abs(w - h) < min(w, h) * 0.15:
                characteristics['face_angle'] = 'perfect_frontal'
            elif abs(w - h) < min(w, h) * 0.3:
                characteristics['face_angle'] = 'slight_angle'
                
            # Visibility assessment
            contrast = np.std(gray_face)
            if contrast < 20:
                characteristics['visibility'] = 'poor'
            elif contrast > 50:
                characteristics['visibility'] = 'excellent'
                
        except Exception:
            pass
            
        return characteristics
    
    def draw_optimized_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw optimized detection results with comprehensive information"""
        result_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            method = detection['method']
            ensemble_info = detection.get('ensemble_info', {})
            
            # Dynamic color coding
            if confidence > 0.9:
                color = (0, 255, 0)  # Bright green for excellent
            elif confidence > 0.7:
                color = (0, 255, 255)  # Yellow for good
            elif confidence > 0.5:
                color = (0, 165, 255)  # Orange for moderate
            else:
                color = (0, 100, 255)  # Red for low
            
            # Ensemble gets special treatment
            if ensemble_info.get('size', 1) > 1:
                color = (255, 0, 255)  # Magenta for ensemble
            
            # Dynamic thickness based on confidence
            thickness = max(2, int(confidence * 8))
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
            
            # Comprehensive labeling
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Main label with ensemble info
            ensemble_suffix = f" (E{ensemble_info.get('size', 1)})" if ensemble_info.get('size', 1) > 1 else ""
            main_text = f"{detection['person_name']}{ensemble_suffix} ({confidence:.2f})"
            
            # Draw main label
            (text_w, text_h), baseline = cv2.getTextSize(main_text, font, 0.65, 2)
            cv2.rectangle(result_image, (x, y - text_h - 18), (x + text_w + 12, y), color, -1)
            cv2.putText(result_image, main_text, (x + 6, y - 6), font, 0.65, (0, 0, 0), 2)
            
            # Gaze information
            gaze_text = f"Gaze: {detection['gaze_direction']}"
            (gaze_w, gaze_h), _ = cv2.getTextSize(gaze_text, font, 0.5, 2)
            cv2.rectangle(result_image, (x, y + h + 5), (x + gaze_w + 10, y + h + gaze_h + 15), (255, 255, 0), -1)
            cv2.putText(result_image, gaze_text, (x + 5, y + h + gaze_h + 10), font, 0.5, (0, 0, 0), 2)
            
            # Emotion information
            emotion_text = f"Emotion: {detection['emotion']} ({detection['emotion_confidence']:.2f})"
            (emo_w, emo_h), _ = cv2.getTextSize(emotion_text, font, 0.45, 1)
            cv2.rectangle(result_image, (x, y + h + gaze_h + 20), 
                        (x + emo_w + 10, y + h + gaze_h + emo_h + 30), (0, 255, 255), -1)
            cv2.putText(result_image, emotion_text, (x + 5, y + h + gaze_h + emo_h + 25), 
                      font, 0.45, (0, 0, 0), 1)
            
            # Method and quality info
            characteristics = detection.get('characteristics', {})
            quality_text = f"Quality: {characteristics.get('sharpness', 'good')}, Light: {characteristics.get('lighting_quality', 'normal')}"
            cv2.putText(result_image, quality_text, (x + 5, y + h + gaze_h + emo_h + 45), 
                      font, 0.35, (128, 128, 128), 1)
            
            method_text = f"Method: {method}"
            cv2.putText(result_image, method_text, (x + 5, y + h + gaze_h + emo_h + 60), 
                      font, 0.35, color, 1)
        
        # Comprehensive summary
        total_faces = len(detections)
        ensemble_count = sum(1 for d in detections if d.get('ensemble_info', {}).get('size', 1) > 1)
        avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
        methods_used = len(set(d['method'].split('_')[0] for d in detections))
        
        summary_lines = [
            f"Optimized Detection Results",
            f"Faces: {total_faces} | Ensembles: {ensemble_count} | Methods: {methods_used}",
            f"Average Confidence: {avg_confidence:.2f}"
        ]
        
        summary_font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 25
        
        for line in summary_lines:
            (sum_w, sum_h), _ = cv2.getTextSize(line, summary_font, 0.6, 2)
            cv2.rectangle(result_image, (10, y_offset - sum_h - 8), (sum_w + 20, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(result_image, line, (15, y_offset), summary_font, 0.6, (255, 255, 255), 2)
            y_offset += sum_h + 10
        
        return result_image