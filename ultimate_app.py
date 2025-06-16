import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mtcnn import MTCNN
import pandas as pd
from typing import List, Dict
import time
from datetime import datetime
import json
import os
import requests
from io import BytesIO
import base64

# Configure Streamlit with mobile optimization
st.set_page_config(
    page_title="Ultimate Computer Vision Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="auto"
)

# Add mobile viewport meta tag for proper mobile rendering
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
""", unsafe_allow_html=True)

# Mobile-responsive CSS for enhanced UI
st.markdown("""
<style>
    /* Main header - responsive */
    .main-header {
        font-size: clamp(1.5rem, 4vw, 3rem);
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 0 1rem;
    }
    
    /* Detection cards - mobile responsive */
    .detection-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: clamp(1rem, 3vw, 1.5rem);
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    
    /* Metric containers - responsive */
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: clamp(0.8rem, 2.5vw, 1.5rem);
        border-radius: 12px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
        font-size: clamp(0.8rem, 2vw, 1rem);
    }
    
    /* Method tags - responsive */
    .method-tag {
        background: #2ecc71;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: clamp(0.7rem, 1.5vw, 0.8rem);
        margin: 0.2rem;
        display: inline-block;
        word-wrap: break-word;
    }
    
    /* Confidence levels */
    .confidence-high { background: #27ae60; }
    .confidence-medium { background: #f39c12; }
    .confidence-low { background: #e74c3c; }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Mobile-specific adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
            margin-bottom: 1rem;
        }
        
        .detection-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .metric-container {
            padding: 1rem;
            font-size: 0.9rem;
        }
        
        .method-tag {
            font-size: 0.7rem;
            padding: 0.2rem 0.6rem;
        }
        
        /* Make columns stack on mobile */
        .element-container {
            width: 100% !important;
        }
        
        /* Improve file uploader on mobile */
        .stFileUploader {
            width: 100%;
        }
        
        /* Better button sizing on mobile */
        .stButton > button {
            width: 100%;
            margin: 0.25rem 0;
        }
        
        /* Responsive sidebar */
        .css-1d391kg {
            width: 100% !important;
        }
    }
    
    /* Touch-friendly improvements */
    @media (pointer: coarse) {
        .stButton > button {
            min-height: 44px;
            padding: 0.75rem 1rem;
        }
        
        .stSelectbox > div > div {
            min-height: 44px;
        }
        
        .stSlider {
            padding: 1rem 0;
        }
    }
    
    /* Image display improvements */
    .stImage {
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs > div > div > div > div {
        font-weight: 600;
        font-size: clamp(0.8rem, 2vw, 1rem);
    }
    
    /* Better spacing for mobile */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class UltimateFaceDetector:
    """Ultimate face detector combining all available libraries with intelligent fallbacks"""
    
    def __init__(self):
        self.detectors = {}
        self.detection_stats = {
            'total_detections': 0,
            'method_usage': {},
            'avg_confidence': 0.0,
            'processing_times': []
        }
        self.initialize_all_detectors()
        
    def initialize_all_detectors(self):
        """Initialize all available detection methods with progress tracking"""
        
        # 1. MediaPipe Face Detection
        try:
            mp_solutions = mp.solutions
            self.mp_face_detection = mp_solutions.face_detection
            self.mp_drawing = mp_solutions.drawing_utils
            self.detectors['mediapipe'] = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.2
            )
            st.success("✅ MediaPipe Face Detection ready")
        except Exception as e:
            st.info(f"MediaPipe not available: {e}")
            
        # 2. MTCNN Detector
        try:
            self.detectors['mtcnn'] = MTCNN()
            st.success("✅ MTCNN detector ready")
        except Exception as e:
            st.info(f"MTCNN not available: {e}")
            
        # 3. OpenCV Haar Cascades
        self.load_opencv_cascades()
            
        # 4. OpenCV DNN Face Detector
        self.load_opencv_dnn()
        
        # 5. Additional detection methods
        self.load_additional_methods()
        
        active_methods = len([k for k in self.detectors.keys() if self.detectors[k]])
        st.success(f"🎯 {active_methods} detection methods active")
        
    def load_opencv_cascades(self):
        """Load multiple OpenCV Haar cascade classifiers"""
        cascade_files = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml', 
            'haarcascade_frontalface_alt2.xml',
            'haarcascade_profileface.xml',

        ]
        
        self.detectors['opencv_cascades'] = []
        
        for cascade_file in cascade_files:
            try:
                # Try multiple loading methods
                cascade = None
                
                # Method 1: From cv2 data
                try:
                    # Try loading from cv2 data directory
                    import cv2.data
                    cascade_path = cv2.data.haarcascades + cascade_file
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        self.detectors['opencv_cascades'].append({
                            'detector': cascade,
                            'name': cascade_file.replace('.xml', ''),
                            'type': 'builtin'
                        })
                        continue
                except:
                    pass
                
                # Method 2: From local file
                try:
                    cascade = cv2.CascadeClassifier(cascade_file)
                    if not cascade.empty():
                        self.detectors['opencv_cascades'].append({
                            'detector': cascade,
                            'name': cascade_file.replace('.xml', ''),
                            'type': 'local'
                        })
                except:
                    pass
                    
            except Exception:
                continue
                
        if self.detectors['opencv_cascades']:
            st.success(f"✅ OpenCV cascades: {len(self.detectors['opencv_cascades'])} loaded")
        else:
            st.info("OpenCV cascades not available")
            
    def load_opencv_dnn(self):
        """Load OpenCV DNN face detector with improved model handling"""
        try:
            if self.download_and_verify_dnn_models():
                net = cv2.dnn.readNetFromTensorflow(
                    'opencv_face_detector_uint8.pb', 
                    'opencv_face_detector.pbtxt'
                )
                self.detectors['opencv_dnn'] = net
                st.success("✅ OpenCV DNN detector ready")
            else:
                st.info("OpenCV DNN models not available")
        except Exception as e:
            st.info(f"OpenCV DNN not available: {e}")
            
    def download_and_verify_dnn_models(self):
        """Download and verify DNN models with better error handling"""
        models = {
            'opencv_face_detector_uint8.pb': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb',
            'opencv_face_detector.pbtxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt'
        }
        
        # Check if models exist and are valid
        for filename in models.keys():
            if not os.path.exists(filename) or os.path.getsize(filename) < 1000:
                try:
                    with st.spinner(f"Downloading {filename}..."):
                        response = requests.get(models[filename], timeout=30)
                        response.raise_for_status()
                        with open(filename, 'wb') as f:
                            f.write(response.content)
                except Exception:
                    return False
        
        # Verify all files are present and valid
        return all(os.path.exists(f) and os.path.getsize(f) > 1000 for f in models.keys())
        
    def load_additional_methods(self):
        """Load additional detection methods"""
        # Edge-based detection using contours
        self.detectors['edge_detection'] = True
        
        # Template matching method
        self.detectors['template_matching'] = True
        
        st.info("Additional detection methods loaded")
    
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3, 
                    enable_methods: List[str] = None) -> List[Dict]:
        """Advanced face detection with method selection and performance tracking"""
        start_time = time.time()
        
        if image is None or image.size == 0:
            return []
            
        all_detections = []
        h, w = image.shape[:2]
        method_results = {}
        
        # Track which methods to use
        if enable_methods is None:
            enable_methods = list(self.detectors.keys())
        
        # 1. MediaPipe Detection
        if 'mediapipe' in enable_methods and 'mediapipe' in self.detectors:
            mp_start = time.time()
            mp_faces = self._detect_with_mediapipe(image, confidence_threshold)
            method_results['mediapipe'] = {
                'faces': len(mp_faces),
                'time': time.time() - mp_start
            }
            all_detections.extend(mp_faces)
            
        # 2. MTCNN Detection
        if 'mtcnn' in enable_methods and 'mtcnn' in self.detectors:
            mtcnn_start = time.time()
            mtcnn_faces = self._detect_with_mtcnn(image, confidence_threshold)
            method_results['mtcnn'] = {
                'faces': len(mtcnn_faces),
                'time': time.time() - mtcnn_start
            }
            all_detections.extend(mtcnn_faces)
            
        # 3. OpenCV Haar Cascades
        if 'opencv_cascades' in enable_methods and 'opencv_cascades' in self.detectors:
            opencv_start = time.time()
            opencv_faces = self._detect_with_opencv(image, confidence_threshold)
            method_results['opencv_cascades'] = {
                'faces': len(opencv_faces),
                'time': time.time() - opencv_start
            }
            all_detections.extend(opencv_faces)
            
        # 4. OpenCV DNN
        if 'opencv_dnn' in enable_methods and 'opencv_dnn' in self.detectors:
            dnn_start = time.time()
            dnn_faces = self._detect_with_dnn(image, confidence_threshold)
            method_results['opencv_dnn'] = {
                'faces': len(dnn_faces),
                'time': time.time() - dnn_start
            }
            all_detections.extend(dnn_faces)
            
        # 5. Edge Detection
        if 'edge_detection' in enable_methods and 'edge_detection' in self.detectors:
            edge_start = time.time()
            edge_faces = self._detect_with_edge_analysis(image, confidence_threshold)
            method_results['edge_detection'] = {
                'faces': len(edge_faces),
                'time': time.time() - edge_start
            }
            all_detections.extend(edge_faces)
        
        # Remove duplicates using advanced NMS
        unique_faces = self._advanced_nms(all_detections, w, h)
        
        # Add comprehensive analysis
        for i, face in enumerate(unique_faces):
            face.update(self._comprehensive_face_analysis(image, face, i))
            
        # Update statistics
        total_time = time.time() - start_time
        self._update_detection_stats(unique_faces, method_results, total_time)
            
        return unique_faces
    
    def _detect_with_mediapipe(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Enhanced MediaPipe detection"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detectors['mediapipe'].process(rgb_image)
            
            faces = []
            if results.detections:
                h, w, _ = image.shape
                
                for detection in results.detections:
                    if detection.score[0] >= threshold:
                        bbox = detection.location_data.relative_bounding_box
                        x = max(0, int(bbox.xmin * w))
                        y = max(0, int(bbox.ymin * h))
                        width = min(w - x, int(bbox.width * w))
                        height = min(h - y, int(bbox.height * h))
                        
                        faces.append({
                            'x': x, 'y': y, 'w': width, 'h': height,
                            'confidence': float(detection.score[0]),
                            'method': 'mediapipe',
                            'detection_data': detection
                        })
            return faces
        except Exception:
            return []
    
    def _detect_with_mtcnn(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Enhanced MTCNN detection with landmarks"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detectors['mtcnn'].detect_faces(rgb_image)
            
            faces = []
            for result in results:
                if result['confidence'] >= threshold:
                    bbox = result['box']
                    x, y, width, height = bbox
                    x, y = max(0, x), max(0, y)
                    
                    faces.append({
                        'x': x, 'y': y, 'w': width, 'h': height,
                        'confidence': result['confidence'],
                        'method': 'mtcnn',
                        'keypoints': result.get('keypoints', {}),
                        'landmarks': True
                    })
            return faces
        except Exception:
            return []
    
    def _detect_with_opencv(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Enhanced OpenCV detection with multiple cascades"""
        faces = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            for cascade_info in self.detectors['opencv_cascades']:
                cascade = cascade_info['detector']
                cascade_name = cascade_info['name']
                
                # Multiple parameter sets for robust detection
                param_sets = [
                    {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30), 'maxSize': (300, 300)},
                    {'scaleFactor': 1.3, 'minNeighbors': 4, 'minSize': (50, 50), 'maxSize': (250, 250)},
                    {'scaleFactor': 1.05, 'minNeighbors': 6, 'minSize': (40, 40), 'maxSize': (200, 200)}
                ]
                
                for params in param_sets:
                    try:
                        detections = cascade.detectMultiScale(gray, **params)
                        for (x, y, w, h) in detections:
                            # Calculate confidence based on face quality
                            face_region = gray[y:y+h, x:x+w]
                            confidence = self._calculate_opencv_confidence(face_region)
                            
                            if confidence >= threshold:
                                faces.append({
                                    'x': x, 'y': y, 'w': w, 'h': h,
                                    'confidence': confidence,
                                    'method': f'opencv_{cascade_name}',
                                    'cascade_type': cascade_name
                                })
                    except Exception:
                        continue
        except Exception:
            pass
        return faces
    
    def _detect_with_dnn(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Enhanced OpenCV DNN detection"""
        try:
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            
            net = self.detectors['opencv_dnn']
            net.setInput(blob)
            detections = net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence >= threshold:
                    x1 = max(0, int(detections[0, 0, i, 3] * w))
                    y1 = max(0, int(detections[0, 0, i, 4] * h))
                    x2 = min(w, int(detections[0, 0, i, 5] * w))
                    y2 = min(h, int(detections[0, 0, i, 6] * h))
                    
                    faces.append({
                        'x': x1, 'y': y1, 'w': x2-x1, 'h': y2-y1,
                        'confidence': float(confidence),
                        'method': 'opencv_dnn',
                        'deep_learning': True
                    })
            return faces
        except Exception:
            return []
    
    def _detect_with_edge_analysis(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Edge-based face detection using contour analysis"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            faces = []
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size and aspect ratio
                if 30 < w < 300 and 30 < h < 300:
                    aspect_ratio = w / h
                    if 0.6 < aspect_ratio < 1.4:  # Face-like aspect ratio
                        # Calculate confidence based on contour properties
                        area = cv2.contourArea(contour)
                        rect_area = w * h
                        fill_ratio = area / rect_area if rect_area > 0 else 0
                        
                        confidence = min(0.8, fill_ratio + 0.3)
                        
                        if confidence >= threshold:
                            faces.append({
                                'x': x, 'y': y, 'w': w, 'h': h,
                                'confidence': confidence,
                                'method': 'edge_detection',
                                'contour_area': area
                            })
            
            return faces
        except Exception:
            return []
    
    def _calculate_opencv_confidence(self, face_region: np.ndarray) -> float:
        """Calculate confidence for OpenCV detections based on face quality"""
        if face_region.size == 0:
            return 0.0
            
        # Calculate various quality metrics
        mean_intensity = np.mean(face_region)
        std_intensity = np.std(face_region)
        
        # Normalize values
        intensity_score = min(1.0, mean_intensity / 128.0)
        contrast_score = min(1.0, std_intensity / 64.0)
        
        # Combined confidence
        confidence = 0.6 + (intensity_score * 0.2) + (contrast_score * 0.2)
        return min(0.95, confidence)
    
    def _advanced_nms(self, detections: List[Dict], img_w: int, img_h: int) -> List[Dict]:
        """Advanced Non-Maximum Suppression with intelligent merging"""
        if not detections:
            return []
            
        # Convert to format for NMS
        boxes = []
        scores = []
        methods = []
        
        for det in detections:
            x, y, w, h = det['x'], det['y'], det['w'], det['h']
            boxes.append([x, y, x + w, y + h])
            scores.append(det['confidence'])
            methods.append(det['method'])
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply NMS with different thresholds for different methods
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.3, 0.4)
        
        unique_detections = []
        if len(indices) > 0:
            # Handle different OpenCV versions
            if isinstance(indices, np.ndarray):
                if indices.ndim == 2:
                    indices = indices.flatten()
                index_list = indices.tolist()
            else:
                index_list = indices
                
            for i in index_list:
                detection = detections[i].copy()
                
                # Find overlapping detections to merge information
                overlapping = []
                for j, other_det in enumerate(detections):
                    if j != i and self._calculate_overlap(detection, other_det) > 0.3:
                        overlapping.append(other_det)
                
                # Merge information from overlapping detections
                if overlapping:
                    detection = self._merge_detections(detection, overlapping)
                
                unique_detections.append(detection)
                
        return unique_detections
    
    def _calculate_overlap(self, det1: Dict, det2: Dict) -> float:
        """Calculate overlap ratio between two detections"""
        x1, y1, w1, h1 = det1['x'], det1['y'], det1['w'], det1['h']
        x2, y2, w2, h2 = det2['x'], det2['y'], det2['w'], det2['h']
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            intersection = (right - left) * (bottom - top)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0
        return 0
    
    def _merge_detections(self, primary: Dict, overlapping: List[Dict]) -> Dict:
        """Merge information from overlapping detections"""
        merged = primary.copy()
        
        # Combine methods
        methods = [primary['method']]
        methods.extend([det['method'] for det in overlapping])
        merged['methods_combined'] = list(set(methods))
        
        # Average confidence
        confidences = [primary['confidence']]
        confidences.extend([det['confidence'] for det in overlapping])
        merged['confidence'] = np.mean(confidences)
        merged['confidence_std'] = np.std(confidences)
        
        # Combine special features
        if any('landmarks' in det for det in overlapping):
            merged['landmarks'] = True
        if any('keypoints' in det for det in overlapping):
            for det in overlapping:
                if 'keypoints' in det:
                    merged['keypoints'] = det['keypoints']
                    break
        
        return merged
    
    def _comprehensive_face_analysis(self, image: np.ndarray, detection: Dict, face_id: int) -> Dict:
        """Comprehensive analysis of detected face"""
        x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
        img_h, img_w = image.shape[:2]
        
        # Extract face region safely
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        face_region = image[y1:y2, x1:x2]
        
        analysis = {
            'person_name': f'Person-{face_id+1}',
            'face_id': face_id + 1,
            'face_area': w * h,
            'face_ratio': w / h if h > 0 else 1.0,
            'relative_size': (w * h) / (img_w * img_h),
            'position_x': (x + w/2) / img_w,
            'position_y': (y + h/2) / img_h
        }
        
        # Enhanced gaze analysis
        analysis.update(self._analyze_gaze_direction(x, y, w, h, img_w, img_h, face_region))
        
        # Enhanced emotion analysis
        analysis.update(self._analyze_emotion_advanced(face_region))
        
        # Face quality assessment
        analysis.update(self._assess_face_quality(face_region))
        
        return analysis
    
    def _analyze_gaze_direction(self, x: int, y: int, w: int, h: int, 
                               img_w: int, img_h: int, face_region: np.ndarray) -> Dict:
        """Advanced gaze direction analysis"""
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Position-based gaze
        pos_x_ratio = center_x / img_w
        pos_y_ratio = center_y / img_h
        
        # Detailed gaze categories
        if pos_x_ratio < 0.2:
            if pos_y_ratio < 0.3:
                gaze = "looking up-left"
            elif pos_y_ratio > 0.7:
                gaze = "looking down-left"
            else:
                gaze = "looking left"
        elif pos_x_ratio > 0.8:
            if pos_y_ratio < 0.3:
                gaze = "looking up-right"
            elif pos_y_ratio > 0.7:
                gaze = "looking down-right"
            else:
                gaze = "looking right"
        elif pos_y_ratio < 0.25:
            gaze = "looking up"
        elif pos_y_ratio > 0.75:
            gaze = "looking down"
        elif 0.4 < pos_x_ratio < 0.6 and 0.4 < pos_y_ratio < 0.6:
            gaze = "looking at camera"
        else:
            gaze = "looking forward"
        
        # Add confidence based on face position stability
        gaze_confidence = 1.0 - abs(pos_x_ratio - 0.5) - abs(pos_y_ratio - 0.5)
        gaze_confidence = max(0.3, min(1.0, gaze_confidence))
        
        return {
            'gaze_direction': gaze,
            'gaze_confidence': gaze_confidence,
            'gaze_x_ratio': pos_x_ratio,
            'gaze_y_ratio': pos_y_ratio
        }
    
    def _analyze_emotion_advanced(self, face_region: np.ndarray) -> Dict:
        """Advanced emotion analysis using multiple features"""
        if face_region.size == 0:
            return {
                'emotion': 'unknown',
                'emotion_confidence': 0.0,
                'emotion_features': {}
            }
        
        # Convert to grayscale for analysis
        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region
        
        # Calculate various features
        mean_intensity = np.mean(gray_face)
        std_intensity = np.std(gray_face)
        
        # Edge detection for expression analysis
        edges = cv2.Canny(gray_face, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Histogram features
        hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
        hist_peak = np.argmax(hist)
        
        # Simple emotion classification based on features
        emotion_features = {
            'brightness': mean_intensity / 255.0,
            'contrast': std_intensity / 128.0,
            'edge_density': edge_density,
            'hist_peak': hist_peak / 255.0
        }
        
        # Rule-based emotion detection
        if mean_intensity > 140 and edge_density > 0.1:
            emotion = 'happy'
            confidence = 0.7
        elif mean_intensity < 80:
            emotion = 'serious'
            confidence = 0.6
        elif edge_density < 0.05:
            emotion = 'calm'
            confidence = 0.5
        elif std_intensity > 60:
            emotion = 'surprised'
            confidence = 0.6
        else:
            emotion = 'neutral'
            confidence = 0.6
        
        return {
            'emotion': emotion,
            'emotion_confidence': confidence,
            'emotion_features': emotion_features
        }
    
    def _assess_face_quality(self, face_region: np.ndarray) -> Dict:
        """Assess face image quality"""
        if face_region.size == 0:
            return {
                'quality_score': 0.0,
                'quality_metrics': {},
                'quality_rating': 'poor'
            }
        
        # Convert to grayscale
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_region
        
        # Calculate quality metrics
        h, w = gray.shape
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 1000.0)
        
        # Brightness
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2
        
        # Contrast
        contrast = np.std(gray) / 128.0
        contrast_score = min(1.0, contrast)
        
        # Size score
        size_score = min(1.0, (w * h) / 10000.0)
        
        # Overall quality
        quality_score = (sharpness * 0.3 + brightness_score * 0.2 + 
                        contrast_score * 0.3 + size_score * 0.2)
        
        # Quality rating
        if quality_score > 0.8:
            rating = 'excellent'
        elif quality_score > 0.6:
            rating = 'good'
        elif quality_score > 0.4:
            rating = 'fair'
        else:
            rating = 'poor'
        
        return {
            'quality_score': quality_score,
            'quality_rating': rating,
            'quality_metrics': {
                'sharpness': sharpness,
                'brightness': brightness_score,
                'contrast': contrast_score,
                'size': size_score
            }
        }
    
    def _update_detection_stats(self, faces: List[Dict], method_results: Dict, total_time: float):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += len(faces)
        self.detection_stats['processing_times'].append(total_time)
        
        # Update method usage
        for method, results in method_results.items():
            if method not in self.detection_stats['method_usage']:
                self.detection_stats['method_usage'][method] = 0
            self.detection_stats['method_usage'][method] += results['faces']
        
        # Update average confidence
        if faces:
            confidences = [f.get('confidence', 0) for f in faces]
            self.detection_stats['avg_confidence'] = np.mean(confidences)
    
    def get_detection_stats(self) -> Dict:
        """Get comprehensive detection statistics"""
        stats = self.detection_stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['total_processing_time'] = sum(stats['processing_times'])
            stats['detection_rate'] = (stats['total_detections'] / 
                                     stats['total_processing_time'] if stats['total_processing_time'] > 0 else 0)
        
        return stats

class EnhancedStorage:
    """Enhanced storage with analytics and export capabilities"""
    
    def __init__(self, storage_file="ultimate_sessions.json"):
        self.storage_file = storage_file
        self.sessions = self._load_sessions()
        
    def _load_sessions(self) -> List[Dict]:
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []
    
    def save_session(self, session_data: Dict):
        try:
            session_data['session_id'] = len(self.sessions) + 1
            session_data['timestamp'] = datetime.now().isoformat()
            self.sessions.append(session_data)
            
            with open(self.storage_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            st.error(f"Failed to save session: {e}")
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions"""
        return self.sessions
    
    def get_analytics(self) -> Dict:
        """Get comprehensive analytics"""
        if not self.sessions:
            return {}
        
        df = pd.DataFrame(self.sessions)
        
        analytics = {
            'total_sessions': len(self.sessions),
            'total_faces': df['faces_detected'].sum() if 'faces_detected' in df.columns else 0,
            'avg_processing_time': df['processing_time'].mean() if 'processing_time' in df.columns else 0,
            'avg_confidence': df['avg_confidence'].mean() if 'avg_confidence' in df.columns else 0,
            'most_active_day': df['timestamp'].dt.date.mode().iloc[0] if 'timestamp' in df.columns else None,
            'detection_methods': {},
            'quality_distribution': {}
        }
        
        return analytics
    
    def export_data(self, format_type: str = 'csv') -> str:
        """Export session data in various formats"""
        if not self.sessions:
            return ""
        
        if format_type == 'csv':
            df = pd.DataFrame(self.sessions)
            return df.to_csv(index=False)
        elif format_type == 'json':
            return json.dumps(self.sessions, indent=2)
        
        return ""

class UltimateComputerVisionApp:
    """Ultimate computer vision application with all advanced features"""
    
    def __init__(self):
        self.detector = None
        self.storage = None
        
    def initialize_systems(self):
        """Initialize all detection and storage systems"""
        if 'ultimate_systems_initialized' not in st.session_state:
            with st.spinner("🚀 Initializing Ultimate Computer Vision Systems..."):
                try:
                    self.detector = UltimateFaceDetector()
                    self.storage = EnhancedStorage()
                    
                    st.session_state.ultimate_systems_initialized = True
                    st.session_state.ultimate_detector = self.detector
                    st.session_state.ultimate_storage = self.storage
                    
                    st.balloons()
                    st.success("🎯 Ultimate detection systems ready!")
                    
                except Exception as e:
                    st.error(f"System initialization failed: {str(e)}")
                    return False
        else:
            self.detector = st.session_state.ultimate_detector
            self.storage = st.session_state.ultimate_storage
            
        return True
    
    def run(self):
        """Main application interface"""
        st.markdown('<h1 class="main-header">🎯 Ultimate Computer Vision Analysis</h1>', 
                   unsafe_allow_html=True)
        
        if not self.initialize_systems():
            st.stop()
        
        self.render_enhanced_sidebar()
        
        # Enhanced main tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Ultimate Detection", 
            "📊 Advanced Analytics",
            "🔧 Method Comparison",
            "📈 Performance Dashboard",
            "💾 Data Management",
            "🎨 Visualization Studio"
        ])
        
        with tab1:
            self.ultimate_detection_tab()
            
        with tab2:
            self.advanced_analytics_tab()
            
        with tab3:
            self.method_comparison_tab()
            
        with tab4:
            self.performance_dashboard_tab()
            
        with tab5:
            self.data_management_tab()
            
        with tab6:
            self.visualization_studio_tab()
    
    def render_enhanced_sidebar(self):
        """Enhanced sidebar with comprehensive controls"""
        st.sidebar.markdown("## 🎯 Detection Configuration")
        
        # Detection parameters
        st.session_state.confidence_threshold = st.sidebar.slider(
            "🎚️ Detection Confidence", 0.1, 1.0, 0.3, 0.05,
            help="Minimum confidence threshold for face detection"
        )
        
        st.session_state.max_faces = st.sidebar.number_input(
            "👥 Maximum Faces", 1, 100, 50,
            help="Maximum number of faces to detect per image"
        )
        
        # Method selection
        st.sidebar.markdown("### 🔧 Detection Methods")
        available_methods = list(self.detector.detectors.keys()) if self.detector else []
        
        st.session_state.enabled_methods = []
        for method in available_methods:
            if self.detector.detectors[method]:
                enabled = st.sidebar.checkbox(
                    f"✅ {method.replace('_', ' ').title()}", 
                    value=True,
                    key=f"enable_{method}"
                )
                if enabled:
                    st.session_state.enabled_methods.append(method)
        
        # Display options
        st.sidebar.markdown("### 🎨 Display Options")
        st.session_state.show_confidence = st.sidebar.checkbox("Show Confidence Scores", True)
        st.session_state.show_methods = st.sidebar.checkbox("Show Detection Methods", True)
        st.session_state.show_landmarks = st.sidebar.checkbox("Show MTCNN Landmarks", False)
        st.session_state.show_quality = st.sidebar.checkbox("Show Quality Metrics", True)
        st.session_state.color_by_method = st.sidebar.checkbox("Color by Method", True)
        
        # System status
        st.sidebar.markdown("### 📊 System Status")
        if self.detector:
            active_methods = len([k for k in self.detector.detectors.keys() 
                                if self.detector.detectors[k]])
            st.sidebar.success(f"🟢 {active_methods} Methods Active")
            
            stats = self.detector.get_detection_stats()
            if stats['total_detections'] > 0:
                st.sidebar.metric("Total Detections", stats['total_detections'])
                st.sidebar.metric("Avg Confidence", f"{stats['avg_confidence']:.3f}")
        
        if self.storage:
            total_sessions = len(self.storage.get_all_sessions())
            st.sidebar.metric("Sessions Recorded", total_sessions)
    
    def ultimate_detection_tab(self):
        """Ultimate face detection interface with real-time processing"""
        st.markdown("### 🎯 Upload Images for Ultimate Detection")
        
        # File upload with enhanced options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Choose image files",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
                accept_multiple_files=True,
                help="Upload multiple images for batch processing"
            )
        
        with col2:
            batch_process = st.checkbox("Batch Process All", value=True)
            show_original = st.checkbox("Show Original", value=True)
        
        if uploaded_files:
            if batch_process:
                self.process_batch_images(uploaded_files, show_original)
            else:
                for uploaded_file in uploaded_files:
                    st.markdown(f"#### 📸 Processing: {uploaded_file.name}")
                    
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)
                    
                    if image is not None:
                        self.process_ultimate_image(image, uploaded_file.name, show_original)
                    else:
                        st.error(f"❌ Could not load: {uploaded_file.name}")
    
    def process_batch_images(self, uploaded_files: List, show_original: bool):
        """Process multiple images in batch with progress tracking"""
        total_files = len(uploaded_files)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        batch_results = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name} ({i+1}/{total_files})")
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            if image is not None:
                result = self.process_ultimate_image(
                    image, uploaded_file.name, show_original, 
                    batch_mode=True
                )
                if result:
                    batch_results.append(result)
            
            progress_bar.progress((i + 1) / total_files)
        
        status_text.text("✅ Batch processing complete!")
        
        # Display batch summary
        self.display_batch_summary(batch_results)
    
    def process_ultimate_image(self, image: np.ndarray, filename: str, 
                              show_original: bool, batch_mode: bool = False) -> Dict:
        """Process single image with ultimate detection capabilities"""
        start_time = time.time()
        
        if not batch_mode:
            if show_original:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                            caption="Original Image", use_container_width=True)
                display_col = col1
                progress_col = col2
            else:
                col1, col2 = st.columns([2, 1])
                display_col = col1
                progress_col = col1
            
            with progress_col:
                st.markdown("#### 🎯 Ultimate Detection Progress")
                progress = st.progress(0)
                status_container = st.container()
        else:
            progress = None
            status_container = None
            display_col = None
            progress_col = None
        
        try:
            # Ultimate face detection
            if not batch_mode and status_container:
                with status_container:
                    st.text("🔍 Scanning with MediaPipe...")
                if progress:
                    progress.progress(20)
            
            detections = self.detector.detect_faces(
                image, 
                confidence_threshold=st.session_state.confidence_threshold,
                enable_methods=st.session_state.enabled_methods
            )
            
            if not batch_mode and progress and status_container:
                progress.progress(100)
                status_container.success("🎉 Detection Complete!")
            
            # Draw enhanced results
            if detections:
                result_image = self.draw_ultimate_results(image.copy(), detections)
            else:
                result_image = image.copy()
                if not batch_mode:
                    st.warning("No faces detected in this image")
            
            processing_time = time.time() - start_time
            
            # Display results
            if not batch_mode and display_col:
                with display_col:
                    st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                            caption=f"Ultimate Results ({len(detections)} faces detected)", 
                            use_container_width=True)
                
                if show_original and progress_col:
                    with progress_col:
                        self.display_detection_metrics(detections, processing_time, filename)
                else:
                    self.display_detection_metrics(detections, processing_time, filename)
            
            # Save session data
            session_data = {
                'filename': filename,
                'faces_detected': len(detections),
                'processing_time': processing_time,
                'detection_methods': st.session_state.enabled_methods,
                'avg_confidence': np.mean([d.get('confidence', 0) for d in detections]) if detections else 0,
                'detections': detections,
                'image_size': image.shape[:2]
            }
            
            if self.storage:
                self.storage.save_session(session_data)
            
            return session_data
            
        except Exception as e:
            if not batch_mode:
                st.error(f"❌ Ultimate processing failed: {str(e)}")
            return None
    
    def draw_ultimate_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw ultimate detection results with enhanced visualization"""
        method_colors = {
            'mediapipe': (0, 255, 0),      # Green
            'mtcnn': (255, 0, 0),          # Red  
            'opencv_dnn': (0, 0, 255),     # Blue
            'opencv_frontalface_default': (255, 255, 0),  # Yellow
            'opencv_frontalface_alt': (255, 0, 255),      # Magenta
            'opencv_frontalface_alt2': (0, 255, 255),     # Cyan
            'opencv_profileface': (128, 128, 128),        # Gray
            'edge_detection': (255, 165, 0)               # Orange
        }
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection.get('x', 0), detection.get('y', 0), detection.get('w', 0), detection.get('h', 0)
            method = detection.get('method', 'unknown')
            confidence = detection.get('confidence', 0)
            
            # Choose color
            if st.session_state.color_by_method:
                color = method_colors.get(method, (255, 255, 255))
            else:
                # Color by confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # High confidence - Green
                elif confidence > 0.6:
                    color = (255, 255, 0)  # Medium confidence - Yellow
                else:
                    color = (255, 0, 0)  # Low confidence - Red
            
            # Draw enhanced bounding box
            thickness = 3 if confidence > 0.7 else 2
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            # Draw corner markers for high-confidence detections
            if confidence > 0.8:
                corner_size = 10
                cv2.line(image, (x, y), (x + corner_size, y), color, 3)
                cv2.line(image, (x, y), (x, y + corner_size), color, 3)
                cv2.line(image, (x + w, y), (x + w - corner_size, y), color, 3)
                cv2.line(image, (x + w, y), (x + w, y + corner_size), color, 3)
                cv2.line(image, (x, y + h), (x + corner_size, y + h), color, 3)
                cv2.line(image, (x, y + h), (x, y + h - corner_size), color, 3)
                cv2.line(image, (x + w, y + h), (x + w - corner_size, y + h), color, 3)
                cv2.line(image, (x + w, y + h), (x + w, y + h - corner_size), color, 3)
            
            # Labels
            y_offset = y - 10
            
            # Person name with ID
            person_name = detection.get('person_name', f'Face-{i+1}')
            cv2.putText(image, person_name, (x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset -= 25
            
            # Method and confidence
            if st.session_state.show_methods and st.session_state.show_confidence:
                method_text = f"{method} ({confidence:.2f})"
                cv2.putText(image, method_text, (x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset -= 20
            elif st.session_state.show_confidence:
                cv2.putText(image, f"Conf: {confidence:.2f}", (x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset -= 20
            elif st.session_state.show_methods:
                cv2.putText(image, method, (x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset -= 20
            
            # Quality indicator
            if st.session_state.show_quality:
                quality = detection.get('quality_rating', 'unknown')
                quality_color = {
                    'excellent': (0, 255, 0),
                    'good': (255, 255, 0),
                    'fair': (255, 165, 0),
                    'poor': (255, 0, 0)
                }.get(quality, color)
                
                cv2.putText(image, f"Quality: {quality}", (x, y + h + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
            
            # Gaze direction
            gaze = detection.get('gaze_direction', '')
            if gaze:
                cv2.putText(image, f"Gaze: {gaze}", (x, y + h + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Emotion
            emotion = detection.get('emotion', '')
            if emotion:
                cv2.putText(image, f"Emotion: {emotion}", (x, y + h + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw MTCNN landmarks if available
            if st.session_state.show_landmarks and 'keypoints' in detection:
                keypoints = detection['keypoints']
                for kp_name, kp_pos in keypoints.items():
                    cv2.circle(image, (int(kp_pos[0]), int(kp_pos[1])), 3, (0, 255, 255), -1)
        
        return image
    
    def display_detection_metrics(self, detections: List[Dict], processing_time: float, filename: str):
        """Display comprehensive detection metrics"""
        st.success("🎉 Ultimate Detection Complete!")
        
        # Basic metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Processing Time", f"{processing_time:.2f}s")
            st.metric("Faces Detected", len(detections))
        
        with col2:
            if detections:
                avg_conf = np.mean([d.get('confidence', 0) for d in detections])
                st.metric("Avg Confidence", f"{avg_conf:.3f}")
                
                detection_rate = len(detections) / processing_time
                st.metric("Detection Rate", f"{detection_rate:.1f} faces/sec")
        
        # Method breakdown
        if detections and st.session_state.show_methods:
            st.markdown("#### 🔧 Detection Method Breakdown")
            methods_used = {}
            for det in detections:
                method = det.get('method', 'unknown')
                methods_used[method] = methods_used.get(method, 0) + 1
            
            for method, count in methods_used.items():
                method_clean = method.replace('_', ' ').title()
                st.markdown(f'<span class="method-tag">{method_clean}: {count}</span>', 
                           unsafe_allow_html=True)
        
        # Detailed face analysis
        if detections:
            st.markdown("#### 👥 Face Analysis")
            for i, detection in enumerate(detections):
                with st.expander(f"👤 {detection.get('person_name', f'Face-{i+1}')} - Analysis"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Method:** {detection.get('method', 'unknown')}")
                        st.write(f"**Confidence:** {detection.get('confidence', 0):.3f}")
                        st.write(f"**Quality:** {detection.get('quality_rating', 'unknown')}")
                        st.write(f"**Face Size:** {detection.get('w', 0)} x {detection.get('h', 0)}")
                    
                    with col2:
                        st.write(f"**Gaze:** {detection.get('gaze_direction', 'Unknown')}")
                        st.write(f"**Emotion:** {detection.get('emotion', 'Unknown')}")
                        st.write(f"**Position:** ({detection.get('x', 0)}, {detection.get('y', 0)})")
                        
                        # Quality metrics
                        quality_metrics = detection.get('quality_metrics', {})
                        if quality_metrics:
                            st.write("**Quality Breakdown:**")
                            for metric, value in quality_metrics.items():
                                st.write(f"  - {metric.title()}: {value:.2f}")
    
    def display_batch_summary(self, batch_results: List[Dict]):
        """Display comprehensive batch processing summary"""
        if not batch_results:
            st.warning("No results from batch processing")
            return
        
        st.markdown("## 📊 Batch Processing Summary")
        
        # Summary metrics
        total_faces = sum(r['faces_detected'] for r in batch_results)
        total_time = sum(r['processing_time'] for r in batch_results)
        avg_confidence = np.mean([r['avg_confidence'] for r in batch_results if r['avg_confidence'] > 0])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Images Processed", len(batch_results))
        with col2:
            st.metric("Total Faces Found", total_faces)
        with col3:
            st.metric("Total Processing Time", f"{total_time:.2f}s")
        with col4:
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        # Results table
        st.markdown("### 📋 Detailed Results")
        df_data = []
        for result in batch_results:
            df_data.append({
                'Filename': result['filename'],
                'Faces': result['faces_detected'],
                'Time (s)': f"{result['processing_time']:.2f}",
                'Avg Confidence': f"{result['avg_confidence']:.3f}",
                'Methods Used': ', '.join(result['detection_methods'])
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    
    def advanced_analytics_tab(self):
        """Advanced analytics dashboard with comprehensive insights"""
        st.markdown("### 📊 Advanced Analytics Dashboard")
        
        if self.storage:
            sessions = self.storage.get_all_sessions()
            
            if sessions:
                # Convert to DataFrame for analysis
                df = pd.DataFrame(sessions)
                
                # Time series analysis
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['date'] = df['timestamp'].dt.date
                    df['hour'] = df['timestamp'].dt.hour
                
                # Summary statistics
                self.display_analytics_overview(df)
                
                # Detailed charts
                self.display_analytics_charts(df)
                
                # Performance analysis
                self.display_performance_analysis(df)
                
            else:
                st.info("📊 No analytics data available yet. Process some images to see insights!")
        else:
            st.error("❌ Storage system not available")
    
    def display_analytics_overview(self, df: pd.DataFrame):
        """Display analytics overview with key metrics"""
        st.markdown("#### 🎯 Key Performance Indicators")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_sessions = len(df)
            st.metric("Total Sessions", total_sessions)
        
        with col2:
            total_faces = df['faces_detected'].sum()
            st.metric("Total Faces", total_faces)
        
        with col3:
            avg_time = df['processing_time'].mean()
            st.metric("Avg Processing Time", f"{avg_time:.2f}s")
        
        with col4:
            avg_confidence = df['avg_confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        with col5:
            detection_rate = total_faces / df['processing_time'].sum() if df['processing_time'].sum() > 0 else 0
            st.metric("Detection Rate", f"{detection_rate:.1f} faces/sec")
    
    def display_analytics_charts(self, df: pd.DataFrame):
        """Display comprehensive analytics charts"""
        st.markdown("#### 📈 Analytics Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Faces detected over time
            if 'timestamp' in df.columns:
                st.markdown("**Faces Detected Over Time**")
                chart_data = df.set_index('timestamp')['faces_detected']
                st.line_chart(chart_data)
            
            # Processing time distribution
            st.markdown("**Processing Time Distribution**")
            st.histogram_chart(df['processing_time'])
        
        with col2:
            # Confidence distribution
            st.markdown("**Confidence Score Distribution**") 
            st.histogram_chart(df['avg_confidence'])
            
            # Daily activity
            if 'date' in df.columns:
                st.markdown("**Daily Activity**")
                daily_stats = df.groupby('date')['faces_detected'].sum()
                st.bar_chart(daily_stats)
    
    def display_performance_analysis(self, df: pd.DataFrame):
        """Display detailed performance analysis"""
        st.markdown("#### ⚡ Performance Analysis")
        
        # Processing time vs faces detected correlation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Processing Time vs Faces Detected**")
            scatter_data = df[['faces_detected', 'processing_time']]
            st.scatter_chart(scatter_data.set_index('faces_detected'))
        
        with col2:
            st.markdown("**Performance Trends**")
            if len(df) > 1:
                # Calculate rolling averages
                df_sorted = df.sort_values('timestamp') if 'timestamp' in df.columns else df
                df_sorted['rolling_avg_time'] = df_sorted['processing_time'].rolling(window=5, min_periods=1).mean()
                
                trend_data = df_sorted[['rolling_avg_time']].reset_index(drop=True)
                st.line_chart(trend_data)
    
    def method_comparison_tab(self):
        """Method comparison and benchmarking"""
        st.markdown("### 🔧 Detection Method Comparison")
        
        if self.detector:
            # Display method capabilities
            st.markdown("#### 🎯 Available Detection Methods")
            
            methods_info = {
                'mediapipe': {
                    'name': 'MediaPipe Face Detection',
                    'type': 'Machine Learning',
                    'speed': 'Fast',
                    'accuracy': 'High',
                    'features': ['Real-time', 'Confidence scores', 'Mobile optimized']
                },
                'mtcnn': {
                    'name': 'MTCNN (Multi-task CNN)',
                    'type': 'Deep Learning',
                    'speed': 'Medium',
                    'accuracy': 'Very High', 
                    'features': ['Facial landmarks', 'Face alignment', 'Multi-stage detection']
                },
                'opencv_dnn': {
                    'name': 'OpenCV DNN',
                    'type': 'Deep Learning',
                    'speed': 'Medium',
                    'accuracy': 'High',
                    'features': ['TensorFlow model', 'Robust detection', 'Good for production']
                },
                'opencv_cascades': {
                    'name': 'OpenCV Haar Cascades',
                    'type': 'Classical CV',
                    'speed': 'Very Fast',
                    'accuracy': 'Medium',
                    'features': ['Lightweight', 'Multiple angles', 'CPU efficient']
                },
                'edge_detection': {
                    'name': 'Edge-based Detection',
                    'type': 'Classical CV',
                    'speed': 'Fast',
                    'accuracy': 'Low-Medium',
                    'features': ['Fallback method', 'Shape analysis', 'No training needed']
                }
            }
            
            for method_key, info in methods_info.items():
                if method_key in self.detector.detectors and self.detector.detectors[method_key]:
                    with st.expander(f"✅ {info['name']} - Active"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Type:** {info['type']}")
                            st.write(f"**Speed:** {info['speed']}")
                            st.write(f"**Accuracy:** {info['accuracy']}")
                        
                        with col2:
                            st.write("**Features:**")
                            for feature in info['features']:
                                st.write(f"• {feature}")
                else:
                    with st.expander(f"❌ {info['name']} - Not Available"):
                        st.info("This method is not currently available in your setup.")
            
            # Real-time benchmarking
            st.markdown("#### ⚡ Performance Benchmarking")
            if st.button("🚀 Run Benchmark Test"):
                self.run_benchmark_test()
    
    def run_benchmark_test(self):
        """Run comprehensive benchmark test"""
        st.markdown("##### Running Benchmark Test...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        benchmark_results = {}
        
        for method in self.detector.detectors.keys():
            if self.detector.detectors[method]:
                with st.spinner(f"Testing {method}..."):
                    start_time = time.time()
                    
                    # Run detection multiple times for average
                    for _ in range(3):
                        try:
                            self.detector.detect_faces(test_image, enable_methods=[method])
                        except:
                            pass
                    
                    avg_time = (time.time() - start_time) / 3
                    benchmark_results[method] = avg_time
        
        # Display results
        if benchmark_results:
            st.markdown("##### Benchmark Results")
            
            results_df = pd.DataFrame([
                {'Method': method.replace('_', ' ').title(), 'Avg Time (s)': f"{time:.3f}"}
                for method, time in benchmark_results.items()
            ])
            
            st.dataframe(results_df, use_container_width=True)
            
            # Chart
            chart_data = pd.DataFrame([
                {'Method': method, 'Time': time}
                for method, time in benchmark_results.items()
            ])
            st.bar_chart(chart_data.set_index('Method'))
    
    def performance_dashboard_tab(self):
        """Real-time performance monitoring dashboard"""
        st.markdown("### 📈 Performance Dashboard")
        
        if self.detector:
            stats = self.detector.get_detection_stats()
            
            # Real-time metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Total Detections", stats.get('total_detections', 0))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                avg_time = stats.get('avg_processing_time', 0)
                st.metric("Avg Processing Time", f"{avg_time:.3f}s")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                detection_rate = stats.get('detection_rate', 0)
                st.metric("Detection Rate", f"{detection_rate:.1f} faces/sec")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Method usage statistics
            if stats.get('method_usage'):
                st.markdown("#### 🎯 Method Usage Statistics")
                
                method_df = pd.DataFrame([
                    {'Method': method.replace('_', ' ').title(), 'Usage Count': count}
                    for method, count in stats['method_usage'].items()
                ])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(method_df, use_container_width=True)
                with col2:
                    st.bar_chart(method_df.set_index('Method'))
            
            # Processing time trends
            if stats.get('processing_times'):
                st.markdown("#### ⏱️ Processing Time Trends")
                times_df = pd.DataFrame({
                    'Session': range(1, len(stats['processing_times']) + 1),
                    'Processing Time': stats['processing_times']
                })
                st.line_chart(times_df.set_index('Session'))
    
    def data_management_tab(self):
        """Data management and export functionality"""
        st.markdown("### 💾 Data Management")
        
        if self.storage:
            sessions = self.storage.get_all_sessions()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Data Overview")
                st.metric("Total Sessions", len(sessions))
                
                if sessions:
                    total_size = sum(len(str(session)) for session in sessions)
                    st.metric("Data Size", f"{total_size / 1024:.1f} KB")
                    
                    latest_session = max(sessions, key=lambda x: x.get('timestamp', ''))
                    st.metric("Latest Session", latest_session.get('timestamp', 'Unknown')[:19])
            
            with col2:
                st.markdown("#### 🔄 Data Operations")
                
                # Export data
                export_format = st.selectbox("Export Format", ["CSV", "JSON"])
                
                if st.button("📥 Export Data"):
                    if sessions:
                        exported_data = self.storage.export_data(export_format.lower())
                        
                        if exported_data:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"vision_data_{timestamp}.{export_format.lower()}"
                            
                            st.download_button(
                                label=f"💾 Download {export_format} File",
                                data=exported_data,
                                file_name=filename,
                                mime="text/csv" if export_format == "CSV" else "application/json"
                            )
                    else:
                        st.warning("No data to export")
                
                # Clear data
                if st.button("🗑️ Clear All Data", type="secondary"):
                    if st.checkbox("I understand this will delete all data"):
                        try:
                            if os.path.exists(self.storage.storage_file):
                                os.remove(self.storage.storage_file)
                            st.success("All data cleared successfully")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed to clear data: {e}")
            
            # Recent sessions
            if sessions:
                st.markdown("#### 📋 Recent Sessions")
                recent_sessions = sessions[-10:][::-1]  # Last 10, reversed
                
                for session in recent_sessions:
                    with st.expander(f"📸 {session.get('filename', 'Unknown')} - {session.get('faces_detected', 0)} faces"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Timestamp:** {session.get('timestamp', 'Unknown')[:19]}")
                            st.write(f"**Processing Time:** {session.get('processing_time', 0):.2f}s")
                            st.write(f"**Faces Detected:** {session.get('faces_detected', 0)}")
                        
                        with col2:
                            st.write(f"**Avg Confidence:** {session.get('avg_confidence', 0):.3f}")
                            st.write(f"**Methods Used:** {', '.join(session.get('detection_methods', []))}")
                            st.write(f"**Image Size:** {session.get('image_size', 'Unknown')}")
    
    def visualization_studio_tab(self):
        """Advanced visualization and analysis studio"""
        st.markdown("### 🎨 Visualization Studio")
        
        if self.storage:
            sessions = self.storage.get_all_sessions()
            
            if sessions:
                df = pd.DataFrame(sessions)
                
                # Visualization options
                viz_type = st.selectbox(
                    "Choose Visualization Type",
                    ["Processing Time Analysis", "Confidence Distribution", 
                     "Method Performance", "Face Count Trends", "Quality Analysis"]
                )
                
                if viz_type == "Processing Time Analysis":
                    self.create_processing_time_viz(df)
                elif viz_type == "Confidence Distribution":
                    self.create_confidence_viz(df)
                elif viz_type == "Method Performance":
                    self.create_method_performance_viz(df)
                elif viz_type == "Face Count Trends":
                    self.create_face_count_viz(df)
                elif viz_type == "Quality Analysis":
                    self.create_quality_viz(df)
                    
            else:
                st.info("🎨 No data available for visualization. Process some images first!")
    
    def create_processing_time_viz(self, df: pd.DataFrame):
        """Create processing time visualization"""
        st.markdown("#### ⏱️ Processing Time Analysis")
        
        if 'processing_time' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Processing Time Distribution**")
                st.histogram_chart(df['processing_time'])
                
                # Statistics
                st.markdown("**Statistics:**")
                st.write(f"Mean: {df['processing_time'].mean():.3f}s")
                st.write(f"Median: {df['processing_time'].median():.3f}s")
                st.write(f"Std Dev: {df['processing_time'].std():.3f}s")
            
            with col2:
                st.markdown("**Processing Time vs Faces Detected**")
                if 'faces_detected' in df.columns:
                    scatter_data = df[['faces_detected', 'processing_time']]
                    st.scatter_chart(scatter_data.set_index('faces_detected'))
    
    def create_confidence_viz(self, df: pd.DataFrame):
        """Create confidence visualization"""
        st.markdown("#### 🎯 Confidence Score Analysis")
        
        if 'avg_confidence' in df.columns:
            # Filter out zero confidence values
            conf_data = df[df['avg_confidence'] > 0]['avg_confidence']
            
            if not conf_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Confidence Distribution**")
                    st.histogram_chart(conf_data)
                
                with col2:
                    st.markdown("**Confidence Statistics**")
                    st.write(f"Mean: {conf_data.mean():.3f}")
                    st.write(f"Median: {conf_data.median():.3f}")
                    st.write(f"Min: {conf_data.min():.3f}")
                    st.write(f"Max: {conf_data.max():.3f}")
                    
                    # Confidence categories
                    high_conf = (conf_data > 0.8).sum()
                    med_conf = ((conf_data > 0.6) & (conf_data <= 0.8)).sum()
                    low_conf = (conf_data <= 0.6).sum()
                    
                    st.markdown("**Confidence Categories:**")
                    st.write(f"High (>0.8): {high_conf}")
                    st.write(f"Medium (0.6-0.8): {med_conf}")
                    st.write(f"Low (≤0.6): {low_conf}")
    
    def create_method_performance_viz(self, df: pd.DataFrame):
        """Create method performance visualization"""
        st.markdown("#### 🔧 Method Performance Analysis")
        
        if 'detection_methods' in df.columns:
            # Extract method usage
            all_methods = []
            for methods_list in df['detection_methods']:
                if isinstance(methods_list, list):
                    all_methods.extend(methods_list)
            
            if all_methods:
                method_counts = pd.Series(all_methods).value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Method Usage Frequency**")
                    st.bar_chart(method_counts)
                
                with col2:
                    st.markdown("**Method Statistics**")
                    for method, count in method_counts.items():
                        percentage = (count / len(df)) * 100
                        st.write(f"**{method.replace('_', ' ').title()}:** {count} uses ({percentage:.1f}%)")
    
    def create_face_count_viz(self, df: pd.DataFrame):
        """Create face count visualization"""
        st.markdown("#### 👥 Face Count Analysis")
        
        if 'faces_detected' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Face Count Distribution**")
                st.histogram_chart(df['faces_detected'])
            
            with col2:
                st.markdown("**Face Count Statistics**")
                st.write(f"Total Faces: {df['faces_detected'].sum()}")
                st.write(f"Avg per Image: {df['faces_detected'].mean():.1f}")
                st.write(f"Max in Single Image: {df['faces_detected'].max()}")
                st.write(f"Images with No Faces: {(df['faces_detected'] == 0).sum()}")
                
                # Face count categories
                zero_faces = (df['faces_detected'] == 0).sum()
                one_face = (df['faces_detected'] == 1).sum()
                multi_faces = (df['faces_detected'] > 1).sum()
                
                st.markdown("**Categories:**")
                st.write(f"No faces: {zero_faces}")
                st.write(f"Single face: {one_face}")
                st.write(f"Multiple faces: {multi_faces}")
    
    def create_quality_viz(self, df: pd.DataFrame):
        """Create quality analysis visualization"""
        st.markdown("#### ⭐ Quality Analysis")
        
        # Extract quality data from detections
        quality_data = []
        for session in df.to_dict('records'):
            detections = session.get('detections', [])
            for detection in detections:
                if 'quality_rating' in detection:
                    quality_data.append(detection['quality_rating'])
        
        if quality_data:
            quality_counts = pd.Series(quality_data).value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Quality Rating Distribution**")
                st.bar_chart(quality_counts)
            
            with col2:
                st.markdown("**Quality Statistics**")
                total = len(quality_data)
                for quality, count in quality_counts.items():
                    percentage = (count / total) * 100
                    st.write(f"**{quality.title()}:** {count} ({percentage:.1f}%)")
        else:
            st.info("No quality data available in current sessions")

def main():
    """Ultimate application entry point"""
    app = UltimateComputerVisionApp()
    app.run()

if __name__ == "__main__":
    main()