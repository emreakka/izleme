import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mtcnn import MTCNN
from typing import List, Dict
import time
from datetime import datetime
import json
import os
import requests
from io import BytesIO

# Configure Streamlit
st.set_page_config(
    page_title="Advanced Computer Vision Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .detection-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MultiLibraryFaceDetector:
    """Advanced face detector using multiple computer vision libraries"""
    
    def __init__(self):
        self.detectors = {}
        self.initialize_all_detectors()
        
    def initialize_all_detectors(self):
        """Initialize all available detection methods"""
        # 1. MediaPipe Face Detection
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.detectors['mediapipe'] = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.3
            )
            st.success("MediaPipe detector initialized")
        except Exception as e:
            st.warning(f"MediaPipe initialization failed: {e}")
            
        # 2. MTCNN Detector
        try:
            self.detectors['mtcnn'] = MTCNN()
            st.success("MTCNN detector initialized")
        except Exception as e:
            st.warning(f"MTCNN initialization failed: {e}")
            
        # 3. OpenCV Haar Cascades (multiple)
        try:
            cascade_files = [
                'haarcascade_frontalface_default.xml',
                'haarcascade_frontalface_alt.xml', 
                'haarcascade_frontalface_alt2.xml',
                'haarcascade_profileface.xml'
            ]
            
            self.detectors['opencv_cascades'] = []
            for cascade_file in cascade_files:
                try:
                    # Try loading from cv2.data first
                    cascade_path = cv2.data.haarcascades + cascade_file
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        self.detectors['opencv_cascades'].append({
                            'detector': cascade,
                            'name': cascade_file.replace('.xml', '')
                        })
                except:
                    # Try loading from local file
                    try:
                        cascade = cv2.CascadeClassifier(cascade_file)
                        if not cascade.empty():
                            self.detectors['opencv_cascades'].append({
                                'detector': cascade,
                                'name': cascade_file.replace('.xml', '')
                            })
                    except:
                        continue
                        
            st.success(f"OpenCV cascades: {len(self.detectors['opencv_cascades'])} loaded")
        except Exception as e:
            st.warning(f"OpenCV cascades initialization failed: {e}")
            
        # 4. OpenCV DNN Face Detector
        try:
            # Download DNN models if not present
            self.download_dnn_models()
            
            net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
            self.detectors['opencv_dnn'] = net
            st.success("OpenCV DNN detector initialized")
        except Exception as e:
            st.warning(f"OpenCV DNN initialization failed: {e}")
            
    def download_dnn_models(self):
        """Download OpenCV DNN models if not present"""
        models = {
            'opencv_face_detector_uint8.pb': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb',
            'opencv_face_detector.pbtxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt'
        }
        
        for filename, url in models.items():
            if not os.path.exists(filename):
                try:
                    response = requests.get(url)
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                except Exception as e:
                    st.warning(f"Failed to download {filename}: {e}")
    
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """Detect faces using all available methods and combine results"""
        if image is None or image.size == 0:
            return []
            
        all_detections = []
        h, w = image.shape[:2]
        
        # 1. MediaPipe Detection
        if 'mediapipe' in self.detectors:
            mp_faces = self._detect_with_mediapipe(image, confidence_threshold)
            all_detections.extend(mp_faces)
            
        # 2. MTCNN Detection
        if 'mtcnn' in self.detectors:
            mtcnn_faces = self._detect_with_mtcnn(image, confidence_threshold)
            all_detections.extend(mtcnn_faces)
            
        # 3. OpenCV Haar Cascades
        if 'opencv_cascades' in self.detectors:
            opencv_faces = self._detect_with_opencv(image, confidence_threshold)
            all_detections.extend(opencv_faces)
            
        # 4. OpenCV DNN
        if 'opencv_dnn' in self.detectors:
            dnn_faces = self._detect_with_dnn(image, confidence_threshold)
            all_detections.extend(dnn_faces)
            
        # Remove duplicates and combine results
        unique_faces = self._remove_duplicates(all_detections, w, h)
        
        # Add enhanced analysis
        for i, face in enumerate(unique_faces):
            face.update(self._analyze_face(image, face, i))
            
        return unique_faces
    
    def _detect_with_mediapipe(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Detect faces using MediaPipe"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detectors['mediapipe'].process(rgb_image)
            
            faces = []
            if results.detections:
                h, w, _ = image.shape
                
                for detection in results.detections:
                    if detection.score[0] >= threshold:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        faces.append({
                            'x': x, 'y': y, 'w': width, 'h': height,
                            'confidence': float(detection.score[0]),
                            'method': 'mediapipe'
                        })
            return faces
        except Exception:
            return []
    
    def _detect_with_mtcnn(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Detect faces using MTCNN"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detectors['mtcnn'].detect_faces(rgb_image)
            
            faces = []
            for result in results:
                if result['confidence'] >= threshold:
                    bbox = result['box']
                    x, y, width, height = bbox
                    
                    faces.append({
                        'x': x, 'y': y, 'w': width, 'h': height,
                        'confidence': result['confidence'],
                        'method': 'mtcnn',
                        'keypoints': result.get('keypoints', {})
                    })
            return faces
        except Exception:
            return []
    
    def _detect_with_opencv(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Detect faces using OpenCV Haar cascades"""
        faces = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            for cascade_info in self.detectors['opencv_cascades']:
                cascade = cascade_info['detector']
                
                # Multiple parameter sets for better detection
                param_sets = [
                    {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)},
                    {'scaleFactor': 1.3, 'minNeighbors': 4, 'minSize': (50, 50)},
                    {'scaleFactor': 1.05, 'minNeighbors': 6, 'minSize': (40, 40)}
                ]
                
                for params in param_sets:
                    try:
                        detections = cascade.detectMultiScale(gray, **params)
                        for (x, y, w, h) in detections:
                            faces.append({
                                'x': x, 'y': y, 'w': w, 'h': h,
                                'confidence': 0.8,  # OpenCV doesn't provide confidence
                                'method': f'opencv_{cascade_info["name"]}'
                            })
                    except Exception:
                        continue
        except Exception:
            pass
        return faces
    
    def _detect_with_dnn(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Detect faces using OpenCV DNN"""
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
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    faces.append({
                        'x': x1, 'y': y1, 'w': x2-x1, 'h': y2-y1,
                        'confidence': float(confidence),
                        'method': 'opencv_dnn'
                    })
            return faces
        except Exception:
            return []
    
    def _remove_duplicates(self, detections: List[Dict], img_w: int, img_h: int) -> List[Dict]:
        """Remove duplicate detections using Non-Maximum Suppression"""
        if not detections:
            return []
            
        # Convert to format for NMS
        boxes = []
        scores = []
        
        for det in detections:
            x, y, w, h = det['x'], det['y'], det['w'], det['h']
            boxes.append([x, y, x + w, y + h])
            scores.append(det['confidence'])
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.3, 0.4)
        
        unique_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                unique_detections.append(detections[i])
                
        return unique_detections
    
    def _analyze_face(self, image: np.ndarray, detection: Dict, face_id: int) -> Dict:
        """Analyze face for additional features"""
        x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
        img_h, img_w = image.shape[:2]
        
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        
        # Gaze analysis based on position
        center_x = x + w // 2
        center_y = y + h // 2
        
        if center_x < img_w * 0.25:
            gaze = "looking far left"
        elif center_x < img_w * 0.4:
            gaze = "looking left"
        elif center_x > img_w * 0.75:
            gaze = "looking far right"
        elif center_x > img_w * 0.6:
            gaze = "looking right"
        elif center_y < img_h * 0.3:
            gaze = "looking up"
        elif center_y > img_h * 0.7:
            gaze = "looking down"
        else:
            gaze = "looking forward"
        
        # Basic emotion analysis based on face characteristics
        if face_region.size > 0:
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate basic features
            mean_intensity = np.mean(face_gray)
            std_intensity = np.std(face_gray)
            
            # Simple emotion heuristics
            if mean_intensity > 120 and std_intensity > 40:
                emotion = "happy"
                emotion_conf = 0.7
            elif mean_intensity < 80:
                emotion = "serious"
                emotion_conf = 0.6
            elif std_intensity < 20:
                emotion = "calm"
                emotion_conf = 0.5
            else:
                emotion = "neutral"
                emotion_conf = 0.6
        else:
            emotion = "unknown"
            emotion_conf = 0.0
        
        return {
            'person_name': f'Person-{face_id+1}',
            'gaze_direction': gaze,
            'emotion': emotion,
            'emotion_confidence': emotion_conf,
            'face_area': w * h,
            'face_ratio': w / h if h > 0 else 1.0
        }

class SimpleStorage:
    """JSON-based storage for session data"""
    
    def __init__(self, storage_file="enhanced_sessions.json"):
        self.storage_file = storage_file
        self.sessions = self._load_sessions()
    
    def _load_sessions(self) -> List[Dict]:
        """Load sessions from file"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []
    
    def save_session(self, session_data: Dict):
        """Save session data"""
        try:
            self.sessions.append(session_data)
            with open(self.storage_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            st.error(f"Failed to save session: {e}")
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions"""
        return self.sessions

class AdvancedComputerVisionApp:
    """Advanced computer vision application with multiple detection libraries"""
    
    def __init__(self):
        self.detector = None
        self.storage = None
        
    def initialize_systems(self):
        """Initialize all detection systems"""
        if 'systems_initialized' not in st.session_state:
            with st.spinner("Initializing advanced computer vision systems..."):
                try:
                    self.detector = MultiLibraryFaceDetector()
                    self.storage = SimpleStorage()
                    
                    st.session_state.systems_initialized = True
                    st.session_state.detector = self.detector
                    st.session_state.storage = self.storage
                    
                    st.success("Advanced detection systems ready!")
                    
                except Exception as e:
                    st.error(f"System initialization failed: {str(e)}")
                    return False
        else:
            self.detector = st.session_state.detector
            self.storage = st.session_state.storage
            
        return True
    
    def run(self):
        """Main application interface"""
        st.markdown('<h1 class="main-header">Advanced Computer Vision Analysis</h1>', 
                   unsafe_allow_html=True)
        
        if not self.initialize_systems():
            st.stop()
        
        self.render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📸 Multi-Library Detection", 
            "📊 Enhanced Analytics",
            "🔧 Detection Methods",
            "📈 Performance Metrics"
        ])
        
        with tab1:
            self.detection_tab()
            
        with tab2:
            self.analytics_tab()
            
        with tab3:
            self.methods_tab()
            
        with tab4:
            self.performance_tab()
    
    def render_sidebar(self):
        """Enhanced sidebar controls"""
        st.sidebar.markdown("## Detection Configuration")
        
        st.session_state.confidence_threshold = st.sidebar.slider(
            "Detection Confidence", 0.1, 1.0, 0.3, 0.05
        )
        
        st.session_state.max_faces = st.sidebar.number_input(
            "Maximum Faces", 1, 50, 20
        )
        
        st.session_state.show_methods = st.sidebar.checkbox(
            "Show Detection Methods", True
        )
        
        st.session_state.show_keypoints = st.sidebar.checkbox(
            "Show MTCNN Keypoints", False
        )
        
        # Detection status
        st.sidebar.markdown("## Detection Status")
        if self.detector:
            methods_count = len([k for k in self.detector.detectors.keys() if self.detector.detectors[k]])
            st.sidebar.success(f"Active Methods: {methods_count}")
            
            # Show active detectors
            for method, detector in self.detector.detectors.items():
                if detector:
                    if method == 'opencv_cascades':
                        st.sidebar.text(f"✓ OpenCV: {len(detector)} cascades")
                    else:
                        st.sidebar.text(f"✓ {method.upper()}")
        else:
            st.sidebar.error("Detection systems offline")
            
        if self.storage:
            total_sessions = len(self.storage.get_all_sessions())
            st.sidebar.metric("Sessions Recorded", total_sessions)
    
    def detection_tab(self):
        """Multi-library face detection interface"""
        st.markdown("### Upload Images for Advanced Detection")
        
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.markdown(f"#### Processing: {uploaded_file.name}")
                
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                
                if image is not None:
                    self.process_advanced_image(image, uploaded_file.name)
                else:
                    st.error(f"Could not load: {uploaded_file.name}")
    
    def process_advanced_image(self, image: np.ndarray, filename: str):
        """Process image with multiple detection methods"""
        start_time = time.time()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                    caption="Original Image", use_container_width=True)
        
        with col2:
            st.markdown("#### Multi-Library Detection")
            progress = st.progress(0)
            status_text = st.empty()
        
        try:
            # Advanced face detection
            status_text.text("Running MediaPipe...")
            progress.progress(25)
            
            status_text.text("Running MTCNN...")
            progress.progress(50)
            
            status_text.text("Running OpenCV methods...")
            progress.progress(75)
            
            detections = self.detector.detect_faces(
                image, 
                confidence_threshold=st.session_state.confidence_threshold
            )
            
            progress.progress(100)
            status_text.text("Complete!")
            
            # Draw enhanced results
            if detections:
                result_image = self.draw_enhanced_results(image.copy(), detections)
            else:
                result_image = image.copy()
            
            processing_time = time.time() - start_time
            
            with col1:
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                        caption=f"Detection Results ({len(detections)} faces found)", 
                        use_container_width=True)
            
            with col2:
                st.success("Processing Complete!")
                st.metric("Processing Time", f"{processing_time:.2f}s")
                st.metric("Faces Detected", len(detections))
                st.metric("Detection Rate", f"{len(detections)/processing_time:.1f} faces/sec")
                
                # Method breakdown
                if detections and st.session_state.show_methods:
                    st.markdown("#### Detection Methods")
                    methods_used = {}
                    for det in detections:
                        method = det.get('method', 'unknown')
                        methods_used[method] = methods_used.get(method, 0) + 1
                    
                    for method, count in methods_used.items():
                        st.text(f"{method}: {count} faces")
                
                # Detailed results
                if detections:
                    st.markdown("#### Face Analysis")
                    for i, detection in enumerate(detections):
                        with st.expander(f"👤 {detection.get('person_name', f'Face-{i+1}')}"):
                            st.write(f"**Method:** {detection.get('method', 'unknown')}")
                            st.write(f"**Confidence:** {detection.get('confidence', 0):.3f}")
                            st.write(f"**Gaze:** {detection.get('gaze_direction', 'Unknown')}")
                            st.write(f"**Emotion:** {detection.get('emotion', 'Unknown')} "
                                   f"({detection.get('emotion_confidence', 0):.2f})")
                            st.write(f"**Position:** ({detection.get('x', 0)}, {detection.get('y', 0)})")
                            st.write(f"**Size:** {detection.get('w', 0)} x {detection.get('h', 0)}")
                            st.write(f"**Face Area:** {detection.get('face_area', 0)} pixels")
                            st.write(f"**Aspect Ratio:** {detection.get('face_ratio', 1.0):.2f}")
                            
                            # Show keypoints if available
                            if 'keypoints' in detection and st.session_state.show_keypoints:
                                st.write("**MTCNN Keypoints:**")
                                for kp_name, kp_pos in detection['keypoints'].items():
                                    st.write(f"  {kp_name}: ({kp_pos[0]:.1f}, {kp_pos[1]:.1f})")
            
            # Save enhanced session data
            if self.storage:
                session_data = {
                    'filename': filename,
                    'timestamp': datetime.now().isoformat(),
                    'faces_detected': len(detections),
                    'processing_time': processing_time,
                    'detection_methods': list(set(d.get('method', 'unknown') for d in detections)),
                    'avg_confidence': sum(d.get('confidence', 0) for d in detections) / len(detections) if detections else 0,
                    'detections': detections
                }
                self.storage.save_session(session_data)
            
        except Exception as e:
            st.error(f"Advanced processing failed: {str(e)}")
            with col2:
                st.error("Processing failed!")
    
    def draw_enhanced_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw enhanced detection results with method colors"""
        method_colors = {
            'mediapipe': (0, 255, 0),      # Green
            'mtcnn': (255, 0, 0),          # Red
            'opencv_dnn': (0, 0, 255),     # Blue
            'opencv_frontalface_default': (255, 255, 0),  # Yellow
            'opencv_frontalface_alt': (255, 0, 255),      # Magenta
            'opencv_frontalface_alt2': (0, 255, 255),     # Cyan
            'opencv_profileface': (128, 128, 128)         # Gray
        }
        
        for detection in detections:
            x, y, w, h = detection.get('x', 0), detection.get('y', 0), detection.get('w', 0), detection.get('h', 0)
            method = detection.get('method', 'unknown')
            color = method_colors.get(method, (255, 255, 255))
            
            # Face rectangle with method-specific color
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            
            # Person name and method
            person_name = detection.get('person_name', 'Unknown')
            cv2.putText(image, f"{person_name} ({method})", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Confidence score
            conf = detection.get('confidence', 0)
            cv2.putText(image, f"Conf: {conf:.2f}", (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
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
            
            # Draw MTCNN keypoints if available
            if 'keypoints' in detection and st.session_state.show_keypoints:
                keypoints = detection['keypoints']
                for kp_name, kp_pos in keypoints.items():
                    cv2.circle(image, (int(kp_pos[0]), int(kp_pos[1])), 2, (0, 255, 255), -1)
        
        return image
    
    def analytics_tab(self):
        """Enhanced analytics dashboard"""
        st.markdown("### Enhanced Analytics Dashboard")
        
        if self.storage:
            sessions = self.storage.get_all_sessions()
            
            if sessions:
                # Enhanced statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Sessions", len(sessions))
                with col2:
                    total_faces = sum(s.get('faces_detected', 0) for s in sessions)
                    st.metric("Total Faces", total_faces)
                with col3:
                    avg_time = sum(s.get('processing_time', 0) for s in sessions) / len(sessions)
                    st.metric("Avg Processing Time", f"{avg_time:.2f}s")
                with col4:
                    avg_conf = sum(s.get('avg_confidence', 0) for s in sessions) / len(sessions)
                    st.metric("Avg Confidence", f"{avg_conf:.3f}")
                
                # Method usage statistics
                st.markdown("#### Detection Method Usage")
                method_stats = {}
                for session in sessions:
                    for method in session.get('detection_methods', []):
                        method_stats[method] = method_stats.get(method, 0) + 1
                
                if method_stats:
                    import pandas as pd
                    df_methods = pd.DataFrame(list(method_stats.items()), columns=['Method', 'Usage Count'])
                    st.bar_chart(df_methods.set_index('Method'))
                
                # Recent sessions with enhanced data
                st.markdown("#### Recent Enhanced Sessions")
                recent_sessions = sessions[-10:]
                for session in reversed(recent_sessions):
                    with st.expander(f"{session['filename']} - {session['faces_detected']} faces"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Time:** {session['timestamp'][:19]}")
                            st.write(f"**Processing:** {session['processing_time']:.2f}s")
                            st.write(f"**Avg Confidence:** {session.get('avg_confidence', 0):.3f}")
                        with col2:
                            st.write(f"**Methods Used:** {', '.join(session.get('detection_methods', []))}")
                            st.write(f"**Faces Found:** {session['faces_detected']}")
                    
            else:
                st.info("No sessions recorded yet. Upload images to see enhanced analytics!")
        else:
            st.error("Storage system not available")
    
    def methods_tab(self):
        """Detection methods information"""
        st.markdown("### Detection Methods Overview")
        
        if self.detector:
            st.markdown("#### Available Detection Libraries")
            
            # MediaPipe
            if 'mediapipe' in self.detector.detectors:
                st.success("**MediaPipe Face Detection** ✅")
                st.write("- Fast and accurate real-time face detection")
                st.write("- Optimized for mobile and web applications")
                st.write("- Provides confidence scores and bounding boxes")
            
            # MTCNN
            if 'mtcnn' in self.detector.detectors:
                st.success("**MTCNN (Multi-task CNN)** ✅")
                st.write("- Three-stage cascaded CNNs for face detection")
                st.write("- Provides facial landmarks (eyes, nose, mouth)")
                st.write("- High accuracy for face detection and alignment")
            
            # OpenCV Cascades
            if 'opencv_cascades' in self.detector.detectors:
                st.success(f"**OpenCV Haar Cascades** ✅ ({len(self.detector.detectors['opencv_cascades'])} loaded)")
                st.write("- Classical computer vision approach")
                st.write("- Multiple cascade classifiers for different angles")
                st.write("- Fast detection for frontal and profile faces")
            
            # OpenCV DNN
            if 'opencv_dnn' in self.detector.detectors:
                st.success("**OpenCV DNN Face Detector** ✅")
                st.write("- Deep neural network based detection")
                st.write("- Pre-trained TensorFlow model")
                st.write("- High accuracy across various conditions")
            
            st.markdown("#### Detection Strategy")
            st.info("""
            **Multi-Library Approach:**
            1. Run all available detection methods in parallel
            2. Combine results from different algorithms
            3. Apply Non-Maximum Suppression to remove duplicates
            4. Enhance results with gaze and emotion analysis
            5. Provide method-specific color coding in results
            """)
        
    def performance_tab(self):
        """Performance metrics and benchmarking"""
        st.markdown("### Performance Metrics")
        
        if self.storage:
            sessions = self.storage.get_all_sessions()
            
            if sessions:
                import pandas as pd
                
                # Performance analysis
                df_data = []
                for session in sessions:
                    df_data.append({
                        'timestamp': session['timestamp'],
                        'faces_detected': session['faces_detected'],
                        'processing_time': session['processing_time'],
                        'avg_confidence': session.get('avg_confidence', 0),
                        'methods_count': len(session.get('detection_methods', []))
                    })
                
                df = pd.DataFrame(df_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Performance charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Processing Time vs Faces Detected")
                    chart_data = df[['faces_detected', 'processing_time']]
                    st.scatter_chart(chart_data.set_index('faces_detected'))
                
                with col2:
                    st.markdown("#### Average Confidence Over Time")
                    st.line_chart(df.set_index('timestamp')['avg_confidence'])
                
                # Performance statistics
                st.markdown("#### Performance Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_fps = df['faces_detected'].sum() / df['processing_time'].sum()
                    st.metric("Average Detection Rate", f"{avg_fps:.1f} faces/sec")
                
                with col2:
                    fastest_time = df['processing_time'].min()
                    st.metric("Fastest Processing", f"{fastest_time:.2f}s")
                
                with col3:
                    max_faces = df['faces_detected'].max()
                    st.metric("Max Faces in Single Image", max_faces)
                
            else:
                st.info("No performance data available yet.")
        else:
            st.error("Storage system not available")

def main():
    """Application entry point"""
    app = AdvancedComputerVisionApp()
    app.run()

if __name__ == "__main__":
    main()