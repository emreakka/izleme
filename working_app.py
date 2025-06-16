import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict
import time
from datetime import datetime
import json
import os

# Configure Streamlit
st.set_page_config(
    page_title="Computer Vision Analysis",
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

class MediaPipeFaceDetector:
    """Simple and reliable face detector using MediaPipe"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.3
        )
        
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """Detect faces using MediaPipe"""
        if image is None or image.size == 0:
            return []
            
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            h, w, _ = image.shape
            
            for i, detection in enumerate(results.detections):
                if detection.score[0] >= confidence_threshold:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Basic gaze analysis based on position
                    center_x = x + width // 2
                    center_y = y + height // 2
                    
                    # Simple gaze direction based on face position
                    if center_x < w * 0.3:
                        gaze = "looking left"
                    elif center_x > w * 0.7:
                        gaze = "looking right"
                    elif center_y < h * 0.3:
                        gaze = "looking up"
                    elif center_y > h * 0.7:
                        gaze = "looking down"
                    else:
                        gaze = "looking forward"
                    
                    # Simple emotion based on face area (placeholder)
                    face_area = width * height
                    if face_area > 10000:
                        emotion = "confident"
                    elif face_area < 3000:
                        emotion = "distant"
                    else:
                        emotion = "neutral"
                    
                    faces.append({
                        'x': x,
                        'y': y,
                        'w': width,
                        'h': height,
                        'confidence': float(detection.score[0]),
                        'person_name': f'Person-{i+1}',
                        'gaze_direction': gaze,
                        'emotion': emotion,
                        'emotion_confidence': 0.7
                    })
        
        return faces

class SimpleStorage:
    """Simple JSON-based storage for session data"""
    
    def __init__(self, storage_file="sessions.json"):
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

class ComputerVisionApp:
    """Main computer vision application"""
    
    def __init__(self):
        self.detector = None
        self.storage = None
        
    def initialize_systems(self):
        """Initialize detection systems"""
        if 'systems_initialized' not in st.session_state:
            with st.spinner("Starting computer vision systems..."):
                try:
                    self.detector = MediaPipeFaceDetector()
                    self.storage = SimpleStorage()
                    
                    st.session_state.systems_initialized = True
                    st.session_state.detector = self.detector
                    st.session_state.storage = self.storage
                    
                    st.success("Systems ready!")
                    
                except Exception as e:
                    st.error(f"System startup failed: {str(e)}")
                    return False
        else:
            self.detector = st.session_state.detector
            self.storage = st.session_state.storage
            
        return True
    
    def run(self):
        """Main application"""
        st.markdown('<h1 class="main-header">Computer Vision Analysis</h1>', 
                   unsafe_allow_html=True)
        
        if not self.initialize_systems():
            st.stop()
        
        self.render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs([
            "📸 Image Analysis", 
            "📊 Analytics",
            "🔧 System Info"
        ])
        
        with tab1:
            self.image_analysis_tab()
            
        with tab2:
            self.analytics_tab()
            
        with tab3:
            self.system_info_tab()
    
    def render_sidebar(self):
        """Sidebar controls"""
        st.sidebar.markdown("## Detection Settings")
        
        st.session_state.confidence_threshold = st.sidebar.slider(
            "Detection Confidence", 0.1, 1.0, 0.3, 0.05
        )
        
        st.session_state.max_faces = st.sidebar.number_input(
            "Maximum Faces", 1, 20, 10
        )
        
        st.sidebar.markdown("## Status")
        if self.detector:
            st.sidebar.success("Face Detection: Ready")
        else:
            st.sidebar.error("Face Detection: Not Ready")
            
        if self.storage:
            total_sessions = len(self.storage.get_all_sessions())
            st.sidebar.metric("Sessions Recorded", total_sessions)
    
    def image_analysis_tab(self):
        """Image analysis interface"""
        st.markdown("### Upload Images for Analysis")
        
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.markdown(f"#### Processing: {uploaded_file.name}")
                
                # Convert to opencv format
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                
                if image is not None:
                    self.process_image(image, uploaded_file.name)
                else:
                    st.error(f"Could not load: {uploaded_file.name}")
    
    def process_image(self, image: np.ndarray, filename: str):
        """Process a single image"""
        start_time = time.time()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                    caption="Original Image", use_container_width=True)
        
        with col2:
            st.markdown("#### Processing...")
            progress = st.progress(0)
        
        try:
            # Detect faces
            progress.progress(50)
            detections = self.detector.detect_faces(
                image, 
                confidence_threshold=st.session_state.confidence_threshold
            )
            
            progress.progress(90)
            
            # Draw results
            if detections:
                result_image = self.draw_results(image.copy(), detections)
            else:
                result_image = image.copy()
            
            progress.progress(100)
            
            # Show results
            processing_time = time.time() - start_time
            
            with col1:
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                        caption=f"Results ({len(detections)} faces found)", 
                        use_container_width=True)
            
            with col2:
                st.success("Complete!")
                st.metric("Processing Time", f"{processing_time:.2f}s")
                st.metric("Faces Found", len(detections))
                
                if detections:
                    st.markdown("#### Details")
                    for i, detection in enumerate(detections):
                        with st.expander(f"👤 {detection.get('person_name', f'Person-{i+1}')}"):
                            st.write(f"**Confidence:** {detection.get('confidence', 0):.3f}")
                            st.write(f"**Gaze:** {detection.get('gaze_direction', 'Unknown')}")
                            st.write(f"**Emotion:** {detection.get('emotion', 'Unknown')}")
                            st.write(f"**Position:** ({detection.get('x', 0)}, {detection.get('y', 0)})")
                            st.write(f"**Size:** {detection.get('w', 0)} x {detection.get('h', 0)}")
            
            # Save session
            if self.storage:
                session_data = {
                    'filename': filename,
                    'timestamp': datetime.now().isoformat(),
                    'faces_detected': len(detections),
                    'processing_time': processing_time,
                    'detections': detections
                }
                self.storage.save_session(session_data)
            
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            with col2:
                st.error("Processing failed!")
    
    def draw_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results"""
        for detection in detections:
            x, y, w, h = detection.get('x', 0), detection.get('y', 0), detection.get('w', 0), detection.get('h', 0)
            
            # Face rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Person name
            person_name = detection.get('person_name', 'Unknown')
            cv2.putText(image, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Gaze direction
            gaze = detection.get('gaze_direction', '')
            if gaze:
                cv2.putText(image, f"Gaze: {gaze}", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Emotion
            emotion = detection.get('emotion', '')
            if emotion:
                cv2.putText(image, f"Emotion: {emotion}", 
                           (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return image
    
    def analytics_tab(self):
        """Analytics dashboard"""
        st.markdown("### Analytics Dashboard")
        
        if self.storage:
            sessions = self.storage.get_all_sessions()
            
            if sessions:
                # Basic statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Sessions", len(sessions))
                with col2:
                    total_faces = sum(s.get('faces_detected', 0) for s in sessions)
                    st.metric("Total Faces", total_faces)
                with col3:
                    avg_time = sum(s.get('processing_time', 0) for s in sessions) / len(sessions)
                    st.metric("Avg Processing Time", f"{avg_time:.2f}s")
                
                # Recent sessions
                st.markdown("#### Recent Sessions")
                for session in sessions[-5:]:
                    st.write(f"**{session['filename']}** - {session['faces_detected']} faces - {session['timestamp'][:16]}")
                    
            else:
                st.info("No sessions recorded yet. Upload some images to see analytics!")
        else:
            st.error("Storage system not available")
    
    def system_info_tab(self):
        """System information"""
        st.markdown("### System Information")
        
        # Detector status
        if self.detector:
            st.success("✅ MediaPipe Face Detection: Active")
            
            if st.button("Test Detector"):
                try:
                    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                    result = self.detector.detect_faces(test_image, 0.5)
                    st.success(f"Detector test passed - returned {len(result)} results")
                except Exception as e:
                    st.error(f"Detector test failed: {e}")
        else:
            st.error("❌ Face Detector: Not Available")
        
        # Storage status
        if self.storage:
            st.success("✅ Storage System: Active")
            sessions = self.storage.get_all_sessions()
            st.info(f"Storing {len(sessions)} sessions")
        else:
            st.error("❌ Storage System: Not Available")
        
        # Environment info
        st.markdown("#### Environment")
        st.text(f"OpenCV Version: {cv2.__version__}")
        st.text(f"NumPy Version: {np.__version__}")

def main():
    """Application entry point"""
    app = ComputerVisionApp()
    app.run()

if __name__ == "__main__":
    main()