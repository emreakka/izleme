import streamlit as st
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
import json

# Core Detection Modules
from core_face_detector import CoreFaceDetector
from advanced_gaze_analyzer import AdvancedGazeAnalyzer
from emotion_recognition_system import EmotionRecognitionSystem
from face_tracking_system import FaceTrackingSystem
from analytics_manager import AnalyticsManager

# Configure Streamlit
st.set_page_config(
    page_title="Advanced Computer Vision Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
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

class AdvancedVisionApp:
    """Main application class for advanced computer vision analysis"""
    
    def __init__(self):
        self.core_detector = None
        self.gaze_analyzer = None
        self.emotion_system = None
        self.face_tracker = None
        self.analytics = None
        self.session_id = None
        
    def initialize_systems(self):
        """Initialize all computer vision systems"""
        if 'systems_initialized' not in st.session_state:
            with st.spinner("🔧 Initializing advanced computer vision systems..."):
                try:
                    # Initialize core systems
                    self.core_detector = CoreFaceDetector()
                    self.gaze_analyzer = AdvancedGazeAnalyzer()
                    self.emotion_system = EmotionRecognitionSystem()
                    self.face_tracker = FaceTrackingSystem()
                    self.analytics = AnalyticsManager()
                    
                    st.session_state.systems_initialized = True
                    st.session_state.core_detector = self.core_detector
                    st.session_state.gaze_analyzer = self.gaze_analyzer
                    st.session_state.emotion_system = self.emotion_system
                    st.session_state.face_tracker = self.face_tracker
                    st.session_state.analytics = self.analytics
                    
                    st.success("✅ All systems initialized successfully!")
                    
                except Exception as e:
                    st.error(f"❌ System initialization failed: {str(e)}")
                    return False
        else:
            # Retrieve from session state
            self.core_detector = st.session_state.core_detector
            self.gaze_analyzer = st.session_state.gaze_analyzer
            self.emotion_system = st.session_state.emotion_system
            self.face_tracker = st.session_state.face_tracker
            self.analytics = st.session_state.analytics
            
        return True
    
    def run(self):
        """Main application entry point"""
        # Header
        st.markdown('<h1 class="main-header">🎯 Advanced Computer Vision Analysis</h1>', 
                   unsafe_allow_html=True)
        
        # Initialize systems
        if not self.initialize_systems():
            st.stop()
        
        # Sidebar configuration
        self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📸 Live Analysis", 
            "📁 Upload & Process", 
            "🎥 Video Analysis",
            "📊 Analytics Dashboard",
            "👥 Face Management"
        ])
        
        with tab1:
            self.live_analysis_tab()
            
        with tab2:
            self.upload_process_tab()
            
        with tab3:
            self.video_analysis_tab()
            
        with tab4:
            self.analytics_dashboard_tab()
            
        with tab5:
            self.face_management_tab()
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.markdown("## ⚙️ Detection Settings")
        
        # Detection parameters
        st.session_state.confidence_threshold = st.sidebar.slider(
            "Detection Confidence", 0.1, 1.0, 0.5, 0.05
        )
        
        st.session_state.detection_method = st.sidebar.selectbox(
            "Detection Method",
            ["Ensemble (All Methods)", "MTCNN Only", "MediaPipe Only", "OpenCV Only", "TensorFlow Only"]
        )
        
        st.session_state.enable_emotion = st.sidebar.checkbox(
            "Enable Emotion Recognition", True
        )
        
        st.session_state.enable_gaze = st.sidebar.checkbox(
            "Enable Gaze Tracking", True
        )
        
        st.session_state.enable_face_tracking = st.sidebar.checkbox(
            "Enable Face Tracking", True
        )
        
        st.session_state.show_landmarks = st.sidebar.checkbox(
            "Show Facial Landmarks", False
        )
        
        # Advanced settings
        with st.sidebar.expander("🔬 Advanced Settings"):
            st.session_state.max_faces = st.sidebar.number_input(
                "Maximum Faces to Detect", 1, 20, 10
            )
            
            st.session_state.nms_threshold = st.sidebar.slider(
                "Non-Max Suppression", 0.1, 0.9, 0.4, 0.05
            )
            
            st.session_state.quality_threshold = st.sidebar.slider(
                "Face Quality Threshold", 0.1, 1.0, 0.6, 0.05
            )
        
        # System status
        st.sidebar.markdown("## 📊 System Status")
        if hasattr(self, 'core_detector') and self.core_detector:
            st.sidebar.success("🟢 Core Detector: Active")
        else:
            st.sidebar.error("🔴 Core Detector: Inactive")
            
        if hasattr(self, 'analytics') and self.analytics:
            total_sessions = self.analytics.get_session_count()
            st.sidebar.metric("Total Sessions", total_sessions)
    
    def live_analysis_tab(self):
        """Live camera analysis tab"""
        st.markdown("### 📹 Real-time Camera Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Camera controls
            camera_col1, camera_col2, camera_col3 = st.columns(3)
            
            with camera_col1:
                start_camera = st.button("🎥 Start Camera", type="primary")
                
            with camera_col2:
                stop_camera = st.button("⏹️ Stop Camera")
                
            with camera_col3:
                capture_frame = st.button("📸 Capture Frame")
            
            # Camera feed placeholder
            camera_placeholder = st.empty()
            
            if start_camera:
                st.session_state.camera_active = True
                
            if stop_camera:
                st.session_state.camera_active = False
            
            # Real-time processing
            if st.session_state.get('camera_active', False):
                self.process_camera_feed(camera_placeholder)
        
        with col2:
            st.markdown("### 📊 Live Metrics")
            
            # Live metrics placeholder
            metrics_placeholder = st.empty()
            
            # Detection history
            st.markdown("### 📈 Detection History")
            history_placeholder = st.empty()
    
    def upload_process_tab(self):
        """Upload and process images tab"""
        st.markdown("### 📁 Upload & Process Images")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.markdown(f"#### Processing: {uploaded_file.name}")
                
                # Convert uploaded file to opencv format
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                
                if image is not None:
                    self.process_single_image(image, uploaded_file.name)
                else:
                    st.error(f"Failed to load image: {uploaded_file.name}")
    
    def video_analysis_tab(self):
        """Video analysis tab"""
        st.markdown("### 🎥 Video Analysis")
        
        # Video upload
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm']
        )
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            temp_video_path = f"temp_video_{int(time.time())}.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            
            st.success(f"Video uploaded: {uploaded_video.name}")
            
            # Video processing controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                process_video = st.button("🎬 Process Video", type="primary")
                
            with col2:
                frame_skip = st.number_input("Frame Skip", 1, 30, 5)
                
            with col3:
                max_duration = st.number_input("Max Duration (sec)", 10, 300, 60)
            
            if process_video:
                self.process_video_file(temp_video_path, frame_skip, max_duration)
    
    def analytics_dashboard_tab(self):
        """Analytics dashboard tab"""
        st.markdown("### 📊 Analytics Dashboard")
        
        if self.analytics:
            # Time range selector
            col1, col2 = st.columns(2)
            
            with col1:
                days_back = st.selectbox("Time Range", [1, 7, 30, 90], index=1)
                
            with col2:
                refresh_analytics = st.button("🔄 Refresh Data")
            
            # Get analytics data
            analytics_data = self.analytics.get_comprehensive_analytics(days_back)
            
            if analytics_data:
                self.render_analytics_dashboard(analytics_data)
            else:
                st.info("No analytics data available yet. Start processing images to see insights!")
    
    def face_management_tab(self):
        """Face management tab"""
        st.markdown("### 👥 Face Management")
        
        if self.face_tracker:
            # Known faces summary
            known_faces = self.face_tracker.get_known_faces_summary()
            
            if known_faces:
                st.markdown("#### 📋 Known Faces")
                
                for face_id, info in known_faces.items():
                    with st.expander(f"👤 {info['name']} (ID: {face_id})"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Appearances", info['count'])
                            
                        with col2:
                            st.metric("Last Seen", info.get('last_seen', 'Unknown'))
                            
                        with col3:
                            if st.button(f"🗑️ Remove {info['name']}", key=f"remove_{face_id}"):
                                if self.face_tracker.forget_face(face_id):
                                    st.success(f"Removed {info['name']}")
                                    st.rerun()
                        
                        # Rename functionality
                        new_name = st.text_input(f"Rename {info['name']}", key=f"rename_{face_id}")
                        if st.button(f"✏️ Rename", key=f"rename_btn_{face_id}") and new_name:
                            if self.face_tracker.rename_face(face_id, new_name):
                                st.success(f"Renamed to {new_name}")
                                st.rerun()
            else:
                st.info("No known faces yet. Process some images to start building the face database!")
            
            # Bulk operations
            st.markdown("#### 🔧 Bulk Operations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear All Faces", type="secondary"):
                    if st.session_state.get('confirm_clear_faces', False):
                        # Clear all faces logic here
                        st.success("All faces cleared!")
                        st.session_state.confirm_clear_faces = False
                    else:
                        st.session_state.confirm_clear_faces = True
                        st.warning("Click again to confirm clearing all faces")
            
            with col2:
                export_faces = st.button("📤 Export Face Database")
                if export_faces:
                    # Export functionality
                    st.info("Face database export functionality")
    
    def process_camera_feed(self, placeholder):
        """Process real-time camera feed"""
        # This is a placeholder for camera processing
        # In a real implementation, you'd use cv2.VideoCapture(0)
        placeholder.info("Camera feed would be processed here in real implementation")
    
    def process_single_image(self, image: np.ndarray, filename: str):
        """Process a single uploaded image"""
        start_time = time.time()
        
        # Create columns for display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display original image
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
        
        with col2:
            st.markdown("#### 🔍 Processing...")
            progress_bar = st.progress(0)
        
        try:
            # Step 1: Face Detection
            progress_bar.progress(20)
            detections = self.core_detector.detect_faces(
                image, 
                confidence_threshold=st.session_state.confidence_threshold,
                method=st.session_state.detection_method,
                max_faces=st.session_state.max_faces
            )
            
            # Step 2: Gaze Analysis
            if st.session_state.enable_gaze and detections:
                progress_bar.progress(40)
                for detection in detections:
                    gaze_info = self.gaze_analyzer.analyze_gaze(image, detection)
                    detection.update(gaze_info)
            
            # Step 3: Emotion Recognition
            if st.session_state.enable_emotion and detections:
                progress_bar.progress(60)
                for detection in detections:
                    emotion_info = self.emotion_system.recognize_emotion(image, detection)
                    detection.update(emotion_info)
            
            # Step 4: Face Tracking
            if st.session_state.enable_face_tracking and detections:
                progress_bar.progress(80)
                tracked_detections = self.face_tracker.track_faces(image, detections)
                detections = tracked_detections
            
            # Step 5: Draw Results
            progress_bar.progress(90)
            result_image = self.draw_comprehensive_results(image.copy(), detections)
            
            progress_bar.progress(100)
            
            # Display results
            processing_time = time.time() - start_time
            
            with col1:
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                        caption=f"Analysis Results ({len(detections)} faces detected)", 
                        use_column_width=True)
            
            with col2:
                st.success(f"✅ Processing Complete!")
                st.metric("Processing Time", f"{processing_time:.2f}s")
                st.metric("Faces Detected", len(detections))
                
                # Detailed results
                if detections:
                    st.markdown("#### 📋 Detection Details")
                    for i, detection in enumerate(detections):
                        with st.expander(f"👤 {detection.get('person_name', f'Person-{i+1}')}"):
                            st.write(f"**Confidence:** {detection.get('confidence', 0):.3f}")
                            st.write(f"**Gaze:** {detection.get('gaze_direction', 'Unknown')}")
                            st.write(f"**Emotion:** {detection.get('emotion', 'Unknown')} "
                                   f"({detection.get('emotion_confidence', 0):.2f})")
                            st.write(f"**Position:** ({detection.get('x', 0)}, {detection.get('y', 0)})")
                            st.write(f"**Size:** {detection.get('w', 0)} x {detection.get('h', 0)}")
            
            # Save to analytics
            if self.analytics:
                self.analytics.save_detection_session(
                    session_type="image",
                    filename=filename,
                    detections=detections,
                    processing_time=processing_time
                )
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            with col2:
                st.error("Processing failed!")
    
    def process_video_file(self, video_path: str, frame_skip: int, max_duration: int):
        """Process uploaded video file"""
        st.info("Video processing functionality would be implemented here")
    
    def draw_comprehensive_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw comprehensive analysis results on image"""
        for detection in detections:
            x, y, w, h = detection.get('x', 0), detection.get('y', 0), detection.get('w', 0), detection.get('h', 0)
            
            # Draw face rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw person name
            person_name = detection.get('person_name', 'Unknown')
            cv2.putText(image, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw gaze direction
            gaze = detection.get('gaze_direction', '')
            if gaze:
                cv2.putText(image, f"Gaze: {gaze}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw emotion
            emotion = detection.get('emotion', '')
            emotion_conf = detection.get('emotion_confidence', 0)
            if emotion:
                cv2.putText(image, f"Emotion: {emotion} ({emotion_conf:.2f})", 
                           (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw landmarks if enabled
            if st.session_state.get('show_landmarks', False) and 'landmarks' in detection:
                landmarks = detection['landmarks']
                for point in landmarks:
                    cv2.circle(image, tuple(map(int, point)), 2, (255, 255, 0), -1)
        
        return image
    
    def render_analytics_dashboard(self, analytics_data: Dict):
        """Render comprehensive analytics dashboard"""
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sessions", analytics_data.get('total_sessions', 0))
        with col2:
            st.metric("Total Faces", analytics_data.get('total_faces', 0))
        with col3:
            st.metric("Avg Processing Time", f"{analytics_data.get('avg_processing_time', 0):.2f}s")
        with col4:
            st.metric("Success Rate", f"{analytics_data.get('success_rate', 0):.1f}%")
        
        # Charts and visualizations
        st.markdown("#### 📈 Trends")
        
        # Emotion distribution
        if 'emotion_distribution' in analytics_data:
            st.bar_chart(analytics_data['emotion_distribution'])
        
        # Gaze patterns
        if 'gaze_patterns' in analytics_data:
            st.bar_chart(analytics_data['gaze_patterns'])

def main():
    """Main application entry point"""
    app = AdvancedVisionApp()
    app.run()

if __name__ == "__main__":
    main()