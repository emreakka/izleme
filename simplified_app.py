import streamlit as st
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import time
from datetime import datetime
import os

# Simplified modules that work with available libraries
from precise_face_detector import PreciseFaceDetector
from simple_storage import SimpleStorage

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

class AdvancedVisionApp:
    """Simplified computer vision application using proven libraries"""
    
    def __init__(self):
        self.detector = None
        self.storage = None
        self.session_id = None
        
    def initialize_systems(self):
        """Initialize detection systems"""
        if 'systems_initialized' not in st.session_state:
            with st.spinner("Initializing computer vision systems..."):
                try:
                    # Initialize proven detector
                    self.detector = PreciseFaceDetector()
                    self.storage = SimpleStorage()
                    
                    st.session_state.systems_initialized = True
                    st.session_state.detector = self.detector
                    st.session_state.storage = self.storage
                    
                    st.success("Systems initialized successfully!")
                    
                except Exception as e:
                    st.error(f"System initialization failed: {str(e)}")
                    return False
        else:
            # Retrieve from session state
            self.detector = st.session_state.detector
            self.storage = st.session_state.storage
            
        return True
    
    def run(self):
        """Main application entry point"""
        # Header
        st.markdown('<h1 class="main-header">Advanced Computer Vision Analysis</h1>', 
                   unsafe_allow_html=True)
        
        # Initialize systems
        if not self.initialize_systems():
            st.stop()
        
        # Sidebar configuration
        self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs([
            "📸 Image Analysis", 
            "📊 Analytics Dashboard",
            "👥 Face Database",
            "🔧 System Status"
        ])
        
        with tab1:
            self.image_analysis_tab()
            
        with tab2:
            self.analytics_dashboard_tab()
            
        with tab3:
            self.face_database_tab()
            
        with tab4:
            self.system_status_tab()
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.markdown("## Detection Settings")
        
        # Detection parameters
        st.session_state.confidence_threshold = st.sidebar.slider(
            "Detection Confidence", 0.1, 1.0, 0.3, 0.05
        )
        
        st.session_state.max_faces = st.sidebar.number_input(
            "Maximum Faces to Detect", 1, 20, 10
        )
        
        st.session_state.show_landmarks = st.sidebar.checkbox(
            "Show Analysis Details", True
        )
        
        # System status
        st.sidebar.markdown("## System Status")
        if hasattr(self, 'detector') and self.detector:
            st.sidebar.success("🟢 Face Detector: Active")
        else:
            st.sidebar.error("🔴 Face Detector: Inactive")
            
        if hasattr(self, 'storage') and self.storage:
            total_sessions = len(self.storage.get_all_sessions())
            st.sidebar.metric("Total Sessions", total_sessions)
    
    def image_analysis_tab(self):
        """Image analysis tab"""
        st.markdown("### Upload & Analyze Images")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg', 'bmp'],
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
    
    def process_single_image(self, image: np.ndarray, filename: str):
        """Process a single uploaded image"""
        start_time = time.time()
        
        # Create columns for display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display original image
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
        
        with col2:
            st.markdown("#### Processing...")
            progress_bar = st.progress(0)
        
        try:
            # Face Detection
            progress_bar.progress(50)
            detections = self.detector.detect_faces(
                image, 
                confidence_threshold=st.session_state.confidence_threshold
            )
            
            progress_bar.progress(90)
            
            # Draw results
            if detections:
                result_image = self.draw_results(image.copy(), detections)
            else:
                result_image = image.copy()
            
            progress_bar.progress(100)
            
            # Display results
            processing_time = time.time() - start_time
            
            with col1:
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                        caption=f"Analysis Results ({len(detections)} faces detected)", 
                        use_column_width=True)
            
            with col2:
                st.success("Processing Complete!")
                st.metric("Processing Time", f"{processing_time:.2f}s")
                st.metric("Faces Detected", len(detections))
                
                # Detailed results
                if detections:
                    st.markdown("#### Detection Details")
                    for i, detection in enumerate(detections):
                        with st.expander(f"👤 {detection.get('person_name', f'Person-{i+1}')}"):
                            st.write(f"**Confidence:** {detection.get('confidence', 0):.3f}")
                            st.write(f"**Gaze:** {detection.get('gaze_direction', 'Unknown')}")
                            st.write(f"**Emotion:** {detection.get('emotion', 'Unknown')} "
                                   f"({detection.get('emotion_confidence', 0):.2f})")
                            st.write(f"**Position:** ({detection.get('x', 0)}, {detection.get('y', 0)})")
                            st.write(f"**Size:** {detection.get('w', 0)} x {detection.get('h', 0)}")
            
            # Save to storage
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
            st.error(f"Processing failed: {str(e)}")
            with col2:
                st.error("Processing failed!")
    
    def draw_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results on image"""
        for detection in detections:
            x, y, w, h = detection.get('x', 0), detection.get('y', 0), detection.get('w', 0), detection.get('h', 0)
            
            # Draw face rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Draw person name
            person_name = detection.get('person_name', 'Unknown')
            cv2.putText(image, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Draw gaze direction
            gaze = detection.get('gaze_direction', '')
            if gaze:
                cv2.putText(image, f"Gaze: {gaze}", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw emotion
            emotion = detection.get('emotion', '')
            emotion_conf = detection.get('emotion_confidence', 0)
            if emotion:
                cv2.putText(image, f"Emotion: {emotion} ({emotion_conf:.2f})", 
                           (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return image
    
    def analytics_dashboard_tab(self):
        """Analytics dashboard tab"""
        st.markdown("### Analytics Dashboard")
        
        if self.storage:
            sessions = self.storage.get_all_sessions()
            
            if sessions:
                # Convert to DataFrame for analysis
                df_data = []
                for session in sessions:
                    df_data.append({
                        'timestamp': session['timestamp'],
                        'filename': session['filename'],
                        'faces_detected': session['faces_detected'],
                        'processing_time': session['processing_time']
                    })
                
                df = pd.DataFrame(df_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Sessions", len(sessions))
                with col2:
                    st.metric("Total Faces", df['faces_detected'].sum())
                with col3:
                    st.metric("Avg Processing Time", f"{df['processing_time'].mean():.2f}s")
                with col4:
                    st.metric("Avg Faces per Image", f"{df['faces_detected'].mean():.1f}")
                
                # Charts
                st.markdown("#### Processing Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.bar_chart(df.set_index('timestamp')['faces_detected'])
                    
                with col2:
                    st.line_chart(df.set_index('timestamp')['processing_time'])
                
                # Recent sessions table
                st.markdown("#### Recent Sessions")
                st.dataframe(df.sort_values('timestamp', ascending=False).head(10))
                
            else:
                st.info("No sessions recorded yet. Process some images to see analytics!")
        else:
            st.error("Storage system not available")
    
    def face_database_tab(self):
        """Face database management tab"""
        st.markdown("### Face Database")
        
        if self.storage:
            sessions = self.storage.get_all_sessions()
            
            # Extract unique faces
            all_faces = {}
            for session in sessions:
                for detection in session.get('detections', []):
                    person_name = detection.get('person_name', 'Unknown')
                    if person_name not in all_faces:
                        all_faces[person_name] = {
                            'count': 0,
                            'last_seen': session['timestamp'],
                            'avg_confidence': 0
                        }
                    
                    all_faces[person_name]['count'] += 1
                    all_faces[person_name]['avg_confidence'] += detection.get('confidence', 0)
            
            # Calculate averages
            for face_data in all_faces.values():
                if face_data['count'] > 0:
                    face_data['avg_confidence'] /= face_data['count']
            
            if all_faces:
                st.markdown("#### Detected Faces")
                
                for person_name, face_data in all_faces.items():
                    with st.expander(f"👤 {person_name}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Appearances", face_data['count'])
                            
                        with col2:
                            st.metric("Avg Confidence", f"{face_data['avg_confidence']:.3f}")
                            
                        with col3:
                            st.metric("Last Seen", face_data['last_seen'][:10])
            else:
                st.info("No faces detected yet. Process some images to build the face database!")
        else:
            st.error("Storage system not available")
    
    def system_status_tab(self):
        """System status and diagnostics"""
        st.markdown("### System Status")
        
        # Detector status
        st.markdown("#### Detection System")
        if hasattr(self, 'detector') and self.detector:
            st.success("✅ Precise Face Detector: Active")
            
            # Test detector
            if st.button("🔧 Test Detector"):
                try:
                    # Create test image
                    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                    result = self.detector.detect_faces(test_image, 0.5)
                    st.success(f"Detector test passed - returned {len(result)} results")
                except Exception as e:
                    st.error(f"Detector test failed: {e}")
        else:
            st.error("❌ Face Detector: Not Available")
        
        # Storage status
        st.markdown("#### Storage System")
        if hasattr(self, 'storage') and self.storage:
            st.success("✅ Storage System: Active")
            sessions = self.storage.get_all_sessions()
            st.info(f"Currently storing {len(sessions)} sessions")
        else:
            st.error("❌ Storage System: Not Available")
        
        # Environment info
        st.markdown("#### Environment Information")
        env_info = {
            "OpenCV Version": cv2.__version__,
            "NumPy Version": np.__version__,
            "Pandas Version": pd.__version__,
            "Python Version": f"{st.version}",
        }
        
        for key, value in env_info.items():
            st.text(f"{key}: {value}")

def main():
    """Main application entry point"""
    app = AdvancedVisionApp()
    app.run()

if __name__ == "__main__":
    main()