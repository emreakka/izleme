import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from gaze_detector import GazeDetector
from emotion_detector import EmotionDetector
from face_recognition import FaceRecognitionSystem
from utils import process_frame, draw_results
from enhanced_utils import enhanced_process_frame
from robust_face_detector import RobustFaceDetector
from simple_storage import simple_storage
from simple_robust_processing import simple_robust_process_frame

# Initialize detectors
@st.cache_resource
def load_detectors():
    gaze_detector = GazeDetector()
    emotion_detector = EmotionDetector()
    face_recognition = FaceRecognitionSystem()
    robust_detector = RobustFaceDetector()
    return gaze_detector, emotion_detector, face_recognition, robust_detector

def main():
    st.title("👁️ Gaze Direction & Emotion Detection")
    st.markdown("Upload an image/video or use your webcam to detect gaze direction and emotions in real-time.")
    
    # Load detectors
    gaze_detector, emotion_detector, face_recognition, robust_detector = load_detectors()
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    show_landmarks = st.sidebar.checkbox("Show Facial Landmarks", value=True)
    
    # Database controls
    st.sidebar.header("Database")
    if st.sidebar.button("View Analytics"):
        show_analytics()
    
    if st.sidebar.button("View Session History"):
        show_session_history()
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Webcam", "🖼️ Upload Image", "🎥 Upload Video", "📊 Analytics"])
    
    with tab1:
        st.header("Real-time Webcam Detection")
        
        # Webcam section
        if st.button("Start Webcam Detection"):
            run_webcam_detection(gaze_detector, emotion_detector, face_recognition, confidence_threshold, show_landmarks)
    
    with tab2:
        st.header("Image Upload")
        uploaded_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image is not None:
            process_uploaded_image(uploaded_image, gaze_detector, emotion_detector, face_recognition, confidence_threshold, show_landmarks)
    
    with tab3:
        st.header("Video Upload")
        uploaded_video = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            process_uploaded_video(uploaded_video, gaze_detector, emotion_detector, face_recognition, confidence_threshold, show_landmarks)
    
    with tab4:
        st.header("Analytics Dashboard")
        show_analytics_dashboard()
    
    # Add Face Management section to sidebar
    if face_recognition:
        st.sidebar.header("👤 Face Management")
        known_faces = face_recognition.get_known_faces_summary()
        
        if known_faces:
            st.sidebar.subheader("Known People")
            for face_id, face_data in known_faces.items():
                with st.sidebar.expander(f"{face_data['name']} ({face_data['encounter_count']} times)"):
                    new_name = st.text_input(f"Rename", value=face_data['name'], key=f"rename_{face_id}")
                    if st.button(f"Update Name", key=f"update_{face_id}"):
                        if new_name and new_name.strip():
                            if face_recognition.rename_face(face_id, new_name.strip()):
                                st.success(f"Renamed to {new_name}")
                                st.rerun()
                    
                    if st.button(f"Forget Person", key=f"forget_{face_id}"):
                        if face_recognition.forget_face(face_id):
                            st.success("Person forgotten")
                            st.rerun()
                    
                    st.write(f"First seen: {face_data['first_seen'].strftime('%m/%d %H:%M')}")
                    st.write(f"Last seen: {face_data['last_seen'].strftime('%m/%d %H:%M')}")
        else:
            st.sidebar.info("No known faces yet")

def run_webcam_detection(gaze_detector, emotion_detector, face_recognition, confidence_threshold, show_landmarks):
    """Run real-time webcam detection"""
    st.info("Starting webcam... Press 'Stop' to end detection.")
    
    # Create storage session
    session_id = simple_storage.create_session("webcam", confidence_threshold, show_landmarks)
    
    # Create placeholders for video and results
    video_placeholder = st.empty()
    results_placeholder = st.empty()
    stop_button = st.button("Stop Detection")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not access webcam. Please check your camera permissions.")
        return
    
    total_faces = 0
    frame_count = 0
    
    try:
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam")
                break
            
            frame_count += 1
            
            # Process frame
            processed_frame, results = process_frame(
                frame, gaze_detector, emotion_detector, confidence_threshold, show_landmarks
            )
            
            # Save results to storage
            if results:
                simple_storage.save_detection_results(session_id, results, frame_count)
                total_faces += len(results)
            
            # Display video
            video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
            
            # Display results
            if results:
                display_detection_results(results, results_placeholder)
    
    finally:
        cap.release()
        # End storage session
        simple_storage.end_session(session_id, total_faces, frame_count)
        st.success(f"Session completed: {total_faces} faces detected in {frame_count} frames")

def process_uploaded_image(uploaded_image, gaze_detector, emotion_detector, face_recognition, confidence_threshold, show_landmarks):
    """Process uploaded image"""
    # Create storage session
    session_id = simple_storage.create_session("image", confidence_threshold, show_landmarks)
    
    # Load image
    image = Image.open(uploaded_image)
    image_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    st.subheader("Original Image")
    st.image(image, use_container_width=True)
    
    # Process image with robust detection
    with st.spinner("Processing image..."):
        robust_detector = RobustFaceDetector()
        processed_image, results = simple_robust_process_frame(
            image_array, robust_detector, confidence_threshold
        )
    
    # Save results to storage
    total_faces = 0
    if results:
        simple_storage.save_detection_results(session_id, results)
        total_faces = len(results)
    
    # End storage session
    simple_storage.end_session(session_id, total_faces, 1)
    
    # Display results
    st.subheader("Detection Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(processed_image, channels="BGR", caption="Processed Image", use_container_width=True)
    
    with col2:
        if results:
            display_detection_results(results)
            st.success(f"Detected {total_faces} face(s) and saved to database")
        else:
            st.warning("No faces detected in the image.")

def process_uploaded_video(uploaded_video, gaze_detector, emotion_detector, face_recognition, confidence_threshold, show_landmarks):
    """Process uploaded video"""
    # Create storage session
    session_id = simple_storage.create_session("video", confidence_threshold, show_landmarks)
    
    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_video.read())
        tmp_path = tmp_file.name
    
    total_faces = 0
    frames_processed = 0
    
    try:
        # Open video
        cap = cv2.VideoCapture(tmp_path)
        
        if not cap.isOpened():
            st.error("Could not open video file")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.info(f"Video: {total_frames} frames at {fps} FPS")
        
        # Process video frames
        frame_placeholder = st.empty()
        results_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame for performance
            if frame_count % 5 == 0:
                processed_frame, results = enhanced_process_frame(
                    frame, gaze_detector, emotion_detector, face_recognition, confidence_threshold, show_landmarks
                )
                
                # Save results to storage
                if results:
                    simple_storage.save_detection_results(session_id, results, frame_count)
                    total_faces += len(results)
                
                frames_processed += 1
                
                # Display frame
                frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                
                # Display results
                if results:
                    display_detection_results(results, results_placeholder)
            
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
        
        cap.release()
        
        # End storage session
        simple_storage.end_session(session_id, total_faces, frames_processed)
        st.success(f"Video processing completed! {total_faces} faces detected in {frames_processed} processed frames")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def display_detection_results(results, placeholder=None):
    """Display detection results in a formatted way"""
    if placeholder:
        with placeholder.container():
            display_results_content(results)
    else:
        display_results_content(results)

def display_results_content(results):
    """Display the actual results content"""
    for i, result in enumerate(results):
        face_id = result.get('face_id', i+1)
        st.subheader(f"Face {face_id}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎭 Emotion Detection**")
            emotion = result.get('emotion', 'Unknown')
            emotion_confidence = result.get('emotion_confidence', 0.0)
            
            # Emotion with confidence bar
            st.metric("Detected Emotion", emotion)
            st.progress(emotion_confidence)
            st.caption(f"Confidence: {emotion_confidence:.2%}")
            
            # Show all emotion scores if available
            if 'emotion_scores' in result and result['emotion_scores']:
                st.markdown("**All Emotions:**")
                for emotion_name, score in result['emotion_scores'].items():
                    st.text(f"{emotion_name}: {score:.3f}")
        
        with col2:
            st.markdown("**👁️ Gaze Direction**")
            gaze_direction = result.get('gaze_direction', 'Unknown')
            gaze_confidence = result.get('gaze_confidence', 0.0)
            
            st.metric("Gaze Direction", gaze_direction)
            st.progress(gaze_confidence)
            st.caption(f"Confidence: {gaze_confidence:.2%}")
            
            # Show gaze angles if available
            if 'gaze_angles' in result:
                pitch, yaw = result['gaze_angles']
                st.text(f"Eye Pitch: {pitch:.1f}°")
                st.text(f"Eye Yaw: {yaw:.1f}°")
            
            # Show head pose angles if available
            if 'head_pose_angles' in result:
                head_pitch, head_yaw, head_roll = result['head_pose_angles']
                st.markdown("**Head Pose:**")
                st.text(f"Head Pitch: {head_pitch:.1f}°")
                st.text(f"Head Yaw: {head_yaw:.1f}°")
                st.text(f"Head Roll: {head_roll:.1f}°")
        
        if i < len(results) - 1:
            st.divider()

def show_analytics_dashboard():
    """Display analytics dashboard"""
    st.subheader("📊 Detection Analytics")
    
    # Time period selector
    col1, col2 = st.columns(2)
    with col1:
        days = st.selectbox("Time Period", [1, 7, 30], index=1, format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}")
    
    with col2:
        if st.button("Refresh Data"):
            st.rerun()
    
    # Get analytics data
    emotion_analytics = simple_storage.get_emotion_analytics(days)
    gaze_analytics = simple_storage.get_gaze_analytics(days)
    
    # Display emotion analytics
    if emotion_analytics:
        st.subheader("🎭 Emotion Distribution")
        emotion_col1, emotion_col2 = st.columns(2)
        
        with emotion_col1:
            # Create emotion chart data
            emotions = list(emotion_analytics.keys())
            counts = [data['total_detections'] for data in emotion_analytics.values()]
            
            try:
                import pandas as pd
                emotion_df = pd.DataFrame({
                    'Emotion': emotions,
                    'Count': counts
                })
                st.bar_chart(emotion_df.set_index('Emotion'))
            except ImportError:
                # Fallback to simple chart display
                for emotion, count in zip(emotions, counts):
                    st.text(f"{emotion}: {count}")
        
        with emotion_col2:
            st.markdown("**Emotion Statistics:**")
            for emotion, data in emotion_analytics.items():
                st.metric(
                    f"{emotion.capitalize()}", 
                    f"{data['total_detections']} detections",
                    f"{data['average_confidence']:.1%} avg confidence"
                )
    else:
        st.info("No emotion data available for the selected period")
    
    # Display gaze analytics
    if gaze_analytics:
        st.subheader("👁️ Gaze Direction Distribution")
        gaze_col1, gaze_col2 = st.columns(2)
        
        with gaze_col1:
            # Create gaze chart data
            directions = list(gaze_analytics.keys())
            counts = [data['total_detections'] for data in gaze_analytics.values()]
            
            try:
                import pandas as pd
                gaze_df = pd.DataFrame({
                    'Direction': directions,
                    'Count': counts
                })
                st.bar_chart(gaze_df.set_index('Direction'))
            except ImportError:
                # Fallback to simple chart display
                for direction, count in zip(directions, counts):
                    st.text(f"{direction}: {count}")
        
        with gaze_col2:
            st.markdown("**Gaze Statistics:**")
            for direction, data in gaze_analytics.items():
                st.metric(
                    f"{direction}", 
                    f"{data['total_detections']} detections",
                    f"{data['average_confidence']:.1%} avg confidence"
                )
    else:
        st.info("No gaze data available for the selected period")

def show_analytics():
    """Show analytics in sidebar"""
    with st.sidebar:
        st.subheader("📈 Quick Analytics")
        
        # Get recent analytics
        emotion_data = simple_storage.get_emotion_analytics(7)
        gaze_data = simple_storage.get_gaze_analytics(7)
        
        if emotion_data:
            st.markdown("**Top Emotions (7 days):**")
            sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1]['total_detections'], reverse=True)
            for emotion, data in sorted_emotions[:3]:
                st.text(f"{emotion}: {data['total_detections']}")
        
        if gaze_data:
            st.markdown("**Top Gaze Directions (7 days):**")
            sorted_gaze = sorted(gaze_data.items(), key=lambda x: x[1]['total_detections'], reverse=True)
            for direction, data in sorted_gaze[:3]:
                st.text(f"{direction}: {data['total_detections']}")

def show_session_history():
    """Show session history in sidebar"""
    with st.sidebar:
        st.subheader("📋 Recent Sessions")
        
        sessions = simple_storage.get_session_history(5)
        
        if sessions:
            for session in sessions:
                with st.expander(f"{session['type'].title()} - {session['start_time'].strftime('%m/%d %H:%M')}"):
                    st.text(f"Faces: {session['faces_detected']}")
                    st.text(f"Frames: {session['frames_processed']}")
                    st.text(f"Confidence: {session['confidence_threshold']}")
                    if session['end_time']:
                        duration = session['end_time'] - session['start_time']
                        st.text(f"Duration: {duration.total_seconds():.1f}s")
                    
                    if st.button(f"View Details", key=f"details_{session['id']}"):
                        show_session_details(session['id'])
        else:
            st.info("No sessions found")

def show_session_details(session_id):
    """Show detailed session information"""
    st.subheader("Session Details")
    
    details = simple_storage.get_detection_details(session_id)
    
    if details:
        st.success(f"Found {len(details)} detection records")
        
        # Create DataFrame for detailed view
        try:
            import pandas as pd
            
            records = []
            for detail in details:
                records.append({
                    'Timestamp': detail['timestamp'].strftime('%H:%M:%S'),
                    'Face ID': detail['face_id'],
                    'Emotion': detail['emotion'],
                    'Emotion Conf': f"{detail['emotion_confidence']:.2f}",
                    'Gaze Direction': detail['gaze_direction'],
                    'Gaze Conf': f"{detail['gaze_confidence']:.2f}",
                    'Head Yaw': f"{detail['head_pose'][1]:.1f}°",
                    'Head Pitch': f"{detail['head_pose'][0]:.1f}°",
                    'Frame': detail['frame_number'] or 'N/A',
                    'Processing (ms)': f"{detail['processing_time_ms']:.1f}" if detail['processing_time_ms'] else 'N/A'
                })
            
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"session_{session_id}_details.csv",
                mime="text/csv"
            )
        except ImportError:
            # Fallback to simple display
            for detail in details:
                st.text(f"{detail['timestamp'].strftime('%H:%M:%S')} - Face {detail['face_id']}: {detail['emotion']} ({detail['emotion_confidence']:.2f}), Gaze: {detail['gaze_direction']} ({detail['gaze_confidence']:.2f})")
    else:
        st.warning("No detection details found for this session")

if __name__ == "__main__":
    main()
