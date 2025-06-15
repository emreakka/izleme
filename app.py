import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from gaze_detector import GazeDetector
from emotion_detector import EmotionDetector
from utils import process_frame, draw_results

# Initialize detectors
@st.cache_resource
def load_detectors():
    gaze_detector = GazeDetector()
    emotion_detector = EmotionDetector()
    return gaze_detector, emotion_detector

def main():
    st.title("👁️ Gaze Direction & Emotion Detection")
    st.markdown("Upload an image/video or use your webcam to detect gaze direction and emotions in real-time.")
    
    # Load detectors
    gaze_detector, emotion_detector = load_detectors()
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    show_landmarks = st.sidebar.checkbox("Show Facial Landmarks", value=True)
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📷 Webcam", "🖼️ Upload Image", "🎥 Upload Video"])
    
    with tab1:
        st.header("Real-time Webcam Detection")
        
        # Webcam section
        if st.button("Start Webcam Detection"):
            run_webcam_detection(gaze_detector, emotion_detector, confidence_threshold, show_landmarks)
    
    with tab2:
        st.header("Image Upload")
        uploaded_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image is not None:
            process_uploaded_image(uploaded_image, gaze_detector, emotion_detector, confidence_threshold, show_landmarks)
    
    with tab3:
        st.header("Video Upload")
        uploaded_video = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            process_uploaded_video(uploaded_video, gaze_detector, emotion_detector, confidence_threshold, show_landmarks)

def run_webcam_detection(gaze_detector, emotion_detector, confidence_threshold, show_landmarks):
    """Run real-time webcam detection"""
    st.info("Starting webcam... Press 'Stop' to end detection.")
    
    # Create placeholders for video and results
    video_placeholder = st.empty()
    results_placeholder = st.empty()
    stop_button = st.button("Stop Detection")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not access webcam. Please check your camera permissions.")
        return
    
    try:
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam")
                break
            
            # Process frame
            processed_frame, results = process_frame(
                frame, gaze_detector, emotion_detector, confidence_threshold, show_landmarks
            )
            
            # Display video
            video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
            
            # Display results
            if results:
                display_detection_results(results, results_placeholder)
    
    finally:
        cap.release()

def process_uploaded_image(uploaded_image, gaze_detector, emotion_detector, confidence_threshold, show_landmarks):
    """Process uploaded image"""
    # Load image
    image = Image.open(uploaded_image)
    image_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    st.subheader("Original Image")
    st.image(image, use_container_width=True)
    
    # Process image
    with st.spinner("Processing image..."):
        processed_image, results = process_frame(
            image_array, gaze_detector, emotion_detector, confidence_threshold, show_landmarks
        )
    
    # Display results
    st.subheader("Detection Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(processed_image, channels="BGR", caption="Processed Image", use_container_width=True)
    
    with col2:
        if results:
            display_detection_results(results)
        else:
            st.warning("No faces detected in the image.")

def process_uploaded_video(uploaded_video, gaze_detector, emotion_detector, confidence_threshold, show_landmarks):
    """Process uploaded video"""
    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_video.read())
        tmp_path = tmp_file.name
    
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
                processed_frame, results = process_frame(
                    frame, gaze_detector, emotion_detector, confidence_threshold, show_landmarks
                )
                
                # Display frame
                frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                
                # Display results
                if results:
                    display_detection_results(results, results_placeholder)
            
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
        
        cap.release()
        st.success("Video processing completed!")
    
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

if __name__ == "__main__":
    main()
