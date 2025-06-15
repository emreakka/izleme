import cv2
import numpy as np
from typing import List, Dict, Tuple
from gaze_detector import GazeDetector
from emotion_detector import EmotionDetector

def process_frame(frame: np.ndarray, gaze_detector: GazeDetector, emotion_detector: EmotionDetector, 
                 face_recognition=None, confidence_threshold: float = 0.5, show_landmarks: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Process a single frame for gaze detection, emotion detection, and face recognition
    
    Args:
        frame: Input frame in BGR format
        gaze_detector: GazeDetector instance
        emotion_detector: EmotionDetector instance
        face_recognition: FaceRecognitionSystem instance (optional)
        confidence_threshold: Minimum confidence for detections
        show_landmarks: Whether to show facial landmarks
        
    Returns:
        Tuple of (processed_frame, detection_results)
    """
    # Make a copy of the frame
    processed_frame = frame.copy()
    results = []
    
    try:
        # Face recognition first (if available)
        face_info = []
        if face_recognition:
            face_info = face_recognition.recognize_faces(frame)
        
        # Detect gaze
        gaze_results = gaze_detector.detect_gaze(frame, confidence_threshold)
        
        # Detect emotions
        emotion_results = emotion_detector.detect_emotion(frame, confidence_threshold)
        
        # Improved face matching using spatial proximity
        all_faces = []
        
        # Collect all face detections with their positions
        for i, gaze_info in enumerate(gaze_results):
            face_center = np.mean(gaze_info['landmarks'], axis=0) if 'landmarks' in gaze_info else None
            all_faces.append({
                'type': 'gaze',
                'index': i,
                'data': gaze_info,
                'center': face_center
            })
        
        for i, emotion_info in enumerate(emotion_results):
            face_center = np.mean(emotion_info['landmarks'], axis=0) if 'landmarks' in emotion_info else None
            all_faces.append({
                'type': 'emotion',
                'index': i,
                'data': emotion_info,
                'center': face_center
            })
        
        # Group faces by proximity
        matched_faces = []
        used_emotion_indices = set()
        used_gaze_indices = set()
        
        for gaze_face in [f for f in all_faces if f['type'] == 'gaze']:
            if gaze_face['index'] in used_gaze_indices:
                continue
                
            result = {
                'face_id': len(matched_faces) + 1,
                'gaze_direction': gaze_face['data']['gaze_direction'],
                'gaze_angles': gaze_face['data']['gaze_angles'],
                'gaze_confidence': gaze_face['data']['confidence'],
                'head_pose_angles': gaze_face['data'].get('head_pose_angles', (0.0, 0.0, 0.0))
            }
            
            # Find closest emotion detection
            best_emotion_match = None
            min_distance = float('inf')
            
            if gaze_face['center'] is not None:
                for emotion_face in [f for f in all_faces if f['type'] == 'emotion' and f['index'] not in used_emotion_indices]:
                    if emotion_face['center'] is not None:
                        distance = np.linalg.norm(gaze_face['center'] - emotion_face['center'])
                        if distance < min_distance and distance < 100:  # Within 100 pixels
                            min_distance = distance
                            best_emotion_match = emotion_face
            
            if best_emotion_match:
                result.update({
                    'emotion': best_emotion_match['data']['emotion'],
                    'emotion_confidence': best_emotion_match['data']['confidence'],
                    'emotion_scores': best_emotion_match['data']['emotion_scores']
                })
                used_emotion_indices.add(best_emotion_match['index'])
                # Draw emotion overlay
                processed_frame = emotion_detector.draw_emotion_overlay(processed_frame, best_emotion_match['data'])
            else:
                result.update({
                    'emotion': 'Unknown',
                    'emotion_confidence': 0.0,
                    'emotion_scores': {}
                })
            
            # Draw gaze overlay
            processed_frame = gaze_detector.draw_gaze_overlay(processed_frame, gaze_face['data'])
            used_gaze_indices.add(gaze_face['index'])
            matched_faces.append(result)
        
        # Add remaining emotion detections that weren't matched
        for emotion_face in [f for f in all_faces if f['type'] == 'emotion' and f['index'] not in used_emotion_indices]:
            result = {
                'face_id': len(matched_faces) + 1,
                'gaze_direction': 'Unknown',
                'gaze_angles': (0.0, 0.0),
                'gaze_confidence': 0.0,
                'head_pose_angles': (0.0, 0.0, 0.0),
                'emotion': emotion_face['data']['emotion'],
                'emotion_confidence': emotion_face['data']['confidence'],
                'emotion_scores': emotion_face['data']['emotion_scores']
            }
            # Draw emotion overlay
            processed_frame = emotion_detector.draw_emotion_overlay(processed_frame, emotion_face['data'])
            matched_faces.append(result)
        
        results = matched_faces
        
        # Draw additional landmarks if requested
        if show_landmarks and (gaze_results or emotion_results):
            processed_frame = draw_facial_landmarks(processed_frame, gaze_results, emotion_results)
    
    except Exception as e:
        # If processing fails, return original frame with error info
        cv2.putText(processed_frame, f"Processing Error: {str(e)[:50]}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return processed_frame, results

def draw_facial_landmarks(image: np.ndarray, gaze_results: List[Dict], emotion_results: List[Dict]) -> np.ndarray:
    """Draw facial landmarks on the image"""
    result_image = image.copy()
    
    # Draw landmarks from gaze results
    for gaze_info in gaze_results:
        if 'landmarks' in gaze_info:
            landmarks = gaze_info['landmarks']
            # Draw key facial points
            for point in landmarks[::10]:  # Draw every 10th point to avoid clutter
                cv2.circle(result_image, tuple(point.astype(int)), 1, (0, 255, 255), -1)
    
    # Draw landmarks from emotion results (if not already drawn)
    if not gaze_results:  # Only if gaze didn't draw any landmarks
        for emotion_info in emotion_results:
            if 'landmarks' in emotion_info:
                landmarks = emotion_info['landmarks']
                for point in landmarks[::10]:
                    cv2.circle(result_image, tuple(point.astype(int)), 1, (255, 255, 0), -1)
    
    return result_image

def draw_results(image: np.ndarray, results: List[Dict]) -> np.ndarray:
    """Draw detection results summary on image"""
    result_image = image.copy()
    
    if not results:
        cv2.putText(result_image, "No faces detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return result_image
    
    # Draw summary for each detected face
    y_offset = 30
    for i, result in enumerate(results):
        # Face number
        face_text = f"Face {i+1}:"
        cv2.putText(result_image, face_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Emotion
        emotion = result.get('emotion', 'Unknown')
        emotion_conf = result.get('emotion_confidence', 0.0)
        emotion_text = f"  Emotion: {emotion} ({emotion_conf:.2f})"
        cv2.putText(result_image, emotion_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
        y_offset += 20
        
        # Gaze
        gaze = result.get('gaze_direction', 'Unknown')
        gaze_conf = result.get('gaze_confidence', 0.0)
        gaze_text = f"  Gaze: {gaze} ({gaze_conf:.2f})"
        cv2.putText(result_image, gaze_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
        y_offset += 30
    
    return result_image

def validate_image(image: np.ndarray) -> bool:
    """Validate if image is suitable for processing"""
    if image is None:
        return False
    
    if len(image.shape) != 3:
        return False
    
    height, width, channels = image.shape
    if height < 50 or width < 50:
        return False
    
    if channels != 3:
        return False
    
    return True

def resize_image_for_processing(image: np.ndarray, max_width: int = 1200) -> np.ndarray:
    """Resize image for optimal processing performance"""
    height, width = image.shape[:2]
    
    if width > max_width:
        # Calculate new height maintaining aspect ratio
        new_height = int(height * (max_width / width))
        resized_image = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image
    
    return image

def get_emotion_color(emotion: str) -> Tuple[int, int, int]:
    """Get color associated with emotion for visualization"""
    emotion_colors = {
        'happy': (0, 255, 0),      # Green
        'sad': (255, 0, 0),        # Blue
        'angry': (0, 0, 255),      # Red
        'surprised': (0, 255, 255), # Yellow
        'fearful': (128, 0, 128),   # Purple
        'disgusted': (0, 128, 128), # Teal
        'neutral': (128, 128, 128)  # Gray
    }
    
    return emotion_colors.get(emotion.lower(), (255, 255, 255))  # White as default

def format_confidence_score(confidence: float) -> str:
    """Format confidence score for display"""
    if confidence >= 0.8:
        return f"{confidence:.1%} (High)"
    elif confidence >= 0.6:
        return f"{confidence:.1%} (Medium)"
    else:
        return f"{confidence:.1%} (Low)"
