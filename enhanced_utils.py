import cv2
import numpy as np
import time
from typing import List, Dict, Tuple

def enhanced_process_frame(frame: np.ndarray, gaze_detector, emotion_detector, 
                          face_recognition=None, confidence_threshold: float = 0.3, 
                          show_landmarks: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Enhanced frame processing with better face detection and clear labeling
    """
    start_time = time.time()
    processed_frame = frame.copy()
    
    try:
        # Face recognition first (if available)
        face_info = []
        if face_recognition:
            face_info = face_recognition.recognize_faces(frame)
        
        # Detect gaze with very low threshold for better detection
        gaze_results = gaze_detector.detect_gaze(frame, 0.1)
        
        # Detect emotions with very low threshold
        emotion_results = emotion_detector.detect_emotion(frame, 0.1)
        
        # Combine all detections
        all_results = []
        
        # Process gaze results first
        for i, gaze_result in enumerate(gaze_results):
            result = {
                'face_id': i + 1,
                'person_name': f'Person-{i + 1}',
                'gaze_direction': gaze_result['gaze_direction'],
                'gaze_pitch': gaze_result['gaze_angles'][0],
                'gaze_yaw': gaze_result['gaze_angles'][1],
                'gaze_confidence': gaze_result['confidence'],
                'head_pose': gaze_result.get('head_pose', (0.0, 0.0, 0.0)),
                'emotion': 'neutral',
                'emotion_confidence': 0.0,
                'emotion_scores': {},
                'is_known_person': False,
                'face_similarity': 0.0,
                'encounter_count': 1,
                'processing_time_ms': (time.time() - start_time) * 1000
            }
            
            # Add face recognition info if available
            if i < len(face_info):
                face = face_info[i]
                result.update({
                    'person_name': face['name'],
                    'is_known_person': face['is_known'],
                    'face_similarity': face['similarity'],
                    'encounter_count': face['encounter_count']
                })
            
            # Find matching emotion result
            if i < len(emotion_results):
                emotion_result = emotion_results[i]
                result.update({
                    'emotion': emotion_result['emotion'],
                    'emotion_confidence': emotion_result['confidence'],
                    'emotion_scores': emotion_result['emotion_scores']
                })
            
            all_results.append(result)
        
        # Add any additional emotion detections that weren't matched
        for i in range(len(gaze_results), len(emotion_results)):
            emotion_result = emotion_results[i]
            result = {
                'face_id': i + 1,
                'person_name': f'Person-{i + 1}',
                'gaze_direction': 'Unknown',
                'gaze_pitch': 0.0,
                'gaze_yaw': 0.0,
                'gaze_confidence': 0.0,
                'head_pose': (0.0, 0.0, 0.0),
                'emotion': emotion_result['emotion'],
                'emotion_confidence': emotion_result['confidence'],
                'emotion_scores': emotion_result['emotion_scores'],
                'is_known_person': False,
                'face_similarity': 0.0,
                'encounter_count': 1,
                'processing_time_ms': (time.time() - start_time) * 1000
            }
            
            # Add face recognition info if available
            if i < len(face_info):
                face = face_info[i]
                result.update({
                    'person_name': face['name'],
                    'is_known_person': face['is_known'],
                    'face_similarity': face['similarity'],
                    'encounter_count': face['encounter_count']
                })
            
            all_results.append(result)
        
        # Add any additional face recognition detections
        for i in range(max(len(gaze_results), len(emotion_results)), len(face_info)):
            face = face_info[i]
            result = {
                'face_id': i + 1,
                'person_name': face['name'],
                'gaze_direction': 'Unknown',
                'gaze_pitch': 0.0,
                'gaze_yaw': 0.0,
                'gaze_confidence': 0.0,
                'head_pose': (0.0, 0.0, 0.0),
                'emotion': 'Unknown',
                'emotion_confidence': 0.0,
                'emotion_scores': {},
                'is_known_person': face['is_known'],
                'face_similarity': face['similarity'],
                'encounter_count': face['encounter_count'],
                'processing_time_ms': (time.time() - start_time) * 1000
            }
            all_results.append(result)
        
        # Draw enhanced overlays
        processed_frame = draw_enhanced_results(processed_frame, all_results, gaze_results, 
                                              emotion_results, face_info, show_landmarks)
        
        return processed_frame, all_results
        
    except Exception as e:
        print(f"Error in enhanced_process_frame: {e}")
        return frame, []

def draw_enhanced_results(image: np.ndarray, combined_results: List[Dict], 
                         gaze_results: List[Dict], emotion_results: List[Dict], 
                         face_info: List[Dict], show_landmarks: bool = True) -> np.ndarray:
    """
    Draw enhanced results with clear face labeling and gaze directions
    """
    result_image = image.copy()
    
    try:
        # Draw gaze overlays first
        for i, gaze_result in enumerate(gaze_results):
            if 'landmarks' in gaze_result:
                # Draw face box
                landmarks = gaze_result['landmarks']
                x_coords = [int(p[0]) for p in landmarks]
                y_coords = [int(p[1]) for p in landmarks]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Expand box slightly
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(image.shape[1], x_max + padding)
                y_max = min(image.shape[0], y_max + padding)
                
                # Draw face box
                cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                
                # Get person info
                person_name = f'Person-{i + 1}'
                gaze_direction = gaze_result['gaze_direction']
                
                if i < len(combined_results):
                    person_name = combined_results[i]['person_name']
                    gaze_direction = combined_results[i]['gaze_direction']
                
                # Draw text background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                
                # Person name
                (text_w, text_h), baseline = cv2.getTextSize(person_name, font, font_scale, thickness)
                cv2.rectangle(result_image, (x_min, y_min - text_h - 10), 
                            (x_min + text_w + 10, y_min), (0, 255, 0), -1)
                cv2.putText(result_image, person_name, (x_min + 5, y_min - 5), 
                          font, font_scale, (0, 0, 0), thickness)
                
                # Gaze direction
                gaze_text = f"Gaze: {gaze_direction}"
                (gaze_w, gaze_h), _ = cv2.getTextSize(gaze_text, font, 0.6, 2)
                cv2.rectangle(result_image, (x_min, y_max), 
                            (x_min + gaze_w + 10, y_max + gaze_h + 10), (255, 255, 0), -1)
                cv2.putText(result_image, gaze_text, (x_min + 5, y_max + gaze_h + 5), 
                          font, 0.6, (0, 0, 0), 2)
                
                # Emotion if available
                if i < len(emotion_results):
                    emotion = emotion_results[i]['emotion']
                    confidence = emotion_results[i]['confidence']
                    emotion_text = f"Emotion: {emotion} ({confidence:.2f})"
                    (emo_w, emo_h), _ = cv2.getTextSize(emotion_text, font, 0.5, 1)
                    cv2.rectangle(result_image, (x_min, y_max + gaze_h + 15), 
                                (x_min + emo_w + 10, y_max + gaze_h + emo_h + 25), (0, 255, 255), -1)
                    cv2.putText(result_image, emotion_text, (x_min + 5, y_max + gaze_h + emo_h + 20), 
                              font, 0.5, (0, 0, 0), 1)
        
        # Draw any additional face recognition boxes
        for i in range(len(gaze_results), len(face_info)):
            face = face_info[i]
            bbox = face['bbox']
            x, y, w, h = bbox
            
            # Draw face box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 165, 0), 3)
            
            # Draw person name
            person_name = face['name']
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(person_name, font, 0.8, 2)
            cv2.rectangle(result_image, (x, y - text_h - 10), 
                        (x + text_w + 10, y), (255, 165, 0), -1)
            cv2.putText(result_image, person_name, (x + 5, y - 5), 
                      font, 0.8, (0, 0, 0), 2)
            
            # Add "No gaze data" text
            no_gaze_text = "Gaze: Not detected"
            cv2.putText(result_image, no_gaze_text, (x + 5, y + h + 20), 
                      font, 0.6, (255, 165, 0), 2)
        
        return result_image
        
    except Exception as e:
        print(f"Error in draw_enhanced_results: {e}")
        return image

def get_emotion_color(emotion: str) -> Tuple[int, int, int]:
    """Get color for emotion visualization"""
    color_map = {
        'happy': (0, 255, 0),      # Green
        'sad': (255, 0, 0),        # Blue  
        'angry': (0, 0, 255),      # Red
        'surprised': (0, 255, 255), # Yellow
        'neutral': (128, 128, 128), # Gray
        'fearful': (255, 0, 255),   # Magenta
        'disgusted': (0, 128, 255)  # Orange
    }
    return color_map.get(emotion.lower(), (255, 255, 255))

def format_confidence_score(confidence: float) -> str:
    """Format confidence score for display"""
    return f"{confidence:.2f}"