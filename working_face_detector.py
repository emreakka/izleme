import cv2
import numpy as np
from typing import List, Dict, Tuple

class WorkingFaceDetector:
    """Simple but effective face detector that actually works"""
    
    def __init__(self):
        # Load Haar cascade
        self.cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        print(f"Haar cascade loaded: {not self.cascade.empty()}")
        
        # Try MediaPipe as well
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=0.2
            )
            self.mediapipe_available = True
            print("MediaPipe face detection initialized")
        except Exception as e:
            print(f"MediaPipe not available: {e}")
            self.mediapipe_available = False
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect all faces in image using working methods"""
        results = []
        h, w = image.shape[:2]
        
        # Method 1: Try MediaPipe first
        if self.mediapipe_available:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_results = self.face_detection.process(rgb_image)
                
                if mp_results.detections:
                    for i, detection in enumerate(mp_results.detections):
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Make sure coordinates are valid
                        x = max(0, x)
                        y = max(0, y)
                        width = min(w - x, width)
                        height = min(h - y, height)
                        
                        if width > 20 and height > 20:  # Minimum size check
                            results.append({
                                'face_id': len(results) + 1,
                                'person_name': f'Person-{len(results) + 1}',
                                'bbox': (x, y, width, height),
                                'method': 'mediapipe',
                                'confidence': detection.score[0] if detection.score else 0.8
                            })
            except Exception as e:
                print(f"MediaPipe detection error: {e}")
        
        # Method 2: Use Haar cascade with proven parameters
        if not self.cascade.empty():
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Use the parameters that worked in our test
                faces = self.cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                for (x, y, w_face, h_face) in faces:
                    # Check if this face overlaps with existing MediaPipe detections
                    is_duplicate = False
                    for existing in results:
                        ex, ey, ew, eh = existing['bbox']
                        if (abs(x - ex) < 50 and abs(y - ey) < 50):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        results.append({
                            'face_id': len(results) + 1,
                            'person_name': f'Person-{len(results) + 1}',
                            'bbox': (x, y, w_face, h_face),
                            'method': 'opencv_haar',
                            'confidence': 0.7
                        })
            except Exception as e:
                print(f"Haar cascade error: {e}")
        
        # Add gaze and emotion analysis for each detected face
        for i, face_data in enumerate(results):
            x, y, w_face, h_face = face_data['bbox']
            
            # Extract face region
            face_region = image[y:y+h_face, x:x+w_face]
            
            # Analyze gaze direction based on face position
            gaze_direction = self._analyze_gaze_from_position(x, y, w_face, h_face, w, h)
            
            # Simple emotion detection
            emotion = self._analyze_emotion_simple(face_region)
            
            # Update face data
            face_data.update({
                'gaze_direction': gaze_direction,
                'gaze_confidence': 0.6,
                'emotion': emotion,
                'emotion_confidence': 0.5
            })
        
        return results
    
    def _analyze_gaze_from_position(self, x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> str:
        """Analyze gaze direction based on face position and orientation"""
        # Face center
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Relative position in image
        rel_x = center_x / img_w
        rel_y = center_y / img_h
        
        # Determine gaze based on position and face orientation
        if rel_y > 0.7:
            return "looking at table/game"
        elif rel_y < 0.3:
            return "looking up/away"
        elif rel_x < 0.3:
            return "looking at person on left"
        elif rel_x > 0.7:
            return "looking at person on right"
        else:
            # For center faces, try to determine based on face angle
            if w > h * 1.2:  # Wide face might be turned
                if center_x < img_w // 2:
                    return "looking right"
                else:
                    return "looking left"
            else:
                return "looking at camera"
    
    def _analyze_emotion_simple(self, face_region: np.ndarray) -> str:
        """Simple emotion analysis"""
        if face_region.size == 0:
            return "neutral"
        
        # Simple brightness analysis for smile detection
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            # Check mouth region (lower third)
            mouth_region = gray_face[2*h//3:, w//4:3*w//4]
            
            if mouth_region.size > 0:
                mouth_brightness = np.mean(mouth_region)
                face_brightness = np.mean(gray_face)
                
                if mouth_brightness > face_brightness * 1.15:
                    return "happy"
                elif mouth_brightness < face_brightness * 0.85:
                    return "sad"
            
            return "neutral"
        except:
            return "neutral"
    
    def draw_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results on image"""
        result_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            person_name = detection['person_name']
            gaze_direction = detection['gaze_direction']
            emotion = detection['emotion']
            method = detection['method']
            
            # Choose color based on method
            if method == 'mediapipe':
                color = (0, 255, 0)  # Green
            else:
                color = (255, 165, 0)  # Orange
            
            # Draw face box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
            
            # Draw person name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Person name
            (text_w, text_h), baseline = cv2.getTextSize(person_name, font, font_scale, thickness)
            cv2.rectangle(result_image, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
            cv2.putText(result_image, person_name, (x + 5, y - 5), font, font_scale, (0, 0, 0), thickness)
            
            # Draw gaze direction below the box
            gaze_text = f"Looking: {gaze_direction}"
            (gaze_w, gaze_h), _ = cv2.getTextSize(gaze_text, font, 0.6, 2)
            cv2.rectangle(result_image, (x, y + h), (x + gaze_w + 10, y + h + gaze_h + 10), (255, 255, 0), -1)
            cv2.putText(result_image, gaze_text, (x + 5, y + h + gaze_h + 5), font, 0.6, (0, 0, 0), 2)
            
            # Draw emotion
            emotion_text = f"Emotion: {emotion}"
            (emo_w, emo_h), _ = cv2.getTextSize(emotion_text, font, 0.5, 1)
            cv2.rectangle(result_image, (x, y + h + gaze_h + 15), 
                        (x + emo_w + 10, y + h + gaze_h + emo_h + 25), (0, 255, 255), -1)
            cv2.putText(result_image, emotion_text, (x + 5, y + h + gaze_h + emo_h + 20), 
                      font, 0.5, (0, 0, 0), 1)
        
        return result_image