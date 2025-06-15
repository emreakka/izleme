import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple
import time

class RobustFaceDetector:
    """Robust face detection using multiple methods"""
    
    def __init__(self):
        # Initialize MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Primary face detection (more sensitive)
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # Short range model for close faces
            min_detection_confidence=0.3
        )
        
        # Secondary face detection (long range)
        self.face_detection_long = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Long range model
            min_detection_confidence=0.3
        )
        
        # Face mesh for detailed analysis
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=20,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # OpenCV Haar cascade as backup
        try:
            self.haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            self.haar_face_cascade = None
    
    def detect_all_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using multiple methods"""
        results = []
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Method 1: MediaPipe face detection (short range)
        mp_faces_short = []
        detection_results = self.face_detection.process(rgb_image)
        if detection_results.detections:
            for i, detection in enumerate(detection_results.detections):
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                mp_faces_short.append({
                    'method': 'mediapipe_short',
                    'bbox': (x, y, width, height),
                    'confidence': detection.score[0] if detection.score else 0.8,
                    'face_id': len(results) + 1
                })
        
        # Method 2: MediaPipe face detection (long range)
        mp_faces_long = []
        detection_results_long = self.face_detection_long.process(rgb_image)
        if detection_results_long.detections:
            for i, detection in enumerate(detection_results_long.detections):
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Check if this face is already detected
                is_duplicate = False
                for existing_face in mp_faces_short:
                    ex, ey, ew, eh = existing_face['bbox']
                    if (abs(x - ex) < 50 and abs(y - ey) < 50):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    mp_faces_long.append({
                        'method': 'mediapipe_long',
                        'bbox': (x, y, width, height),
                        'confidence': detection.score[0] if detection.score else 0.8,
                        'face_id': len(results) + len(mp_faces_short) + 1
                    })
        
        # Method 3: OpenCV Haar cascade as backup
        haar_faces = []
        if self.haar_face_cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.haar_face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=3, 
                minSize=(30, 30)
            )
            
            for (x, y, w_face, h_face) in faces:
                # Check if this face is already detected
                is_duplicate = False
                all_existing = mp_faces_short + mp_faces_long
                for existing_face in all_existing:
                    ex, ey, ew, eh = existing_face['bbox']
                    if (abs(x - ex) < 50 and abs(y - ey) < 50):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    haar_faces.append({
                        'method': 'opencv_haar',
                        'bbox': (x, y, w_face, h_face),
                        'confidence': 0.7,
                        'face_id': len(results) + len(mp_faces_short) + len(mp_faces_long) + 1
                    })
        
        # Combine all detections
        all_faces = mp_faces_short + mp_faces_long + haar_faces
        
        # Process each detected face
        for face_data in all_faces:
            x, y, w_face, h_face = face_data['bbox']
            
            # Extract face region for analysis
            face_region = image[max(0, y):min(h, y + h_face), max(0, x):min(w, x + w_face)]
            
            if face_region.size > 0:
                # Analyze gaze direction (simplified)
                gaze_direction = self._analyze_simple_gaze(face_region, (x, y, w_face, h_face), image.shape)
                
                # Analyze emotion (simplified)
                emotion = self._analyze_simple_emotion(face_region)
                
                result = {
                    'face_id': face_data['face_id'],
                    'person_name': f'Person-{face_data["face_id"]}',
                    'bbox': face_data['bbox'],
                    'method': face_data['method'],
                    'confidence': face_data['confidence'],
                    'gaze_direction': gaze_direction,
                    'gaze_confidence': 0.6,
                    'emotion': emotion,
                    'emotion_confidence': 0.5,
                    'is_known_person': False,
                    'processing_time_ms': 0
                }
                
                results.append(result)
        
        return results
    
    def _analyze_simple_gaze(self, face_region: np.ndarray, bbox: Tuple[int, int, int, int], 
                           image_shape: Tuple[int, int, int]) -> str:
        """Simple gaze analysis based on face position and orientation"""
        x, y, w, h = bbox
        img_h, img_w = image_shape[:2]
        
        # Face center relative to image
        face_center_x = (x + w/2) / img_w
        face_center_y = (y + h/2) / img_h
        
        # Simple heuristics based on position
        if face_center_y > 0.7:
            return "looking at table/floor"
        elif face_center_y < 0.3:
            return "looking up/ceiling"
        elif face_center_x < 0.3:
            return "looking at person/object on left"
        elif face_center_x > 0.7:
            return "looking at person/object on right"
        else:
            # Try to detect eye direction using simple edge detection
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Look for eye regions (upper third of face)
            eye_region = gray_face[:h//3, :]
            
            # Simple horizontal gradient analysis
            if eye_region.size > 0:
                grad_x = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
                mean_grad = np.mean(grad_x)
                
                if mean_grad > 10:
                    return "looking right"
                elif mean_grad < -10:
                    return "looking left"
                else:
                    return "looking at camera"
            else:
                return "looking straight ahead"
    
    def _analyze_simple_emotion(self, face_region: np.ndarray) -> str:
        """Simple emotion analysis based on facial features"""
        if face_region.size == 0:
            return "neutral"
        
        # Simple heuristics based on image properties
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Detect if mouth region is brighter (potential smile)
        h, w = gray_face.shape
        mouth_region = gray_face[2*h//3:, w//4:3*w//4]
        
        if mouth_region.size > 0:
            mouth_brightness = np.mean(mouth_region)
            face_brightness = np.mean(gray_face)
            
            if mouth_brightness > face_brightness * 1.1:
                return "happy"
            elif mouth_brightness < face_brightness * 0.9:
                return "sad"
        
        return "neutral"
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results on image"""
        result_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            face_id = detection['face_id']
            person_name = detection['person_name']
            gaze_direction = detection['gaze_direction']
            emotion = detection['emotion']
            method = detection['method']
            
            # Choose color based on detection method
            if method == 'mediapipe_short':
                color = (0, 255, 0)  # Green
            elif method == 'mediapipe_long':
                color = (255, 255, 0)  # Cyan
            else:
                color = (255, 165, 0)  # Orange
            
            # Draw face box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
            
            # Draw person name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            (text_w, text_h), baseline = cv2.getTextSize(person_name, font, font_scale, thickness)
            cv2.rectangle(result_image, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
            cv2.putText(result_image, person_name, (x + 5, y - 5), font, font_scale, (0, 0, 0), thickness)
            
            # Draw gaze direction
            gaze_text = f"Gaze: {gaze_direction}"
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
            
            # Draw detection method
            method_text = f"Method: {method}"
            cv2.putText(result_image, method_text, (x + 5, y + h + gaze_h + emo_h + 40), 
                      font, 0.4, color, 1)
        
        return result_image