import cv2
import numpy as np
from typing import List, Dict, Tuple
import time

class RobustFaceDetector:
    """Robust face detection using multiple methods"""
    
    def __init__(self):
        # Initialize MediaPipe components safely
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            
            # Primary face detection (more sensitive)
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # Short range model for close faces
                min_detection_confidence=0.2
            )
            
            # Secondary face detection (long range)
            self.face_detection_long = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Long range model
                min_detection_confidence=0.2
            )
        except Exception as e:
            print(f"MediaPipe initialization error: {e}")
            self.face_detection = None
            self.face_detection_long = None
        
        # OpenCV Haar cascade as backup
        try:
            import os
            # Try multiple possible paths for the cascade file
            possible_paths = [
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                'haarcascade_frontalface_default.xml'
            ]
            
            self.haar_face_cascade = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.haar_face_cascade = cv2.CascadeClassifier(path)
                    break
        except Exception as e:
            print(f"Haar cascade initialization error: {e}")
            self.haar_face_cascade = None
    
    def detect_all_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using multiple methods"""
        results = []
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Method 1: MediaPipe face detection (short range)
        mp_faces_short = []
        if self.face_detection:
            try:
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
            except Exception as e:
                print(f"MediaPipe short range detection error: {e}")
        
        # Method 2: MediaPipe face detection (long range)
        mp_faces_long = []
        if self.face_detection_long:
            try:
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
            except Exception as e:
                print(f"MediaPipe long range detection error: {e}")
        
        # Method 3: OpenCV Haar cascade with multiple scales
        haar_faces = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try different scale factors to catch more faces
            scale_factors = [1.05, 1.1, 1.2, 1.3]
            min_neighbors_values = [2, 3, 4, 5]
            min_sizes = [(20, 20), (30, 30), (40, 40)]
            
            all_haar_detections = []
            
            for scale_factor in scale_factors:
                for min_neighbors in min_neighbors_values:
                    for min_size in min_sizes:
                        if self.haar_face_cascade is not None:
                            faces = self.haar_face_cascade.detectMultiScale(
                                gray, 
                                scaleFactor=scale_factor, 
                                minNeighbors=min_neighbors, 
                                minSize=min_size,
                                flags=cv2.CASCADE_SCALE_IMAGE
                            )
                            
                            for (x, y, w_face, h_face) in faces:
                                # Add some padding to improve detection accuracy
                                padding = 10
                                x = max(0, x - padding)
                                y = max(0, y - padding)
                                w_face = min(w - x, w_face + 2*padding)
                                h_face = min(h - y, h_face + 2*padding)
                                
                                all_haar_detections.append((x, y, w_face, h_face, 0.6))
            
            # Remove duplicate detections
            unique_detections = []
            for (x, y, w_face, h_face, conf) in all_haar_detections:
                is_duplicate = False
                
                # Check against existing MediaPipe detections
                all_existing = mp_faces_short + mp_faces_long
                for existing_face in all_existing:
                    ex, ey, ew, eh = existing_face['bbox']
                    if (abs(x - ex) < 60 and abs(y - ey) < 60):
                        is_duplicate = True
                        break
                
                # Check against other Haar detections
                for (ux, uy, uw, uh, _) in unique_detections:
                    if (abs(x - ux) < 40 and abs(y - uy) < 40):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_detections.append((x, y, w_face, h_face, conf))
            
            # Convert to standard format
            for i, (x, y, w_face, h_face, conf) in enumerate(unique_detections):
                haar_faces.append({
                    'method': 'opencv_haar',
                    'bbox': (x, y, w_face, h_face),
                    'confidence': conf,
                    'face_id': len(results) + len(mp_faces_short) + len(mp_faces_long) + i + 1
                })
                
        except Exception as e:
            print(f"Haar cascade detection error: {e}")
            
            # Fallback: Simple face detection using contours
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (15, 15), 0)
                thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if 1000 < area < 50000:  # Reasonable face size range
                        x, y, w_face, h_face = cv2.boundingRect(contour)
                        aspect_ratio = float(w_face) / h_face
                        if 0.7 < aspect_ratio < 1.3:  # Face-like aspect ratio
                            haar_faces.append({
                                'method': 'contour_fallback',
                                'bbox': (x, y, w_face, h_face),
                                'confidence': 0.4,
                                'face_id': len(results) + len(mp_faces_short) + len(mp_faces_long) + len(haar_faces) + 1
                            })
            except Exception as fallback_error:
                print(f"Fallback detection error: {fallback_error}")
        
        # Combine all detections
        all_faces = mp_faces_short + mp_faces_long + haar_faces
        
        # Process each detected face
        for face_data in all_faces:
            x, y, w_face, h_face = face_data['bbox']
            
            # Extract face region for analysis
            face_region = image[max(0, y):min(h, y + h_face), max(0, x):min(w, x + w_face)]
            
            if face_region.size > 0:
                # Analyze gaze direction (simplified)
                img_shape = (image.shape[0], image.shape[1], image.shape[2] if len(image.shape) == 3 else 1)
                gaze_direction = self._analyze_simple_gaze(face_region, (x, y, w_face, h_face), img_shape)
                
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