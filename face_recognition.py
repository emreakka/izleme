import cv2
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import mediapipe as mp

class FaceRecognitionSystem:
    """Face recognition and tracking system using MediaPipe face detection and feature extraction"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Face detection model
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Use full range model for better accuracy
            min_detection_confidence=0.7
        )
        
        # Face mesh for feature extraction
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Storage for known faces
        self.known_faces_file = "detection_data/known_faces.json"
        self.face_encodings_file = "detection_data/face_encodings.npy"
        self.known_faces = self._load_known_faces()
        self.face_counter = len(self.known_faces)
        
        # Ensure detection_data directory exists
        os.makedirs("detection_data", exist_ok=True)
    
    def _load_known_faces(self) -> Dict:
        """Load known faces from storage"""
        if os.path.exists(self.known_faces_file):
            try:
                with open(self.known_faces_file, 'r') as f:
                    data = json.load(f)
                    # Convert datetime strings back to datetime objects
                    for face_id, face_data in data.items():
                        face_data['first_seen'] = datetime.fromisoformat(face_data['first_seen'])
                        face_data['last_seen'] = datetime.fromisoformat(face_data['last_seen'])
                    return data
            except Exception as e:
                print(f"Error loading known faces: {e}")
        return {}
    
    def _save_known_faces(self):
        """Save known faces to storage"""
        try:
            # Convert datetime objects to strings for JSON serialization
            data_to_save = {}
            for face_id, face_data in self.known_faces.items():
                data_copy = face_data.copy()
                data_copy['first_seen'] = face_data['first_seen'].isoformat()
                data_copy['last_seen'] = face_data['last_seen'].isoformat()
                data_to_save[face_id] = data_copy
            
            with open(self.known_faces_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            print(f"Error saving known faces: {e}")
    
    def _extract_face_features(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract facial features for recognition using MediaPipe landmarks"""
        x, y, w, h = bbox
        face_region = image[max(0, y):min(image.shape[0], y+h), 
                          max(0, x):min(image.shape[1], x+w)]
        
        if face_region.size == 0:
            return None
        
        # Process with MediaPipe face mesh
        rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_face)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Extract key facial points for feature vector
            key_points = [0, 9, 10, 151, 33, 263, 61, 291, 39, 269, 130, 359]  # Key facial landmarks
            features = []
            
            for point_idx in key_points:
                if point_idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[point_idx]
                    features.extend([landmark.x, landmark.y, landmark.z])
            
            # Add geometric features
            if len(features) >= 36:  # Ensure we have enough points
                features_array = np.array(features)
                
                # Add distances between key points for better recognition
                eye_distance = np.linalg.norm(features_array[9:12] - features_array[21:24])  # Inter-eye distance
                nose_mouth_distance = np.linalg.norm(features_array[3:6] - features_array[15:18])  # Nose to mouth
                face_width = np.linalg.norm(features_array[12:15] - features_array[24:27])  # Face width
                
                features.extend([eye_distance, nose_mouth_distance, face_width])
                
                return np.array(features)
        
        return None
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature vectors"""
        try:
            # Ensure same length
            min_len = min(len(features1), len(features2))
            f1 = features1[:min_len]
            f2 = features2[:min_len]
            
            # Normalize features
            f1_norm = f1 / (np.linalg.norm(f1) + 1e-8)
            f2_norm = f2 / (np.linalg.norm(f2) + 1e-8)
            
            # Calculate cosine similarity
            similarity = np.dot(f1_norm, f2_norm)
            return float(similarity)
        except Exception:
            return 0.0
    
    def recognize_faces(self, image: np.ndarray) -> List[Dict]:
        """Recognize faces in the image and assign IDs"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        recognized_faces = []
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Extract features
                features = self._extract_face_features(image, (x, y, width, height))
                
                if features is not None:
                    # Try to match with known faces
                    best_match_id = None
                    best_similarity = 0.0
                    similarity_threshold = 0.75  # Adjust for sensitivity
                    
                    for face_id, face_data in self.known_faces.items():
                        if 'features' in face_data:
                            known_features = np.array(face_data['features'])
                            similarity = self._calculate_similarity(features, known_features)
                            
                            if similarity > best_similarity and similarity > similarity_threshold:
                                best_similarity = similarity
                                best_match_id = face_id
                    
                    # Create face info
                    if best_match_id:
                        # Update existing face
                        self.known_faces[best_match_id]['last_seen'] = datetime.now()
                        self.known_faces[best_match_id]['encounter_count'] += 1
                        face_info = {
                            'face_id': best_match_id,
                            'name': self.known_faces[best_match_id]['name'],
                            'is_known': True,
                            'similarity': best_similarity,
                            'bbox': (x, y, width, height),
                            'encounter_count': self.known_faces[best_match_id]['encounter_count'],
                            'first_seen': self.known_faces[best_match_id]['first_seen'],
                            'last_seen': self.known_faces[best_match_id]['last_seen']
                        }
                    else:
                        # New face
                        self.face_counter += 1
                        new_face_id = f"Person_{self.face_counter}"
                        now = datetime.now()
                        
                        self.known_faces[new_face_id] = {
                            'name': new_face_id,
                            'features': features.tolist(),
                            'first_seen': now,
                            'last_seen': now,
                            'encounter_count': 1
                        }
                        
                        face_info = {
                            'face_id': new_face_id,
                            'name': new_face_id,
                            'is_known': False,
                            'similarity': 1.0,
                            'bbox': (x, y, width, height),
                            'encounter_count': 1,
                            'first_seen': now,
                            'last_seen': now
                        }
                    
                    recognized_faces.append(face_info)
        
        # Save updated face data
        self._save_known_faces()
        
        return recognized_faces
    
    def rename_face(self, face_id: str, new_name: str) -> bool:
        """Rename a known face"""
        if face_id in self.known_faces:
            self.known_faces[face_id]['name'] = new_name
            self._save_known_faces()
            return True
        return False
    
    def get_known_faces_summary(self) -> Dict:
        """Get summary of all known faces"""
        summary = {}
        for face_id, face_data in self.known_faces.items():
            summary[face_id] = {
                'name': face_data['name'],
                'encounter_count': face_data['encounter_count'],
                'first_seen': face_data['first_seen'],
                'last_seen': face_data['last_seen']
            }
        return summary
    
    def forget_face(self, face_id: str) -> bool:
        """Remove a face from known faces"""
        if face_id in self.known_faces:
            del self.known_faces[face_id]
            self._save_known_faces()
            return True
        return False
    
    def draw_face_recognition_overlay(self, image: np.ndarray, face_info: Dict, 
                                    color: Tuple[int, int, int] = (255, 165, 0)) -> np.ndarray:
        """Draw face recognition overlay on image"""
        result_image = image.copy()
        x, y, w, h = face_info['bbox']
        
        # Draw bounding box
        thickness = 3 if face_info['is_known'] else 2
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
        
        # Prepare text
        name = face_info['name']
        encounter_text = f"Seen {face_info['encounter_count']} times"
        similarity_text = f"Match: {face_info['similarity']:.2f}" if face_info['is_known'] else "New person"
        
        # Draw background for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        
        # Calculate text size for background
        (text_w, text_h), baseline = cv2.getTextSize(name, font, font_scale, text_thickness)
        cv2.rectangle(result_image, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
        
        # Draw text
        cv2.putText(result_image, name, (x + 5, y - 5), font, font_scale, (255, 255, 255), text_thickness)
        
        # Draw additional info below
        info_y = y + h + 20
        cv2.putText(result_image, encounter_text, (x, info_y), font, 0.5, color, 1)
        cv2.putText(result_image, similarity_text, (x, info_y + 15), font, 0.5, color, 1)
        
        return result_image