import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import pickle
import os
import time
from typing import Dict, List, Tuple, Optional
import logging
import json

class FaceTrackingSystem:
    """
    Advanced face tracking and recognition system using:
    - Feature extraction and comparison
    - Persistent face database
    - scikit-learn clustering for face matching
    - Temporal tracking across frames
    """
    
    def __init__(self, database_path: str = "face_database.pkl"):
        self.logger = logging.getLogger(__name__)
        self.database_path = database_path
        self.known_faces = {}
        self.face_counter = 0
        self.similarity_threshold = 0.7
        self.temporal_tracks = {}  # For tracking faces across frames
        
        # Load existing face database
        self._load_face_database()
        
        self.logger.info("Face tracking system initialized")
    
    def _load_face_database(self):
        """Load known faces from persistent storage"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('known_faces', {})
                    self.face_counter = data.get('face_counter', 0)
                    
                self.logger.info(f"Loaded {len(self.known_faces)} known faces from database")
            else:
                self.known_faces = {}
                self.face_counter = 0
                self.logger.info("No existing face database found, starting fresh")
                
        except Exception as e:
            self.logger.warning(f"Failed to load face database: {e}")
            self.known_faces = {}
            self.face_counter = 0
    
    def _save_face_database(self):
        """Save known faces to persistent storage"""
        try:
            data = {
                'known_faces': self.known_faces,
                'face_counter': self.face_counter
            }
            
            with open(self.database_path, 'wb') as f:
                pickle.dump(data, f)
                
            self.logger.info("Face database saved successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to save face database: {e}")
    
    def track_faces(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Track and identify faces across frames
        
        Args:
            image: Input image in BGR format
            detections: List of face detections
            
        Returns:
            List of detections with tracking information
        """
        try:
            if not detections:
                return detections
            
            # Extract features for each detected face
            for detection in detections:
                features = self._extract_face_features(image, detection)
                
                if features is not None:
                    # Find matching known face
                    matched_face_id = self._find_matching_face(features)
                    
                    if matched_face_id:
                        # Update existing face
                        self._update_known_face(matched_face_id, features, detection)
                        detection['person_name'] = self.known_faces[matched_face_id]['name']
                        detection['face_id'] = matched_face_id
                        detection['recognition_confidence'] = self.known_faces[matched_face_id].get('confidence', 0.8)
                    else:
                        # Register new face
                        new_face_id = self._register_new_face(features, detection)
                        detection['person_name'] = f"Person-{new_face_id}"
                        detection['face_id'] = new_face_id
                        detection['recognition_confidence'] = 0.9  # High confidence for new registration
                else:
                    # Fallback if feature extraction fails
                    detection['person_name'] = f"Unknown-{len(detections)}"
                    detection['face_id'] = None
                    detection['recognition_confidence'] = 0.1
            
            # Save updated database
            self._save_face_database()
            
            return detections
            
        except Exception as e:
            self.logger.warning(f"Face tracking failed: {e}")
            # Add fallback names
            for i, detection in enumerate(detections):
                if 'person_name' not in detection:
                    detection['person_name'] = f"Person-{i+1}"
                    detection['face_id'] = None
                    detection['recognition_confidence'] = 0.1
            
            return detections
    
    def _extract_face_features(self, image: np.ndarray, detection: Dict) -> Optional[np.ndarray]:
        """Extract features from face region for recognition"""
        try:
            x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            if face_region.size == 0:
                return None
            
            # Resize to standard size
            face_resized = cv2.resize(face_region, (100, 100))
            
            # Convert to grayscale
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Extract multiple types of features
            features = []
            
            # 1. Histogram features
            hist_features = cv2.calcHist([face_gray], [0], None, [256], [0, 256]).flatten()
            hist_features = hist_features / np.sum(hist_features)  # Normalize
            features.extend(hist_features[:50])  # Use first 50 bins
            
            # 2. LBP (Local Binary Pattern) features
            lbp_features = self._extract_lbp_features(face_gray)
            features.extend(lbp_features)
            
            # 3. Geometric features
            geometric_features = self._extract_geometric_features_simple(face_gray)
            features.extend(geometric_features)
            
            # 4. Texture features
            texture_features = self._extract_texture_features(face_gray)
            features.extend(texture_features)
            
            return np.array(features)
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            return None
    
    def _extract_lbp_features(self, face_gray: np.ndarray) -> List[float]:
        """Extract Local Binary Pattern features"""
        try:
            # Simple LBP implementation
            h, w = face_gray.shape
            lbp_image = np.zeros_like(face_gray)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = face_gray[i, j]
                    binary_pattern = 0
                    
                    # 8-neighbor LBP
                    neighbors = [
                        face_gray[i-1, j-1], face_gray[i-1, j], face_gray[i-1, j+1],
                        face_gray[i, j+1], face_gray[i+1, j+1], face_gray[i+1, j],
                        face_gray[i+1, j-1], face_gray[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            binary_pattern += 2**k
                    
                    lbp_image[i, j] = binary_pattern
            
            # Calculate histogram of LBP
            hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(float)
            hist = hist / (np.sum(hist) + 1e-8)  # Normalize
            
            return hist[:50].tolist()  # Return first 50 features
            
        except Exception:
            return [0.0] * 50
    
    def _extract_geometric_features_simple(self, face_gray: np.ndarray) -> List[float]:
        """Extract simple geometric features"""
        try:
            h, w = face_gray.shape
            
            # Divide face into regions and calculate statistics
            regions = [
                face_gray[0:h//3, 0:w//3],      # Top-left
                face_gray[0:h//3, w//3:2*w//3], # Top-center
                face_gray[0:h//3, 2*w//3:w],    # Top-right
                face_gray[h//3:2*h//3, 0:w//3], # Middle-left
                face_gray[h//3:2*h//3, w//3:2*w//3], # Center
                face_gray[h//3:2*h//3, 2*w//3:w],    # Middle-right
                face_gray[2*h//3:h, 0:w//3],    # Bottom-left
                face_gray[2*h//3:h, w//3:2*w//3], # Bottom-center
                face_gray[2*h//3:h, 2*w//3:w],  # Bottom-right
            ]
            
            features = []
            for region in regions:
                if region.size > 0:
                    features.extend([
                        float(np.mean(region)),
                        float(np.std(region)),
                        float(np.var(region))
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
            
            return features
            
        except Exception:
            return [0.0] * 27  # 9 regions * 3 features
    
    def _extract_texture_features(self, face_gray: np.ndarray) -> List[float]:
        """Extract texture features using gradients"""
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude and direction
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)
            
            # Statistics
            features = [
                float(np.mean(magnitude)),
                float(np.std(magnitude)),
                float(np.mean(direction)),
                float(np.std(direction)),
                float(np.max(magnitude)),
                float(np.min(magnitude))
            ]
            
            return features
            
        except Exception:
            return [0.0] * 6
    
    def _find_matching_face(self, features: np.ndarray) -> Optional[str]:
        """Find matching face in known faces database"""
        try:
            if not self.known_faces:
                return None
            
            best_similarity = 0.0
            best_match_id = None
            
            for face_id, face_data in self.known_faces.items():
                stored_features = np.array(face_data['features'])
                
                # Ensure feature vectors have same length
                min_length = min(len(features), len(stored_features))
                features_truncated = features[:min_length]
                stored_features_truncated = stored_features[:min_length]
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    features_truncated.reshape(1, -1),
                    stored_features_truncated.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_match_id = face_id
            
            if best_match_id:
                # Update confidence based on similarity
                self.known_faces[best_match_id]['confidence'] = best_similarity
                self.known_faces[best_match_id]['last_seen'] = time.time()
                self.known_faces[best_match_id]['appearance_count'] += 1
                
                return best_match_id
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Face matching failed: {e}")
            return None
    
    def _register_new_face(self, features: np.ndarray, detection: Dict) -> str:
        """Register a new face in the database"""
        try:
            self.face_counter += 1
            face_id = f"face_{self.face_counter:04d}"
            
            self.known_faces[face_id] = {
                'features': features.tolist(),
                'name': f"Person-{self.face_counter}",
                'first_seen': time.time(),
                'last_seen': time.time(),
                'appearance_count': 1,
                'confidence': 0.9,
                'detection_info': {
                    'width': detection.get('w', 0),
                    'height': detection.get('h', 0),
                    'quality': detection.get('sharpness', 0)
                }
            }
            
            self.logger.info(f"Registered new face: {face_id}")
            return face_id
            
        except Exception as e:
            self.logger.warning(f"Face registration failed: {e}")
            return f"unknown_{int(time.time())}"
    
    def _update_known_face(self, face_id: str, features: np.ndarray, detection: Dict):
        """Update known face with new observation"""
        try:
            if face_id in self.known_faces:
                face_data = self.known_faces[face_id]
                
                # Update features with weighted average
                old_features = np.array(face_data['features'])
                alpha = 0.1  # Learning rate
                
                # Ensure same length
                min_length = min(len(features), len(old_features))
                new_features = (1 - alpha) * old_features[:min_length] + alpha * features[:min_length]
                
                face_data['features'] = new_features.tolist()
                face_data['last_seen'] = time.time()
                face_data['appearance_count'] += 1
                
                # Update quality metrics
                current_quality = detection.get('sharpness', 0)
                if current_quality > face_data['detection_info'].get('quality', 0):
                    face_data['detection_info']['quality'] = current_quality
                    face_data['detection_info']['width'] = detection.get('w', 0)
                    face_data['detection_info']['height'] = detection.get('h', 0)
                
        except Exception as e:
            self.logger.warning(f"Face update failed: {e}")
    
    def rename_face(self, face_id: str, new_name: str) -> bool:
        """Rename a known face"""
        try:
            if face_id in self.known_faces:
                self.known_faces[face_id]['name'] = new_name
                self._save_face_database()
                self.logger.info(f"Renamed face {face_id} to {new_name}")
                return True
            else:
                self.logger.warning(f"Face ID {face_id} not found")
                return False
                
        except Exception as e:
            self.logger.warning(f"Face rename failed: {e}")
            return False
    
    def forget_face(self, face_id: str) -> bool:
        """Remove a face from the known faces database"""
        try:
            if face_id in self.known_faces:
                del self.known_faces[face_id]
                self._save_face_database()
                self.logger.info(f"Removed face {face_id} from database")
                return True
            else:
                self.logger.warning(f"Face ID {face_id} not found")
                return False
                
        except Exception as e:
            self.logger.warning(f"Face removal failed: {e}")
            return False
    
    def get_known_faces_summary(self) -> Dict:
        """Get summary of all known faces"""
        try:
            summary = {}
            
            for face_id, face_data in self.known_faces.items():
                summary[face_id] = {
                    'name': face_data['name'],
                    'count': face_data['appearance_count'],
                    'first_seen': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(face_data['first_seen'])),
                    'last_seen': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(face_data['last_seen'])),
                    'confidence': face_data.get('confidence', 0.0),
                    'quality': face_data['detection_info'].get('quality', 0)
                }
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"Face summary generation failed: {e}")
            return {}
    
    def get_face_statistics(self) -> Dict:
        """Get face tracking statistics"""
        try:
            if not self.known_faces:
                return {
                    'total_faces': 0,
                    'active_faces': 0,
                    'average_appearances': 0,
                    'most_frequent_face': None
                }
            
            total_faces = len(self.known_faces)
            current_time = time.time()
            active_threshold = 24 * 60 * 60  # 24 hours
            
            active_faces = sum(1 for face_data in self.known_faces.values() 
                             if current_time - face_data['last_seen'] < active_threshold)
            
            total_appearances = sum(face_data['appearance_count'] for face_data in self.known_faces.values())
            average_appearances = total_appearances / total_faces
            
            # Find most frequent face
            most_frequent_face = max(self.known_faces.items(), 
                                   key=lambda x: x[1]['appearance_count'])
            
            return {
                'total_faces': total_faces,
                'active_faces': active_faces,
                'average_appearances': round(average_appearances, 2),
                'most_frequent_face': {
                    'id': most_frequent_face[0],
                    'name': most_frequent_face[1]['name'],
                    'appearances': most_frequent_face[1]['appearance_count']
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Statistics generation failed: {e}")
            return {
                'total_faces': 0,
                'active_faces': 0,
                'average_appearances': 0,
                'most_frequent_face': None
            }
    
    def export_face_database(self, export_path: str = "face_database_export.json") -> bool:
        """Export face database to JSON format"""
        try:
            export_data = {}
            
            for face_id, face_data in self.known_faces.items():
                export_data[face_id] = {
                    'name': face_data['name'],
                    'first_seen': face_data['first_seen'],
                    'last_seen': face_data['last_seen'],
                    'appearance_count': face_data['appearance_count'],
                    'confidence': face_data.get('confidence', 0.0),
                    'detection_info': face_data['detection_info']
                    # Note: Features are excluded for privacy/size reasons
                }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Face database exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Database export failed: {e}")
            return False
    
    def clear_all_faces(self) -> bool:
        """Clear all known faces from database"""
        try:
            self.known_faces = {}
            self.face_counter = 0
            self._save_face_database()
            self.logger.info("All known faces cleared")
            return True
            
        except Exception as e:
            self.logger.warning(f"Database clear failed: {e}")
            return False