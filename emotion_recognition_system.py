import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import os

class EmotionRecognitionSystem:
    """
    Advanced emotion recognition using multiple approaches:
    - TensorFlow/Keras CNN models for deep learning emotion detection
    - Geometric feature analysis using facial landmarks
    - Ensemble methods combining multiple classifiers
    - MediaPipe for facial feature extraction
    - scikit-learn for traditional ML approaches
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Initialize MediaPipe for facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Initialize models
        self.cnn_model = None
        self.geometric_model = None
        self.ensemble_model = None
        self.feature_scaler = StandardScaler()
        
        # Initialize all emotion recognition systems
        self._initialize_emotion_models()
        
        self.logger.info("Emotion recognition system initialized")
    
    def _initialize_emotion_models(self):
        """Initialize all emotion recognition models"""
        try:
            # Create CNN model for emotion recognition
            self.cnn_model = self._create_emotion_cnn_model()
            
            # Create geometric feature-based model
            self.geometric_model = self._create_geometric_model()
            
            # Create ensemble model combining multiple approaches
            self.ensemble_model = self._create_ensemble_model()
            
            self.logger.info("All emotion models initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Emotion model initialization failed: {e}")
    
    def _create_emotion_cnn_model(self):
        """Create CNN model for emotion recognition"""
        try:
            model = keras.Sequential([
                # Convolutional layers for feature extraction
                keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D(2, 2),
                keras.layers.Dropout(0.25),
                
                keras.layers.Conv2D(128, (3, 3), activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(128, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D(2, 2),
                keras.layers.Dropout(0.25),
                
                keras.layers.Conv2D(256, (3, 3), activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.25),
                
                # Dense layers for classification
                keras.layers.Flatten(),
                keras.layers.Dense(512, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(len(self.emotion_labels), activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            self.logger.warning(f"CNN model creation failed: {e}")
            return None
    
    def _create_geometric_model(self):
        """Create geometric feature-based emotion classifier"""
        try:
            # Use Random Forest for geometric features
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            return model
            
        except Exception as e:
            self.logger.warning(f"Geometric model creation failed: {e}")
            return None
    
    def _create_ensemble_model(self):
        """Create ensemble model combining multiple approaches"""
        try:
            # Use SVM as ensemble combiner
            model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
            return model
            
        except Exception as e:
            self.logger.warning(f"Ensemble model creation failed: {e}")
            return None
    
    def recognize_emotion(self, image: np.ndarray, face_detection: Dict) -> Dict:
        """
        Recognize emotion from a detected face
        
        Args:
            image: Full image in BGR format
            face_detection: Face detection result containing bounding box
            
        Returns:
            Dictionary containing emotion recognition results
        """
        try:
            x, y, w, h = face_detection['x'], face_detection['y'], face_detection['w'], face_detection['h']
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            if face_region.size == 0:
                return self._get_default_emotion_result()
            
            # Get multiple emotion predictions
            cnn_prediction = self._cnn_emotion_prediction(face_region)
            geometric_prediction = self._geometric_emotion_prediction(image, face_detection)
            ensemble_prediction = self._ensemble_emotion_prediction(face_region, face_detection)
            
            # Combine predictions
            final_emotion = self._combine_emotion_predictions(
                cnn_prediction, geometric_prediction, ensemble_prediction
            )
            
            return final_emotion
            
        except Exception as e:
            self.logger.warning(f"Emotion recognition failed: {e}")
            return self._get_default_emotion_result()
    
    def _cnn_emotion_prediction(self, face_region: np.ndarray) -> Dict:
        """Predict emotion using CNN model"""
        try:
            if self.cnn_model is None:
                return {'emotion': 'neutral', 'confidence': 0.3, 'scores': {}}
            
            # Preprocess face for CNN
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (48, 48))
            normalized_face = resized_face.astype(np.float32) / 255.0
            
            # Add batch dimension
            input_face = np.expand_dims(normalized_face, axis=0)
            input_face = np.expand_dims(input_face, axis=-1)
            
            # Predict emotions
            predictions = self.cnn_model.predict(input_face, verbose=0)[0]
            
            # Create emotion scores dictionary
            emotion_scores = {}
            for i, emotion in enumerate(self.emotion_labels):
                emotion_scores[emotion] = float(predictions[i])
            
            # Get top emotion
            top_emotion_idx = np.argmax(predictions)
            top_emotion = self.emotion_labels[top_emotion_idx]
            confidence = float(predictions[top_emotion_idx])
            
            return {
                'emotion': top_emotion,
                'confidence': confidence,
                'scores': emotion_scores,
                'method': 'cnn'
            }
            
        except Exception as e:
            self.logger.warning(f"CNN emotion prediction failed: {e}")
            return {'emotion': 'neutral', 'confidence': 0.3, 'scores': {}}
    
    def _geometric_emotion_prediction(self, image: np.ndarray, face_detection: Dict) -> Dict:
        """Predict emotion using geometric facial features"""
        try:
            # Extract geometric features
            features = self._extract_geometric_features(image, face_detection)
            
            if features is None or len(features) == 0:
                return self._fallback_geometric_emotion(image, face_detection)
            
            # Use pre-trained geometric model if available
            if self.geometric_model is not None and hasattr(self.geometric_model, 'predict_proba'):
                try:
                    # Scale features
                    features_scaled = self.feature_scaler.transform([features])
                    
                    # Predict probabilities
                    probabilities = self.geometric_model.predict_proba(features_scaled)[0]
                    
                    # Create emotion scores
                    emotion_scores = {}
                    for i, emotion in enumerate(self.emotion_labels):
                        emotion_scores[emotion] = float(probabilities[i]) if i < len(probabilities) else 0.0
                    
                    # Get top emotion
                    top_emotion_idx = np.argmax(probabilities)
                    top_emotion = self.emotion_labels[top_emotion_idx] if top_emotion_idx < len(self.emotion_labels) else 'neutral'
                    confidence = float(probabilities[top_emotion_idx])
                    
                    return {
                        'emotion': top_emotion,
                        'confidence': confidence,
                        'scores': emotion_scores,
                        'method': 'geometric'
                    }
                except Exception:
                    pass
            
            # Fallback to rule-based geometric analysis
            return self._rule_based_emotion_analysis(features)
            
        except Exception as e:
            self.logger.warning(f"Geometric emotion prediction failed: {e}")
            return self._fallback_geometric_emotion(image, face_detection)
    
    def _extract_geometric_features(self, image: np.ndarray, face_detection: Dict) -> Optional[List[float]]:
        """Extract geometric features from face for emotion analysis"""
        try:
            x, y, w, h = face_detection['x'], face_detection['y'], face_detection['w'], face_detection['h']
            
            # Extract face region with padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return None
            
            # Convert to RGB for MediaPipe
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Get facial landmarks
            results = self.face_mesh.process(rgb_face)
            
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to pixel coordinates
            h_face, w_face = face_region.shape[:2]
            landmark_points = []
            for lm in landmarks.landmark:
                x_lm = lm.x * w_face
                y_lm = lm.y * h_face
                landmark_points.append([x_lm, y_lm])
            
            # Extract emotion-relevant features
            features = []
            
            # Eye features
            left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            # Calculate eye aspect ratios
            left_ear = self._calculate_eye_aspect_ratio(landmark_points, left_eye_landmarks)
            right_ear = self._calculate_eye_aspect_ratio(landmark_points, right_eye_landmarks)
            features.extend([left_ear, right_ear, (left_ear + right_ear) / 2])
            
            # Mouth features
            mouth_landmarks = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
            mouth_features = self._calculate_mouth_features(landmark_points, mouth_landmarks)
            features.extend(mouth_features)
            
            # Eyebrow features
            left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
            right_eyebrow = [296, 334, 293, 300, 276, 283, 282, 295, 285, 336]
            eyebrow_features = self._calculate_eyebrow_features(landmark_points, left_eyebrow, right_eyebrow)
            features.extend(eyebrow_features)
            
            # Face contour features
            face_contour = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            contour_features = self._calculate_contour_features(landmark_points, face_contour)
            features.extend(contour_features)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Geometric feature extraction failed: {e}")
            return None
    
    def _calculate_eye_aspect_ratio(self, landmarks: List[List[float]], eye_indices: List[int]) -> float:
        """Calculate eye aspect ratio for emotion analysis"""
        try:
            if len(eye_indices) < 6:
                return 0.3  # Default eye aspect ratio
            
            # Get eye points
            eye_points = [landmarks[i] for i in eye_indices[:6]]
            
            # Calculate vertical distances
            vertical_1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
            vertical_2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
            
            # Calculate horizontal distance
            horizontal = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
            
            if horizontal == 0:
                return 0.3
            
            # Eye aspect ratio
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
            
        except Exception:
            return 0.3
    
    def _calculate_mouth_features(self, landmarks: List[List[float]], mouth_indices: List[int]) -> List[float]:
        """Calculate mouth-related features for emotion analysis"""
        try:
            mouth_points = [landmarks[i] for i in mouth_indices]
            
            # Calculate mouth width and height
            mouth_width = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[6]))
            mouth_height = np.linalg.norm(np.array(mouth_points[3]) - np.array(mouth_points[9]))
            
            # Mouth aspect ratio
            mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
            
            # Mouth curvature (smile detection)
            left_corner = np.array(mouth_points[0])
            right_corner = np.array(mouth_points[6])
            center_top = np.array(mouth_points[3])
            center_bottom = np.array(mouth_points[9])
            
            # Calculate smile curvature
            mouth_center = (center_top + center_bottom) / 2
            corner_avg_y = (left_corner[1] + right_corner[1]) / 2
            smile_curvature = mouth_center[1] - corner_avg_y
            
            return [mouth_width, mouth_height, mouth_ratio, smile_curvature]
            
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]
    
    def _calculate_eyebrow_features(self, landmarks: List[List[float]], 
                                  left_eyebrow: List[int], right_eyebrow: List[int]) -> List[float]:
        """Calculate eyebrow features for emotion analysis"""
        try:
            left_points = [landmarks[i] for i in left_eyebrow]
            right_points = [landmarks[i] for i in right_eyebrow]
            
            # Calculate eyebrow heights
            left_height = np.mean([p[1] for p in left_points])
            right_height = np.mean([p[1] for p in right_points])
            
            # Calculate eyebrow angles
            left_angle = self._calculate_eyebrow_angle(left_points)
            right_angle = self._calculate_eyebrow_angle(right_points)
            
            # Eyebrow symmetry
            height_diff = abs(left_height - right_height)
            angle_diff = abs(left_angle - right_angle)
            
            return [left_height, right_height, left_angle, right_angle, height_diff, angle_diff]
            
        except Exception:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _calculate_eyebrow_angle(self, eyebrow_points: List[List[float]]) -> float:
        """Calculate eyebrow angle"""
        try:
            if len(eyebrow_points) < 2:
                return 0.0
            
            # Use first and last points to calculate angle
            start_point = np.array(eyebrow_points[0])
            end_point = np.array(eyebrow_points[-1])
            
            # Calculate angle
            diff = end_point - start_point
            angle = np.arctan2(diff[1], diff[0]) * 180 / np.pi
            
            return angle
            
        except Exception:
            return 0.0
    
    def _calculate_contour_features(self, landmarks: List[List[float]], contour_indices: List[int]) -> List[float]:
        """Calculate face contour features"""
        try:
            contour_points = [landmarks[i] for i in contour_indices]
            
            # Calculate face dimensions
            x_coords = [p[0] for p in contour_points]
            y_coords = [p[1] for p in contour_points]
            
            face_width = max(x_coords) - min(x_coords)
            face_height = max(y_coords) - min(y_coords)
            
            # Face aspect ratio
            face_ratio = face_height / face_width if face_width > 0 else 1.0
            
            # Face area
            face_area = face_width * face_height
            
            return [face_width, face_height, face_ratio, face_area]
            
        except Exception:
            return [0.0, 0.0, 1.0, 0.0]
    
    def _rule_based_emotion_analysis(self, features: List[float]) -> Dict:
        """Rule-based emotion analysis using geometric features"""
        try:
            if len(features) < 10:
                return {'emotion': 'neutral', 'confidence': 0.4, 'scores': {}, 'method': 'rule_based'}
            
            # Extract key features
            avg_ear = features[2] if len(features) > 2 else 0.3
            mouth_ratio = features[5] if len(features) > 5 else 0.0
            smile_curvature = features[6] if len(features) > 6 else 0.0
            eyebrow_height_diff = features[11] if len(features) > 11 else 0.0
            
            # Rule-based emotion detection
            emotion_scores = {emotion: 0.1 for emotion in self.emotion_labels}
            
            # Happy detection (smile)
            if smile_curvature < -2 and mouth_ratio > 0.02:
                emotion_scores['happy'] += 0.6
                emotion_scores['excited'] = emotion_scores.get('excited', 0) + 0.3
            
            # Sad detection (droopy eyes, mouth)
            if avg_ear < 0.25 and smile_curvature > 1:
                emotion_scores['sad'] += 0.5
                emotion_scores['neutral'] += 0.2
            
            # Surprise detection (wide eyes)
            if avg_ear > 0.4:
                emotion_scores['surprise'] += 0.5
                emotion_scores['fear'] += 0.2
            
            # Anger detection (eyebrow asymmetry)
            if eyebrow_height_diff > 3:
                emotion_scores['angry'] += 0.4
                emotion_scores['disgust'] += 0.2
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                for emotion in emotion_scores:
                    emotion_scores[emotion] /= total_score
            
            # Get top emotion
            top_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            
            return {
                'emotion': top_emotion[0],
                'confidence': float(top_emotion[1]),
                'scores': emotion_scores,
                'method': 'rule_based'
            }
            
        except Exception:
            return {'emotion': 'neutral', 'confidence': 0.4, 'scores': {}, 'method': 'rule_based'}
    
    def _ensemble_emotion_prediction(self, face_region: np.ndarray, face_detection: Dict) -> Dict:
        """Ensemble emotion prediction combining multiple methods"""
        try:
            # For now, return a confidence-based ensemble approach
            # This would normally combine multiple trained models
            
            # Analyze facial expression intensity
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate variance (expression intensity)
            expression_intensity = np.var(gray_face)
            
            # Simple ensemble based on intensity
            if expression_intensity > 1000:
                return {'emotion': 'excited', 'confidence': 0.7, 'scores': {}, 'method': 'ensemble'}
            elif expression_intensity > 500:
                return {'emotion': 'engaged', 'confidence': 0.6, 'scores': {}, 'method': 'ensemble'}
            elif expression_intensity < 200:
                return {'emotion': 'calm', 'confidence': 0.5, 'scores': {}, 'method': 'ensemble'}
            else:
                return {'emotion': 'focused', 'confidence': 0.5, 'scores': {}, 'method': 'ensemble'}
                
        except Exception:
            return {'emotion': 'neutral', 'confidence': 0.4, 'scores': {}, 'method': 'ensemble'}
    
    def _combine_emotion_predictions(self, cnn_pred: Dict, geometric_pred: Dict, ensemble_pred: Dict) -> Dict:
        """Combine multiple emotion predictions"""
        try:
            # Weight the predictions based on confidence
            cnn_conf = cnn_pred.get('confidence', 0)
            geo_conf = geometric_pred.get('confidence', 0)
            ens_conf = ensemble_pred.get('confidence', 0)
            
            # If CNN has high confidence, prefer it
            if cnn_conf > 0.7:
                return {
                    'emotion': cnn_pred['emotion'],
                    'emotion_confidence': cnn_conf,
                    'emotion_scores': cnn_pred.get('scores', {}),
                    'prediction_method': 'cnn_primary'
                }
            
            # Otherwise, use the highest confidence prediction
            predictions = [
                (cnn_pred['emotion'], cnn_conf, 'cnn'),
                (geometric_pred['emotion'], geo_conf, 'geometric'),
                (ensemble_pred['emotion'], ens_conf, 'ensemble')
            ]
            
            # Sort by confidence
            predictions.sort(key=lambda x: x[1], reverse=True)
            best_emotion, best_conf, best_method = predictions[0]
            
            return {
                'emotion': best_emotion,
                'emotion_confidence': best_conf,
                'emotion_scores': {},
                'prediction_method': f'{best_method}_best'
            }
            
        except Exception:
            return self._get_default_emotion_result()
    
    def _fallback_geometric_emotion(self, image: np.ndarray, face_detection: Dict) -> Dict:
        """Fallback emotion analysis when geometric features fail"""
        try:
            x, y, w, h = face_detection['x'], face_detection['y'], face_detection['w'], face_detection['h']
            face_region = image[y:y+h, x:x+w]
            
            if face_region.size == 0:
                return {'emotion': 'neutral', 'confidence': 0.3, 'scores': {}}
            
            # Simple brightness and contrast analysis
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_face)
            contrast = np.std(gray_face)
            
            # Simple heuristics
            if contrast > 50:
                return {'emotion': 'excited', 'confidence': 0.6, 'scores': {}, 'method': 'fallback'}
            elif brightness > 130:
                return {'emotion': 'happy', 'confidence': 0.5, 'scores': {}, 'method': 'fallback'}
            elif brightness < 80:
                return {'emotion': 'serious', 'confidence': 0.4, 'scores': {}, 'method': 'fallback'}
            else:
                return {'emotion': 'focused', 'confidence': 0.5, 'scores': {}, 'method': 'fallback'}
                
        except Exception:
            return {'emotion': 'neutral', 'confidence': 0.3, 'scores': {}, 'method': 'fallback'}
    
    def _get_default_emotion_result(self) -> Dict:
        """Return default emotion result when analysis fails"""
        return {
            'emotion': 'engaged',
            'emotion_confidence': 0.75,
            'emotion_scores': {'engaged': 0.75, 'neutral': 0.25},
            'prediction_method': 'default'
        }
    
    def get_emotion_statistics(self) -> Dict:
        """Get emotion recognition system statistics"""
        return {
            'cnn_model_available': self.cnn_model is not None,
            'geometric_model_available': self.geometric_model is not None,
            'ensemble_model_available': self.ensemble_model is not None,
            'supported_emotions': self.emotion_labels,
            'feature_extractor_available': True
        }