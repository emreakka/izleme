import cv2
import numpy as np
from typing import Dict, List, Tuple
import mediapipe as mp

class EmotionDetector:
    """Emotion detection using facial landmarks and geometric features"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Define emotion categories
        self.emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fearful', 'disgusted']
        
        # Key facial landmarks for emotion detection
        self.MOUTH_LANDMARKS = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        self.EYEBROW_LEFT = [70, 63, 105, 66, 107, 55, 65]
        self.EYEBROW_RIGHT = [296, 334, 293, 300, 276, 283, 282]
        self.EYE_LEFT = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.EYE_RIGHT = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
    def detect_emotion(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect emotions from facial expressions
        
        Args:
            image: Input image in BGR format
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of emotion detection results for each face
        """
        results = []
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        face_results = self.face_mesh.process(rgb_image)
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                emotion_info = self._analyze_emotion(face_landmarks, (image.shape[0], image.shape[1], image.shape[2]))
                
                if emotion_info['confidence'] >= confidence_threshold:
                    results.append(emotion_info)
        
        return results
    
    def _analyze_emotion(self, landmarks, image_shape: Tuple[int, int, int]) -> Dict:
        """Analyze emotion from facial landmarks"""
        h, w = image_shape[:2]
        
        # Convert landmarks to pixel coordinates
        landmarks_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_points.append([x, y])
        
        landmarks_array = np.array(landmarks_points)
        
        # Extract facial features
        features = self._extract_facial_features(landmarks_array)
        
        # Classify emotion based on features
        emotion_scores = self._classify_emotion(features)
        
        # Get dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        return {
            'emotion': dominant_emotion[0],
            'confidence': dominant_emotion[1],
            'emotion_scores': emotion_scores,
            'features': features,
            'landmarks': landmarks_array
        }
    
    def _extract_facial_features(self, landmarks: np.ndarray) -> Dict:
        """Extract geometric features from facial landmarks"""
        features = {}
        
        try:
            # Mouth features
            mouth_points = landmarks[self.MOUTH_LANDMARKS]
            mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0])
            mouth_height = np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1])
            mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
            
            # Mouth corners (smile detection)
            left_corner = landmarks[61]  # Left mouth corner
            right_corner = landmarks[291]  # Right mouth corner
            mouth_center_y = np.mean(mouth_points[:, 1])
            mouth_curve = (left_corner[1] + right_corner[1]) / 2 - mouth_center_y
            
            features['mouth_ratio'] = mouth_ratio
            features['mouth_curve'] = mouth_curve
            features['mouth_width'] = mouth_width
            
            # Eye features
            left_eye_points = landmarks[self.EYE_LEFT]
            right_eye_points = landmarks[self.EYE_RIGHT]
            
            left_eye_height = np.max(left_eye_points[:, 1]) - np.min(left_eye_points[:, 1])
            left_eye_width = np.max(left_eye_points[:, 0]) - np.min(left_eye_points[:, 0])
            left_eye_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 0
            
            right_eye_height = np.max(right_eye_points[:, 1]) - np.min(right_eye_points[:, 1])
            right_eye_width = np.max(right_eye_points[:, 0]) - np.min(right_eye_points[:, 0])
            right_eye_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 0
            
            features['left_eye_ratio'] = left_eye_ratio
            features['right_eye_ratio'] = right_eye_ratio
            features['avg_eye_ratio'] = (left_eye_ratio + right_eye_ratio) / 2
            
            # Eyebrow features
            left_eyebrow_points = landmarks[self.EYEBROW_LEFT]
            right_eyebrow_points = landmarks[self.EYEBROW_RIGHT]
            
            # Distance between eyebrows and eyes
            left_eyebrow_eye_dist = np.mean(left_eyebrow_points[:, 1]) - np.mean(left_eye_points[:, 1])
            right_eyebrow_eye_dist = np.mean(right_eyebrow_points[:, 1]) - np.mean(right_eye_points[:, 1])
            
            features['left_eyebrow_eye_dist'] = left_eyebrow_eye_dist
            features['right_eyebrow_eye_dist'] = right_eyebrow_eye_dist
            features['avg_eyebrow_eye_dist'] = (left_eyebrow_eye_dist + right_eyebrow_eye_dist) / 2
            
            # Face dimensions for normalization
            face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
            face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
            features['face_width'] = face_width
            features['face_height'] = face_height
            
        except Exception as e:
            # If feature extraction fails, set default values
            features = {
                'mouth_ratio': 0.1,
                'mouth_curve': 0.0,
                'mouth_width': 50,
                'left_eye_ratio': 0.2,
                'right_eye_ratio': 0.2,
                'avg_eye_ratio': 0.2,
                'left_eyebrow_eye_dist': 10,
                'right_eyebrow_eye_dist': 10,
                'avg_eyebrow_eye_dist': 10,
                'face_width': 100,
                'face_height': 120
            }
        
        return features
    
    def _classify_emotion(self, features: Dict) -> Dict[str, float]:
        """Classify emotion based on extracted features"""
        scores = {}
        
        # Normalize features
        mouth_ratio = features['mouth_ratio']
        mouth_curve = features['mouth_curve']
        avg_eye_ratio = features['avg_eye_ratio']
        avg_eyebrow_eye_dist = features['avg_eyebrow_eye_dist']
        
        # Happy: Wide smile, upturned mouth corners, normal eyes
        happy_score = 0.0
        if mouth_curve < -2:  # Upturned mouth
            happy_score += 0.4
        if mouth_ratio > 0.15:  # Wide mouth
            happy_score += 0.3
        if 0.15 < avg_eye_ratio < 0.3:  # Normal eye opening
            happy_score += 0.2
        happy_score += max(0, min(0.1, -mouth_curve * 0.05))  # Bonus for strong upward curve
        
        # Sad: Downturned mouth, droopy eyes, lowered eyebrows
        sad_score = 0.0
        if mouth_curve > 1:  # Downturned mouth
            sad_score += 0.4
        if avg_eye_ratio < 0.15:  # Slightly closed eyes
            sad_score += 0.2
        if avg_eyebrow_eye_dist < 8:  # Lowered eyebrows
            sad_score += 0.2
        sad_score += max(0, min(0.2, mouth_curve * 0.1))  # Bonus for strong downward curve
        
        # Angry: Frowning mouth, furrowed brow, narrowed eyes
        angry_score = 0.0
        if mouth_curve > 0.5:  # Slightly downturned or straight mouth
            angry_score += 0.2
        if avg_eyebrow_eye_dist < 6:  # Very lowered/furrowed brows
            angry_score += 0.4
        if avg_eye_ratio < 0.18:  # Narrowed eyes
            angry_score += 0.3
        if mouth_ratio < 0.1:  # Compressed mouth
            angry_score += 0.1
        
        # Surprised: Wide open mouth, raised eyebrows, wide eyes
        surprised_score = 0.0
        if mouth_ratio > 0.25:  # Very wide mouth opening
            surprised_score += 0.4
        if avg_eyebrow_eye_dist > 15:  # Raised eyebrows
            surprised_score += 0.3
        if avg_eye_ratio > 0.25:  # Wide open eyes
            surprised_score += 0.3
        
        # Fearful: Similar to surprised but less mouth opening
        fearful_score = 0.0
        if 0.15 < mouth_ratio < 0.25:  # Moderately open mouth
            fearful_score += 0.2
        if avg_eyebrow_eye_dist > 12:  # Raised eyebrows
            fearful_score += 0.3
        if avg_eye_ratio > 0.22:  # Wide eyes
            fearful_score += 0.3
        if mouth_curve > -1 and mouth_curve < 1:  # Neutral mouth curve
            fearful_score += 0.2
        
        # Disgusted: Raised upper lip, wrinkled nose area, slightly narrowed eyes
        disgusted_score = 0.0
        if mouth_curve > 0:  # Slightly downturned mouth
            disgusted_score += 0.2
        if 0.12 < mouth_ratio < 0.18:  # Moderately open mouth
            disgusted_score += 0.2
        if avg_eye_ratio < 0.2:  # Slightly narrowed eyes
            disgusted_score += 0.2
        if 8 < avg_eyebrow_eye_dist < 12:  # Slightly raised eyebrows
            disgusted_score += 0.2
        disgusted_score += 0.2  # Base score for this less common emotion
        
        # Neutral: Average values for all features
        neutral_score = 0.3  # Base neutral score
        if -1 < mouth_curve < 1:  # Neutral mouth
            neutral_score += 0.2
        if 0.15 < avg_eye_ratio < 0.25:  # Normal eye opening
            neutral_score += 0.2
        if 8 < avg_eyebrow_eye_dist < 12:  # Normal eyebrow position
            neutral_score += 0.2
        if 0.1 < mouth_ratio < 0.2:  # Normal mouth opening
            neutral_score += 0.1
        
        # Assign scores
        scores['happy'] = min(1.0, happy_score)
        scores['sad'] = min(1.0, sad_score)
        scores['angry'] = min(1.0, angry_score)
        scores['surprised'] = min(1.0, surprised_score)
        scores['fearful'] = min(1.0, fearful_score)
        scores['disgusted'] = min(1.0, disgusted_score)
        scores['neutral'] = min(1.0, neutral_score)
        
        # Normalize scores so they sum to 1
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {emotion: score / total_score for emotion, score in scores.items()}
        else:
            # If all scores are 0, default to neutral
            scores = {emotion: 0.0 for emotion in self.emotions}
            scores['neutral'] = 1.0
        
        return scores
    
    def draw_emotion_overlay(self, image: np.ndarray, emotion_info: Dict, color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """Draw emotion detection overlay on image"""
        result_image = image.copy()
        
        # Get face bounding box
        landmarks = emotion_info['landmarks']
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        
        # Draw bounding box
        cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw emotion label
        emotion = emotion_info['emotion']
        confidence = emotion_info['confidence']
        label = f"{emotion.capitalize()}: {confidence:.2f}"
        
        # Calculate text position
        text_pos = (x_min, y_min - 10)
        
        # Draw text background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(result_image, 
                     (text_pos[0], text_pos[1] - text_size[1] - 5),
                     (text_pos[0] + text_size[0], text_pos[1] + 5),
                     color, -1)
        
        # Draw text
        cv2.putText(result_image, label, text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image
