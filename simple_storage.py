import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import uuid

class SimpleStorage:
    """Simple file-based storage as fallback when database is unavailable"""
    
    def __init__(self):
        self.storage_dir = "detection_data"
        os.makedirs(self.storage_dir, exist_ok=True)
        self.sessions_file = os.path.join(self.storage_dir, "sessions.json")
        self.detections_file = os.path.join(self.storage_dir, "detections.json")
        
    def _load_json(self, filename: str) -> List[Dict]:
        """Load JSON data from file"""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_json(self, filename: str, data: List[Dict]):
        """Save JSON data to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, default=str, indent=2)
        except Exception as e:
            print(f"Error saving to {filename}: {e}")
    
    def create_session(self, session_type: str, confidence_threshold: float, show_landmarks: bool) -> str:
        """Create a new detection session"""
        session_id = str(uuid.uuid4())
        session = {
            'id': session_id,
            'session_start': datetime.utcnow().isoformat(),
            'session_end': None,
            'session_type': session_type,
            'total_faces_detected': 0,
            'total_frames_processed': 0,
            'confidence_threshold': confidence_threshold,
            'show_landmarks': show_landmarks
        }
        
        sessions = self._load_json(self.sessions_file)
        sessions.append(session)
        self._save_json(self.sessions_file, sessions)
        
        return session_id
    
    def end_session(self, session_id: str, total_faces: int, total_frames: int):
        """End a detection session"""
        sessions = self._load_json(self.sessions_file)
        for session in sessions:
            if session['id'] == session_id:
                session['session_end'] = datetime.utcnow().isoformat()
                session['total_faces_detected'] = total_faces
                session['total_frames_processed'] = total_frames
                break
        self._save_json(self.sessions_file, sessions)
    
    def save_detection_results(self, session_id: str, results: List[Dict], frame_number: Optional[int] = None):
        """Save detection results"""
        detections = self._load_json(self.detections_file)
        
        for result in results:
            detection = {
                'id': str(uuid.uuid4()),
                'session_id': session_id,
                'timestamp': datetime.utcnow().isoformat(),
                'face_id': result.get('face_id', 1),
                'gaze_direction': result.get('gaze_direction', 'Unknown'),
                'gaze_pitch': result.get('gaze_angles', (0.0, 0.0))[0],
                'gaze_yaw': result.get('gaze_angles', (0.0, 0.0))[1],
                'gaze_confidence': result.get('gaze_confidence', 0.0),
                'head_pitch': result.get('head_pose_angles', (0.0, 0.0, 0.0))[0],
                'head_yaw': result.get('head_pose_angles', (0.0, 0.0, 0.0))[1],
                'head_roll': result.get('head_pose_angles', (0.0, 0.0, 0.0))[2],
                'emotion': result.get('emotion', 'Unknown'),
                'emotion_confidence': result.get('emotion_confidence', 0.0),
                'emotion_scores': result.get('emotion_scores', {}),
                'frame_number': frame_number
            }
            detections.append(detection)
        
        self._save_json(self.detections_file, detections)
    
    def get_session_history(self, limit: int = 10) -> List[Dict]:
        """Get recent detection sessions"""
        sessions = self._load_json(self.sessions_file)
        # Sort by start time, newest first
        sessions.sort(key=lambda x: x['session_start'], reverse=True)
        
        result = []
        for session in sessions[:limit]:
            start_time = datetime.fromisoformat(session['session_start'].replace('Z', '+00:00'))
            end_time = None
            if session['session_end']:
                end_time = datetime.fromisoformat(session['session_end'].replace('Z', '+00:00'))
            
            result.append({
                'id': session['id'],
                'start_time': start_time,
                'end_time': end_time,
                'type': session['session_type'],
                'faces_detected': session['total_faces_detected'],
                'frames_processed': session['total_frames_processed'],
                'confidence_threshold': session['confidence_threshold']
            })
        
        return result
    
    def get_emotion_analytics(self, days: int = 7) -> Dict:
        """Get emotion analytics"""
        detections = self._load_json(self.detections_file)
        
        # Filter by date range
        from datetime import timedelta
        since_date = datetime.utcnow() - timedelta(days=days)
        
        emotion_data = {}
        for detection in detections:
            detection_time = datetime.fromisoformat(detection['timestamp'].replace('Z', '+00:00'))
            if detection_time >= since_date:
                emotion = detection['emotion']
                if emotion != 'Unknown':
                    if emotion not in emotion_data:
                        emotion_data[emotion] = {
                            'total_detections': 0,
                            'average_confidence': 0.0,
                            'confidence_sum': 0.0
                        }
                    
                    emotion_data[emotion]['total_detections'] += 1
                    emotion_data[emotion]['confidence_sum'] += detection['emotion_confidence']
                    emotion_data[emotion]['average_confidence'] = (
                        emotion_data[emotion]['confidence_sum'] / emotion_data[emotion]['total_detections']
                    )
        
        # Clean up the data structure
        for emotion in emotion_data:
            del emotion_data[emotion]['confidence_sum']
        
        return emotion_data
    
    def get_gaze_analytics(self, days: int = 7) -> Dict:
        """Get gaze analytics"""
        detections = self._load_json(self.detections_file)
        
        # Filter by date range
        from datetime import timedelta
        since_date = datetime.utcnow() - timedelta(days=days)
        
        gaze_data = {}
        for detection in detections:
            detection_time = datetime.fromisoformat(detection['timestamp'].replace('Z', '+00:00'))
            if detection_time >= since_date:
                gaze_direction = detection['gaze_direction']
                if gaze_direction != 'Unknown':
                    if gaze_direction not in gaze_data:
                        gaze_data[gaze_direction] = {
                            'total_detections': 0,
                            'average_confidence': 0.0,
                            'confidence_sum': 0.0
                        }
                    
                    gaze_data[gaze_direction]['total_detections'] += 1
                    gaze_data[gaze_direction]['confidence_sum'] += detection['gaze_confidence']
                    gaze_data[gaze_direction]['average_confidence'] = (
                        gaze_data[gaze_direction]['confidence_sum'] / gaze_data[gaze_direction]['total_detections']
                    )
        
        # Clean up the data structure
        for direction in gaze_data:
            del gaze_data[direction]['confidence_sum']
        
        return gaze_data
    
    def get_detection_details(self, session_id: str) -> List[Dict]:
        """Get detailed detection results for a session"""
        detections = self._load_json(self.detections_file)
        
        session_detections = []
        for detection in detections:
            if detection['session_id'] == session_id:
                timestamp = datetime.fromisoformat(detection['timestamp'].replace('Z', '+00:00'))
                session_detections.append({
                    'timestamp': timestamp,
                    'face_id': detection['face_id'],
                    'gaze_direction': detection['gaze_direction'],
                    'gaze_angles': (detection['gaze_pitch'], detection['gaze_yaw']),
                    'gaze_confidence': detection['gaze_confidence'],
                    'head_pose': (detection['head_pitch'], detection['head_yaw'], detection['head_roll']),
                    'emotion': detection['emotion'],
                    'emotion_confidence': detection['emotion_confidence'],
                    'emotion_scores': detection['emotion_scores'],
                    'frame_number': detection['frame_number'],
                    'processing_time_ms': None
                })
        
        # Sort by timestamp
        session_detections.sort(key=lambda x: x['timestamp'])
        return session_detections

# Create global instance
simple_storage = SimpleStorage()