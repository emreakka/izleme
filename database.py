import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import json

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DetectionSession(Base):
    """Table to store detection sessions"""
    __tablename__ = "detection_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_start = Column(DateTime, default=datetime.utcnow)
    session_end = Column(DateTime, nullable=True)
    session_type = Column(String(50))  # 'webcam', 'image', 'video'
    total_faces_detected = Column(Integer, default=0)
    total_frames_processed = Column(Integer, default=0)
    confidence_threshold = Column(Float, default=0.5)
    show_landmarks = Column(Boolean, default=True)

class FaceDetection(Base):
    """Table to store individual face detection results"""
    __tablename__ = "face_detections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    face_id = Column(Integer)  # Face ID within the frame
    
    # Gaze detection data
    gaze_direction = Column(String(50))
    gaze_pitch = Column(Float)
    gaze_yaw = Column(Float)
    gaze_confidence = Column(Float)
    
    # Head pose data
    head_pitch = Column(Float)
    head_yaw = Column(Float)
    head_roll = Column(Float)
    
    # Emotion detection data
    emotion = Column(String(50))
    emotion_confidence = Column(Float)
    emotion_scores = Column(JSON)  # Store all emotion scores as JSON
    
    # Additional metadata
    frame_number = Column(Integer, nullable=True)  # For video processing
    processing_time_ms = Column(Float, nullable=True)

class EmotionStatistics(Base):
    """Table to store emotion statistics over time"""
    __tablename__ = "emotion_statistics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(DateTime, default=datetime.utcnow)
    emotion = Column(String(50))
    total_detections = Column(Integer, default=1)
    average_confidence = Column(Float)
    session_type = Column(String(50))

class GazeStatistics(Base):
    """Table to store gaze direction statistics"""
    __tablename__ = "gaze_statistics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(DateTime, default=datetime.utcnow)
    gaze_direction = Column(String(50))
    total_detections = Column(Integer, default=1)
    average_confidence = Column(Float)
    session_type = Column(String(50))

class DatabaseManager:
    """Database manager for handling detection data"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        Base.metadata.create_all(bind=engine)
    
    def get_db(self) -> Session:
        """Get database session"""
        db = self.SessionLocal()
        try:
            return db
        except Exception:
            db.close()
            raise
    
    def create_session(self, session_type: str, confidence_threshold: float, show_landmarks: bool) -> str:
        """Create a new detection session"""
        db = self.get_db()
        try:
            session = DetectionSession(
                session_type=session_type,
                confidence_threshold=confidence_threshold,
                show_landmarks=show_landmarks
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return str(session.id)
        finally:
            db.close()
    
    def end_session(self, session_id: str, total_faces: int, total_frames: int):
        """End a detection session"""
        db = self.get_db()
        try:
            session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
            if session:
                session.session_end = datetime.utcnow()
                session.total_faces_detected = total_faces
                session.total_frames_processed = total_frames
                db.commit()
        finally:
            db.close()
    
    def save_detection_results(self, session_id: str, results: List[Dict], frame_number: Optional[int] = None):
        """Save detection results to database"""
        db = self.get_db()
        try:
            detections = []
            start_time = datetime.utcnow()
            
            for result in results:
                detection = FaceDetection(
                    session_id=session_id,
                    face_id=result.get('face_id', 1),
                    gaze_direction=result.get('gaze_direction', 'Unknown'),
                    gaze_pitch=result.get('gaze_angles', (0.0, 0.0))[0],
                    gaze_yaw=result.get('gaze_angles', (0.0, 0.0))[1],
                    gaze_confidence=result.get('gaze_confidence', 0.0),
                    head_pitch=result.get('head_pose_angles', (0.0, 0.0, 0.0))[0],
                    head_yaw=result.get('head_pose_angles', (0.0, 0.0, 0.0))[1],
                    head_roll=result.get('head_pose_angles', (0.0, 0.0, 0.0))[2],
                    emotion=result.get('emotion', 'Unknown'),
                    emotion_confidence=result.get('emotion_confidence', 0.0),
                    emotion_scores=result.get('emotion_scores', {}),
                    frame_number=frame_number
                )
                detections.append(detection)
            
            db.add_all(detections)
            db.commit()
            
            # Update statistics
            self._update_statistics(db, results)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            for detection in detections:
                detection.processing_time_ms = processing_time
            db.commit()
            
        finally:
            db.close()
    
    def _update_statistics(self, db: Session, results: List[Dict]):
        """Update emotion and gaze statistics"""
        today = datetime.utcnow().date()
        
        for result in results:
            # Update emotion statistics
            emotion = result.get('emotion')
            emotion_confidence = result.get('emotion_confidence', 0.0)
            
            if emotion and emotion != 'Unknown':
                emotion_stat = db.query(EmotionStatistics).filter(
                    EmotionStatistics.emotion == emotion,
                    EmotionStatistics.date >= today
                ).first()
                
                if emotion_stat:
                    # Update existing record
                    new_total = emotion_stat.total_detections + 1
                    emotion_stat.average_confidence = (
                        (emotion_stat.average_confidence * emotion_stat.total_detections + emotion_confidence) / new_total
                    )
                    emotion_stat.total_detections = new_total
                else:
                    # Create new record
                    emotion_stat = EmotionStatistics(
                        emotion=emotion,
                        total_detections=1,
                        average_confidence=emotion_confidence
                    )
                    db.add(emotion_stat)
            
            # Update gaze statistics
            gaze_direction = result.get('gaze_direction')
            gaze_confidence = result.get('gaze_confidence', 0.0)
            
            if gaze_direction and gaze_direction != 'Unknown':
                gaze_stat = db.query(GazeStatistics).filter(
                    GazeStatistics.gaze_direction == gaze_direction,
                    GazeStatistics.date >= today
                ).first()
                
                if gaze_stat:
                    # Update existing record
                    new_total = gaze_stat.total_detections + 1
                    gaze_stat.average_confidence = (
                        (gaze_stat.average_confidence * gaze_stat.total_detections + gaze_confidence) / new_total
                    )
                    gaze_stat.total_detections = new_total
                else:
                    # Create new record
                    gaze_stat = GazeStatistics(
                        gaze_direction=gaze_direction,
                        total_detections=1,
                        average_confidence=gaze_confidence
                    )
                    db.add(gaze_stat)
    
    def get_session_history(self, limit: int = 10) -> List[Dict]:
        """Get recent detection sessions"""
        db = self.get_db()
        try:
            sessions = db.query(DetectionSession).order_by(
                DetectionSession.session_start.desc()
            ).limit(limit).all()
            
            return [
                {
                    'id': str(session.id),
                    'start_time': session.session_start,
                    'end_time': session.session_end,
                    'type': session.session_type,
                    'faces_detected': session.total_faces_detected,
                    'frames_processed': session.total_frames_processed,
                    'confidence_threshold': session.confidence_threshold
                }
                for session in sessions
            ]
        finally:
            db.close()
    
    def get_emotion_analytics(self, days: int = 7) -> Dict:
        """Get emotion analytics for the last N days"""
        db = self.get_db()
        try:
            from datetime import timedelta
            since_date = datetime.utcnow() - timedelta(days=days)
            
            stats = db.query(EmotionStatistics).filter(
                EmotionStatistics.date >= since_date
            ).all()
            
            emotion_data = {}
            for stat in stats:
                if stat.emotion not in emotion_data:
                    emotion_data[stat.emotion] = {
                        'total_detections': 0,
                        'average_confidence': 0.0
                    }
                emotion_data[stat.emotion]['total_detections'] += stat.total_detections
                emotion_data[stat.emotion]['average_confidence'] = stat.average_confidence
            
            return emotion_data
        finally:
            db.close()
    
    def get_gaze_analytics(self, days: int = 7) -> Dict:
        """Get gaze analytics for the last N days"""
        db = self.get_db()
        try:
            from datetime import timedelta
            since_date = datetime.utcnow() - timedelta(days=days)
            
            stats = db.query(GazeStatistics).filter(
                GazeStatistics.date >= since_date
            ).all()
            
            gaze_data = {}
            for stat in stats:
                if stat.gaze_direction not in gaze_data:
                    gaze_data[stat.gaze_direction] = {
                        'total_detections': 0,
                        'average_confidence': 0.0
                    }
                gaze_data[stat.gaze_direction]['total_detections'] += stat.total_detections
                gaze_data[stat.gaze_direction]['average_confidence'] = stat.average_confidence
            
            return gaze_data
        finally:
            db.close()
    
    def get_detection_details(self, session_id: str) -> List[Dict]:
        """Get detailed detection results for a session"""
        db = self.get_db()
        try:
            detections = db.query(FaceDetection).filter(
                FaceDetection.session_id == session_id
            ).order_by(FaceDetection.timestamp).all()
            
            return [
                {
                    'timestamp': detection.timestamp,
                    'face_id': detection.face_id,
                    'gaze_direction': detection.gaze_direction,
                    'gaze_angles': (detection.gaze_pitch, detection.gaze_yaw),
                    'gaze_confidence': detection.gaze_confidence,
                    'head_pose': (detection.head_pitch, detection.head_yaw, detection.head_roll),
                    'emotion': detection.emotion,
                    'emotion_confidence': detection.emotion_confidence,
                    'emotion_scores': detection.emotion_scores,
                    'frame_number': detection.frame_number,
                    'processing_time_ms': detection.processing_time_ms
                }
                for detection in detections
            ]
        finally:
            db.close()

# Global database manager instance
db_manager = DatabaseManager()