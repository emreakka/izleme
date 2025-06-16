import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json
import os
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict, Counter
import time

class AnalyticsManager:
    """
    Analytics and data management system using pandas and SQLite:
    - Session tracking and metrics
    - Emotion and gaze analytics
    - Performance monitoring
    - Data visualization support
    - Export capabilities
    """
    
    def __init__(self, db_path: str = "analytics.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.connection = None
        
        # Initialize database
        self._init_database()
        
        self.logger.info("Analytics manager initialized")
    
    def _init_database(self):
        """Initialize SQLite database for analytics"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            # Create tables
            self._create_tables()
            
            self.logger.info("Analytics database initialized")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables for analytics"""
        try:
            cursor = self.connection.cursor()
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    session_type TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    faces_detected INTEGER DEFAULT 0,
                    frames_processed INTEGER DEFAULT 0,
                    processing_time REAL DEFAULT 0,
                    confidence_threshold REAL DEFAULT 0.5,
                    detection_method TEXT,
                    filename TEXT,
                    image_width INTEGER,
                    image_height INTEGER,
                    created_at REAL DEFAULT (datetime('now'))
                )
            ''')
            
            # Face detections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_detections (
                    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    face_id TEXT,
                    person_name TEXT,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    detection_method TEXT,
                    gaze_direction TEXT,
                    gaze_confidence REAL,
                    emotion TEXT,
                    emotion_confidence REAL,
                    sharpness REAL,
                    brightness REAL,
                    contrast REAL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # System events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_description TEXT,
                    event_data TEXT,
                    severity TEXT DEFAULT 'info',
                    timestamp REAL NOT NULL
                )
            ''')
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Table creation failed: {e}")
            raise
    
    def create_session(self, session_type: str, filename: str = None, 
                      detection_method: str = "ensemble", confidence_threshold: float = 0.5,
                      image_dimensions: Tuple[int, int] = None) -> str:
        """Create a new analytics session"""
        try:
            session_id = f"session_{int(time.time() * 1000)}"
            start_time = time.time()
            
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO sessions (
                    session_id, session_type, start_time, confidence_threshold,
                    detection_method, filename, image_width, image_height
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, session_type, start_time, confidence_threshold,
                detection_method, filename,
                image_dimensions[0] if image_dimensions else None,
                image_dimensions[1] if image_dimensions else None
            ))
            
            self.connection.commit()
            
            self.logger.info(f"Created session: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Session creation failed: {e}")
            return f"session_error_{int(time.time())}"
    
    def end_session(self, session_id: str, faces_detected: int = 0, 
                   frames_processed: int = 1, processing_time: float = 0):
        """End an analytics session"""
        try:
            end_time = time.time()
            
            cursor = self.connection.cursor()
            cursor.execute('''
                UPDATE sessions 
                SET end_time = ?, faces_detected = ?, frames_processed = ?, processing_time = ?
                WHERE session_id = ?
            ''', (end_time, faces_detected, frames_processed, processing_time, session_id))
            
            self.connection.commit()
            
            self.logger.info(f"Ended session: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Session end failed: {e}")
    
    def save_detection_session(self, session_type: str, filename: str, 
                             detections: List[Dict], processing_time: float):
        """Save a complete detection session with results"""
        try:
            # Create session
            session_id = self.create_session(
                session_type=session_type,
                filename=filename,
                detection_method="ensemble"
            )
            
            # Save detections
            if detections:
                for detection in detections:
                    self.save_face_detection(session_id, detection)
            
            # End session
            self.end_session(
                session_id=session_id,
                faces_detected=len(detections),
                frames_processed=1,
                processing_time=processing_time
            )
            
            # Save performance metrics
            self.save_performance_metric(session_id, "processing_time", processing_time, "seconds")
            self.save_performance_metric(session_id, "faces_detected", len(detections), "count")
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Detection session save failed: {e}")
            return None
    
    def save_face_detection(self, session_id: str, detection: Dict):
        """Save individual face detection result"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO face_detections (
                    session_id, face_id, person_name, x, y, width, height,
                    confidence, detection_method, gaze_direction, gaze_confidence,
                    emotion, emotion_confidence, sharpness, brightness, contrast, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                detection.get('face_id'),
                detection.get('person_name'),
                detection.get('x', 0),
                detection.get('y', 0),
                detection.get('w', 0),
                detection.get('h', 0),
                detection.get('confidence', 0.0),
                detection.get('method', 'unknown'),
                detection.get('gaze_direction'),
                detection.get('gaze_confidence', 0.0) if 'gaze_confidence' in detection else None,
                detection.get('emotion'),
                detection.get('emotion_confidence', 0.0),
                detection.get('sharpness', 0.0),
                detection.get('brightness', 0.0),
                detection.get('contrast', 0.0),
                time.time()
            ))
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Face detection save failed: {e}")
    
    def save_performance_metric(self, session_id: str, metric_name: str, 
                              metric_value: float, metric_unit: str = ""):
        """Save performance metric"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (
                    session_id, metric_name, metric_value, metric_unit, timestamp
                ) VALUES (?, ?, ?, ?, ?)
            ''', (session_id, metric_name, metric_value, metric_unit, time.time()))
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Performance metric save failed: {e}")
    
    def log_system_event(self, event_type: str, description: str, 
                        event_data: Dict = None, severity: str = "info"):
        """Log system event"""
        try:
            cursor = self.connection.cursor()
            
            event_data_json = json.dumps(event_data) if event_data else None
            
            cursor.execute('''
                INSERT INTO system_events (
                    event_type, event_description, event_data, severity, timestamp
                ) VALUES (?, ?, ?, ?, ?)
            ''', (event_type, description, event_data_json, severity, time.time()))
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"System event logging failed: {e}")
    
    def get_comprehensive_analytics(self, days_back: int = 7) -> Dict:
        """Get comprehensive analytics for the specified time period"""
        try:
            end_time = time.time()
            start_time = end_time - (days_back * 24 * 60 * 60)
            
            analytics = {}
            
            # Session statistics
            session_stats = self._get_session_statistics(start_time, end_time)
            analytics.update(session_stats)
            
            # Detection statistics
            detection_stats = self._get_detection_statistics(start_time, end_time)
            analytics.update(detection_stats)
            
            # Emotion analytics
            emotion_analytics = self._get_emotion_analytics(start_time, end_time)
            analytics.update(emotion_analytics)
            
            # Gaze analytics
            gaze_analytics = self._get_gaze_analytics(start_time, end_time)
            analytics.update(gaze_analytics)
            
            # Performance analytics
            performance_analytics = self._get_performance_analytics(start_time, end_time)
            analytics.update(performance_analytics)
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Analytics generation failed: {e}")
            return {}
    
    def _get_session_statistics(self, start_time: float, end_time: float) -> Dict:
        """Get session-related statistics"""
        try:
            cursor = self.connection.cursor()
            
            # Total sessions
            cursor.execute('''
                SELECT COUNT(*) as total_sessions FROM sessions 
                WHERE start_time BETWEEN ? AND ?
            ''', (start_time, end_time))
            total_sessions = cursor.fetchone()['total_sessions']
            
            # Sessions by type
            cursor.execute('''
                SELECT session_type, COUNT(*) as count FROM sessions 
                WHERE start_time BETWEEN ? AND ?
                GROUP BY session_type
            ''', (start_time, end_time))
            sessions_by_type = dict(cursor.fetchall())
            
            # Average processing time
            cursor.execute('''
                SELECT AVG(processing_time) as avg_processing_time FROM sessions 
                WHERE start_time BETWEEN ? AND ? AND processing_time > 0
            ''', (start_time, end_time))
            result = cursor.fetchone()
            avg_processing_time = result['avg_processing_time'] if result['avg_processing_time'] else 0
            
            return {
                'total_sessions': total_sessions,
                'sessions_by_type': sessions_by_type,
                'avg_processing_time': round(avg_processing_time, 3)
            }
            
        except Exception as e:
            self.logger.error(f"Session statistics failed: {e}")
            return {}
    
    def _get_detection_statistics(self, start_time: float, end_time: float) -> Dict:
        """Get face detection statistics"""
        try:
            cursor = self.connection.cursor()
            
            # Total faces detected
            cursor.execute('''
                SELECT COUNT(*) as total_faces FROM face_detections fd
                JOIN sessions s ON fd.session_id = s.session_id
                WHERE s.start_time BETWEEN ? AND ?
            ''', (start_time, end_time))
            total_faces = cursor.fetchone()['total_faces']
            
            # Average confidence
            cursor.execute('''
                SELECT AVG(confidence) as avg_confidence FROM face_detections fd
                JOIN sessions s ON fd.session_id = s.session_id
                WHERE s.start_time BETWEEN ? AND ?
            ''', (start_time, end_time))
            result = cursor.fetchone()
            avg_confidence = result['avg_confidence'] if result['avg_confidence'] else 0
            
            # Detection methods
            cursor.execute('''
                SELECT detection_method, COUNT(*) as count FROM face_detections fd
                JOIN sessions s ON fd.session_id = s.session_id
                WHERE s.start_time BETWEEN ? AND ?
                GROUP BY detection_method
            ''', (start_time, end_time))
            detection_methods = dict(cursor.fetchall())
            
            # Face quality metrics
            cursor.execute('''
                SELECT 
                    AVG(sharpness) as avg_sharpness,
                    AVG(brightness) as avg_brightness,
                    AVG(contrast) as avg_contrast
                FROM face_detections fd
                JOIN sessions s ON fd.session_id = s.session_id
                WHERE s.start_time BETWEEN ? AND ?
            ''', (start_time, end_time))
            quality_result = cursor.fetchone()
            
            return {
                'total_faces': total_faces,
                'avg_confidence': round(avg_confidence, 3),
                'detection_methods': detection_methods,
                'avg_sharpness': round(quality_result['avg_sharpness'] or 0, 2),
                'avg_brightness': round(quality_result['avg_brightness'] or 0, 2),
                'avg_contrast': round(quality_result['avg_contrast'] or 0, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Detection statistics failed: {e}")
            return {}
    
    def _get_emotion_analytics(self, start_time: float, end_time: float) -> Dict:
        """Get emotion recognition analytics"""
        try:
            cursor = self.connection.cursor()
            
            # Emotion distribution
            cursor.execute('''
                SELECT emotion, COUNT(*) as count FROM face_detections fd
                JOIN sessions s ON fd.session_id = s.session_id
                WHERE s.start_time BETWEEN ? AND ? AND emotion IS NOT NULL
                GROUP BY emotion
                ORDER BY count DESC
            ''', (start_time, end_time))
            emotion_distribution = dict(cursor.fetchall())
            
            # Average emotion confidence
            cursor.execute('''
                SELECT AVG(emotion_confidence) as avg_emotion_confidence FROM face_detections fd
                JOIN sessions s ON fd.session_id = s.session_id
                WHERE s.start_time BETWEEN ? AND ? AND emotion_confidence > 0
            ''', (start_time, end_time))
            result = cursor.fetchone()
            avg_emotion_confidence = result['avg_emotion_confidence'] if result['avg_emotion_confidence'] else 0
            
            # Emotion trends over time
            cursor.execute('''
                SELECT 
                    DATE(s.start_time, 'unixepoch') as date,
                    emotion,
                    COUNT(*) as count
                FROM face_detections fd
                JOIN sessions s ON fd.session_id = s.session_id
                WHERE s.start_time BETWEEN ? AND ? AND emotion IS NOT NULL
                GROUP BY date, emotion
                ORDER BY date
            ''', (start_time, end_time))
            emotion_trends = cursor.fetchall()
            
            return {
                'emotion_distribution': emotion_distribution,
                'avg_emotion_confidence': round(avg_emotion_confidence, 3),
                'emotion_trends': [dict(row) for row in emotion_trends]
            }
            
        except Exception as e:
            self.logger.error(f"Emotion analytics failed: {e}")
            return {}
    
    def _get_gaze_analytics(self, start_time: float, end_time: float) -> Dict:
        """Get gaze tracking analytics"""
        try:
            cursor = self.connection.cursor()
            
            # Gaze patterns
            cursor.execute('''
                SELECT gaze_direction, COUNT(*) as count FROM face_detections fd
                JOIN sessions s ON fd.session_id = s.session_id
                WHERE s.start_time BETWEEN ? AND ? AND gaze_direction IS NOT NULL
                GROUP BY gaze_direction
                ORDER BY count DESC
            ''', (start_time, end_time))
            gaze_patterns = dict(cursor.fetchall())
            
            # Average gaze confidence
            cursor.execute('''
                SELECT AVG(gaze_confidence) as avg_gaze_confidence FROM face_detections fd
                JOIN sessions s ON fd.session_id = s.session_id
                WHERE s.start_time BETWEEN ? AND ? AND gaze_confidence > 0
            ''', (start_time, end_time))
            result = cursor.fetchone()
            avg_gaze_confidence = result['avg_gaze_confidence'] if result['avg_gaze_confidence'] else 0
            
            return {
                'gaze_patterns': gaze_patterns,
                'avg_gaze_confidence': round(avg_gaze_confidence, 3)
            }
            
        except Exception as e:
            self.logger.error(f"Gaze analytics failed: {e}")
            return {}
    
    def _get_performance_analytics(self, start_time: float, end_time: float) -> Dict:
        """Get system performance analytics"""
        try:
            cursor = self.connection.cursor()
            
            # Performance metrics
            cursor.execute('''
                SELECT 
                    metric_name,
                    AVG(metric_value) as avg_value,
                    MIN(metric_value) as min_value,
                    MAX(metric_value) as max_value
                FROM performance_metrics pm
                JOIN sessions s ON pm.session_id = s.session_id
                WHERE s.start_time BETWEEN ? AND ?
                GROUP BY metric_name
            ''', (start_time, end_time))
            performance_metrics = {
                row['metric_name']: {
                    'avg': round(row['avg_value'], 3),
                    'min': round(row['min_value'], 3),
                    'max': round(row['max_value'], 3)
                }
                for row in cursor.fetchall()
            }
            
            # Success rate (sessions with faces detected / total sessions)
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN faces_detected > 0 THEN 1 ELSE 0 END) as successful
                FROM sessions
                WHERE start_time BETWEEN ? AND ?
            ''', (start_time, end_time))
            result = cursor.fetchone()
            total_sessions = result['total']
            successful_sessions = result['successful']
            success_rate = (successful_sessions / total_sessions * 100) if total_sessions > 0 else 0
            
            return {
                'performance_metrics': performance_metrics,
                'success_rate': round(success_rate, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Performance analytics failed: {e}")
            return {}
    
    def get_session_count(self) -> int:
        """Get total number of sessions"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM sessions')
            return cursor.fetchone()['count']
            
        except Exception as e:
            self.logger.error(f"Session count failed: {e}")
            return 0
    
    def export_analytics_data(self, export_path: str = "analytics_export.csv", 
                            days_back: int = 30) -> bool:
        """Export analytics data to CSV"""
        try:
            end_time = time.time()
            start_time = end_time - (days_back * 24 * 60 * 60)
            
            # Export face detections with session info
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT 
                    s.session_id,
                    s.session_type,
                    datetime(s.start_time, 'unixepoch') as session_start,
                    s.filename,
                    fd.person_name,
                    fd.x, fd.y, fd.width, fd.height,
                    fd.confidence,
                    fd.detection_method,
                    fd.gaze_direction,
                    fd.gaze_confidence,
                    fd.emotion,
                    fd.emotion_confidence,
                    fd.sharpness,
                    fd.brightness,
                    fd.contrast,
                    datetime(fd.timestamp, 'unixepoch') as detection_time
                FROM face_detections fd
                JOIN sessions s ON fd.session_id = s.session_id
                WHERE s.start_time BETWEEN ? AND ?
                ORDER BY s.start_time DESC
            ''', (start_time, end_time))
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(cursor.fetchall())
            
            if not df.empty:
                df.to_csv(export_path, index=False)
                self.logger.info(f"Analytics data exported to {export_path}")
                return True
            else:
                self.logger.warning("No data to export")
                return False
                
        except Exception as e:
            self.logger.error(f"Analytics export failed: {e}")
            return False
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """Get recent session information"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT 
                    session_id,
                    session_type,
                    datetime(start_time, 'unixepoch') as start_time,
                    datetime(end_time, 'unixepoch') as end_time,
                    faces_detected,
                    frames_processed,
                    processing_time,
                    filename
                FROM sessions
                ORDER BY start_time DESC
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            self.logger.error(f"Recent sessions query failed: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old analytics data"""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
            
            cursor = self.connection.cursor()
            
            # Delete old sessions and related data
            cursor.execute('DELETE FROM face_detections WHERE session_id IN (SELECT session_id FROM sessions WHERE start_time < ?)', (cutoff_time,))
            cursor.execute('DELETE FROM performance_metrics WHERE session_id IN (SELECT session_id FROM sessions WHERE start_time < ?)', (cutoff_time,))
            cursor.execute('DELETE FROM sessions WHERE start_time < ?', (cutoff_time,))
            cursor.execute('DELETE FROM system_events WHERE timestamp < ?', (cutoff_time,))
            
            self.connection.commit()
            
            self.logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
    
    def __del__(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()