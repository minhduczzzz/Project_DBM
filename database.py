import sqlite3
from datetime import datetime
from typing import List, Dict, Tuple


class ChatDatabase:
    """SQLite database for storing chat history"""
    
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create chat sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    breed_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL
                )
            """)
            
            # Create chat messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                )
            """)
            
            conn.commit()
    
    def create_session(self, breed_name: str, confidence: float) -> int:
        """Create a new chat session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_sessions (breed_name, confidence) VALUES (?, ?)",
                (breed_name, confidence)
            )
            conn.commit()
            return cursor.lastrowid
    
    def save_message(self, session_id: int, role: str, content: str):
        """Save a message to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )
            conn.commit()
    
    def get_session_history(self, session_id: int) -> List[Dict]:
        """Get all messages in a session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content, timestamp FROM chat_messages WHERE session_id = ? ORDER BY message_id ASC",
                (session_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_breed_sessions(self, breed_name: str) -> List[Dict]:
        """Get all sessions for a specific breed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, breed_name, created_at, confidence 
                FROM chat_sessions 
                WHERE breed_name = ? 
                ORDER BY created_at DESC
                """,
                (breed_name,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all chat sessions"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, breed_name, created_at, confidence 
                FROM chat_sessions 
                ORDER BY created_at DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_session(self, session_id: int):
        """Delete a chat session and all its messages"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
