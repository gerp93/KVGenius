"""
Chat history database management using SQLite.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ChatHistoryDB:
    """Manages chat history in SQLite database."""
    
    def __init__(self, db_path: str = "./chat_history.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self._init_database()
    
    def _connect(self):
        """Open a SQLite connection with WAL mode and timeout to prevent locking."""
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                model TEXT NOT NULL,
                ai_character_id INTEGER,
                user_persona_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ai_character_id) REFERENCES ai_characters(id),
                FOREIGN KEY (user_persona_id) REFERENCES user_personas(id)
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
            )
        """)
        
        # AI Characters table (what the AI acts as)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                system_prompt TEXT,
                temperature REAL DEFAULT 0.7,
                top_p REAL DEFAULT 0.95,
                top_k INTEGER DEFAULT 50,
                description TEXT,
                avatar TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User Personas table (who the user is roleplaying as)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                background TEXT,
                avatar TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_time ON messages(timestamp)")
        
        # Prompt Library table for saved image generation prompts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_library (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                prompt TEXT NOT NULL,
                negative_prompt TEXT,
                categories TEXT,
                steps INTEGER DEFAULT 25,
                guidance_scale REAL DEFAULT 7.5,
                width INTEGER DEFAULT 512,
                height INTEGER DEFAULT 768,
                lora TEXT,
                lora_strength REAL DEFAULT 0.8,
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prompt_name ON prompt_library(name)")
        
        # Add categories column if it doesn't exist (migration for existing databases)
        try:
            cursor.execute("ALTER TABLE prompt_library ADD COLUMN categories TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Add lora columns if they don't exist (migration)
        try:
            cursor.execute("ALTER TABLE prompt_library ADD COLUMN lora TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE prompt_library ADD COLUMN lora_strength REAL DEFAULT 0.8")
        except sqlite3.OperationalError:
            pass
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def create_conversation(self, model: str, ai_character_id: Optional[int] = None, user_persona_id: Optional[int] = None, title: Optional[str] = None) -> int:
        """Create a new conversation."""
        if not title:
            title = f"Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (title, model, ai_character_id, user_persona_id)
                VALUES (?, ?, ?, ?)
            """, (title, model, ai_character_id, user_persona_id))
            conv_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()
        logger.info(f"Created conversation {conv_id}: {title}")
        return conv_id
    
    def add_message(self, conversation_id: int, role: str, content: str):
        """Add a message to a conversation."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (conversation_id, role, content)
                VALUES (?, ?, ?)
            """, (conversation_id, role, content))
            cursor.execute("""
                UPDATE conversations
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (conversation_id,))
            conn.commit()
        finally:
            conn.close()
    
    def get_conversation_messages(self, conversation_id: int) -> List[Tuple[str, str]]:
        """Get all messages from a conversation in Gradio format [(user, assistant), ...]."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            """, (conversation_id,))
            messages = cursor.fetchall()
        finally:
            conn.close()
        history = []
        user_msg = None
        for role, content in messages:
            if role == "user":
                user_msg = content
            elif role == "assistant" and user_msg:
                history.append((user_msg, content))
                user_msg = None
        return history
    
    def get_recent_conversations(self, limit: int = 20) -> List[Dict]:
        """Get recent conversations."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, model, ai_character_id, user_persona_id, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
        finally:
            conn.close()
        conversations = []
        for row in rows:
            conversations.append({
                "id": row[0],
                "title": row[1],
                "model": row[2],
                "ai_character_id": row[3],
                "user_persona_id": row[4],
                "created_at": row[5],
                "updated_at": row[6]
            })
        return conversations
    
    def get_conversation(self, conversation_id: int) -> Optional[Dict]:
        """Get a single conversation by ID."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, model, ai_character_id, user_persona_id, created_at, updated_at
                FROM conversations
                WHERE id = ?
            """, (conversation_id,))
            row = cursor.fetchone()
        finally:
            conn.close()
        if row:
            return {
                "id": row[0],
                "title": row[1],
                "model": row[2],
                "ai_character_id": row[3],
                "user_persona_id": row[4],
                "created_at": row[5],
                "updated_at": row[6]
            }
        return None
    
    def update_conversation_title(self, conversation_id: int, title: str):
        """Update conversation title."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversations
                SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (title, conversation_id))
            conn.commit()
        finally:
            conn.close()
    
    def delete_conversation(self, conversation_id: int):
        """Delete a conversation and all its messages."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
        finally:
            conn.close()
        logger.info(f"Deleted conversation {conversation_id}")
    
    def delete_all_conversations(self):
        """Delete all conversations and their messages."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages")
            cursor.execute("DELETE FROM conversations")
            conn.commit()
        finally:
            conn.close()
        logger.info("Deleted all conversations")
    
    def export_conversation(self, conversation_id: int) -> Dict:
        """Export conversation as JSON."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT title, model, ai_character_id, user_persona_id, created_at, updated_at
                FROM conversations
                WHERE id = ?
            """, (conversation_id,))
            conv = cursor.fetchone()
            cursor.execute("""
                SELECT role, content, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            """, (conversation_id,))
            messages = cursor.fetchall()
        finally:
            conn.close()
        return {
            "id": conversation_id,
            "title": conv[0],
            "model": conv[1],
            "ai_character_id": conv[2],
            "user_persona_id": conv[3],
            "created_at": conv[4],
            "updated_at": conv[5],
            "messages": [
                {"role": msg[0], "content": msg[1], "timestamp": msg[2]}
                for msg in messages
            ]
        }
    
    # Persona management methods
    
    # AI Character Methods
    def create_ai_character(self, name: str, system_prompt: str = "", temperature: float = 0.7,
                      top_p: float = 0.95, top_k: int = 50, description: str = "",
                      avatar: str = "🤖") -> int:
        """Create a new AI character."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ai_characters (name, system_prompt, temperature, top_p, top_k, description, avatar)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, system_prompt, temperature, top_p, top_k, description, avatar))
            char_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()
        logger.info(f"Created AI character {char_id}: {name}")
        return char_id
    
    def get_ai_character(self, character_id: int) -> Optional[Dict]:
        """Get AI character by ID."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, system_prompt, temperature, top_p, top_k, description, avatar
                FROM ai_characters
                WHERE id = ?
            """, (character_id,))
            row = cursor.fetchone()
        finally:
            conn.close()
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "system_prompt": row[2],
                "temperature": row[3],
                "top_p": row[4],
                "top_k": row[5],
                "description": row[6],
                "avatar": row[7]
            }
        return None
    
    def get_all_ai_characters(self) -> List[Dict]:
        """Get all AI characters."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, system_prompt, temperature, top_p, top_k, description, avatar
                FROM ai_characters
                ORDER BY name ASC
            """)
            rows = cursor.fetchall()
        finally:
            conn.close()
        characters = []
        for row in rows:
            characters.append({
                "id": row[0],
                "name": row[1],
                "system_prompt": row[2],
                "temperature": row[3],
                "top_p": row[4],
                "top_k": row[5],
                "description": row[6],
                "avatar": row[7]
            })
        return characters
    
    def update_ai_character(self, character_id: int, **kwargs):
        """Update AI character fields."""
        fields = []
        values = []
        for key, value in kwargs.items():
            if key in ['name', 'system_prompt', 'temperature', 'top_p', 'top_k', 'description', 'avatar']:
                fields.append(f"{key} = ?")
                values.append(value)
        if not fields:
            return
        conn = self._connect()
        try:
            cursor = conn.cursor()
            query = f"UPDATE ai_characters SET {', '.join(fields)} WHERE id = ?"
            values.append(character_id)
            cursor.execute(query, values)
            conn.commit()
        finally:
            conn.close()
    
    def delete_ai_character(self, character_id: int):
        """Delete an AI character."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ai_characters WHERE id = ?", (character_id,))
            conn.commit()
        finally:
            conn.close()
        logger.info(f"Deleted AI character {character_id}")
    
    # User Persona Methods
    def create_user_persona(self, name: str, description: str = "", background: str = "",
                           avatar: str = "👤") -> int:
        """Create a new user persona."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_personas (name, description, background, avatar)
                VALUES (?, ?, ?, ?)
            """, (name, description, background, avatar))
            persona_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()
        logger.info(f"Created user persona {persona_id}: {name}")
        return persona_id
    
    def get_user_persona(self, persona_id: int) -> Optional[Dict]:
        """Get user persona by ID."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, background, avatar
                FROM user_personas
                WHERE id = ?
            """, (persona_id,))
            row = cursor.fetchone()
        finally:
            conn.close()
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "background": row[3],
                "avatar": row[4]
            }
        return None
    
    def get_all_user_personas(self) -> List[Dict]:
        """Get all user personas."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, background, avatar
                FROM user_personas
                ORDER BY name ASC
            """)
            rows = cursor.fetchall()
        finally:
            conn.close()
        personas = []
        for row in rows:
            personas.append({
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "background": row[3],
                "avatar": row[4]
            })
        return personas
    
    def update_user_persona(self, persona_id: int, **kwargs):
        """Update user persona fields."""
        fields = []
        values = []
        for key, value in kwargs.items():
            if key in ['name', 'description', 'background', 'avatar']:
                fields.append(f"{key} = ?")
                values.append(value)
        if not fields:
            return
        conn = self._connect()
        try:
            cursor = conn.cursor()
            query = f"UPDATE user_personas SET {', '.join(fields)} WHERE id = ?"
            values.append(persona_id)
            cursor.execute(query, values)
            conn.commit()
        finally:
            conn.close()
    
    def delete_user_persona(self, persona_id: int):
        """Delete a user persona."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_personas WHERE id = ?", (persona_id,))
            conn.commit()
        finally:
            conn.close()
        logger.info(f"Deleted user persona {persona_id}")
    
    def init_defaults(self):
        """Initialize default AI characters and user personas if none exist."""
        ai_characters = self.get_all_ai_characters()
        if not ai_characters:
            self.create_ai_character(
                name="Default Assistant",
                system_prompt="You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.",
                description="Standard helpful AI assistant",
                avatar="🤖"
            )
            self.create_ai_character(
                name="Creative Writer",
                system_prompt="You are a creative writing assistant specialized in fiction, storytelling, and vivid prose. Focus on engaging narratives, rich descriptions, and compelling characters.",
                temperature=0.9,
                description="Fiction and creative writing specialist",
                avatar="✍️"
            )
            self.create_ai_character(
                name="Code Helper",
                system_prompt="You are an expert programming assistant. Provide clean, efficient, well-documented code. Explain technical concepts clearly and suggest best practices.",
                temperature=0.3,
                description="Programming and technical assistance",
                avatar="💻"
            )
            self.create_ai_character(
                name="Friendly Companion",
                system_prompt="You are a friendly, empathetic companion who engages in casual conversation. You're interested in the user's life, ask follow-up questions, and provide emotional support.",
                temperature=0.85,
                description="Casual conversational companion",
                avatar="😊"
            )
            logger.info("Initialized default AI characters")
        
        user_personas = self.get_all_user_personas()
        if not user_personas:
            self.create_user_persona(
                name="Myself (Default)",
                description="Just being myself",
                avatar="👤"
            )
            self.create_user_persona(
                name="Fantasy Adventurer",
                description="A brave adventurer in a fantasy world",
                background="You are a skilled warrior/mage on a quest for glory and treasure.",
                avatar="⚔️"
            )
            self.create_user_persona(
                name="Space Explorer",
                description="A starship captain exploring the cosmos",
                background="You command a starship in the far future, exploring unknown galaxies.",
                avatar="🚀"
            )
            self.create_user_persona(
                name="Detective",
                description="A hard-boiled detective solving mysteries",
                background="You're a private investigator in a noir-style city, solving cases.",
                avatar="🕵️"
            )
            logger.info("Initialized default user personas")

    # ==================== Prompt Library Methods ====================
    
    def save_prompt(self, name: str, prompt: str, negative_prompt: str = "", 
                    categories: List[str] = None, steps: int = 25, guidance_scale: float = 7.5,
                    width: int = 512, height: int = 768, lora: str = None,
                    lora_strength: float = 0.8, model: str = None) -> int:
        """Save a prompt to the library."""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Convert categories list to JSON string
        categories_json = json.dumps(categories) if categories else None
        
        try:
            cursor.execute("""
                INSERT INTO prompt_library (name, prompt, negative_prompt, categories, steps, guidance_scale, width, height, lora, lora_strength, model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, prompt, negative_prompt, categories_json, steps, guidance_scale, width, height, lora, lora_strength, model))
            
            prompt_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Saved prompt '{name}' with id {prompt_id}")
            return prompt_id
        except sqlite3.IntegrityError:
            # Name already exists, update instead
            cursor.execute("""
                UPDATE prompt_library 
                SET prompt = ?, negative_prompt = ?, categories = ?, steps = ?, guidance_scale = ?, 
                    width = ?, height = ?, lora = ?, lora_strength = ?, model = ?, updated_at = CURRENT_TIMESTAMP
                WHERE name = ?
            """, (prompt, negative_prompt, categories_json, steps, guidance_scale, width, height, lora, lora_strength, model, name))
            conn.commit()
            logger.info(f"Updated existing prompt '{name}'")
            return self.get_prompt_by_name(name)['id']
        finally:
            conn.close()
    
    def get_all_prompts(self) -> List[Dict]:
        """Get all saved prompts."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM prompt_library ORDER BY updated_at DESC
            """)
            rows = cursor.fetchall()
        finally:
            conn.close()
        prompts = []
        for row in rows:
            prompt = dict(row)
            if prompt.get('categories'):
                try:
                    prompt['categories'] = json.loads(prompt['categories'])
                except (json.JSONDecodeError, TypeError):
                    prompt['categories'] = []
            else:
                prompt['categories'] = []
            prompts.append(prompt)
        return prompts
    
    def get_prompt_by_id(self, prompt_id: int) -> Optional[Dict]:
        """Get a prompt by ID."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prompt_library WHERE id = ?", (prompt_id,))
            row = cursor.fetchone()
        finally:
            conn.close()
        return dict(row) if row else None
    
    def get_prompt_by_name(self, name: str) -> Optional[Dict]:
        """Get a prompt by name."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prompt_library WHERE name = ?", (name,))
            row = cursor.fetchone()
        finally:
            conn.close()
        return dict(row) if row else None
    
    def update_prompt(self, prompt_id: int, **kwargs) -> bool:
        """Update a prompt's fields."""
        if not kwargs:
            return False
        
        conn = self._connect()
        cursor = conn.cursor()
        
        # Build update query dynamically
        allowed_fields = ['name', 'prompt', 'negative_prompt', 'categories', 'steps', 'guidance_scale', 'width', 'height', 'lora', 'lora_strength', 'model']
        updates = []
        values = []
        for field, value in kwargs.items():
            if field in allowed_fields:
                # Convert categories list to JSON
                if field == 'categories' and isinstance(value, list):
                    value = json.dumps(value)
                updates.append(f"{field} = ?")
                values.append(value)
        
        if not updates:
            conn.close()
            return False
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        values.append(prompt_id)
        
        try:
            cursor.execute(f"""
                UPDATE prompt_library SET {', '.join(updates)} WHERE id = ?
            """, values)
            conn.commit()
            success = cursor.rowcount > 0
            logger.info(f"Updated prompt {prompt_id}: {success}")
            return success
        except Exception as e:
            logger.error(f"Error updating prompt {prompt_id}: {e}")
            return False
        finally:
            conn.close()
    
    def delete_prompt(self, prompt_id: int) -> bool:
        """Delete a prompt from the library."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prompt_library WHERE id = ?", (prompt_id,))
            conn.commit()
            success = cursor.rowcount > 0
        finally:
            conn.close()
        logger.info(f"Deleted prompt {prompt_id}: {success}")
        return success