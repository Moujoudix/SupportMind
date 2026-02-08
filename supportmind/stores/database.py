"""
SQLite Database with FTS5 support.
"""

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from supportmind.config.settings import get_config


class Database:
    """Thread-safe SQLite database with FTS5."""

    _instances: Dict[str, "Database"] = {}
    _lock = threading.Lock()

    def __new__(cls, db_path: str = None):
        """Singleton per database path."""
        db_path = db_path or get_config().db_path

        if db_path not in cls._instances:
            with cls._lock:
                if db_path not in cls._instances:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instances[db_path] = instance
        return cls._instances[db_path]

    def __init__(self, db_path: str = None):
        if self._initialized:
            return

        self.db_path = db_path or get_config().db_path
        self._local = threading.local()
        self._create_tables()
        self._initialized = True

    @property
    def conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def transaction(self):
        """Transaction context manager."""
        try:
            yield self.conn
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e

    def _create_tables(self):
        """Create all database tables."""
        schema = """
        -- Documents (unified storage)
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            doc_type TEXT NOT NULL,
            source_id TEXT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            version INTEGER DEFAULT 1,
            created_at TEXT,
            updated_at TEXT,
            content_hash TEXT,
            is_active INTEGER DEFAULT 1
        );

        -- FTS5 for keyword search
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            doc_id, title, content, doc_type,
            content='documents', content_rowid='rowid'
        );

        -- FTS triggers
        CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(doc_id, title, content, doc_type)
            VALUES (new.doc_id, new.title, new.content, new.doc_type);
        END;

        CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, doc_id, title, content, doc_type)
            VALUES ('delete', old.doc_id, old.title, old.content, old.doc_type);
        END;

        -- Tickets
        CREATE TABLE IF NOT EXISTS tickets (
            Ticket_Number TEXT PRIMARY KEY,
            Conversation_ID TEXT,
            Created_At TEXT,
            Closed_At TEXT,
            Status TEXT,
            Priority TEXT,
            Tier INTEGER,
            Product TEXT,
            Module TEXT,
            Category TEXT,
            Case_Type TEXT,
            Account_Name TEXT,
            Property_Name TEXT,
            Property_City TEXT,
            Property_State TEXT,
            Contact_Name TEXT,
            Contact_Role TEXT,
            Contact_Email TEXT,
            Contact_Phone TEXT,
            Subject TEXT,
            Description TEXT,
            Resolution TEXT,
            Root_Cause TEXT,
            Tags TEXT,
            KB_Article_ID TEXT,
            Script_ID TEXT,
            Generated_KB_Article_ID TEXT
        );

        -- Conversations
        CREATE TABLE IF NOT EXISTS conversations (
            Conversation_ID TEXT PRIMARY KEY,
            Ticket_Number TEXT,
            Channel TEXT,
            Conversation_Start TEXT,
            Conversation_End TEXT,
            Customer_Role TEXT,
            Agent_Name TEXT,
            Product TEXT,
            Category TEXT,
            Issue_Summary TEXT,
            Transcript TEXT,
            Sentiment TEXT
        );

        -- Questions
        CREATE TABLE IF NOT EXISTS questions (
            Question_ID TEXT PRIMARY KEY,
            Source TEXT,
            Product TEXT,
            Category TEXT,
            Module TEXT,
            Difficulty TEXT,
            Question_Text TEXT,
            Answer_Type TEXT,
            Target_ID TEXT,
            Target_Title TEXT
        );

        -- Knowledge gaps
        CREATE TABLE IF NOT EXISTS knowledge_gaps (
            gap_id TEXT PRIMARY KEY,
            gap_type TEXT,
            detected_from_ticket TEXT,
            detected_from_query TEXT,
            retrieval_confidence REAL,
            score_margin REAL,
            repeated_count INTEGER DEFAULT 1,
            triggered_by_auto_zero INTEGER DEFAULT 0,
            topic TEXT,
            priority TEXT DEFAULT 'medium',
            should_update_existing INTEGER DEFAULT 0,
            existing_kb_to_update TEXT,
            status TEXT DEFAULT 'open',
            detected_at TEXT
        );

        -- Draft KB articles
        CREATE TABLE IF NOT EXISTS draft_kb_articles (
            draft_id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            category TEXT,
            module TEXT,
            product TEXT,
            tags TEXT,
            source_tickets TEXT,
            source_gap_id TEXT,
            is_update INTEGER DEFAULT 0,
            updating_kb_id TEXT,
            status TEXT DEFAULT 'pending',
            reviewer TEXT,
            review_notes TEXT,
            generation_confidence REAL,
            created_at TEXT,
            reviewed_at TEXT,
            published_at TEXT,
            published_kb_id TEXT
        );

        -- KB Lineage
        CREATE TABLE IF NOT EXISTS kb_lineage (
            lineage_id INTEGER PRIMARY KEY AUTOINCREMENT,
            KB_Article_ID TEXT,
            Source_Type TEXT,
            Source_ID TEXT,
            Relationship TEXT,
            Evidence_Snippet TEXT,
            Event_Timestamp TEXT
        );

        -- Learning Events
        CREATE TABLE IF NOT EXISTS learning_events (
            Event_ID TEXT PRIMARY KEY,
            Trigger_Ticket_Number TEXT,
            Trigger_Conversation_ID TEXT,
            Detected_Gap TEXT,
            Proposed_KB_Article_ID TEXT,
            Draft_Summary TEXT,
            Final_Status TEXT,
            Reviewer_Role TEXT,
            Event_Timestamp TEXT
        );

        -- QA Scores
        CREATE TABLE IF NOT EXISTS qa_scores (
            score_id TEXT PRIMARY KEY,
            ticket_number TEXT,
            conversation_id TEXT,
            response_trace_id TEXT,
            tone_score REAL,
            accuracy_score REAL,
            completeness_score REAL,
            compliance_score REAL,
            overall_score REAL,
            auto_zero INTEGER DEFAULT 0,
            auto_zero_reason TEXT,
            violations TEXT,
            action TEXT,
            suggestions TEXT,
            evaluated_at TEXT
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(doc_type);
        CREATE INDEX IF NOT EXISTS idx_docs_source ON documents(source_id);
        CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(Status);
        CREATE INDEX IF NOT EXISTS idx_questions_type ON questions(Answer_Type);
        CREATE INDEX IF NOT EXISTS idx_gaps_status ON knowledge_gaps(status);
        """
        with self.transaction():
            self.conn.executescript(schema)

    def insert(self, table: str, data: Dict[str, Any]) -> bool:
        """Insert or replace a record."""
        processed = {}
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                processed[k] = json.dumps(v)
            elif isinstance(v, bool):
                processed[k] = 1 if v else 0
            else:
                processed[k] = v

        cols = ", ".join(processed.keys())
        vals = ", ".join(["?" for _ in processed])
        sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({vals})"

        with self.transaction():
            self.conn.execute(sql, list(processed.values()))
        return True

    def get(self, table: str, id_col: str, id_val: str) -> Optional[Dict]:
        """Get a single record by ID."""
        sql = f"SELECT * FROM {table} WHERE {id_col} = ?"
        cur = self.conn.execute(sql, [id_val])
        row = cur.fetchone()
        return self._parse_row(row) if row else None

    def get_all(self, table: str, where: str = None,
                params: List = None, limit: int = 1000) -> List[Dict]:
        """Get multiple records."""
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        sql += f" LIMIT {limit}"
        cur = self.conn.execute(sql, params or [])
        return [self._parse_row(r) for r in cur.fetchall()]

    def query(self, sql: str, params: List = None) -> List[Dict]:
        """Execute raw SQL query."""
        cur = self.conn.execute(sql, params or [])
        return [self._parse_row(r) for r in cur.fetchall()]

    def execute(self, sql: str, params: List = None) -> int:
        """Execute raw SQL statement."""
        with self.transaction():
            cur = self.conn.execute(sql, params or [])
        return cur.rowcount

    def _parse_row(self, row) -> Dict:
        """Parse row, deserializing JSON fields."""
        if not row:
            return {}
        result = dict(row)
        json_fields = ['metadata', 'tags', 'violations', 'suggestions', 'source_tickets']
        for field in json_fields:
            if field in result and result[field]:
                try:
                    result[field] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return result

    def fts_search(self, query: str, doc_type: str = None,
                   limit: int = 10) -> List[Dict]:
        """Full-text search using FTS5."""
        clean = re.sub(r'[^\w\s]', ' ', query)
        terms = clean.split()
        if not terms:
            return []

        fts_query = " OR ".join(terms)
        sql = """
        SELECT d.*, bm25(documents_fts) as bm25_score
        FROM documents_fts fts
        JOIN documents d ON fts.doc_id = d.doc_id
        WHERE documents_fts MATCH ?
        """
        params = [fts_query]

        if doc_type:
            sql += " AND d.doc_type = ?"
            params.append(doc_type)

        sql += " ORDER BY bm25_score LIMIT ?"
        params.append(limit)

        try:
            return self.query(sql, params)
        except sqlite3.OperationalError:
            return []
