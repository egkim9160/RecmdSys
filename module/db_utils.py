#!/usr/bin/env python3
"""Database utility functions for MySQL connection"""
import os
import logging
from mysql.connector.connection import MySQLConnection
import mysql.connector
from dotenv import load_dotenv
from pathlib import Path

# Setup logger
logger = logging.getLogger(__name__)


def get_connection() -> MySQLConnection:
    """Create and return a MySQL database connection using environment variables"""
    # Load .env from project root (RecmdSys/)
    try:
        project_root = Path(__file__).resolve().parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(str(env_path))
    except Exception:
        load_dotenv()
    cfg = dict(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        charset="utf8mb4",
    )
    logger.info(f"[DB] Connecting to MySQL: host={cfg['host']}, user={cfg['user']}, database={cfg['database']}")
    try:
        conn = mysql.connector.connect(**cfg)
        logger.info("[DB] Connection established successfully")
        return conn
    except Exception as e:
        logger.error(f"[DB] Connection failed: {e}")
        raise
