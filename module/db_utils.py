#!/usr/bin/env python3
"""Database utility functions for MySQL connection"""
import os
from mysql.connector.connection import MySQLConnection
import mysql.connector
from dotenv import load_dotenv
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

try:
    from sqlalchemy import create_engine
except Exception:
    create_engine = None  # type: ignore


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
    return mysql.connector.connect(**cfg)


def get_sqlalchemy_connection() -> Any:
    """Return a SQLAlchemy Connection for pandas.read_sql (preferred). Fallbacks to MySQLConnection.

    - Uses env vars: DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
    - Requires SQLAlchemy; if unavailable or creation fails, returns mysql.connector connection.
    """
    # Load .env from project root (RecmdSys/)
    try:
        project_root = Path(__file__).resolve().parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(str(env_path))
    except Exception:
        load_dotenv()

    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD", "")
    database = os.getenv("DB_NAME")

    if create_engine is not None and all([host, user, database]):
        try:
            url = f"mysql+mysqlconnector://{user}:{quote_plus(password)}@{host}/{database}?charset=utf8mb4"
            engine = create_engine(url, pool_pre_ping=True)
            return engine.connect()
        except Exception:
            # Fall back to mysql.connector
            pass

    return get_connection()
