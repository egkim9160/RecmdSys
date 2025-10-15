#!/usr/bin/env python3
"""Database utility functions for MySQL connection"""
import os
from mysql.connector.connection import MySQLConnection
import mysql.connector
from dotenv import load_dotenv


def get_connection() -> MySQLConnection:
    """Create and return a MySQL database connection using environment variables"""
    load_dotenv()
    cfg = dict(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        charset="utf8mb4",
    )
    return mysql.connector.connect(**cfg)
