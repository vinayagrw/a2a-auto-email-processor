import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.resolve()

# Path to ChromaDB
CHROMA_DB_PATH = (BASE_DIR.parent / "data" / "chroma").resolve()

# Server configuration
HOST = "0.0.0.0"
PORT = 8004

# Static files
STATIC_DIR = (BASE_DIR / "static").resolve()
TEMPLATES_DIR = (BASE_DIR / "templates").resolve()

# Ensure directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Print paths for debugging
print(f"BASE_DIR: {BASE_DIR}")
print(f"STATIC_DIR: {STATIC_DIR} (exists: {STATIC_DIR.exists()})")
print(f"TEMPLATES_DIR: {TEMPLATES_DIR} (exists: {TEMPLATES_DIR.exists()})")
print(f"CHROMA_DB_PATH: {CHROMA_DB_PATH} (exists: {CHROMA_DB_PATH.exists()})")
