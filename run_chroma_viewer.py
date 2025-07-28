#!/usr/bin/env python3
"""
Run the ChromaDB Viewer application.
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the chroma_viewer package
from chroma_viewer import run

if __name__ == "__main__":
    run()
