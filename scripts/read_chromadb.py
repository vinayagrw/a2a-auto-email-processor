#!/usr/bin/env python3
"""
Simple ChromaDB Data Reader

This script reads and displays data from a ChromaDB collection in the specified directory.
"""
import os
import json
import chromadb
from typing import List, Dict, Any
from pathlib import Path

# Configuration
DATA_DIR = r"c:\Users\khush\Documents\a2a-\a2a-mcp-contractor-automation\data\chroma"
COLLECTION_NAME = "emails"

def list_collections() -> List[str]:
    """List all available collections in the ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=DATA_DIR)
        return [c.name for c in client.list_collections()]
    except Exception as e:
        print(f"Error listing collections: {e}")
        return []

def print_collection_data(collection_name: str):
    """Print all data from the specified collection."""
    try:
        # Initialize Chroma client
        client = chromadb.PersistentClient(path=DATA_DIR)
        
        # Get the collection
        collection = client.get_collection(name=collection_name)
        
        # Get all items (limit to 1000 for safety)
        results = collection.get(limit=1000, include=["documents", "metadatas"])
        
        if not results["ids"]:
            print(f"No documents found in collection '{collection_name}'")
            return
            
        print(f"\nFound {len(results['ids'])} documents in collection '{collection_name}':")
        print("-" * 80)
        
        # Print each document
        for i, (doc_id, doc, metadata) in enumerate(zip(
            results["ids"], 
            results["documents"], 
            results["metadatas"]
        ), 1):
            print(f"\nDocument {i} (ID: {doc_id})")
            print("-" * 40)
            
            # Print metadata
            if metadata:
                print("Metadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            
            # Print document content (first 500 chars)
            if doc:
                print("\nContent:")
                print(doc[:500] + ("..." if len(doc) > 500 else ""))
            
            print("\n" + "=" * 80 + "\n")
            
    except Exception as e:
        print(f"Error reading collection '{collection_name}': {e}")

def main():
    print("ChromaDB Data Reader")
    print("=" * 40 + "\n")
    
    # List available collections
    collections = list_collections()
    
    if not collections:
        print(f"No collections found in {DATA_DIR}")
        return
    
    print("Available collections:")
    for i, name in enumerate(collections, 1):
        print(f"  {i}. {name}")
    
    # Use the default collection if it exists
    if COLLECTION_NAME in collections:
        print(f"\nReading from default collection: {COLLECTION_NAME}")
        print_collection_data(COLLECTION_NAME)
    else:
        print(f"\nDefault collection '{COLLECTION_NAME}' not found.")
        print("Please specify a collection name to read from.")

if __name__ == "__main__":
    main()
