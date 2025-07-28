"""
ChromaDB Viewer - A web interface for browsing ChromaDB collections.
"""
from logging import Logger
import os
import json
from re import L
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import chromadb
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
from pathlib import Path

from .config import CHROMA_DB_PATH, STATIC_DIR, TEMPLATES_DIR, HOST, PORT

# Create FastAPI app
app = FastAPI(title="ChromaDB Viewer", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mount static files
app.mount(
    "/static", 
    StaticFiles(directory=str(STATIC_DIR)), 
    name="static"
)

# Add a route to serve the main.js file directly (helpful for debugging)
@app.get("/js/main.js")
async def get_js():
    js_path = STATIC_DIR / "js" / "main.js"
    
    if not js_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"JavaScript file not found at {js_path}. Static dir: {STATIC_DIR}"
        )
    return FileResponse(str(js_path))

# Add a test endpoint to verify static files
@app.get("/test-static")
async def test_static():
    """Test if static files are being served correctly."""
    # Define paths
    js_path = STATIC_DIR / "js" / "main.js"
    css_path = STATIC_DIR / "css" / "style.css"
    template_path = TEMPLATES_DIR / "index.html"
    js_dir = STATIC_DIR / "js"
    css_dir = STATIC_DIR / "css"
    
    return {
        "status": "ok",
        "static_dir": str(STATIC_DIR),
        "static_dir_exists": STATIC_DIR.exists(),
        "templates_dir": str(TEMPLATES_DIR),
        "templates_dir_exists": TEMPLATES_DIR.exists(),
        "js_exists": js_path.exists(),
        "js_path": str(js_path),
        "css_exists": css_path.exists(),
        "template_exists": template_path.exists(),
        "files_in_static": os.listdir(str(STATIC_DIR)) if STATIC_DIR.exists() else [],
        "files_in_js_dir": os.listdir(str(js_dir)) if js_dir.exists() else [],
        "files_in_css_dir": os.listdir(str(css_dir)) if css_dir.exists() else []
    }

print("ChromaDB path: ", CHROMA_DB_PATH)

# ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Models
class CollectionInfo(BaseModel):
    name: str
    count: int
    metadata: Optional[dict] = None

class DocumentResponse(BaseModel):
    id: str
    document: Any
    metadata: Optional[dict] = None

class CollectionResponse(BaseModel):
    name: str
    total: int
    limit: int
    offset: int
    data: List[DocumentResponse]

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page."""
    try:
        # Verify template exists
        template_path = TEMPLATES_DIR / "index.html"
        if not template_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Template not found at {template_path}"
            )
            
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "static_url": "/static"
            }
        )
    except Exception as e:
        # Provide detailed error information
        import traceback
        template_files = os.listdir(str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else 'Directory not found'
        
        return HTMLResponse(
            content=f"""
            <h1>Error loading template</h1>
            <p>Error: {str(e)}</p>
            <h3>Debug Information:</h3>
            <pre>Template path: {TEMPLATES_DIR}
            Type: {type(TEMPLATES_DIR)}
            
            Files in template directory: {template_files}
            
            {traceback.format_exc()}</pre>
            """,
            status_code=500
        )


@app.get("/api/collections", response_model=List[CollectionInfo])
async def list_collections():
    """List all collections in the ChromaDB."""
    collections = []
    for collection in chroma_client.list_collections():
        try:
            col = chroma_client.get_collection(collection.name)
            collections.append({
                "name": collection.name,
                "count": col.count(),
                "metadata": collection.metadata or {}
            })
        except Exception as e:
            print(f"Error getting collection {collection.name}: {e}")
    return collections

@app.get("/api/collections/{collection_name}", response_model=CollectionResponse)
async def get_collection(
    collection_name: str,
    limit: int = 10,
    offset: int = 0
):
    """Get documents from a specific collection."""
    try:
        collection = chroma_client.get_collection(collection_name)
        results = collection.get(limit=limit, offset=offset)
        
        # Get total count
        total = collection.count()
        
        return {
            "name": collection_name,
            "total": total,
            "limit": limit,
            "offset": offset,
            "data": [
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta or {}
                }
                for doc_id, doc, meta in zip(
                    results.get('ids', []),
                    results.get('documents', []),
                    results.get('metadatas', [{}] * len(results.get('ids', [])))
                )
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

def run():
    """Run the ChromaDB viewer server."""
    import uvicorn
    print(f"Starting ChromaDB Viewer at http://{HOST}:{PORT}")
    print(f"ChromaDB path: {CHROMA_DB_PATH}")
    uvicorn.run(app, host=HOST, port=PORT)

if __name__ == "__main__":
    run()
