from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Query
from fastapi_mcp import FastApiMCP
import logging
from contextlib import asynccontextmanager

from mcp_memory_service import Memory
from ..storage.sqlite_vec import SqliteVecMemoryStorage
from ..config import (
    HTTP_PORT,
    HTTP_HOST,
    CORS_ORIGINS,
    DATABASE_PATH,
    EMBEDDING_MODEL_NAME,
    MDNS_ENABLED,
    HTTPS_ENABLED
)

logger = logging.getLogger(__name__)
storage: Optional[SqliteVecMemoryStorage] = None

from .dependencies import set_storage, get_storage
from ..utils.hashing import generate_content_hash


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global storage, mdns_advertiser
    
    # Startup
    logger.info("Starting MCP Memory Service HTTP interface...")
    try:
        storage = SqliteVecMemoryStorage(
            db_path=DATABASE_PATH,
            embedding_model=EMBEDDING_MODEL_NAME
        )
        await storage.initialize()
        set_storage(storage)  # Set the global storage instance
        logger.info(f"SQLite-vec storage initialized at {DATABASE_PATH}")
            
    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP Memory Service HTTP interface...")
    
    if storage:
        await storage.close()

app = FastAPI(lifespan=lifespan)

@app.post("/store_memory", operation_id="store_memory")
async def store_memory(content: str = Query(description="The memory content to store"), memory_type: Optional[str] = Query(None, description="Optional memory type"), tags: List[str] = Query([], description="Optional tags for the memory")):
    """Store a new memory with optional tags and metadata"""

    from mcp_memory_service.models.memory import Memory

    content_hash = generate_content_hash(content, {})
    memory = Memory(
        content=content,
        content_hash=content_hash,
        tags=tags,
        memory_type=memory_type
    )

    storage = get_storage()
    success, message = await storage.store(memory)
        
    return {
        "success": success,
        "message": message,
        "content_hash": memory.content_hash if success else None
    }


@app.get("/retrieve_memory", operation_id="retrieve_memory")
async def retrieve_memory(query: str = Query(description="Search query for finding relevant memories"), limit: int = Query(10, description="Maximum number of memories to return")):
    """RSearch and retrieve memories using semantic similarity"""

    storage = get_storage()
    results = await storage.retrieve(query=query, n_results=limit)
        
    return {
        "results": [
            {
                "content": r.memory.content,
                "content_hash": r.memory.content_hash,
                "tags": r.memory.tags,
                "similarity_score": r.relevance_score,
                "created_at": r.memory.created_at_iso
            }
            for r in results
        ],
        "total_found": len(results)
    }

@app.get("/search_by_tag", operation_id="search_by_tag")
async def search_by_tag(tags: list[str] = Query(description="Tags to search for")):
    """Search memories by specific tags"""

    storage = get_storage()
    results = await storage.search_by_tag(tags=tags)
        
    return {
        "results": [
            {
                "content": memory.content,
                "content_hash": memory.content_hash,
                "tags": memory.tags,
                "created_at": memory.created_at_iso
            }
            for memory in results
        ],
        "total_found": len(results)
    }

@app.post("/delete_memory", operation_id="delete_memory")
async def delete_memory(content_hash: str = Query(description="Hash of the memory to delete")):
    """Delete a specific memory by content hash"""

    storage = get_storage()
    success,message = await storage.delete(content_hash)
        
    return {
        "success": success,
        "message": message
    }

# Convert FastAPI app to MCP server
mcp = FastApiMCP(app, name="Memory API", description="API for managing memories")
mcp.mount_http()  # Mounts MCP endpoint at /mcp
