"""
FastAPI service for processing educational materials.
REST API endpoints for interaction with LangGraph workflow.
"""

import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
#from langfuse import Langfuse

from ..core.graph_manager import GraphManager
from ..config.settings import get_settings
from ..services.file_utils import ImageFileManager, ensure_temp_storage
from ..config.config_manager import initialize_config_manager
from ..models.model_factory import initialize_model_factory
from ..services.hitl_manager import get_hitl_manager
from ..models.hitl_config import HITLConfig


# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler("logs/core.log", encoding="utf-8"),  # File
    ],
)
logger = logging.getLogger(__name__)

# Global manager instance
graph_manager: Optional[GraphManager] = None


class ProcessRequest(BaseModel):
    """Request model for processing"""

    thread_id: Optional[str] = Field(
        default=None, description="Thread ID (optional)"
    )
    message: str = Field(..., description="Message for processing")
    image_paths: Optional[List[str]] = Field(
        default=None, description="Paths to uploaded images (optional)"
    )


class ProcessResponse(BaseModel):
    """Processing response model"""

    thread_id: str = Field(..., description="Thread ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    result: Any = Field(..., description="Processing result")


class UploadResponse(BaseModel):
    """Image upload response model"""

    thread_id: str = Field(..., description="Thread ID")
    uploaded_files: List[str] = Field(..., description="Paths to uploaded files")
    message: str = Field(..., description="Result message")


class StateResponse(BaseModel):
    """State response model"""

    thread_id: str = Field(..., description="Thread ID")
    state: Optional[Dict[str, Any]] = Field(
        default=None, description="Current state"
    )
    current_step: Dict[str, Any] = Field(..., description="Current step")


class NodeSettingRequest(BaseModel):
    """Request model for updating node setting"""

    enabled: bool = Field(..., description="Enable/disable HITL for node")


class BulkUpdateRequest(BaseModel):
    """Request model for bulk HITL update"""

    enable_all: bool = Field(..., description="Enable/disable HITL for all nodes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global graph_manager

    logger.info("Starting core AI service...")

    # Initialize settings
    settings = get_settings()

    # Initialize configuration manager
    try:
        config_manager = initialize_config_manager(settings.graph_config_path)
        logger.info(f"Graph configuration loaded from {settings.graph_config_path}")
    except Exception as e:
        logger.error(f"Failed to load graph configuration: {e}")
        raise

    # Initialize model factory
    try:
        initialize_model_factory(settings.openai_api_key, config_manager)
        logger.info("Model factory initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model factory: {e}")
        raise

    # Create temporary directories
    ensure_temp_storage()
    logger.info("Temporary storage initialized")

    # Check LangFuse connection
    #try:
        #langfuse = Langfuse()
       # if langfuse.auth_check():
           # logger.info("LangFuse client authenticated successfully")
       #     else:
       #         logger.warning("LangFuse authentication failed")
   # except Exception as e:
   #         logger.warning(f"LangFuse initialization error: {e}")

    # Initialize GraphManager
    graph_manager = GraphManager()
    logger.info("GraphManager initialized successfully")

    yield

    logger.info("Shutting down core AI service...")

    # Clean up temporary files on shutdown
    # Can add cleanup logic here if needed


# Create FastAPI application
app = FastAPI(
    title="core AI",
    description="Educational materials preparation system based on LangGraph with image support",
    version="1.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Service health check"""
    return {
        "message": "core AI service is running",
        "status": "ok",
        "features": ["text_processing", "image_recognition"],
    }


@app.get("/health")
async def health_check():
    """Health probe for monitoring"""
    try:
        # Check GraphManager availability
        if graph_manager is None:
            raise HTTPException(status_code=503, detail="GraphManager not initialized")

        return {"status": "healthy", "service": "core-ai"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/upload-images/{thread_id}", response_model=UploadResponse)
async def upload_images(thread_id: str, files: List[UploadFile] = File(...)):
    """
    Upload images for thread_id.

    Args:
        thread_id: Thread ID
        files: List of uploaded image files

    Returns:
        UploadResponse: Information about uploaded files

    Raises:
        HTTPException: On upload or validation errors
    """
    try:
        logger.info(f"Uploading {len(files)} images for thread {thread_id}")

        # Check number of files
        settings = get_settings()
        if len(files) > settings.max_images_per_request:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files: {len(files)} > {settings.max_images_per_request}",
            )

        # Check each file
        image_data_list = []
        for file in files:
            # Check file type
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.content_type}. Only images are allowed.",
                )

            # Read file content
            content = await file.read()

            # Check size
            if len(content) > settings.max_image_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} too large: {len(content)} > {settings.max_image_size}",
                )

            image_data_list.append(content)

        # Save images
        file_manager = ImageFileManager()
        saved_paths = file_manager.save_uploaded_images(thread_id, image_data_list)

        logger.info(
            f"Successfully uploaded {len(saved_paths)} images for thread {thread_id}"
        )

        return UploadResponse(
            thread_id=thread_id,
            uploaded_files=saved_paths,
            message=f"Successfully uploaded {len(saved_paths)} images",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading images for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/process", response_model=ProcessResponse)
async def process_request(
    question: str = Form(..., description="Question or task for processing"),
    settings: Optional[str] = Form(None, description="JSON processing settings"),
    thread_id: Optional[str] = Form(None, description="Thread ID (optional)"),
    images: Optional[List[UploadFile]] = File(None, description="Images for processing"),
    wallet_address: Optional[str] = Form(None, description="User wallet address"),
    user_id: Optional[str] = Form(None, description="User ID")
):
    """
    Universal endpoint for processing educational material.
    Supports all scenarios: with images, text notes, and without them.

    Args:
        question: Question or task for creating educational material
        settings: JSON settings (difficulty, subject, volume, enableHITL, etc.)
        thread_id: Thread ID (optional)
        images: List of images for processing (optional)

    Returns:
        ProcessResponse: Processing result with thread_id

    Raises:
        HTTPException: On processing errors
    """
    if graph_manager is None:
        raise HTTPException(status_code=503, detail="GraphManager not available")

    try:
        logger.info(f"Processing request for thread: {thread_id}, question length: {len(question)}")
        
        # Parse settings if provided
        parsed_settings = None
        if settings:
            try:
                parsed_settings = json.loads(settings)
                logger.info(f"Parsed settings: {parsed_settings}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse settings JSON: {e}")
                raise HTTPException(status_code=400, detail="Invalid settings JSON format")
        
        # Process images if they exist
        image_paths = None
        if images and len(images) > 0:
            logger.info(f"Processing {len(images)} uploaded images")
            
            # Check number of files
            settings_obj = get_settings()
            if len(images) > settings_obj.max_images_per_request:
                raise HTTPException(
                    status_code=400,
                    detail=f"Too many files: {len(images)} > {settings_obj.max_images_per_request}",
                )
            
            # Generate thread_id if not provided
            if not thread_id:
                import uuid
                thread_id = str(uuid.uuid4())
                logger.info(f"Generated new thread_id: {thread_id}")
            
            # Save images
            image_data_list = []
            for image_file in images:
                # Check file type
                if not image_file.content_type or not image_file.content_type.startswith("image/"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid file type: {image_file.content_type}. Only images are allowed.",
                    )
                
                # Read content
                content = await image_file.read()
                
                # Check size
                if len(content) > settings_obj.max_image_size:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File {image_file.filename} too large: {len(content)} > {settings_obj.max_image_size}",
                    )
                
                image_data_list.append(content)
            
            # Save images
            file_manager = ImageFileManager()
            image_paths = file_manager.save_uploaded_images(thread_id, image_data_list)
            logger.info(f"Saved {len(image_paths)} images for processing")

        logger.info(f"🔍 [core_API] Processing request with wallet_address: {wallet_address}")
        
        result = await graph_manager.process_step(
            thread_id=thread_id or "", 
            query=question,
            image_paths=image_paths,
            wallet_address=wallet_address,
            user_id=user_id
        )

        return ProcessResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/state/{thread_id}", response_model=StateResponse)
async def get_state(thread_id: str):
    """
    Get current thread state.

    Args:
        thread_id: Thread ID

    Returns:
        StateResponse: Current state and step

    Raises:
        HTTPException: On state retrieval errors
    """
    if graph_manager is None:
        raise HTTPException(status_code=503, detail="GraphManager not available")

    try:
        # Get state and current step
        state = await graph_manager.get_thread_state(thread_id)
        current_step = await graph_manager.get_current_step(thread_id)

        return StateResponse(
            thread_id=thread_id, state=state, current_step=current_step
        )

    except Exception as e:
        logger.error(f"Error getting state for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=f"State retrieval error: {str(e)}")


@app.delete("/thread/{thread_id}")
async def delete_thread(thread_id: str):
    """
    Delete thread and all associated data.

    Args:
        thread_id: Thread ID to delete

    Returns:
        Dict: Deletion result

    Raises:
        HTTPException: On deletion errors
    """
    if graph_manager is None:
        raise HTTPException(status_code=503, detail="GraphManager not available")

    try:
        await graph_manager.delete_thread(thread_id)

        # Clean up temporary files for this thread
        try:
            file_manager = ImageFileManager()
            file_manager.cleanup_temp_directory(thread_id)
        except Exception as cleanup_error:
            logger.warning(
                f"Failed to cleanup temp files for thread {thread_id}: {cleanup_error}"
            )

        logger.info(f"Thread {thread_id} deleted successfully")

        return {"message": f"Thread {thread_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion error: {str(e)}")


# HITL Configuration API Endpoints


@app.get("/api/hitl/{thread_id}", response_model=HITLConfig)
async def get_hitl_config(thread_id: str):
    """
    Get current HITL configuration for thread

    Args:
        thread_id: User thread ID

    Returns:
        HITLConfig: Current HITL configuration
    """
    try:
        hitl_manager = get_hitl_manager()
        config = hitl_manager.get_config(thread_id)
        logger.info(f"Retrieved HITL config for thread {thread_id}: {config.to_dict()}")
        return config

    except Exception as e:
        logger.error(f"Error getting HITL config for thread {thread_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get HITL config: {str(e)}"
        )


@app.put("/api/hitl/{thread_id}", response_model=HITLConfig)
async def set_hitl_config(thread_id: str, config: HITLConfig):
    """
    Set full HITL configuration for thread

    Args:
        thread_id: User thread ID
        config: New HITL configuration

    Returns:
        HITLConfig: Set HITL configuration
    """
    try:
        hitl_manager = get_hitl_manager()
        hitl_manager.set_config(thread_id, config)
        logger.info(f"Set HITL config for thread {thread_id}: {config.to_dict()}")
        return config

    except Exception as e:
        logger.error(f"Error setting HITL config for thread {thread_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to set HITL config: {str(e)}"
        )


@app.patch("/api/hitl/{thread_id}/node/{node_name}", response_model=HITLConfig)
async def update_node_hitl(thread_id: str, node_name: str, request: NodeSettingRequest):
    """
    Update HITL setting for specific node

    Args:
        thread_id: User thread ID
        node_name: Node name
        request: Request with new setting

    Returns:
        HITLConfig: Updated HITL configuration
    """
    try:
        hitl_manager = get_hitl_manager()
        updated_config = hitl_manager.update_node_setting(
            thread_id, node_name, request.enabled
        )
        logger.info(
            f"Updated node {node_name} to {request.enabled} for thread {thread_id}"
        )
        return updated_config

    except Exception as e:
        logger.error(f"Error updating node {node_name} for thread {thread_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update node setting: {str(e)}"
        )


@app.post("/api/hitl/{thread_id}/reset", response_model=HITLConfig)
async def reset_hitl_config(thread_id: str):
    """
    Reset configuration to default values

    Args:
        thread_id: User thread ID

    Returns:
        HITLConfig: Reset HITL configuration
    """
    try:
        hitl_manager = get_hitl_manager()
        hitl_manager.reset_config(thread_id)
        config = hitl_manager.get_config(thread_id)
        logger.info(f"Reset HITL config for thread {thread_id}")
        return config

    except Exception as e:
        logger.error(f"Error resetting HITL config for thread {thread_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to reset HITL config: {str(e)}"
        )


@app.post("/api/hitl/{thread_id}/bulk", response_model=HITLConfig)
async def bulk_update_hitl(thread_id: str, request: BulkUpdateRequest):
    """
    Enable or disable HITL for all nodes

    Args:
        thread_id: User thread ID
        request: Request with flag for all nodes

    Returns:
        HITLConfig: Updated HITL configuration
    """
    try:
        hitl_manager = get_hitl_manager()
        updated_config = hitl_manager.bulk_update(thread_id, request.enable_all)
        logger.info(f"Bulk updated HITL to {request.enable_all} for thread {thread_id}")
        return updated_config

    except Exception as e:
        logger.error(f"Error bulk updating HITL for thread {thread_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to bulk update HITL: {str(e)}"
        )


@app.get("/api/hitl/debug/all-configs")
async def get_all_hitl_configs():
    """
    Get all HITL configurations (for debugging)

    Returns:
        Dict: All HITL configurations by thread_id
    """
    try:
        hitl_manager = get_hitl_manager()
        all_configs = hitl_manager.get_all_configs()
        # Convert HITLConfig objects to dict for JSON serialization
        serialized_configs = {
            thread_id: config.to_dict() for thread_id, config in all_configs.items()
        }
        logger.info(f"Retrieved all HITL configs: {len(serialized_configs)} threads")
        return {"configs": serialized_configs}

    except Exception as e:
        logger.error(f"Error getting all HITL configs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get all configs: {str(e)}"
        )


# Materials API Endpoints


@app.get("/api/materials/{thread_id}/session/{session_id}")
async def get_material_info(thread_id: str, session_id: str):
    """
    Get session information and available material files

    Args:
        thread_id: Thread ID
        session_id: Session ID

    Returns:
        Dict with session metadata
    
    Raises:
        HTTPException: On errors or if session not found
    """
    if graph_manager is None:
        raise HTTPException(status_code=503, detail="GraphManager not available")

    try:
        material_info = graph_manager.get_material_info(thread_id, session_id)
        
        if material_info is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Material not found for thread {thread_id}, session {session_id}"
            )
        
        logger.info(f"Retrieved material info for thread {thread_id}, session {session_id}")
        return material_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting material info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get material info: {str(e)}")


@app.get("/api/materials/{thread_id}/session/{session_id}/file/{file_name}")
async def get_material_file(thread_id: str, session_id: str, file_name: str):
    """
    Get material file content

    Args:
        thread_id: Thread ID
        session_id: Session ID
        file_name: File name (e.g., generated_material.md)

    Returns:
        File content in text or JSON format
    
    Raises:
        HTTPException: On errors or if file not found
    """
    if graph_manager is None:
        raise HTTPException(status_code=503, detail="GraphManager not available")

    try:
        content = graph_manager.get_material_content(thread_id, session_id, file_name)
        
        if content is None:
            raise HTTPException(
                status_code=404, 
                detail=f"File {file_name} not found for thread {thread_id}, session {session_id}"
            )
        
        logger.info(f"Retrieved material file {file_name} for thread {thread_id}, session {session_id}")
        
        # Return content as JSON with content field
        return {
            "thread_id": thread_id,
            "session_id": session_id,
            "file_name": file_name,
            "content": content
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting material file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get material file: {str(e)}")


@app.get("/api/materials/{thread_id}/sessions")
async def get_thread_sessions(thread_id: str):
    """
    Get list of all sessions for thread_id

    Args:
        thread_id: Thread ID

    Returns:
        List of sessions with metadata
    
    Raises:
        HTTPException: On errors or if thread not found
    """
    if graph_manager is None:
        raise HTTPException(status_code=503, detail="GraphManager not available")

    try:
        sessions = graph_manager.get_thread_sessions(thread_id)
        
        if sessions is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Thread {thread_id} not found"
            )
        
        logger.info(f"Retrieved {len(sessions)} sessions for thread {thread_id}")
        
        return {
            "thread_id": thread_id,
            "sessions": sessions,
            "count": len(sessions)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get thread sessions: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "core.api.main:app", 
        host=settings.host, 
        port=settings.port
    )
