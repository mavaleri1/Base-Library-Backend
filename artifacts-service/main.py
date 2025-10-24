"""FastAPI application for Artifacts Service."""

import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, status, Path as PathParam, Query, Depends, Form, UploadFile, File
from fastapi.responses import PlainTextResponse, JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import httpx

from storage import ArtifactsStorage
from auth import auth_service, require_auth, verify_resource_owner
from auth_models_api import AuthCodeRequest, AuthTokenResponse
from web3_auth import router as web3_router, get_current_user, get_db
from models_web3 import Material, User
from models import (
    HealthResponse,
    ThreadsListResponse,
    ThreadDetailResponse,
    SessionFilesResponse,
    FileContent,
    FileOperationResponse,
    ErrorResponse,
    ExportFormat,
    PackageType,
    ExportSettings,
    SessionSummary,
    ExportRequest,
    MaterialUpdateRequest,
    MaterialDeleteResponse,
    LeaderboardEntry,
    LeaderboardResponse,
)
from exceptions import ArtifactsServiceException, map_to_http_exception
from settings import settings
from services.export import MarkdownExporter, PDFExporter, ZIPExporter
from services.permissions import (
    check_material_ownership,
    can_view_material,
    can_edit_material,
    get_material_with_permissions
)
from services.content_hash import calculate_content_hash, ContentHashManager
from services.ownership import check_material_ownership, get_nft_status, sync_material_with_blockchain
from sqlalchemy import func, select, and_, or_, desc, case
from sqlalchemy.ext.asyncio import AsyncSession
import re

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler(log_dir / "artifacts.log", encoding="utf-8")  # File
    ]
)

# Global storage instance
storage = ArtifactsStorage()

# Logger instance
logger = logging.getLogger(__name__)


def validate_content_hash(content_hash: str) -> bool:
    """Validate SHA-256 hash"""
    if not content_hash:
        return False
    
    # SHA-256 should be 64 hex characters
    pattern = r'^[a-fA-F0-9]{64}$'
    return bool(re.match(pattern, content_hash))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    # Ensure data directory exists
    storage.base_path.mkdir(parents=True, exist_ok=True)
    # Connect auth service
    await auth_service.connect()
    yield
    # Shutdown - cleanup if needed
    await auth_service.disconnect()


# FastAPI application
app = FastAPI(
    title="Artifacts Service",
    description="File storage system for core AI artifacts",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for Web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",  # Vite preview
        "http://localhost:3001",  # Alternative dev port
        "http://127.0.0.1:5174",  # IP access preview
        "http://127.0.0.1:3000",  # IP access alternative
        "http://127.0.0.1:3001",  # IP access alternative
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Web3 authentication router
app.include_router(web3_router)


@app.exception_handler(ArtifactsServiceException)
async def service_exception_handler(request, exc: ArtifactsServiceException):
    """Handle service exceptions."""
    http_exc = map_to_http_exception(exc)
    return JSONResponse(
        status_code=http_exc.status_code,
        content=ErrorResponse(error=str(exc)).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if storage is accessible
        storage.base_path.exists()
        return HealthResponse(status="ok")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {str(e)}",
        )


@app.post("/auth/verify", response_model=AuthTokenResponse)
async def verify_auth_code(request: AuthCodeRequest):
    """Verify auth code and return JWT token."""
    if not auth_service.pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    
    user_id = await auth_service.verify_auth_code(request.username, request.code)
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired auth code"
        )
    
    # Create JWT token
    token = auth_service.create_jwt_token(user_id, request.username)
    
    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.jwt_expiration_minutes * 60
    )


@app.get("/threads", response_model=ThreadsListResponse, dependencies=[Depends(require_auth)])
async def get_threads():
    """Get list of all threads."""
    try:
        threads = storage.get_threads()
        return ThreadsListResponse(threads=threads)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get threads: {str(e)}",
        )


@app.get("/threads/{thread_id}", response_model=ThreadDetailResponse)
async def get_thread(
    thread_id: str = PathParam(description="Thread identifier"),
    user_id: str = Depends(verify_resource_owner)
):
    """Get information about a specific thread."""
    try:
        thread_info = storage.get_thread_info(thread_id)
        return ThreadDetailResponse(
            thread_id=thread_info.thread_id,
            sessions=thread_info.sessions,
            created=thread_info.created,
            last_activity=thread_info.last_activity,
            sessions_count=thread_info.sessions_count,
        )
    except ArtifactsServiceException as e:
        raise map_to_http_exception(e)


@app.get(
    "/threads/{thread_id}/sessions/{session_id}", response_model=SessionFilesResponse
)
async def get_session_files(
    thread_id: str = PathParam(description="Thread identifier"),
    session_id: str = PathParam(description="Session identifier"),
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get list of files in a session."""
    try:
        # Check if user has access to this material
        # Find material by thread_id
        result = await db.execute(
            select(Material).where(Material.thread_id == thread_id)
        )
        material = result.scalar_one_or_none()
        
        if material:
            # Check ownership
            if not await check_material_ownership(str(material.id), current_user.wallet_address, db):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access to this resource is forbidden"
                )
        
        files = storage.get_session_files(thread_id, session_id)
        return SessionFilesResponse(
            thread_id=thread_id, session_id=session_id, files=files
        )
    except ArtifactsServiceException as e:
        raise map_to_http_exception(e)


@app.get("/threads/{thread_id}/sessions/{session_id}/files/{file_path:path}")
async def get_file(
    thread_id: str = PathParam(description="Thread identifier"),
    session_id: str = PathParam(description="Session identifier"),
    file_path: str = PathParam(description="File path relative to session"),
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get file content."""
    try:
        # Check if user has access to this material
        # Find material by thread_id
        result = await db.execute(
            select(Material).where(Material.thread_id == thread_id)
        )
        material = result.scalar_one_or_none()
        
        if material:
            # Check ownership
            if not await check_material_ownership(str(material.id), current_user.wallet_address, db):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access to this resource is forbidden"
                )
        
        content = storage.read_file(thread_id, session_id, file_path)

        # Determine response type based on file extension
        if file_path.endswith(".json"):
            return JSONResponse(content=content)
        else:
            return PlainTextResponse(content=content, media_type="text/plain")

    except ArtifactsServiceException as e:
        raise map_to_http_exception(e)


@app.post(
    "/threads/{thread_id}/sessions/{session_id}/files/{file_path:path}",
    response_model=FileOperationResponse,
)
async def create_or_update_file(
    file_content: FileContent,
    thread_id: str = PathParam(description="Thread identifier"),
    session_id: str = PathParam(description="Session identifier"),
    file_path: str = PathParam(description="File path relative to session"),
    user_id: str = Depends(verify_resource_owner)
):
    """Create or update a file in a session."""
    try:
        # Check if file already exists
        file_exists = False
        try:
            storage.read_file(thread_id, session_id, file_path)
            file_exists = True
        except Exception as e:
            logger.debug(f"File {file_path} not found, will create new: {e}")

        # Write the file
        storage.write_file(
            thread_id=thread_id,
            session_id=session_id,
            path=file_path,
            content=file_content.content,
            content_type=file_content.content_type,
        )

        if file_exists:
            return FileOperationResponse(message="File updated", path=file_path)
        else:
            return FileOperationResponse(message="File created", path=file_path)

    except ArtifactsServiceException as e:
        raise map_to_http_exception(e)


@app.delete(
    "/threads/{thread_id}/sessions/{session_id}/files/{file_path:path}",
    response_model=FileOperationResponse,
)
async def delete_file(
    thread_id: str = PathParam(description="Thread identifier"),
    session_id: str = PathParam(description="Session identifier"),
    file_path: str = PathParam(description="File path relative to session"),
    user_id: str = Depends(verify_resource_owner)
):
    """Delete a file from a session."""
    try:
        storage.delete_file(thread_id, session_id, file_path)
        return FileOperationResponse(message="File deleted", path=file_path)
    except ArtifactsServiceException as e:
        raise map_to_http_exception(e)


@app.delete(
    "/threads/{thread_id}/sessions/{session_id}", response_model=FileOperationResponse
)
async def delete_session(
    thread_id: str = PathParam(description="Thread identifier"),
    session_id: str = PathParam(description="Session identifier"),
    user_id: str = Depends(verify_resource_owner)
):
    """Delete an entire session with all files."""
    try:
        storage.delete_session(thread_id, session_id)
        return FileOperationResponse(message="Session deleted")
    except ArtifactsServiceException as e:
        raise map_to_http_exception(e)


@app.delete("/threads/{thread_id}", response_model=FileOperationResponse)
async def delete_thread(
    thread_id: str = PathParam(description="Thread identifier"),
    user_id: str = Depends(verify_resource_owner)
):
    """Delete an entire thread with all sessions."""
    try:
        storage.delete_thread(thread_id)
        return FileOperationResponse(message="Thread deleted")
    except ArtifactsServiceException as e:
        raise map_to_http_exception(e)


# Export API Endpoints
# NOTE: These MUST be defined BEFORE the generic file path routes below
# to avoid route conflicts

# Store user settings in memory (in production, use a database)
user_settings = {}


@app.get("/threads/{thread_id}/sessions/{session_id}/export/single")
async def export_single_document(
    thread_id: str = PathParam(description="Thread identifier"),
    session_id: str = PathParam(description="Session identifier"),
    document_name: str = Query(description="Document name to export"),
    format: ExportFormat = Query(ExportFormat.MARKDOWN, description="Export format"),
    user_id: str = Depends(verify_resource_owner)
):
    """Export a single document."""
    try:
        # Select exporter based on format
        if format == ExportFormat.PDF:
            exporter = PDFExporter(storage.base_path)
        else:
            exporter = MarkdownExporter(storage.base_path)
        
        # Export document
        content = await exporter.export_single_document(
            thread_id, session_id, document_name, format
        )
        
        # Determine file extension and mime type
        if format == ExportFormat.PDF:
            ext = "pdf"
            media_type = "application/pdf"
        else:
            ext = "md"
            media_type = "text/markdown"
        
        # Format filename
        filename = exporter.format_filename(document_name, session_id, ext)
        
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}"
        )


@app.get("/threads/{thread_id}/sessions/{session_id}/export/package")
async def export_package(
    thread_id: str = PathParam(description="Thread identifier"),
    session_id: str = PathParam(description="Session identifier"),
    package_type: PackageType = Query(PackageType.FINAL, description="Package type"),
    format: ExportFormat = Query(ExportFormat.MARKDOWN, description="Export format"),
    user_id: str = Depends(verify_resource_owner)
):
    """Export a package of documents as ZIP archive."""
    logger.info(f"Export package request: thread_id={thread_id}, session_id={session_id}, package_type={package_type}, format={format}, user_id={user_id}")
    
    try:
        # Log storage base path
        logger.debug(f"Storage base path: {storage.base_path}")
        
        # Check if session exists
        session_path = storage.base_path / thread_id / "sessions" / session_id
        logger.debug(f"Checking session path: {session_path}")
        
        if not session_path.exists():
            logger.error(f"Session path does not exist: {session_path}")
            raise FileNotFoundError(f"Session not found: {session_id}")
        
        # List files in session
        files = list(session_path.iterdir())
        logger.debug(f"Files in session: {[f.name for f in files]}")
        
        # Use ZIP exporter
        zip_exporter = ZIPExporter(storage.base_path)
        logger.debug(f"Created ZIPExporter with base_path: {storage.base_path}")
        
        # Export package
        content = await zip_exporter.export_session_archive(
            thread_id, session_id, package_type, format
        )
        logger.info(f"Successfully exported package, size: {len(content)} bytes")
        
        # Format filename
        filename = f"session_{session_id}_export.zip"
        
        return Response(
            content=content,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except FileNotFoundError as e:
        logger.error(f"File not found during export: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Export failed with exception: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}"
        )


@app.get("/users/{user_id}/sessions/recent", response_model=List[SessionSummary])
async def get_recent_sessions(
    user_id: str = PathParam(description="User identifier"),
    limit: int = Query(5, max=5, description="Maximum number of sessions"),
    auth_user_id: str = Depends(require_auth)
):
    """Get list of recent sessions for export."""
    # Verify user can only access their own sessions
    if user_id != auth_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to other users' sessions is forbidden"
        )
    
    try:
        # Get user's thread (thread_id equals user_id in our system)
        threads_to_check = []
        try:
            thread_info = storage.get_thread_info(user_id)
            threads_to_check = [thread_info]
        except:
            # User has no threads yet
            return []
        
        sessions_list = []
        for thread in threads_to_check:
            for session in thread.sessions[:limit]:
                # Create session summary
                summary = SessionSummary(
                    thread_id=thread.thread_id,
                    session_id=session.session_id,
                    input_content=session.input_content,
                    question_preview=session.input_content[:30] + "..." 
                        if len(session.input_content) > 30 else session.input_content,
                    display_name=f"{session.input_content[:30]}... - {session.created.strftime('%d.%m.%Y')}",
                    created_at=session.created,
                    has_synthesized=False,  # Check if synthesized_material.md exists
                    has_questions=False,     # Check if questions.md exists
                    answers_count=0          # Count answer files
                )
                sessions_list.append(summary)
                
                if len(sessions_list) >= limit:
                    break
        
        return sessions_list
    except Exception as e:
        logger.error(f"Failed to get recent sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recent sessions: {str(e)}"
        )


@app.get("/users/{user_id}/export-settings", response_model=ExportSettings)
async def get_export_settings(
    user_id: str = PathParam(description="User identifier"),
    auth_user_id: str = Depends(require_auth)
):
    """Get user export settings."""
    # Verify user can only access their own settings
    if user_id != auth_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to other users' settings is forbidden"
        )
    
    # Return existing settings or create default
    if user_id not in user_settings:
        user_settings[user_id] = ExportSettings(user_id=user_id)
    
    return user_settings[user_id]


@app.put("/users/{user_id}/export-settings", response_model=ExportSettings)
async def update_export_settings(
    user_id: str = PathParam(description="User identifier"),
    settings: ExportSettings = ...,
    auth_user_id: str = Depends(require_auth)
):
    """Update user export settings."""
    # Verify user can only update their own settings
    if user_id != auth_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to other users' settings is forbidden"
        )
    
    settings.user_id = user_id
    settings.modified = datetime.now()
    user_settings[user_id] = settings
    return settings


# =============================================================================
# Materials API - get user materials with blockchain metadata
# =============================================================================

class MaterialResponse(BaseModel):
    """Response model for a single material."""
    id: str
    subject: Optional[str]
    grade: Optional[str]
    topic: Optional[str]
    title: Optional[str]
    content_hash: str
    ipfs_cid: Optional[str]
    word_count: Optional[int]
    status: str
    created_at: str
    updated_at: str
    author_wallet: str
    thread_id: str
    session_id: str
    file_path: str
    # NFT fields
    nft_minted: bool
    nft_token_id: Optional[int]
    is_owner: bool


class MaterialsListResponse(BaseModel):
    """Response model for list of materials."""
    materials: List[MaterialResponse]
    total: int
    page: int
    page_size: int


# NFT API Models
class OwnershipResponse(BaseModel):
    """Response model for material ownership check."""
    isOwner: bool
    nftMinted: bool
    tokenId: Optional[int]
    canMint: bool


class NFTStatusResponse(BaseModel):
    """Response model for NFT status."""
    nftMinted: bool
    tokenId: Optional[int]
    txHash: Optional[str]
    ipfsCid: Optional[str]


class BlockchainSyncRequest(BaseModel):
    """Request model for blockchain sync."""
    ipfsCid: str
    contentHash: str
    txHash: str
    tokenId: int


class BlockchainSyncResponse(BaseModel):
    """Response model for blockchain sync."""
    id: str
    blockchain: dict


class ContentHashCheckResponse(BaseModel):
    """Response model for content hash duplicate check."""
    exists: bool
    tokenId: Optional[int]
    materialId: Optional[str]


class NFTMetadataResponse(BaseModel):
    """Response model for NFT metadata."""
    tokenId: Optional[int]
    ipfsCid: Optional[str]
    contentHash: Optional[str]
    txHash: Optional[str]
    isPublished: bool


class BlockchainStatsResponse(BaseModel):
    """Response model for blockchain statistics."""
    totalNFTs: int
    publishedNFTs: int
    subjects: List[dict]


class UserStatsResponse(BaseModel):
    """Response model for user statistics."""
    totalMaterials: int
    mintedNFTs: int
    publishedMaterials: int
    draftMaterials: int
    subjects: List[dict]


@app.get(
    "/api/materials/my",
    response_model=MaterialsListResponse,
    summary="Get my materials",
    description="Get all materials created by authenticated user with blockchain metadata"
)
async def get_my_materials(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
    subject: Optional[str] = Query(None, description="Filter by subject"),
    grade: Optional[str] = Query(None, description="Filter by grade"),
    status: Optional[str] = Query(None, description="Filter by status"),
    current_user: User = Depends(get_current_user),
):
    """Get materials created by current user."""
    from web3_auth import get_db
    
    async for db in get_db():
        try:
            # Build query
            query = select(Material).where(Material.author_id == current_user.id)
            
            # Apply filters
            if subject:
                query = query.where(Material.subject == subject)
            if grade:
                query = query.where(Material.grade == grade)
            if status:
                query = query.where(Material.status == status)
            
            # Count total
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await db.execute(count_query)
            total = total_result.scalar()
            
            # Apply pagination and ordering
            query = query.order_by(Material.created_at.desc())
            query = query.offset((page - 1) * page_size).limit(page_size)
            
            # Execute query
            result = await db.execute(query)
            materials = result.scalars().all()
            
            # Convert to response model
            materials_response = [
                MaterialResponse(
                    id=str(m.id),
                    subject=m.subject,
                    grade=m.grade,
                    topic=m.topic,
                    title=m.title,
                    content_hash=m.content_hash,
                    ipfs_cid=m.ipfs_cid,
                    word_count=m.word_count,
                    status=m.status,
                    created_at=m.created_at.isoformat(),
                    updated_at=m.updated_at.isoformat(),
                    author_wallet=current_user.wallet_address,
                    thread_id=m.thread_id,
                    session_id=m.session_id,
                    file_path=m.file_path,
                    # NFT fields
                    nft_minted=m.nft_minted or False,
                    nft_token_id=m.nft_token_id,
                    is_owner=True  # User is always owner of their own materials
                )
                for m in materials
            ]
            
            return MaterialsListResponse(
                materials=materials_response,
                total=total,
                page=page,
                page_size=page_size
            )
            
        except Exception as e:
            logger.error(f"Error fetching materials: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch materials"
            )


@app.get(
    "/api/materials/all",
    response_model=MaterialsListResponse,
    summary="Get all published materials",
    description="Get all published materials from all users (public access)"
)
async def get_all_materials(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    subject: Optional[str] = Query(None, description="Filter by subject"),
    grade: Optional[str] = Query(None, description="Filter by grade"),
    status: Optional[str] = Query("published", description="Filter by status"),
):
    """Get all published materials (public endpoint)."""
    async for db in get_db():
        try:
            # Build query - only published materials by default
            query = select(Material).where(Material.status == status)
            
            # Apply filters
            if subject:
                query = query.where(Material.subject == subject)
            if grade:
                query = query.where(Material.grade == grade)
            
            # Count total
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await db.execute(count_query)
            total = total_result.scalar()
            
            # Apply pagination and ordering
            query = query.order_by(Material.created_at.desc())
            query = query.offset((page - 1) * page_size).limit(page_size)
            
            # Execute query
            result = await db.execute(query)
            materials = result.scalars().all()
            
            # Get materials with author wallets
            materials_response = []
            for m in materials:
                # Get author wallet
                author_result = await db.execute(
                    select(User).where(User.id == m.author_id)
                )
                author = author_result.scalar_one_or_none()
                author_wallet = author.wallet_address if author else "unknown"
                
                materials_response.append(
                    MaterialResponse(
                        id=str(m.id),
                        subject=m.subject,
                        grade=m.grade,
                        topic=m.topic,
                        title=m.title,
                        content_hash=m.content_hash,
                        ipfs_cid=m.ipfs_cid,
                        word_count=m.word_count,
                        status=m.status,
                        created_at=m.created_at.isoformat(),
                        updated_at=m.updated_at.isoformat(),
                        author_wallet=author_wallet,
                        thread_id=m.thread_id,
                        session_id=m.session_id,
                        file_path=m.file_path,
                        # NFT fields
                        nft_minted=m.nft_minted or False,
                        nft_token_id=m.nft_token_id,
                        is_owner=False  # Public endpoint, not user's own materials
                    )
                )
            
            return MaterialsListResponse(
                materials=materials_response,
                total=total,
                page=page,
                page_size=page_size
            )
            
        except Exception as e:
            logger.error(f"Error fetching all materials: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch materials"
            )


@app.get(
    "/api/materials/leaderboard",
    response_model=LeaderboardResponse,
    summary="Get materials leaderboard",
    description="Get leaderboard of users by number of created materials and NFTs"
)
async def get_materials_leaderboard(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
):
    """Get leaderboard of users by number of created materials and NFTs."""
    async for db in get_db():
        try:
            # SQL query to get leaderboard data
            # First get total number of users with materials
            total_users_result = await db.execute(
                select(func.count(func.distinct(Material.author_id)))
                .where(Material.id.isnot(None))
            )
            total_users = total_users_result.scalar()
            
            # Main query for leaderboard
            leaderboard_query = select(
                User.id,
                User.wallet_address,
                func.count(Material.id).label('materials_count'),
                #func.count(func.case((Material.nft_minted == True, 1), else_=None)).label('nft_count'),
                func.count(case((Material.nft_minted == True, 1))).label('nft_count'),
                (func.count(Material.id) + func.count(case((Material.nft_minted == True, 1)))).label('total_score')
            ).select_from(
                User.__table__.join(Material.__table__, User.id == Material.author_id)
            ).group_by(
                User.id, User.wallet_address
            ).order_by(
                func.count(Material.id) + func.count(case((Material.nft_minted == True, 1))).desc(),
                func.count(Material.id).desc()
            ).offset((page - 1) * page_size).limit(page_size)
            
            # Execute query
            result = await db.execute(leaderboard_query)
            leaderboard_data = result.fetchall()
            
            # Form response
            entries = []
            for rank_offset, row in enumerate(leaderboard_data):
                rank = (page - 1) * page_size + rank_offset + 1
                entries.append(LeaderboardEntry(
                    rank=rank,
                    walletAddress=row.wallet_address,
                    materialsCount=row.materials_count,
                    nftCount=row.nft_count,
                    totalScore=row.total_score
                ))
            
            return LeaderboardResponse(
                entries=entries,
                total=total_users,
                page=page,
                page_size=page_size
            )
            
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get leaderboard"
            )


@app.get(
    "/api/materials/{material_id}",
    summary="Get material by ID",
    description="Get specific material by ID (public for published, owner-only for drafts)"
)
async def get_material(
    material_id: str = PathParam(description="Material UUID"),
    include_content: bool = Query(True, description="Include full content in response"),
    current_user: Optional[User] = Depends(lambda: None),  # Optional auth
):
    """Get specific material by ID."""
    async for db in get_db():
        try:
            # Get material
            result = await db.execute(
                select(Material).where(Material.id == material_id)
            )
            material = result.scalar_one_or_none()
            
            if not material:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Material not found"
                )
            
            # Check viewing permissions
            user_wallet = current_user.wallet_address if current_user else None
            if not await can_view_material(material, user_wallet, db):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have access to this material"
                )
            
            # Get material with permissions
            response_data = await get_material_with_permissions(
                material, user_wallet, db
            )
            
            # Optionally include content
            if include_content:
                response_data["content"] = material.content
            
            return JSONResponse(content=response_data)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching material: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch material"
            )


@app.get(
    "/api/materials/stats/subjects",
    summary="Get subject statistics",
    description="Get statistics grouped by subject for current user"
)
async def get_subject_stats(
    current_user: User = Depends(get_current_user),
):
    """Get materials statistics by subject."""
    async for db in get_db():
        try:
            # Query materials grouped by subject
            result = await db.execute(
                select(
                    Material.subject,
                    func.count(Material.id).label("count")
                )
                .where(Material.author_id == current_user.id)
                .group_by(Material.subject)
                .order_by(func.count(Material.id).desc())
            )
            
            stats = [
                {"subject": row.subject or "Unknown", "count": row.count}
                for row in result
            ]
            
            return {"subjects": stats}
            
        except Exception as e:
            logger.error(f"Error fetching subject stats: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch statistics"
            )


@app.patch(
    "/api/materials/{material_id}",
    summary="Update material",
    description="Update existing material (only by owner)"
)
async def update_material(
    material_id: str = PathParam(description="Material UUID"),
    updates: MaterialUpdateRequest = ...,
    current_user: User = Depends(get_current_user),
):
    """Update material (only owner can edit)."""
    async for db in get_db():
        try:
            # Get material
            result = await db.execute(
                select(Material).where(Material.id == material_id)
            )
            material = result.scalar_one_or_none()
            
            if not material:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Material not found"
                )
            
            # Check ownership
            if not await check_material_ownership(material_id, current_user.wallet_address, db):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to edit this material"
                )
            
            # Validate status if provided
            if updates.status and updates.status not in ["draft", "published", "archived"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid status. Must be: draft, published, or archived"
                )
            
            # Update fields
            updated_fields = []
            
            if updates.title is not None:
                material.title = updates.title
                updated_fields.append("title")
            
            if updates.content is not None:
                # Validate content size (max 1MB)
                if len(updates.content.encode('utf-8')) > 1000000:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Content too large (max 1MB)"
                    )
                
                material.content = updates.content
                updated_fields.append("content")
                
                # Recalculate hash and word count
                hash_manager = ContentHashManager()
                material.content_hash = calculate_content_hash(updates.content)
                material.word_count = hash_manager.calculate_word_count(updates.content)
                
                # Update file on disk
                try:
                    file_full_path = storage.base_path / material.file_path
                    if file_full_path.exists():
                        with open(file_full_path, 'w', encoding='utf-8') as f:
                            f.write(updates.content)
                        logger.info(f"Updated material file on disk: {file_full_path}")
                except Exception as file_error:
                    logger.error(f"Failed to update file on disk: {file_error}")
            
            if updates.subject is not None:
                material.subject = updates.subject
                updated_fields.append("subject")
            
            if updates.grade is not None:
                material.grade = updates.grade
                updated_fields.append("grade")
            
            if updates.topic is not None:
                material.topic = updates.topic
                updated_fields.append("topic")
            
            if updates.status is not None:
                material.status = updates.status
                updated_fields.append("status")
            
            # Update timestamp
            material.updated_at = datetime.utcnow()
            
            await db.commit()
            await db.refresh(material)
            
            logger.info(
                f"Material {material_id} updated by {current_user.wallet_address}. "
                f"Updated fields: {', '.join(updated_fields)}"
            )
            
            # Return updated material with permissions
            response_data = await get_material_with_permissions(
                material, current_user.wallet_address, db
            )
            response_data["content"] = material.content
            
            return JSONResponse(content=response_data)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating material: {e}", exc_info=True)
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update material"
            )


@app.delete(
    "/api/materials/{material_id}",
    response_model=MaterialDeleteResponse,
    summary="Delete material",
    description="Delete material (only by owner)"
)
async def delete_material(
    material_id: str = PathParam(description="Material UUID"),
    current_user: User = Depends(get_current_user),
):
    """Delete material (only owner can delete)."""
    async for db in get_db():
        try:
            # Get material
            result = await db.execute(
                select(Material).where(Material.id == material_id)
            )
            material = result.scalar_one_or_none()
            
            if not material:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Material not found"
                )
            
            # Check ownership
            if not await check_material_ownership(material_id, current_user.wallet_address, db):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to delete this material"
                )
            
            # Store file path for deletion
            file_path = material.file_path
            
            # Delete from database
            await db.delete(material)
            await db.commit()
            
            # Optionally delete file from disk
            try:
                file_full_path = storage.base_path / file_path
                if file_full_path.exists():
                    file_full_path.unlink()
                    logger.info(f"Deleted material file from disk: {file_full_path}")
            except Exception as file_error:
                logger.warning(f"Failed to delete file from disk: {file_error}")
                # Don't fail the request if file deletion fails
            
            logger.info(
                f"Material {material_id} deleted by {current_user.wallet_address}"
            )
            
            return MaterialDeleteResponse(
                success=True,
                message="Material deleted successfully"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting material: {e}", exc_info=True)
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete material"
            )




@app.post(
    "/api/materials/bulk-classify",
    response_model=dict,
    summary="Bulk classify materials",
    description="Re-classify existing materials using AI"
)
async def bulk_classify_materials(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of materials to classify"),
    current_user: User = Depends(get_current_user)
):
    """Re-classify existing materials using AI."""
    async for db in get_db():
        try:
            # Get materials that need classification (no subject or generic subject)
            result = await db.execute(
                select(Material)
                .where(
                    (Material.subject.is_(None)) | 
                    (Material.subject == "Other") |
                    (Material.subject == "Unclassified")
                )
                .limit(limit)
                .order_by(Material.created_at.desc())
            )
            materials = result.scalars().all()
            
            if not materials:
                return {
                    "message": "No materials need classification",
                    "classified": 0,
                    "errors": 0
                }
            
            from services.material_classifier import get_classifier_service
            classifier = get_classifier_service()
            classified_count = 0
            error_count = 0
            
            for material in materials:
                try:
                    # Classify material
                    classification = await classifier.classify_material(
                        content=material.content,
                        input_query=material.input_query or ""
                    )
                    
                    # Update material
                    material.subject = classification.subject
                    material.grade = classification.grade
                    material.topic = classification.topic
                    material.updated_at = datetime.utcnow()
                    
                    classified_count += 1
                    logger.info(
                        f"Classified material {material.id}: "
                        f"subject={classification.subject}, "
                        f"grade={classification.grade}, "
                        f"topic={classification.topic}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error classifying material {material.id}: {e}")
                    error_count += 1
            
            # Commit all changes
            await db.commit()
            
            return {
                "message": f"Bulk classification completed",
                "total_processed": len(materials),
                "classified": classified_count,
                "errors": error_count
            }
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error during bulk classification: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to classify materials"
            )


# =============================================================================
# core Proxy API - proxying requests to core with authentication
# =============================================================================

@app.post("/api/process")
async def proxy_to_core(
    question: str = Form(..., description="Question or task for processing"),
    settings: Optional[str] = Form(None, description="JSON processing settings"),
    thread_id: Optional[str] = Form(None, description="Thread ID (optional)"),
    images: Optional[List[UploadFile]] = File(None, description="Images for processing"),
    current_user: User = Depends(get_current_user)
):
    """
    Proxy for redirecting requests to core with authentication.
    
    This endpoint:
    1. Validates JWT token and extracts wallet_address
    2. Redirects request to core with wallet_address
    3. Returns result to frontend
    """
    try:
        logger.info(f"Proxy request to core for user {current_user.wallet_address[:10]}...")
        logger.info(f"Question length: {len(question)}")
        logger.info(f"Thread ID: {thread_id}")
        logger.info(f"Images count: {len(images) if images else 0}")
        
        # Prepare data for core
        core_data = {
            "question": question,
            "wallet_address": current_user.wallet_address,  # Pass wallet_address
            "user_id": str(current_user.id),  # Pass user_id
        }
        
        if settings:
            core_data["settings"] = settings
        if thread_id:
            core_data["thread_id"] = thread_id
        
        logger.info(f"🔍 [PROXY] Sending data to core: {core_data}")
        
        # Send request to core
        core_url = "http://core:8000/process"
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Prepare form data
            form_data = {}
            for key, value in core_data.items():
                form_data[key] = value
            
            # Prepare files for core
            files_data = None
            if images:
                files_data = []
                for image in images:
                    # Read file content
                    content = await image.read()
                    files_data.append(("images", (image.filename, content, image.content_type)))
            
            # Send request
            response = await client.post(
                core_url,
                data=form_data,
                files=files_data
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Successfully proxied request to core for user {current_user.wallet_address[:10]}...")
                return result
            else:
                logger.error(f"core returned error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"core error: {response.text}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to core: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="core service is unavailable"
        )
    except Exception as e:
        logger.error(f"Error in proxy to core: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Proxy error: {str(e)}"
        )


# =============================================================================
# NFT API - Material ownership and NFT status
# =============================================================================

@app.get(
    "/api/materials/{material_id}/ownership",
    response_model=OwnershipResponse,
    summary="Check material ownership",
    description="Check if current user owns the material and can mint NFT"
)
async def check_material_ownership_endpoint(
    material_id: str = PathParam(description="Material UUID"),
    current_user: User = Depends(get_current_user)
):
    """Check if current user owns the material and can mint NFT."""
    async for db in get_db():
        try:
            result = await check_material_ownership(
                material_id=material_id,
                current_user_wallet=current_user.wallet_address,
                db=db
            )
            
            if "error" in result:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
            
            return OwnershipResponse(
                isOwner=result["isOwner"],
                nftMinted=result["nftMinted"],
                tokenId=result["tokenId"],
                canMint=result["canMint"]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking material ownership: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to check material ownership"
            )


@app.get(
    "/api/materials/{material_id}/nft-status",
    response_model=NFTStatusResponse,
    summary="Get NFT status",
    description="Get NFT status information for a material"
)
async def get_nft_status_endpoint(
    material_id: str = PathParam(description="Material UUID"),
    current_user: User = Depends(get_current_user)
):
    """Get NFT status for a material."""
    async for db in get_db():
        try:
            result = await get_nft_status(
                material_id=material_id,
                db=db
            )
            
            if "error" in result:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
            
            return NFTStatusResponse(
                nftMinted=result["nftMinted"],
                tokenId=result["tokenId"],
                txHash=result["txHash"],
                ipfsCid=result["ipfsCid"]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting NFT status: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get NFT status"
            )


@app.post(
    "/api/materials/{material_id}/sync-blockchain",
    response_model=BlockchainSyncResponse,
    summary="Sync material with blockchain",
    description="Sync material with blockchain after NFT minting"
)
async def sync_material_with_blockchain_endpoint(
    material_id: str = PathParam(description="Material UUID"),
    sync_data: BlockchainSyncRequest = ...,
    current_user: User = Depends(get_current_user)
):
    """Sync material with blockchain after NFT minting."""
    async for db in get_db():
        try:
            result = await sync_material_with_blockchain(
                material_id=material_id,
                current_user_wallet=current_user.wallet_address,
                ipfs_cid=sync_data.ipfsCid,
                content_hash=sync_data.contentHash,
                tx_hash=sync_data.txHash,
                token_id=sync_data.tokenId,
                db=db
            )
            
            if "error" in result:
                # Check if it's a duplicate content hash error
                if "already exists" in result["error"]:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=result["error"]
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=result["error"]
                    )
            
            return BlockchainSyncResponse(
                id=result["id"],
                blockchain=result["blockchain"]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error syncing material with blockchain: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to sync material with blockchain"
            )


@app.get(
    "/api/materials/check-content-hash",
    response_model=ContentHashCheckResponse,
    summary="Check content hash duplicate",
    description="Check if material with this content hash already exists"
)
async def check_content_hash_exists(
    content_hash: str = Query(..., description="SHA-256 hash of content"),
    current_user: User = Depends(get_current_user)
):
    """Check if material with same contentHash exists"""
    async for db in get_db():
        try:
            # Validate contentHash
            if not validate_content_hash(content_hash):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid content hash format. Must be 64-character hex string."
                )
            
            # Search for existing material
            result = await db.execute(
                select(Material).where(Material.nft_content_hash == content_hash)
            )
            existing_material = result.scalar_one_or_none()
            
            if existing_material:
                return ContentHashCheckResponse(
                    exists=True,
                    tokenId=existing_material.nft_token_id,
                    materialId=str(existing_material.id)
                )
            else:
                return ContentHashCheckResponse(
                    exists=False,
                    tokenId=None,
                    materialId=None
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking content hash: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to check content hash"
            )


@app.get(
    "/api/materials/{material_id}/nft-metadata",
    response_model=NFTMetadataResponse,
    summary="Get NFT metadata",
    description="Get NFT metadata for a material"
)
async def get_material_nft_metadata(
    material_id: str = PathParam(description="Material UUID")
):
    """Get NFT metadata for material"""
    async for db in get_db():
        try:
            # Get material
            result = await db.execute(
                select(Material).where(Material.id == material_id)
            )
            material = result.scalar_one_or_none()
            
            if not material or not material.nft_minted:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="NFT not found"
                )
            
            return NFTMetadataResponse(
                tokenId=material.nft_token_id,
                ipfsCid=material.nft_ipfs_cid,
                contentHash=material.nft_content_hash,
                txHash=material.nft_tx_hash,
                isPublished=material.status == "published"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting NFT metadata: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get NFT metadata"
            )


@app.get(
    "/api/materials/blockchain-stats",
    response_model=BlockchainStatsResponse,
    summary="Get blockchain statistics",
    description="Get statistics about blockchain materials"
)
async def get_blockchain_stats():
    """Statistics for blockchain materials"""
    async for db in get_db():
        try:
            # Count NFTs
            total_nfts_result = await db.execute(
                select(func.count()).select_from(
                    select(Material).where(Material.nft_minted == True).subquery()
                )
            )
            total_nfts = total_nfts_result.scalar()
            
            # Count published NFTs
            published_nfts_result = await db.execute(
                select(func.count()).select_from(
                    select(Material).where(
                        Material.nft_minted == True,
                        Material.status == "published"
                    ).subquery()
                )
            )
            published_nfts = published_nfts_result.scalar()
            
            # Statistics by subjects
            subjects_result = await db.execute(
                select(
                    Material.subject,
                    func.count(Material.id).label('count')
                )
                .where(Material.nft_minted == True)
                .group_by(Material.subject)
                .order_by(func.count(Material.id).desc())
            )
            subjects = [
                {"subject": row.subject or "Unknown", "count": row.count} 
                for row in subjects_result
            ]
            
            return BlockchainStatsResponse(
                totalNFTs=total_nfts,
                publishedNFTs=published_nfts,
                subjects=subjects
            )
            
        except Exception as e:
            logger.error(f"Error getting blockchain stats: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get blockchain statistics"
            )


@app.get(
    "/api/user/my-stats",
    response_model=UserStatsResponse,
    summary="Get my statistics",
    description="Get statistics about current user's materials and NFT count"
)
async def get_my_stats(
    current_user: User = Depends(get_current_user),
):
    """Get user statistics for materials and NFTs."""
    async for db in get_db():
        try:
            # Count total number of user materials
            total_materials_result = await db.execute(
                select(func.count()).select_from(
                    select(Material).where(Material.author_id == current_user.id).subquery()
                )
            )
            total_materials = total_materials_result.scalar()
            
            # Count minted NFTs
            minted_nfts_result = await db.execute(
                select(func.count()).select_from(
                    select(Material).where(
                        Material.author_id == current_user.id,
                        Material.nft_minted == True
                    ).subquery()
                )
            )
            minted_nfts = minted_nfts_result.scalar()
            
            # Count published materials
            published_materials_result = await db.execute(
                select(func.count()).select_from(
                    select(Material).where(
                        Material.author_id == current_user.id,
                        Material.status == "published"
                    ).subquery()
                )
            )
            published_materials = published_materials_result.scalar()
            
            # Count drafts
            draft_materials_result = await db.execute(
                select(func.count()).select_from(
                    select(Material).where(
                        Material.author_id == current_user.id,
                        Material.status == "draft"
                    ).subquery()
                )
            )
            draft_materials = draft_materials_result.scalar()
            
            # Statistics by subjects
            subjects_result = await db.execute(
                select(
                    Material.subject,
                    func.count(Material.id).label('count')
                )
                .where(Material.author_id == current_user.id)
                .group_by(Material.subject)
                .order_by(func.count(Material.id).desc())
            )
            subjects = [
                {"subject": row.subject or "Unknown", "count": row.count} 
                for row in subjects_result
            ]
            
            return UserStatsResponse(
                totalMaterials=total_materials,
                mintedNFTs=minted_nfts,
                publishedMaterials=published_materials,
                draftMaterials=draft_materials,
                subjects=subjects
            )
            
        except Exception as e:
            logger.error(f"Error getting user stats: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get user statistics"
            )


def main():
    """Main entry point for running the server."""
    import uvicorn

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
