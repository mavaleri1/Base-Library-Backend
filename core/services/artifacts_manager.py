"""
Local Artifacts Manager for core AI.
"""

import os
import json
import uuid
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel
import httpx
import asyncio

logger = logging.getLogger(__name__)

try:
    import sys
    import os
    
    # Simple solution: add artifacts-service to path
    # Find project root directory
    current_dir = Path(__file__).parent.parent.parent  # core/services -> core -> backend
    artifacts_service_path = current_dir / "artifacts-service"
    
    logger.info(f"🔍 [DB_INTEGRATION] Looking for artifacts-service at: {artifacts_service_path}")
    logger.info(f"🔍 [DB_INTEGRATION] Current file: {Path(__file__)}")
    logger.info(f"🔍 [DB_INTEGRATION] Current dir: {current_dir}")
    
    # If not found, try relative to current working directory
    if not artifacts_service_path.exists():
        artifacts_service_path = Path.cwd() / "artifacts-service"
        logger.info(f"🔍 [DB_INTEGRATION] Trying current working directory: {artifacts_service_path}")
    
    # Try absolute path as fallback
    if not artifacts_service_path.exists():
        artifacts_service_path = Path("D:/GitHub/1. BASE/Base Library/backend/artifacts-service")
        logger.info(f"🔍 [DB_INTEGRATION] Trying absolute path: {artifacts_service_path}")
    
    logger.info(f"🔍 [DB_INTEGRATION] Final artifacts_service_path: {artifacts_service_path}")
    logger.info(f"🔍 [DB_INTEGRATION] Path exists: {artifacts_service_path.exists()}")
    
    if artifacts_service_path.exists():
        logger.info("✅ [DB_INTEGRATION] Artifacts service path found, proceeding with imports...")
    else:
        logger.error(f"❌ [DB_INTEGRATION] Artifacts service path NOT found: {artifacts_service_path}")
        logger.error(f"❌ [DB_INTEGRATION] Current working directory: {Path.cwd()}")
        logger.error(f"❌ [DB_INTEGRATION] Current file location: {Path(__file__)}")
        logger.error(f"❌ [DB_INTEGRATION] Parent directories:")
        for i, parent in enumerate(Path(__file__).parents):
            logger.error(f"❌ [DB_INTEGRATION]   Parent {i}: {parent}")
    
    if artifacts_service_path.exists():
        # Add artifacts-service to path temporarily for imports
        sys.path.insert(0, str(artifacts_service_path))
        logger.info(f"Added artifacts service path to sys.path: {artifacts_service_path}")
        
        # Import modules with explicit path handling
        import importlib.util
        
        logger.info("🔍 [DB_INTEGRATION] Starting module imports...")
        
        # Import models_web3
        logger.info("🔍 [DB_INTEGRATION] Importing models_web3...")
        models_path = artifacts_service_path / "models_web3.py"
        logger.info(f"🔍 [DB_INTEGRATION] Models path: {models_path}")
        logger.info(f"🔍 [DB_INTEGRATION] Models path exists: {models_path.exists()}")
        
        spec = importlib.util.spec_from_file_location("models_web3", models_path)
        models_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models_module)
        Material = models_module.Material
        User = models_module.User
        UserSession = models_module.UserSession
        logger.info("✅ [DB_INTEGRATION] Models imported successfully")
        
        # Import material_classifier
        logger.info("🔍 [DB_INTEGRATION] Importing material_classifier...")
        classifier_path = artifacts_service_path / "services" / "material_classifier.py"
        logger.info(f"🔍 [DB_INTEGRATION] Classifier path: {classifier_path}")
        logger.info(f"🔍 [DB_INTEGRATION] Classifier path exists: {classifier_path.exists()}")
        
        spec = importlib.util.spec_from_file_location("material_classifier", classifier_path)
        classifier_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(classifier_module)
        get_classifier_service = classifier_module.get_classifier_service
        logger.info("✅ [DB_INTEGRATION] Material classifier imported successfully")
        
        # Import content_hash
        logger.info("🔍 [DB_INTEGRATION] Importing content_hash...")
        content_hash_path = artifacts_service_path / "services" / "content_hash.py"
        logger.info(f"🔍 [DB_INTEGRATION] Content hash path: {content_hash_path}")
        logger.info(f"🔍 [DB_INTEGRATION] Content hash path exists: {content_hash_path.exists()}")
        
        spec = importlib.util.spec_from_file_location("content_hash", content_hash_path)
        content_hash_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(content_hash_module)
        calculate_content_hash = content_hash_module.calculate_content_hash
        ContentHashManager = content_hash_module.ContentHashManager
        logger.info("✅ [DB_INTEGRATION] Content hash imported successfully")
        
        # Import SQLAlchemy (this should work normally)
        logger.info("🔍 [DB_INTEGRATION] Importing SQLAlchemy...")
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
        from sqlalchemy import select
        logger.info("✅ [DB_INTEGRATION] SQLAlchemy imported successfully")
        
        # Remove artifacts-service from path to avoid conflicts
        sys.path.remove(str(artifacts_service_path))
        logger.info("✅ [DB_INTEGRATION] Removed artifacts-service from sys.path")
        
        DB_INTEGRATION_AVAILABLE = True
        logger.info("🎉 [DB_INTEGRATION] Database integration for materials is ENABLED")
    else:
        logger.error(f"❌ [DB_INTEGRATION] Artifacts service path not found: {artifacts_service_path}")
        raise ImportError(f"Artifacts service path not found: {artifacts_service_path}")
        
except ImportError as e:
    logger.error(f"❌ [DB_INTEGRATION] Import error: {e}", exc_info=True)
    logger.error(f"❌ [DB_INTEGRATION] Error type: {type(e).__name__}")
    logger.warning(f"Database integration not available: {e}")
    DB_INTEGRATION_AVAILABLE = False
    Material = None
    User = None
    UserSession = None


class ArtifactsConfig(BaseModel):
    """Configuration for local artifact storage"""

    base_path: str = "data/artifacts"
    ensure_permissions: bool = True
    atomic_writes: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    database_url: Optional[str] = None
    enable_db_storage: bool = True


class LocalArtifactsManager:
    """Class for managing local artifacts."""

    def __init__(self, config: ArtifactsConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self._ensure_base_directory_exists()
        
        self.db_session_maker = None
        logger.info(f"🔍 [INIT] Checking database integration availability...")
        logger.info(f"🔍 [INIT] DB_INTEGRATION_AVAILABLE: {DB_INTEGRATION_AVAILABLE}")
        logger.info(f"🔍 [INIT] config.enable_db_storage: {config.enable_db_storage}")
        logger.info(f"🔍 [INIT] config.database_url: {'SET' if config.database_url else 'NOT SET'}")
        
        if DB_INTEGRATION_AVAILABLE and config.enable_db_storage and config.database_url:
            try:
                logger.info("🔍 [INIT] Attempting to initialize database session maker...")
                async_db_url = config.database_url.replace("postgresql://", "postgresql+asyncpg://")
                logger.info(f"🔍 [INIT] Async DB URL: {async_db_url[:50]}...")
                
                engine = create_async_engine(async_db_url, echo=False)
                logger.info("✅ [INIT] Database engine created successfully")
                
                self.db_session_maker = async_sessionmaker(
                    engine, class_=AsyncSession, expire_on_commit=False
                )
                logger.info("✅ [INIT] Database session maker initialized successfully for materials storage")
            except Exception as e:
                logger.error(f"❌ [INIT] Failed to initialize database session maker: {e}", exc_info=True)
                logger.error(f"❌ [INIT] Error type: {type(e).__name__}")
                self.db_session_maker = None
        else:
            logger.warning("⚠️ [INIT] Database integration conditions not met:")
            logger.warning(f"⚠️ [INIT] - DB_INTEGRATION_AVAILABLE: {DB_INTEGRATION_AVAILABLE}")
            logger.warning(f"⚠️ [INIT] - config.enable_db_storage: {config.enable_db_storage}")
            logger.warning(f"⚠️ [INIT] - config.database_url present: {'YES' if config.database_url else 'NO'}")

    def _ensure_base_directory_exists(self):
        """Creates base directory if it doesn't exist"""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            if self.config.ensure_permissions:
                os.chmod(self.base_path, 0o755)
            logger.info(f"Artifacts base directory ensured: {self.base_path}")
        except Exception as e:
            logger.error(f"Failed to create base directory {self.base_path}: {e}")
            raise

    def _ensure_directory_exists(self, path: Path) -> None:
        """Create directories with proper permissions"""
        try:
            path.mkdir(parents=True, exist_ok=True)
            if self.config.ensure_permissions:
                os.chmod(path, 0o755)
            logger.debug(f"Directory ensured: {path}")
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise

    def _atomic_write_file(self, file_path: Path, content: str) -> None:
        """Atomic file write (temp file + rename)"""
        try:
            # Create temporary file in the same directory
            temp_path = file_path.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")

            # Write content to temporary file
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Atomic rename
            temp_path.rename(file_path)

            # Set permissions if configured
            if self.config.ensure_permissions:
                os.chmod(file_path, 0o666)

            logger.debug(f"File written atomically: {file_path}")

        except Exception as e:
            # Cleanup temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Failed to write file {file_path}: {e}")
            raise

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return f"session-{timestamp}"

    def _create_thread_metadata(
        self, thread_id: str, input_content: str
    ) -> Dict[str, Any]:
        """Create thread-level metadata.json"""
        now = datetime.now().isoformat()
        return {
            "thread_id": thread_id,
            "created": now,
            "last_activity": now,
            "sessions_count": 1,
            "input_content": input_content,
            "user_info": None,
        }

    def _create_session_metadata(
        self,
        session_id: str,
        thread_id: str,
        input_content: str,
        display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create session-level metadata.json"""
        now = datetime.now().isoformat()
        return {
            "session_id": session_id,
            "thread_id": thread_id,
            "input_content": input_content,
            "display_name": display_name,
            "created": now,
            "modified": now,
            "status": "active",
            "workflow_data": None,
            "files": [],
        }

    def _update_thread_metadata(
        self, thread_path: Path, updates: Dict[str, Any]
    ) -> None:
        """Update thread metadata"""
        metadata_file = thread_path / "metadata.json"

        # Load existing metadata
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load thread metadata, creating new: {e}")

        # Update metadata
        metadata.update(updates)
        metadata["last_activity"] = datetime.now().isoformat()

        # Save atomically
        self._atomic_write_file(
            metadata_file, json.dumps(metadata, indent=2, ensure_ascii=False)
        )

    def _update_session_metadata(
        self, session_path: Path, updates: Dict[str, Any]
    ) -> None:
        """Update session metadata"""
        metadata_file = session_path / "session_metadata.json"

        # Load existing metadata
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load session metadata, creating new: {e}")

        # Update metadata
        metadata.update(updates)
        metadata["modified"] = datetime.now().isoformat()

        # Save atomically
        self._atomic_write_file(
            metadata_file, json.dumps(metadata, indent=2, ensure_ascii=False)
        )

    def _create_learning_material_content(
        self,
        input_content: str,
        generated_material: str,
        thread_id: str = "",
        session_id: str = "",
    ) -> str:
        """Creates markdown file content with learning material."""

        content = f"""# Learning Material

## Original Exam Question

{input_content}

## Generated Material

{generated_material}
"""

        return content

    def _create_questions_content(
        self, questions: list, questions_and_answers: list, thread_id: str = ""
    ) -> str:
        """Creates markdown file content with questions and answers."""

        content = f"""# Additional Questions and Answers 

## Additional Questions

"""

        for i, question in enumerate(questions, 1):
            content += f"{i}. {question}\n"

        content += "\n## Questions and Answers\n\n"

        for i, qa in enumerate(questions_and_answers, 1):
            content += f"### {i}. Q&A\n\n{qa}\n\n---\n\n"

        return content

    async def push_learning_material(
        self,
        thread_id: str,
        input_content: str,
        generated_material: str,
        display_name: Optional[str] = None,
        wallet_address: Optional[str] = None,
        user_expert_role: Optional[str] = None,  # NEW: expert_role from user settings
        user_id: Optional[str] = None,  # NEW: user_id for fetching expert_role
    ) -> Dict[str, Any]:
        """
        Creates session and saves generated_material.md

        Args:
            thread_id: Thread identifier
            input_content: Original exam question
            generated_material: Generated learning material

        Returns:
            Dict with information about created file
        """
        
        try:
            # Generate session ID
            session_id = self._generate_session_id()

            # Create paths
            thread_path = self.base_path / thread_id
            session_path = thread_path / "sessions" / session_id

            # Ensure directories exist
            self._ensure_directory_exists(session_path)

            # Create thread metadata
            thread_metadata = self._create_thread_metadata(thread_id, input_content)
            thread_metadata_file = thread_path / "metadata.json"
            self._atomic_write_file(
                thread_metadata_file,
                json.dumps(thread_metadata, indent=2, ensure_ascii=False),
            )

            # Create session metadata
            session_metadata = self._create_session_metadata(
                session_id, thread_id, input_content, display_name
            )
            session_metadata_file = session_path / "session_metadata.json"
            self._atomic_write_file(
                session_metadata_file,
                json.dumps(session_metadata, indent=2, ensure_ascii=False),
            )

            # Write learning material file
            file_path = session_path / "generated_material.md"
            self._atomic_write_file(file_path, generated_material)

            # Update session metadata with new file
            self._update_session_metadata(
                session_path, {"files": ["generated_material.md"]}
            )

            logger.info(
                f"Successfully created learning material for thread {thread_id} session {session_id}"
            )

            relative_file_path = str(file_path.relative_to(self.base_path))
            relative_session_path = str(session_path.relative_to(self.base_path))
            relative_thread_path = str(thread_path.relative_to(self.base_path))

            result = {
                "success": True,
                "file_path": relative_file_path,
                "folder_path": relative_session_path,  # Session folder for compatibility
                "session_id": session_id,
                "thread_path": relative_thread_path,
                "absolute_path": str(file_path),
            }
            
            if self.db_session_maker:
                try:
                    # Get expert_role from user settings if not provided
                    if user_expert_role is None and user_id:
                        user_expert_role = await self._get_user_expert_role(user_id)
                    
                    material_id = await self._save_material_to_db(
                        thread_id=thread_id,
                        session_id=session_id,
                        content=generated_material,
                        input_query=input_content,
                        file_path=relative_file_path,
                        display_name=display_name,
                        wallet_address=wallet_address,
                        user_expert_role=user_expert_role
                    )
                    result["material_id"] = str(material_id)
                except Exception as db_error:
                    logger.error(f"Failed to save material to database: {db_error}", exc_info=True)
            
            return result

        except Exception as e:
            logger.error(
                f"Failed to push learning material for thread {thread_id}: {e}"
            )
            return {"success": False, "error": str(e)}

    async def push_recognized_notes(
        self,
        thread_id: str,
        session_id: str,
        recognized_notes: str,
    ) -> Dict[str, Any]:
        """
        Saves recognized_notes.md to existing session

        Args:
            thread_id: Thread identifier
            session_id: Session identifier
            recognized_notes: Recognized text

        Returns:
            success/error status + file paths
        """
        try:
            # Build session path from thread_id and session_id
            session_path = self.base_path / thread_id / "sessions" / session_id

            if not session_path.exists():
                raise ValueError(f"Session path does not exist: {session_path}")

            # Create recognized notes content
            # Write recognized notes file
            file_path = session_path / "recognized_notes.md"
            self._atomic_write_file(file_path, recognized_notes)

            # Update session metadata
            try:
                with open(
                    session_path / "session_metadata.json", "r", encoding="utf-8"
                ) as f:
                    metadata = json.load(f)
                    current_files = metadata.get("files", [])
                    if "recognized_notes.md" not in current_files:
                        current_files.append("recognized_notes.md")
                    self._update_session_metadata(
                        session_path, {"files": current_files}
                    )
            except Exception as e:
                logger.warning(f"Failed to update session metadata: {e}")

            logger.info(f"Successfully created recognized notes for thread {thread_id}")

            return {
                "success": True,
                "file_path": str(file_path.relative_to(self.base_path)),
                "absolute_path": str(file_path),
            }

        except Exception as e:
            logger.error(f"Failed to push recognized notes for thread {thread_id}: {e}")
            return {"success": False, "error": str(e)}

    async def push_synthesized_material(
        self,
        thread_id: str,
        session_id: str,
        synthesized_material: str,
    ) -> Dict[str, Any]:
        """
        Saves synthesized_material.md

        Args:
            thread_id: Thread identifier
            session_id: Session identifier
            synthesized_material: Synthesized material

        Returns:
            success/error status + file paths
        """
        try:
            # Build session path from thread_id and session_id
            session_path = self.base_path / thread_id / "sessions" / session_id

            if not session_path.exists():
                raise ValueError(f"Session path does not exist: {session_path}")

            # Write synthesized material file
            file_path = session_path / "synthesized_material.md"
            self._atomic_write_file(file_path, synthesized_material)

            # Update session metadata
            try:
                with open(
                    session_path / "session_metadata.json", "r", encoding="utf-8"
                ) as f:
                    metadata = json.load(f)
                    current_files = metadata.get("files", [])
                    if "synthesized_material.md" not in current_files:
                        current_files.append("synthesized_material.md")
                    self._update_session_metadata(
                        session_path, {"files": current_files}
                    )
            except Exception as e:
                logger.warning(f"Failed to update session metadata: {e}")

            logger.info(
                f"Successfully created synthesized material for thread {thread_id}"
            )

            return {
                "success": True,
                "file_path": str(file_path.relative_to(self.base_path)),
                "absolute_path": str(file_path),
            }

        except Exception as e:
            logger.error(
                f"Failed to push synthesized material for thread {thread_id}: {e}"
            )
            return {"success": False, "error": str(e)}

    async def push_questions_and_answers(
        self,
        thread_id: str,
        session_id: str,
        questions: list,
        questions_and_answers: list,
    ) -> Dict[str, Any]:
        """
        Saves questions.md and individual answer files

        Args:
            thread_id: Thread identifier
            session_id: Session identifier
            questions: List of gap questions
            questions_and_answers: List of Q&A pairs

        Returns:
            success/error status + file paths
        """
        try:
            # Build session path from thread_id and session_id
            session_path = self.base_path / thread_id / "sessions" / session_id

            if not session_path.exists():
                raise ValueError(f"Session path does not exist: {session_path}")

            created_files = []

            # Create gap questions content
            markdown_content = self._create_questions_content(
                questions=questions, questions_and_answers=questions_and_answers, thread_id=thread_id
            )

            # Write main questions file
            questions_file = session_path / "questions.md"
            self._atomic_write_file(questions_file, markdown_content)
            created_files.append("questions.md")

            # Create answers directory if there are individual answers
            if questions_and_answers:
                answers_dir = session_path / "answers"
                self._ensure_directory_exists(answers_dir)

                for i, qa in enumerate(questions_and_answers, 1):
                    answer_file = answers_dir / f"answer_{i:03d}.md"
                    answer_content = f"""# Answer {i}

{qa}
"""
                    self._atomic_write_file(answer_file, answer_content)
                    created_files.append(f"answers/answer_{i:03d}.md")

            # Update session metadata
            try:
                with open(
                    session_path / "session_metadata.json", "r", encoding="utf-8"
                ) as f:
                    metadata = json.load(f)
                    current_files = metadata.get("files", [])
                    for file in created_files:
                        if file not in current_files:
                            current_files.append(file)
                    self._update_session_metadata(
                        session_path, {"files": current_files, "status": "completed"}
                    )
            except Exception as e:
                logger.warning(f"Failed to update session metadata: {e}")

            logger.info(
                f"Successfully created questions and answers for thread {thread_id}"
            )

            return {
                "success": True,
                "file_path": str(questions_file.relative_to(self.base_path)),
                "created_files": created_files,
                "absolute_path": str(questions_file),
            }

        except Exception as e:
            logger.error(
                f"Failed to push questions and answers for thread {thread_id}: {e}"
            )
            return {"success": False, "error": str(e)}
    
    def _map_expert_role_to_grade(self, expert_role: str) -> str:
        """Map expert_role from user settings to grade level."""
        mapping = {
            "Advanced": "Advanced",
            "Intermediate": "Intermediate", 
            "Beginner": "Beginner"
        }
        return mapping.get(expert_role, "Intermediate")  # Default fallback

    async def _get_user_expert_role(self, user_id: str) -> Optional[str]:
        """Get user's expert_role from prompt-config-service."""
        try:
            # Get prompt-config-service URL from settings
            from ..config.settings import get_settings
            settings = get_settings()
            prompt_service_url = settings.prompt_service_url
            
            url = f"{prompt_service_url}/api/v1/users/{user_id}/placeholders/expert_role"
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                data = response.json()
                expert_role_value = data.get("display_name")
                
                if expert_role_value:
                    return expert_role_value
                else:
                    logger.warning(f"No display_name found for user {user_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get expert_role for user {user_id}: {e}")
            return None

    async def _save_material_to_db(
        self,
        thread_id: str,
        session_id: str,
        content: str,
        input_query: str,
        file_path: str,
        display_name: Optional[str] = None,
        wallet_address: Optional[str] = None,
        user_expert_role: Optional[str] = None,  # NEW: expert_role from user settings
    ) -> uuid.UUID:
        """Save material to database with automatic classification and blockchain metadata.
        
        Args:
            thread_id: Thread identifier
            session_id: Session identifier
            content: Material content (markdown)
            input_query: Original user query
            file_path: Relative file path
            display_name: Optional display name
            
        Returns:
            UUID of created material
        """
        if not self.db_session_maker:
            logger.error("Database session maker not initialized!")
            raise RuntimeError("Database session maker not initialized")
        
        async with self.db_session_maker() as session:
            try:
                # Step 1: Get user by wallet_address if provided, otherwise use anonymous
                if wallet_address:
                    result_user = await session.execute(
                        select(User).where(User.wallet_address == wallet_address)
                    )
                    author = result_user.scalar_one_or_none()
                    
                    if not author:
                        author = User(wallet_address=wallet_address)
                        session.add(author)
                        await session.flush()
                else:
                    result_user = await session.execute(
                        select(User).where(User.wallet_address == "0x0000000000000000000000000000000000000000")
                    )
                    author = result_user.scalar_one_or_none()
                    
                    if not author:
                        author = User(
                            wallet_address="0x0000000000000000000000000000000000000000"
                        )
                        session.add(author)
                        await session.flush()
                
                # Step 2: Use user settings for grade, AI for subject and topic
                grade = self._map_expert_role_to_grade(user_expert_role) if user_expert_role else "Intermediate"
                
                # Still use AI for subject and topic classification
                classifier = get_classifier_service()
                classification = await classifier.classify_material(
                    content=content,
                    input_query=input_query
                )
                
                # Override grade with user setting
                classification.grade = grade
                
                logger.info(
                    f"✅ [DB_SAVE] Material classified successfully: "
                    f"subject={classification.subject}, "
                    f"grade={classification.grade} (from user expert_role={user_expert_role}), topic={classification.topic}"
                )
                
                # Step 3: Calculate content hash and prepare blockchain metadata
                content_hash = calculate_content_hash(content)
                hash_manager = ContentHashManager()
                
                # Generate title from display_name or extract from content
                title = display_name or self._extract_title_from_content(content)
                word_count = hash_manager.calculate_word_count(content)
                
                # Step 4: Create Material record
                material_id = uuid.uuid4()
                
                ipfs_cid_placeholder = hash_manager.create_blockchain_metadata(
                    content=content,
                    material_id=str(material_id),
                    subject=classification.subject,
                    grade=classification.grade,
                    topic=classification.topic,
                    author_wallet=author.wallet_address
                )["ipfs_cid"]
                
                material = Material(
                    id=material_id,
                    author_id=author.id,
                    thread_id=thread_id,
                    session_id=session_id,
                    subject=classification.subject,
                    grade=classification.grade,
                    topic=classification.topic,
                    content=content,
                    input_query=input_query,
                    content_hash=content_hash,
                    ipfs_cid=ipfs_cid_placeholder,
                    file_path=file_path,
                    title=title,
                    word_count=word_count,
                    status="published",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                session.add(material)
                await session.commit()
                await session.refresh(material)
                
                return material.id
                
            except Exception as e:
                logger.error(f"Error saving material to database: {e}", exc_info=True)
                await session.rollback()
                raise
    
    @staticmethod
    def _extract_title_from_content(content: str, max_length: int = 100) -> str:
        """Extract title from markdown content (first heading or first line).
        
        Args:
            content: Markdown content
            max_length: Maximum title length
            
        Returns:
            Extracted title
        """
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                title = line[2:].strip()
                if len(title) > max_length:
                    title = title[:max_length] + "..."
                return title
        
        # Fallback: use first non-empty line
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if len(line) > max_length:
                    line = line[:max_length] + "..."
                return line
        
        return "Untitled Material"
