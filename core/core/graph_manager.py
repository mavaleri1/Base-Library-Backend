"""
GraphManager – unified wrapper around LangGraph workflow.
Responsible for:
• lazy initialization of DB checkpoints
• graph start / continuation
• passing HITL node messages outside
• pushing artifacts
• tracing in LangFuse
Adapted from project_documentation.md for GeneralState.
"""

import uuid
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable

from langgraph.types import Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
#from langfuse.callback import CallbackHandler

from .graph import create_workflow
from .state import GeneralState
from ..config.settings import get_settings
from ..services.artifacts_manager import LocalArtifactsManager, ArtifactsConfig


NODE_DESCRIPTIONS = { # TODO: reformulate
    "input_processing": "User input processing",
    "generating_content": "Educational material generation",
    "recognition_handwritten": "Handwritten notes recognition",
    "synthesis_material": "Final material synthesis",
    "edit_material": "Iterative material editing",
    "generating_questions": "Assessment questions generation and editing",
    "answer_question": "Question answers generation",
    None: "Ready for new input content",
}

logger = logging.getLogger(__name__)


class GraphManager:
    """
    Manages a single LangGraph instance for multiple users.
    States are separated by thread_id in Postgres-checkpointer.
    """

    # Artifact configuration for each node
    NODE_ARTIFACT_CONFIG: Dict[str, Dict[str, Any]] = {
        "generating_content": {
            "condition": lambda node_data, state: bool(node_data.get("generated_material")),
            "handler": "_save_learning_material"
        },
        "recognition_handwritten": {
            "condition": lambda node_data, state: bool(node_data.get("recognized_notes")),
            "handler": "_save_recognized_notes"
        },
        "synthesis_material": {
            "condition": lambda node_data, state: bool(node_data.get("synthesized_material")),
            "handler": "_save_synthesized_material"
        },
        "edit_material": {
            "condition": lambda node_data, state: node_data.get("last_action") == "edit",
            "handler": "_save_synthesized_material"  # Same method, overwrite
        },
        "generating_questions": {
            "condition": lambda node_data, state: bool(node_data.get("questions")),
            "handler": "_save_questions"
        },
        "answer_question": {
            "condition": lambda node_data, state: bool(node_data.get("questions_and_answers")),
            "handler": "_save_answers"
        }
    }

    def __init__(self) -> None:
        self.workflow = create_workflow()
        self.settings = get_settings()

        self._setup_done = False  # to do DB initialization only once

        # LangFuse integration
        #self.langfuse_handler = CallbackHandler()

        # Dictionary to store session_id for each user
        # Key - thread_id, value - session_id
        self.user_sessions: Dict[str, str] = {}
        
        # Dictionary to store wallet_address for each user
        # Key - thread_id, value - wallet_address
        self.user_wallets: Dict[str, str] = {}
        
        # Dictionary to store user_id for each user
        # Key - thread_id, value - user_id
        self.user_ids: Dict[str, str] = {}

        # Local artifacts manager
        self.artifacts_manager: Optional[LocalArtifactsManager] = None
        
        logger.info(f"🔍 [GRAPH_MANAGER_INIT] Checking artifacts configuration...")
        logger.info(f"🔍 [GRAPH_MANAGER_INIT] artifacts_base_path: {self.settings.artifacts_base_path}")
        logger.info(f"🔍 [GRAPH_MANAGER_INIT] database_url: {'SET' if self.settings.database_url else 'NOT SET'}")
        logger.info(f"🔍 [GRAPH_MANAGER_INIT] is_artifacts_configured(): {self.settings.is_artifacts_configured()}")
        
        if self.settings.is_artifacts_configured():
            logger.info("✅ [GRAPH_MANAGER_INIT] Artifacts are configured, creating LocalArtifactsManager...")
            cfg = ArtifactsConfig(
                base_path=self.settings.artifacts_base_path,
                ensure_permissions=self.settings.artifacts_ensure_permissions,
                atomic_writes=self.settings.artifacts_atomic_writes,
                max_file_size=self.settings.artifacts_max_file_size,
                database_url=self.settings.database_url,
                enable_db_storage=True,
            )
            logger.info(f"🔍 [GRAPH_MANAGER_INIT] ArtifactsConfig created: {cfg}")
            self.artifacts_manager = LocalArtifactsManager(cfg)
            logger.info("✅ [GRAPH_MANAGER_INIT] LocalArtifactsManager created successfully")
            
            # Check db_session_maker after creation
            if self.artifacts_manager.db_session_maker:
                logger.info("✅ [GRAPH_MANAGER_INIT] db_session_maker is available in artifacts_manager")
            else:
                logger.warning("⚠️ [GRAPH_MANAGER_INIT] db_session_maker is NOT available in artifacts_manager")
        else:
            logger.warning("⚠️ [GRAPH_MANAGER_INIT] Artifacts are NOT configured, skipping LocalArtifactsManager creation")

        # user settings storage
        self.user_settings: Dict[str, Dict[str, Any]] = {}

        # artifacts data storage by thread_id
        # Structure: {thread_id: {session_id, pending_urls, sent_urls, web_ui_base_url}}
        self.artifacts_data: Dict[str, Dict[str, Any]] = {}

    # ---------- internal helpers ----------

    async def _ensure_setup(self):
        """DB checkpoints initialization"""
        if self._setup_done:
            return
        async with AsyncPostgresSaver.from_conn_string(
            self.settings.database_url
        ) as saver:
            await saver.setup()
        self._setup_done = True
        logger.info("PostgreSQL checkpointer setup completed")

    async def _get_state(self, thread_id: str):
        """Get state for thread_id"""
        await self._ensure_setup()
        cfg = {"configurable": {"thread_id": thread_id}}
        async with AsyncPostgresSaver.from_conn_string(
            self.settings.database_url
        ) as saver:
            graph = self.workflow.compile(checkpointer=saver)
            return await graph.aget_state(cfg)

    async def delete_thread(self, thread_id: str):
        """Delete thread and all related data"""
        await self._ensure_setup()
        async with AsyncPostgresSaver.from_conn_string(
            self.settings.database_url
        ) as saver:
            await saver.adelete_thread(thread_id)

        # Clear artifacts data from dictionary
        if thread_id in self.artifacts_data:
            del self.artifacts_data[thread_id]

        # Also delete session_id for this user
        self.delete_session(thread_id)

        logger.info(f"Thread {thread_id} deleted successfully")
    
    def get_material_info(self, thread_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session materials information
        
        Args:
            thread_id: Thread identifier
            session_id: Session identifier
            
        Returns:
            Dictionary with metadata and file paths or None
        """
        if not self.artifacts_manager:
            return None
            
        from pathlib import Path
        import json
        
        session_path = Path(self.settings.artifacts_base_path) / thread_id / "sessions" / session_id
        
        if not session_path.exists():
            logger.warning(f"Session path does not exist: {session_path}")
            return None
            
        metadata_file = session_path / "session_metadata.json"
        if not metadata_file.exists():
            logger.warning(f"Metadata file does not exist: {metadata_file}")
            return None
            
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                
            return {
                "thread_id": thread_id,
                "session_id": session_id,
                "metadata": metadata,
                "session_path": str(session_path)
            }
        except Exception as e:
            logger.error(f"Error reading material info: {e}")
            return None
    
    def get_material_content(self, thread_id: str, session_id: str, file_name: str) -> Optional[str]:
        """
        Get material file content
        
        Args:
            thread_id: Thread identifier
            session_id: Session identifier
            file_name: File name
            
        Returns:
            File content or None
        """
        if not self.artifacts_manager:
            return None
            
        from pathlib import Path
        
        file_path = Path(self.settings.artifacts_base_path) / thread_id / "sessions" / session_id / file_name
        
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return None
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file content: {e}")
            return None
    
    def get_thread_sessions(self, thread_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get list of all sessions for thread_id
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            List of dictionaries with session information or None
        """
        if not self.artifacts_manager:
            return None
            
        from pathlib import Path
        import json
        
        thread_path = Path(self.settings.artifacts_base_path) / thread_id
        
        if not thread_path.exists():
            logger.warning(f"Thread path does not exist: {thread_path}")
            return None
        
        sessions_path = thread_path / "sessions"
        if not sessions_path.exists():
            logger.warning(f"Sessions path does not exist: {sessions_path}")
            return []
        
        try:
            sessions = []
            for session_dir in sessions_path.iterdir():
                if session_dir.is_dir():
                    metadata_file = session_dir / "session_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                            sessions.append({
                                "session_id": session_dir.name,
                                "metadata": metadata
                            })
            
            # Sort by creation date (newest first)
            sessions.sort(key=lambda x: x["metadata"].get("created", ""), reverse=True)
            
            return sessions
        except Exception as e:
            logger.error(f"Error reading thread sessions: {e}")
            return None

    # ---------- langfuse session management ----------

    def create_new_session(self, thread_id: str) -> str:
        """
        Creates a new session_id for user.

        Args:
            thread_id: Thread identifier

        Returns:
            str: New session_id
        """
        session_id = str(uuid.uuid4())
        self.user_sessions[thread_id] = session_id
        logger.info(f"Created new session '{session_id}' for user {thread_id}")
        return session_id

    def get_session_id(self, thread_id: str) -> Optional[str]:
        """
        Gets current session_id for user.

        Args:
            thread_id: Thread identifier

        Returns:
            Optional[str]: session_id or None if no session
        """
        return self.user_sessions.get(thread_id)

    def delete_session(self, thread_id: str) -> None:
        """
        Deletes session_id for user.

        Args:
            thread_id: Thread identifier
        """
        if thread_id in self.user_sessions:
            session_id = self.user_sessions.pop(thread_id)
            logger.info(f"Deleted session '{session_id}' for user {thread_id}")

    # ---------- Web UI URL generation ----------

    def _generate_web_ui_url(
        self, thread_id: str, session_id: str, file_name: str
    ) -> str:
        """
        Generates Web UI URL for specific file
        
        Args:
            thread_id: Thread identifier
            session_id: Session identifier
            file_name: File name
        
        Returns:
            Full URL like http://localhost:3001/thread/{thread_id}/session/{session_id}/file/{file_name}
        """
        base_url = self.settings.web_ui_base_url.rstrip('/')
        return f"{base_url}/thread/{thread_id}/session/{session_id}/file/{file_name}"
    
    def _track_artifact_url(
        self, thread_id: str, artifact_type: str, url: str, label: str
    ) -> None:
        """
        Adds URL to pending_urls
        
        Args:
            thread_id: Thread identifier
            artifact_type: Artifact type (learning_material, questions, etc.)
            url: Artifact URL
            label: Display label
        """
        if thread_id not in self.artifacts_data:
            self.artifacts_data[thread_id] = {
                "pending_urls": {},
                "sent_urls": {}
            }
        
        self.artifacts_data[thread_id]["pending_urls"][artifact_type] = {
            "url": url,
            "label": label
        }
        logger.debug(f"Tracked URL for {artifact_type}: {url}")
    
    def _get_pending_urls(self, thread_id: str) -> List[str]:
        """
        Gets list of unsent URLs with labels in Markdown format
        
        Args:
            thread_id: Thread identifier
        
        Returns:
            List of strings with URLs and labels for sending (one message with Markdown links)
        """
        pending = self.artifacts_data.get(thread_id, {}).get("pending_urls", {})
        if not pending:
            logger.debug(f"No pending URLs for thread {thread_id}")
            return []
        
        # Form single message with Markdown links
        links = []
        for artifact_type, data in pending.items():
            # Separate emoji and text
            label = data['label']
            # Find first space after emoji
            if ' ' in label:
                emoji, text = label.split(' ', 1)
                # Format: emoji [text](link)
                link = f"{emoji} [{text}]({data['url']})"
            else:
                # If no space, use as is
                link = f"[{label}]({data['url']})"
            links.append(link)
            logger.debug(f"Adding link for {artifact_type}: {link}")
        
        # Combine all links into one message
        message = "📚 **Materials ready:**\n\n" + "\n".join(links)
        logger.info(f"Generated message with {len(links)} links for thread {thread_id}: {message}")
        return [message]
    
    def _mark_urls_as_sent(self, thread_id: str, artifact_types: List[str]) -> None:
        """
        Moves URLs from pending to sent
        
        Args:
            thread_id: Thread identifier
            artifact_types: List of artifact types to move
        """
        if thread_id not in self.artifacts_data:
            return
        
        pending = self.artifacts_data[thread_id].get("pending_urls", {})
        sent = self.artifacts_data[thread_id].get("sent_urls", {})
        
        for artifact_type in artifact_types:
            if artifact_type in pending:
                sent[artifact_type] = pending.pop(artifact_type)
                logger.debug(f"Marked {artifact_type} URL as sent for thread {thread_id}")
    
    # ---------- local artifacts management ----------


    async def process_step(self, thread_id: str, query: str, image_paths: List[str] = None, wallet_address: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Simplified main method for processing workflow steps.
        
        Args:
            thread_id: Thread identifier
            query: Text query or command
            image_paths: Optional list of image paths
            wallet_address: Optional wallet address of the user
            user_id: Optional user ID for personalization
            
        Returns:
            Processing result with thread_id and messages
        """
        # 1. Preparation
        thread_id, input_state, cfg = await self._prepare_workflow(
            thread_id, query, image_paths, user_id
        )
        
        # Store wallet address for this thread (after thread_id is created)
        if wallet_address:
            self.user_wallets[thread_id] = wallet_address
            logger.info(f"Stored wallet address {wallet_address[:10]}... for thread {thread_id}")
        
        # Store user_id for this thread (after thread_id is created)
        if user_id:
            self.user_ids[thread_id] = user_id
            logger.info(f"Stored user_id {user_id} for thread {thread_id}")
        
        # 2. Workflow execution
        await self._run_workflow(thread_id, input_state, cfg)
        
        # 3. Finalization
        return await self._finalize_workflow(thread_id)

    async def get_current_step(self, thread_id: str) -> Dict[str, str]:
        """Get current workflow step"""
        state = await self._get_state(thread_id)
        node = None
        if state and state.interrupts:
            logger.debug(f"DEBUG LOG: state.next: {state.next[0]}")
            node = state.next[0]

        current_step = {
            "node": node,
            "description": NODE_DESCRIPTIONS.get(node, NODE_DESCRIPTIONS[None]),
        }
        logger.debug(f"Current step for thread {thread_id}: {current_step}")
        return current_step

    async def get_thread_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get full thread state"""
        try:
            state = await self._get_state(thread_id)
            logger.debug(f"State for thread {thread_id}: {state}")
            if state and state.values:
                return state.values
            return None
        except Exception as e:
            logger.error(f"Error getting state for thread {thread_id}: {str(e)}")
            return None

    # ---------- New refactored methods ----------

    async def _prepare_workflow(
        self, thread_id: str, query: str, image_paths: Optional[List[str]], user_id: Optional[str] = None
    ) -> Tuple[str, Any, Dict[str, Any]]:
        """
        Workflow preparation: thread_id, initial state, configuration

        Args:
            thread_id: Thread identifier
            query: Text query
            image_paths: Optional list of image paths
            user_id: Optional user ID for personalization

        Returns:
            Tuple[thread_id, input_state, config]
        """
        # Generate thread_id if not provided
        if not thread_id:
            thread_id = str(uuid.uuid4())
            logger.info(f"Created new thread: {thread_id}")

        # Validate image_paths
        image_paths = image_paths or []
        if image_paths:
            logger.info(f"Processing with {len(image_paths)} images for thread {thread_id}")

        state = await self._get_state(thread_id)

        # Determine input_state and session_id for LangFuse
        if not state.values:  # fresh run - new workflow
            logger.info(f"Starting fresh run for thread {thread_id}")
            input_state = GeneralState(
                input_content=query,
                image_paths=image_paths  # Add images to initial state
            )
            # Create new session_id for new dialogue
            session_id = self.create_new_session(thread_id)
        else:  # continue - existing workflow continuation
            logger.info(f"Continuing run for thread {thread_id}")
            
            if image_paths:
                # Add images via Command.update
                logger.info(f"Adding {len(image_paths)} images to existing workflow")
                input_state = Command(
                    resume=query,
                    update={"image_paths": image_paths}
                )
            else:
                # Regular continuation without images
                input_state = Command(resume=query)
            
            # Use existing session_id
            session_id = self.get_session_id(thread_id) or self.create_new_session(thread_id)

        # Configuration with LangFuse tracing
        cfg = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            },
            #"callbacks": [self.langfuse_handler],
            #"metadata": {
             #   "langfuse_session_id": session_id,
              #  "langfuse_user_id": thread_id
            #},
        }

        return thread_id, input_state, cfg

    async def _run_workflow(
        self, thread_id: str, input_state: Any, cfg: Dict[str, Any]
    ) -> None:
        """
        Workflow execution and event handling

        Args:
            thread_id: Thread identifier
            input_state: Initial state or command
            cfg: Execution configuration
        """
        await self._ensure_setup()
        
        async with AsyncPostgresSaver.from_conn_string(
            self.settings.database_url
        ) as saver:
            graph = self.workflow.compile(checkpointer=saver)
            
            async for event in graph.astream(input_state, cfg, stream_mode="updates"):
                await self._handle_workflow_event(event, thread_id)

    async def _handle_workflow_event(self, event: Dict, thread_id: str) -> None:
        """
        Handle single workflow event

        Args:
            event: Event from graph
            thread_id: Thread identifier
        """
        logger.debug(f"Event: {event}")
        
        for node_name, node_data in event.items():
            await self._process_node_artifacts(node_name, node_data, thread_id)

    async def _process_node_artifacts(
        self, node_name: str, node_data: Dict, thread_id: str
    ) -> None:
        """
        Universal artifact processing for node

        Args:
            node_name: Node name
            node_data: Node data
            thread_id: Thread identifier
        """
        config = self.NODE_ARTIFACT_CONFIG.get(node_name)
        if not config:
            return
        
        # Get current state
        state = await self._get_state(thread_id)
        
        # Check save condition
        if not config["condition"](node_data, state.values):
            return
        
        logger.info(f"Saving artifacts for {node_name}, thread {thread_id}")
        
        # Call appropriate handler
        handler = getattr(self, config["handler"])
        await handler(thread_id, node_data, state.values)

    async def _finalize_workflow(self, thread_id: str) -> Dict[str, Any]:
        """
        Workflow finalization: interrupt handling or final cleanup

        Args:
            thread_id: Thread identifier

        Returns:
            Dict with execution result
        """
        final_state = await self._get_state(thread_id)

        logger.debug(f"final_state interrupts: {final_state.interrupts}")

        if final_state.interrupts:
            interrupt_data = final_state.interrupts[0].value
            logger.debug(f"Interrupt data: {interrupt_data}")
            msgs = interrupt_data.get("message", [str(interrupt_data)])

            # Add unsent URLs to message
            pending_urls = self._get_pending_urls(thread_id)
            if pending_urls:
                # Place links at the beginning, before agent message
                msgs = pending_urls + msgs
                # Mark URLs as sent
                pending_types = list(self.artifacts_data.get(thread_id, {}).get("pending_urls", {}).keys())
                self._mark_urls_as_sent(thread_id, pending_types)
                logger.debug(f"Added {len(pending_urls)} pending URLs to interrupt message for thread {thread_id}")

            logger.info(f"Workflow interrupted for thread {thread_id}, returning messages: {msgs}")
            session_id = self.artifacts_data.get(thread_id, {}).get("session_id")
            return {
                "thread_id": thread_id, 
                "session_id": session_id,
                "result": msgs
            }

        # happy path – everything finished
        logger.info(f"Workflow completed for thread {thread_id}")

        # Form final message with Web UI link
        final_message = ["Done 🎉 – send the next topic for study!"]

        # Generate session link in Web UI
        session_id = self.artifacts_data.get(thread_id, {}).get("session_id")
        if session_id:
            base_url = self.settings.web_ui_base_url.rstrip('/')
            session_url = f"{base_url}/thread/{thread_id}/session/{session_id}"
            final_message.append(
                f"📁 All materials available [here]({session_url})"
            )

        await self.delete_thread(thread_id)

        return_data = {
            "thread_id": thread_id, 
            "session_id": session_id,
            "result": final_message
        }
        logger.debug(f"return_data: {return_data}")

        return return_data

    # ---------- Specialized artifact saving methods ----------

    async def _save_learning_material(
        self, thread_id: str, node_data: Dict, state_values: Dict
    ) -> None:
        """
        Creates new session and saves learning material

        Args:
            thread_id: Thread identifier
            node_data: Data from node
            state_values: Current graph state values
        """
        logger.info(f"🔍 [GRAPH_MANAGER] _save_learning_material called for thread {thread_id}")
        logger.info(f"🔍 [GRAPH_MANAGER] node_data keys: {list(node_data.keys())}")
        logger.info(f"🔍 [GRAPH_MANAGER] state_values keys: {list(state_values.keys())}")
        logger.info(f"🔍 [GRAPH_MANAGER] generated_material length: {len(node_data.get('generated_material', ''))}")
        
        if not self.artifacts_manager:
            logger.warning("⚠️ [GRAPH_MANAGER] Artifacts manager not configured, skipping learning material save")
            return
        
        logger.info("✅ [GRAPH_MANAGER] Artifacts manager is available, proceeding with save...")
        
        wallet_address = self.user_wallets.get(thread_id)
        user_id = self.user_ids.get(thread_id)
        logger.info(f"🔍 [GRAPH_MANAGER] Passing wallet_address to artifacts_manager: {wallet_address}")
        logger.info(f"🔍 [GRAPH_MANAGER] Passing user_id to artifacts_manager: {user_id}")
        
        result = await self.artifacts_manager.push_learning_material(
            thread_id=thread_id,
            input_content=state_values.get("input_content", ""),
            generated_material=node_data.get("generated_material", ""),
            display_name=state_values.get("display_name"),
            wallet_address=wallet_address,
            user_expert_role=None,  # Will be fetched using user_id in artifacts_manager
            user_id=user_id
        )
        
        logger.info(f"🔍 [GRAPH_MANAGER] push_learning_material result: {result}")
        
        if result.get("success"):
            logger.info(
                f"🎉 [GRAPH_MANAGER] Successfully saved learning material for thread {thread_id}: {result.get('file_path')}"
            )
            
            # Initialize data structure for session
            if thread_id not in self.artifacts_data:
                logger.info(f"🔍 [GRAPH_MANAGER] Creating new artifacts_data entry for thread {thread_id}")
                self.artifacts_data[thread_id] = {
                    "pending_urls": {},
                    "sent_urls": {},
                    "session_id": result.get("session_id"),
                    "web_ui_base_url": self.settings.web_ui_base_url
                }
            else:
                logger.info(f"🔍 [GRAPH_MANAGER] Updating existing artifacts_data for thread {thread_id}")
                self.artifacts_data[thread_id]["session_id"] = result.get("session_id")
            
            # Generate and track URL for learning material
            session_id = result.get("session_id")
            if session_id:
                logger.info(f"🔍 [GRAPH_MANAGER] Generating web UI URL for session {session_id}")
                url = self._generate_web_ui_url(
                    thread_id=thread_id,
                    session_id=session_id,
                    file_name="generated_material.md"
                )
                self._track_artifact_url(
                    thread_id=thread_id,
                    artifact_type="learning_material",
                    url=url,
                    label="📚 Generated material"  # Emoji will be separated when formatting
                )
                logger.info(f"✅ [GRAPH_MANAGER] URL generated and tracked: {url}")
            
        else:
            logger.error(
                f"❌ [GRAPH_MANAGER] Failed to save learning material for thread {thread_id}: {result.get('error')}"
            )

    async def _save_recognized_notes(
        self, thread_id: str, node_data: Dict, state_values: Dict
    ) -> None:
        """
        Saves recognized notes to existing session

        Args:
            thread_id: Thread identifier
            node_data: Data from node
            state_values: Current graph state values
        """
        if not self.artifacts_manager:
            logger.debug(
                "Artifacts manager not configured, skipping recognized notes save"
            )
            return
            
        session_id = self.artifacts_data.get(thread_id, {}).get("session_id")
        if not session_id:
            logger.warning(f"No session_id for thread {thread_id}, skipping recognized notes save")
            return
        
        try:
            await self.artifacts_manager.push_recognized_notes(
                thread_id=thread_id,
                session_id=session_id,
                recognized_notes=node_data.get("recognized_notes", "")
            )
            logger.info(f"Successfully saved recognized notes for thread {thread_id}")
            
            # Generate and track URL for recognized notes
            url = self._generate_web_ui_url(
                thread_id=thread_id,
                session_id=session_id,
                file_name="recognized_notes.md"
            )
            self._track_artifact_url(
                thread_id=thread_id,
                artifact_type="recognized_notes",
                url=url,
                label="📝 Recognized notes"
            )
        except Exception as e:
            logger.error(f"Failed to save recognized notes for thread {thread_id}: {e}")

    async def _save_synthesized_material(
        self, thread_id: str, node_data: Dict, state_values: Dict
    ) -> None:
        """
        Saves or overwrites synthesized material

        Args:
            thread_id: Thread identifier
            node_data: Data from node
            state_values: Current graph state values
        """
        if not self.artifacts_manager:
            logger.debug(
                "Artifacts manager not configured, skipping synthesized material save"
            )
            return
            
        session_id = self.artifacts_data.get(thread_id, {}).get("session_id")
        if not session_id:
            logger.warning(f"No session_id for thread {thread_id}, skipping synthesized material save")
            return
        
        # For edit_material take from state, for synthesis_material from node_data
        is_edit_node = node_data.get("last_action") == "edit"
        material = (state_values.get("synthesized_material") 
                    if is_edit_node
                    else node_data.get("synthesized_material", ""))
        
        if not material:
            logger.warning(f"No synthesized material to save for thread {thread_id}")
            return
        
        try:
            await self.artifacts_manager.push_synthesized_material(
                thread_id=thread_id,
                session_id=session_id,
                synthesized_material=material
            )
            action = "edited" if is_edit_node else "synthesized"
            logger.info(f"Successfully saved {action} material for thread {thread_id}")
            
            # Generate and track URL for synthesized material
            session_id = self.artifacts_data.get(thread_id, {}).get("session_id")
            if session_id:
                url = self._generate_web_ui_url(
                    thread_id=thread_id,
                    session_id=session_id,
                    file_name="synthesized_material.md"
                )
                self._track_artifact_url(
                    thread_id=thread_id,
                    artifact_type="synthesized_material",
                    url=url,
                    label="🔄 Synthesized material" if not is_edit_node else "✏️ Edited material"
                )
        except Exception as e:
            logger.error(f"Failed to save synthesized material for thread {thread_id}: {e}")

    async def _save_questions(
        self, thread_id: str, node_data: Dict, state_values: Dict
    ) -> None:
        """
        Saves assessment questions

        Args:
            thread_id: Thread identifier
            node_data: Data from node
            state_values: Current graph state values
        """
        if not self.artifacts_manager:
            logger.debug(
                "Artifacts manager not configured, skipping assessment questions save"
            )
            return
            
        session_id = self.artifacts_data.get(thread_id, {}).get("session_id")
        if not session_id:
            logger.warning(f"No session_id for thread {thread_id}, skipping assessment questions save")
            return
        
        questions = node_data.get("questions", [])
        if not questions:
            logger.warning(f"No assessment questions to save for thread {thread_id}")
            return
        
        try:
            # Save only questions without answers
            await self.artifacts_manager.push_questions_and_answers(
                thread_id=thread_id,
                session_id=session_id,
                questions=questions,
                questions_and_answers=[]  # Empty list, as answers are not yet available
            )
            logger.info(f"Successfully saved assessment questions for thread {thread_id}")
            
            # Generate and track URL for questions
            if session_id:
                url = self._generate_web_ui_url(
                    thread_id=thread_id,
                    session_id=session_id,
                    file_name="questions.md"
                )
                self._track_artifact_url(
                    thread_id=thread_id,
                    artifact_type="questions",
                    url=url,
                    label="❓ Assessment questions"
                )
        except Exception as e:
            logger.error(f"Failed to save assessment questions for thread {thread_id}: {e}")

    async def _save_answers(
        self, thread_id: str, node_data: Dict, state_values: Dict
    ) -> None:
        """
        Saves question answers

        Args:
            thread_id: Thread identifier
            node_data: Data from node
            state_values: Current graph state values
        """
        if not self.artifacts_manager:
            logger.debug(
                "Artifacts manager not configured, skipping answers save"
            )
            return
            
        session_id = self.artifacts_data.get(thread_id, {}).get("session_id")
        if not session_id:
            logger.warning(f"No session_id for thread {thread_id}, skipping answers save")
            return
        
        questions_and_answers = state_values.get("questions_and_answers", [])
        if not questions_and_answers:
            logger.warning(f"No answers to save for thread {thread_id}")
            return
        
        try:
            # Update file with questions and answers
            await self.artifacts_manager.push_questions_and_answers(
                thread_id=thread_id,
                session_id=session_id,
                questions=state_values.get("questions", []),
                questions_and_answers=questions_and_answers
            )
            logger.info(f"Successfully saved answers for thread {thread_id}")
            
            # Generate and track URL for answers
            if session_id:
                url = self._generate_web_ui_url(
                    thread_id=thread_id,
                    session_id=session_id,
                    file_name="questions_and_answers.md"
                )
                self._track_artifact_url(
                    thread_id=thread_id,
                    artifact_type="answers",
                    url=url,
                    label="✅ Questions with answers"
                )
        except Exception as e:
            logger.error(f"Failed to save answers for thread {thread_id}: {e}")
