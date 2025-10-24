"""
Base classes for workflow nodes with LLM model configuration support.
"""

from abc import ABC, abstractmethod
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import Any, Dict
import logging
from ..models.model_factory import create_model_for_node
from ..config.config_models import ModelConfig
from ..config.settings import get_settings
from ..services.prompt_client import PromptConfigClient, WorkflowExecutionError


class BaseWorkflowNode(ABC):
    """
    Base class for all workflow nodes with LLM model configuration support.
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.settings = get_settings()
        self._init_security()
        self._init_prompt_client()
    
    def _init_prompt_client(self):
        """Initialize client for Prompt Configuration Service"""
        try:
            self.prompt_client = PromptConfigClient()
            self.logger.debug(f"Prompt client initialized for {self.__class__.__name__}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize prompt client: {e}")
            self.prompt_client = None

    @abstractmethod
    def get_node_name(self) -> str:
        """Returns node name for configuration lookup in graph.yaml"""
        pass

    def get_model_config(self) -> ModelConfig:
        """Gets model configuration for this node"""
        from ..config_manager import get_config_manager

        config_manager = get_config_manager()
        return config_manager.get_model_config(self.get_node_name())

    def create_model(self) -> ChatOpenAI:
        """Creates model based on configuration for this node"""
        return create_model_for_node(self.get_node_name())

    def _init_security(self):
        """Initialize SecurityGuard with configuration via yaml"""
        self.security_guard = None
        self.logger.debug(
            f"Initializing security guard. Enabled: {self.settings.security_enabled}"
        )

        if self.settings.security_enabled:
            try:
                from ..security.guard import SecurityGuard, InjectionResult
                from ..config.config_manager import get_config_manager
                from ..models.model_factory import get_model_factory

                # Get security guard configuration
                config_manager = get_config_manager()
                security_config = config_manager.get_model_config("security_guard")
                self.logger.debug(f"Got security config: {security_config}")
                
                # Create model through factory for correct provider support
                factory = get_model_factory()
                security_model = factory.create_model(security_config)
                
                # SecurityGuard now gets ready model
                self.security_guard = SecurityGuard(
                    model=security_model.with_structured_output(InjectionResult),
                    fuzzy_threshold=self.settings.security_fuzzy_threshold,
                )
                self.logger.info(
                    f"Security guard initialized successfully for {self.__class__.__name__}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize security guard: {e}")
                self.security_guard = None

    async def validate_input(self, content: str) -> str:
        """
        Universal validation of any user content.
        Always returns valid result (graceful degradation).

        Args:
            content: Content to validate

        Returns:
            Safe content (cleaned or original on error)
        """
        if (
            not self.security_guard
            or not content
            or len(content) < self.settings.security_min_content_length
        ):
            return content

        cleaned = await self.security_guard.validate_and_clean(content)

        if cleaned != content:
            self.logger.info(f"Content sanitized in {self.get_node_name()}")

        return cleaned

    async def get_system_prompt(self, state, config: RunnableConfig, extra_context: Dict[str, Any] = None) -> str:
        """
        Gets system prompt from Prompt Configuration Service.
        
        Args:
            state: Workflow state
            config: LangGraph configuration
            extra_context: Additional context for prompt (not from state)
            
        Returns:
            System prompt
            
        Raises:
            WorkflowExecutionError: When service is unavailable
        """
        if not self.prompt_client:
            raise WorkflowExecutionError("Prompt service is not configured")
        
        try:
            # Get thread_id from configuration
            thread_id = config["configurable"]["thread_id"]
            
            # Try to get user_id from workflow configuration
            user_id = config.get("configurable", {}).get("user_id")
            if user_id:
                self.logger.debug(f"Using user_id from workflow config: {user_id}")
            else:
                # If user_id not passed in configuration, try to extract from thread_id
                try:
                    # First try to interpret as UUID
                    import uuid
                    user_id = uuid.UUID(thread_id)
                    self.logger.debug(f"Using UUID thread_id as user_id: {user_id}")
                except (ValueError, TypeError):
                    try:
                        # If not UUID, try as number
                        user_id = int(thread_id)
                        self.logger.debug(f"Using numeric thread_id as user_id: {user_id}")
                    except (ValueError, TypeError):
                        # If neither, use default user_id = 1
                        self.logger.debug(f"Non-UUID/numeric thread_id format: {thread_id}. Using default user_id=1")
                        user_id = 1
            
            # Form context from workflow state
            context = self._build_context_from_state(state)
            
            # Add additional context if exists
            if extra_context:
                context.update(extra_context)
            
            self.logger.debug(f"Context for prompt generation: {list(context.keys())}")
            
            # Get prompt from service
            prompt = await self.prompt_client.generate_prompt(
                user_id=user_id,
                node_name=self.get_node_name(),
                context=context
            )
            
            self.logger.info(f"Received personalized prompt from service for user {user_id}")
            return prompt
            
        except WorkflowExecutionError:
            # Re-raise service errors without changes
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error getting prompt: {e}")
            raise WorkflowExecutionError(f"Failed to get prompt: {e}")
    
    
    def _build_context_from_state(self, state) -> Dict[str, Any]:
        """
        Builds context for prompt from workflow state.
        Subclasses should override for specific mapping.
        
        Args:
            state: Workflow state
            
        Returns:
            Dictionary with contextual data
        """
        # Base implementation returns empty context
        # Each node should override for its mapping
        return {}


class FeedbackNode(BaseWorkflowNode):
    """
    Abstract base class for nodes implementing the pattern
    "generation — feedback — editing — completion".
    """

    def __init__(self, logger: logging.Logger = None):
        super().__init__(logger)

    @abstractmethod
    def is_initial(self, state) -> bool:
        """Whether to do first generation"""
        pass

    @abstractmethod
    def get_template_type(self) -> str:
        """Returns template type for prompt"""
        pass

    @abstractmethod
    def get_prompt_kwargs(
        self, state, user_feedback: str = None, config=None
    ) -> Dict[str, Any]:
        """Returns parameters for prompt"""
        pass

    async def render_prompt(self, state, user_feedback: str = None, config=None) -> str:
        """Forms prompt for LLM using initial/further logic"""
        # Determine template variant
        if user_feedback or not self.is_initial(state):
            template_variant = "further"
        else:
            template_variant = "initial"

        # Get parameters for prompt
        prompt_kwargs = self.get_prompt_kwargs(state, user_feedback, config)
        
        # Add template_variant to extra_context
        extra_context = {
            "template_variant": template_variant,
            **prompt_kwargs
        }
        
        # Call get_system_prompt with extra_context
        return await self.get_system_prompt(state, config, extra_context)

    @abstractmethod
    def get_model(self):
        """Returns LLM/chain"""
        pass

    @abstractmethod
    def format_initial_response(self, response) -> str:
        pass

    @abstractmethod
    def is_approved(self, response) -> bool:
        pass

    @abstractmethod
    def get_next_node(self, state, approved: bool = False) -> str:
        pass

    @abstractmethod
    def get_user_prompt(self) -> str:
        pass

    @abstractmethod
    def get_update_on_approve(self, state, response) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_current_node_name(self) -> str:
        pass

    # ------- helpers -------
    def get_initial_update(self, response) -> Dict[str, Any]:
        formatted = self.format_initial_response(response)
        return {"feedback_messages": [AIMessage(content=formatted)]}

    def get_continue_update(
        self, state, user_feedback: str, response
    ) -> Dict[str, Any]:
        self.logger.debug(f"User feedback: {user_feedback}")
        self.logger.debug(f"Response: {response}")
        formatted = self.format_initial_response(response)
        self.logger.debug(f"Formatted: {formatted}")
        return {
            "feedback_messages": state.feedback_messages
            + [
                HumanMessage(content=user_feedback),
                AIMessage(content=formatted),
            ]
        }

    async def __call__(self, state, config: RunnableConfig) -> Command:
        thread_id = config["configurable"]["thread_id"]
        self.logger.debug(
            f"Processing {self.__class__.__name__} for thread {thread_id}"
        )

        # 1. First generation
        if self.is_initial(state):
            prompt = await self.render_prompt(state, config=config)
            model = self.get_model()
            response = await model.ainvoke([SystemMessage(content=prompt)])
            return Command(
                goto=self.get_current_node_name(),
                update=self.get_initial_update(response),
            )

        # 2. Request feedback
        messages_for_user = [state.feedback_messages[-1].content]
        if len(state.feedback_messages) == 1:
            messages_for_user.append(self.get_user_prompt())
        interrupt_json = {"message": messages_for_user}
        user_feedback = interrupt(interrupt_json)

        # Validate user feedback with graceful degradation
        if user_feedback and self.security_guard:
            user_feedback = await self.validate_input(user_feedback)

        # 3. Edit with feedback consideration
        prompt = await self.render_prompt(state, user_feedback=user_feedback, config=config)
        model = self.get_model()
        messages = (
            [SystemMessage(content=prompt)]
            + state.feedback_messages
            + [HumanMessage(content=user_feedback)]
        )
        response = await model.ainvoke(messages)
        self.logger.debug(f"Response: {response}")

        # 4. Check approval
        if self.is_approved(response):
            self.logger.debug(f"Approved: {response}")
            return Command(
                goto=self.get_next_node(state, approved=True),
                update=self.get_update_on_approve(state, response),
            )

        self.logger.debug(f"Not approved: {response}")
        goto = self.get_current_node_name()
        self.logger.debug(f"Goto: {goto}")
        update = self.get_continue_update(state, user_feedback, response)
        self.logger.debug(f"Update: {update}")
        return Command(
            goto=goto,
            update=update,
        )
