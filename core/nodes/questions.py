"""
Question generation node with HITL logic.
Adapted from generating_questions_node in main.ipynb using FeedbackNode pattern.
"""

import logging
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import Send
from langgraph.types import Command

from .base import FeedbackNode
from ..core.state import GeneralState, Questions, QuestionsHITL
from ..utils.utils import Config
from ..services.hitl_manager import get_hitl_manager


logger = logging.getLogger(__name__)


class QuestionGenerationNode(FeedbackNode):
    """
    Control question generation node with HITL cycle.
    Uses FeedbackNode pattern for user interaction.
    """

    def __init__(self):
        super().__init__(logger)
        self.config = Config()
        self.model = self.create_model()

    def get_node_name(self) -> str:
        """Returns node name for configuration lookup"""
        return "generating_questions"
    
    async def get_system_prompt(self, state, config, extra_context: Dict[str, Any] = None) -> str:
        """Override to support further prompt variant"""
        # Determine node name considering variant
        node_name = self.get_node_name()
        if extra_context and extra_context.get('template_variant') == 'further':
            node_name = f"{node_name}_further"
            self.logger.debug(f"Using further variant, node name: {node_name}")
        
        # Temporarily replace get_node_name for parent method call
        original_get_node_name = self.get_node_name
        self.get_node_name = lambda: node_name
        
        try:
            # Call parent method with replaced node name
            result = await super().get_system_prompt(state, config, extra_context)
        finally:
            # Restore original method
            self.get_node_name = original_get_node_name
        
        return result
    
    def _build_context_from_state(self, state) -> Dict[str, Any]:
        """Builds context for prompt from workflow state"""
        # FeedbackNode will use prompt_kwargs from get_prompt_kwargs
        # which already contains correct mapping
        return {}

    def is_initial(self, state: GeneralState) -> bool:
        """Checks if first generation is needed"""
        return not state.feedback_messages

    def get_template_type(self) -> str:
        """Returns template type for prompt"""
        return "gen_question"

    def get_prompt_kwargs(
        self, state: GeneralState, user_feedback: str = None, config=None
    ) -> Dict[str, Any]:
        """Returns prompt parameters depending on variant"""
        # Use synthesized_material if available, otherwise generated_material as fallback
        study_material = state.synthesized_material or state.generated_material

        if user_feedback is None:
            # Primary generation (initial variant) # TODO: why is Code unreachable?
            self._current_stage = "initial"
            return {
                "input_content": state.input_content,
                "study_material": study_material,
            }
        else:
            # Refinement based on feedback (further variant)
            self._current_stage = "refine"
            return {
                "input_content": state.input_content,
                "study_material": study_material,
                "current_questions": "\n".join(state.questions),
            }

    def get_model(self):
        """Returns model for generation with structured output"""
        # Use staged logic from get_prompt_kwargs
        if hasattr(self, "_current_stage") and self._current_stage == "refine":
            return self.model.with_structured_output(QuestionsHITL)
        return self.model.with_structured_output(Questions)

    def format_initial_response(self, response) -> str:
        """Formats response for user display"""
        questions = response.questions
        # Format questions as numbered list
        return "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])

    def is_approved(self, response: QuestionsHITL) -> bool:
        """Checks if questions are ready for finalization"""
        return response.next_step == "finalize"

    def get_next_node(self, state: GeneralState, approved: bool = False) -> str:
        """Determines next node"""
        if approved:
            # Use synthesized_material if available, otherwise generated_material
            study_material = state.synthesized_material or state.generated_material
            # Return list of parallel tasks with study_material transfer
            return [
                Send("answer_question", {
                    "question": question,
                    "study_material": study_material
                })
                for question in state.questions
            ]
        return "generating_questions"

    def get_user_prompt(self) -> str:
        """Return user prompt"""
        return "Evaluate the proposed questions. You can request changes or confirm that the questions are ready to be used."

    def get_update_on_approve(self, state: GeneralState, response) -> Dict[str, Any]:
        """Update state on approval"""
        return {
            "questions": response.questions,
            "feedback_messages": [],  # Clear feedback history
        }

    def get_current_node_name(self) -> str:
        """Current node name"""
        return "generating_questions"

    def get_initial_update(self, response) -> Dict[str, Any]:
        """Override to save questions in state"""
        formatted = self.format_initial_response(response)
        return {
            "questions": response.questions,
            "feedback_messages": [AIMessage(content=formatted)],
        }

    def get_continue_update(
        self, state, user_feedback: str, response
    ) -> Dict[str, Any]:
        """Override to update questions"""
        self.logger.debug(f"User feedback: {user_feedback}")
        self.logger.debug(f"Response: {response}")
        formatted = self.format_initial_response(response)
        self.logger.debug(f"Formatted: {formatted}")
        return {
            "questions": response.questions,
            "feedback_messages": state.feedback_messages
            + [
                HumanMessage(content=user_feedback),
                AIMessage(content=formatted),
            ],
        }

    async def __call__(self, state, config) -> Command:
        """Override to check HITL settings before running feedback loop"""
        thread_id = config["configurable"]["thread_id"]
        self.logger.debug(f"Processing QuestionGenerationNode for thread {thread_id}")

        # Check HITL settings
        hitl_manager = get_hitl_manager()
        hitl_enabled = hitl_manager.is_enabled("generating_questions", thread_id)
        self.logger.info(f"HITL for generating_questions: {hitl_enabled}")

        if not hitl_enabled:
            # Run autonomous generation without HITL
            self.logger.info(
                "HITL disabled for generating_questions, running autonomous generation"
            )

            prompt = await self.render_prompt(state, config=config)
            model = self.model.with_structured_output(Questions)
            response = await model.ainvoke([SystemMessage(content=prompt)])

            # Move directly to answer generation
            # Use synthesized_material if available, otherwise generated_material
            study_material = state.synthesized_material or state.generated_material
            return Command(
                goto=[
                    Send("answer_question", {
                        "question": question,
                        "study_material": study_material
                    })
                    for question in response.questions
                ],
                update={
                    "questions": response.questions,
                    "feedback_messages": [],
                    "agent_message": "Questions generated automatically (autonomous mode)",
                },
            )

        # Run normal HITL flow
        return await super().__call__(state, config)
