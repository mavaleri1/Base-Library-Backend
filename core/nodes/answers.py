"""
Answer generation node for test questions.
Adapted from answer_question_node in main.ipynb for parallel processing.
"""

import logging
from typing import Dict, Any, Literal
from langchain_core.messages import SystemMessage
from langgraph.types import Command

# from ..utils.utils import render_system_prompt
from .base import BaseWorkflowNode


logger = logging.getLogger(__name__)


class AnswerGenerationNode(BaseWorkflowNode):
    """
    Node for generating answers to individual test questions.
    Used in parallel tasks via Send.
    """

    def __init__(self):
        super().__init__(logger)
        self.model = self.create_model()

    def get_node_name(self) -> str:
        """Returns node name for configuration lookup"""
        return "answer_question"
    
    def _build_context_from_state(self, state) -> dict:
        """Builds context for prompt from workflow state"""
        # In this node, state is a dict with keys 'question' and 'study_material'
        if isinstance(state, dict) and 'question' in state:
            return {
                "input_content": state['question'],
                "study_material": state.get('study_material', '')
            }
        return {}

    async def __call__(
        self, data: Dict[str, Any], config=None
    ) -> Command[Literal["__end__"]]:
        """
        Generates answer for one test question.

        Args:
            data: Dictionary with keys 'question' and 'study_material' for processing
            config: LangGraph configuration (optional)

        Returns:
            Command with transition to end and generated Q&A
        """
        question = data.get("question", "")
        study_material = data.get("study_material", "")

        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id", "unknown")
        else:
            thread_id = "unknown"

        logger.info(
            f"Generating answer for question in thread {thread_id}: {question[:100]}..."
        )

        try:
            # Create pseudo-state with question and material for passing to get_system_prompt
            state_dict = {
                "question": question,
                "study_material": study_material
            }
            
            # Get personalized prompt from service
            prompt_content = await self.get_system_prompt(state_dict, config)

            messages = [SystemMessage(content=prompt_content)]

            # Generate answer
            response = await self.model.ainvoke(messages)

            # Format Q&A for adding to state
            formatted_qna = f"## {question}\n\n{response.content}"

            logger.info(
                f"Answer generated successfully for question in thread {thread_id}"
            )

            return Command(
                goto="__end__",
                update={
                    "questions_and_answers": [formatted_qna],
                },
            )

        except Exception as e:
            logger.error(
                f"Error generating answer for question in thread {thread_id}: {str(e)}"
            )
            # In case of error, still complete but with error message
            error_qna = f"## {question}\n\n**Answer generation error:** {str(e)}"
            return Command(
                goto="__end__",
                update={
                    "questions_and_answers": [error_qna],
                },
            )
