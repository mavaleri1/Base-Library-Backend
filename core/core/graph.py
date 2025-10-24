"""
LangGraph workflow assembly for processing exam materials.
Combines all nodes into a single graph with proper transitions.
"""

import logging
from langgraph.graph import StateGraph

from .state import GeneralState
from ..nodes import (
    InputProcessingNode,
    ContentGenerationNode,
    RecognitionNode,
    SynthesisNode,
    EditMaterialNode,
    QuestionGenerationNode,
    AnswerGenerationNode,
)


logger = logging.getLogger(__name__)


def create_workflow() -> StateGraph:
    """
    Creates and configures LangGraph workflow for processing exam materials.

    New execution flow with image support and editing:
    1. START -> input_processing (user input analysis)
    2. input_processing -> generating_content (educational material generation)
    3. generating_content -> recognition_handwritten (handwritten notes recognition with HITL)
    4. recognition_handwritten -> synthesis_material (final material synthesis)
    5. synthesis_material -> edit_material (iterative editing with HITL)
    6. edit_material -> generating_questions (control questions generation with HITL)
    7. generating_questions -> answer_question (parallel answer generation)
    8. answer_question -> END

    Returns:
        StateGraph: Configured workflow graph
    """
    logger.info("Creating enhanced exam workflow with image recognition...")

    # Create graph with typed state
    workflow = StateGraph(GeneralState)

    # Initialize all nodes
    input_processing_node = InputProcessingNode()
    content_node = ContentGenerationNode()
    recognition_node = RecognitionNode()
    synthesis_node = SynthesisNode()
    edit_material_node = EditMaterialNode()
    questions_node = QuestionGenerationNode()
    answers_node = AnswerGenerationNode()

    # Add nodes to graph
    workflow.add_node("input_processing", input_processing_node)
    workflow.add_node("generating_content", content_node)
    workflow.add_node("recognition_handwritten", recognition_node)
    workflow.add_node("synthesis_material", synthesis_node)
    workflow.add_node("edit_material", edit_material_node)
    workflow.add_node("generating_questions", questions_node)
    workflow.add_node("answer_question", answers_node)

    # Set entry point
    workflow.set_entry_point("input_processing")

    # Configure transitions between nodes:
    # - input_processing -> generating_content (Command)
    # - generating_content -> recognition_handwritten (Command)
    # - recognition_handwritten -> synthesis_material (Command, with HITL cycle)
    # - synthesis_material -> edit_material (Command)
    # - edit_material -> edit_material (HITL cycle for iterative edits)
    # - edit_material -> generating_questions (Command after completion)
    # - generating_questions -> generating_questions (HITL cycle through Command)
    # - generating_questions -> answer_question (parallel Send through Command)
    # - answer_question -> END (Command)

    logger.info(
        "Enhanced exam workflow created successfully with image recognition support"
    )
    return workflow
