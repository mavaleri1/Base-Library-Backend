"""
State models for LangGraph workflow.
"""

import operator
from typing import List, Any, Annotated, Literal, Optional
from pydantic import BaseModel, Field


class Questions(BaseModel):
    """Model for question generation"""

    questions: List[str] = Field(
        ...,
        description="Questions relevant to the exam question, which is either absent or insufficiently covered in the student's study material.",
    )


class QuestionsHITL(BaseModel):
    """Model for improving current questions"""

    next_step: Literal["clarify", "finalize"] = Field(
        ...,
        description="Indicates whether further clarification is needed (clarify) or if the questions are ready for use (finalize).",
    )
    questions: List[str] = Field(
        ...,
        description="Refined questions relevant to the exam question, which is either absent or insufficiently covered in the student's study material.",
    )


class GeneralState(BaseModel):
    """
    Main state for material processing workflow.
    """

    # Input data
    input_content: str = Field(
        default="", description="Input content for material creation"
    )
    display_name: Optional[str] = Field(
        default=None, description="Brief session name (3-5 words)"
    )

    # New fields for working with images
    image_paths: List[str] = Field(
        default_factory=list,
        description="Paths to uploaded note images (empty list = no images)",
    )
    recognized_notes: str = Field(
        default="", description="Recognized text from handwritten notes"
    )
    synthesized_material: str = Field(
        default="",
        description="Final synthesized material combining generated_material and recognized_notes",
    )

    # Generated content
    generated_material: str = Field(
        default="", description="Generated educational material"
    )

    # Questions
    questions: List[str] = Field(
        default_factory=list, description="List of additional questions"
    )

    # Accumulating fields (use operator.add for combining)
    questions_and_answers: Annotated[List[str], operator.add] = Field(
        default_factory=list, description="List of generated questions and answers"
    )

    # HITL feedback
    feedback_messages: List[Any] = Field(
        default_factory=list, description="Message history for HITL interaction"
    )
    
    # Edit agent fields (minimal for MVP)
    edit_count: int = Field(default=0, description="Total number of edits performed")
    needs_user_input: bool = Field(
        default=True, description="Flag for HITL interaction"
    )
    agent_message: Optional[str] = Field(
        default=None, description="Message from edit agent to user"
    )
    last_action: Optional[str] = Field(
        default=None, description="Type of last action (edit/message/complete)"
    )


# Edit agent structured output models
class ActionDecision(BaseModel):
    """Decision about action type for edit agent"""

    action_type: Literal["edit", "message", "complete"] = Field(
        description="Type of action to perform"
    )


class EditDetails(BaseModel):
    """Details for edit action"""

    old_text: str = Field(description="Exact text to replace")
    new_text: str = Field(description="Replacement text")
    continue_editing: bool = Field(
        default=True, description="Continue editing autonomously after this edit"
    )


class EditMessageDetails(BaseModel):
    """Details for user message from edit agent"""

    content: str = Field(description="Message to send to user")
