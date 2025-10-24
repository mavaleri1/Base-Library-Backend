"""
Node for iterative editing of synthesized material.
Minimal MVP integration based on working code from Jupyter notebook.
"""

import logging
from typing import Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import interrupt, Command
from fuzzysearch import find_near_matches

from .base import BaseWorkflowNode
from ..core.state import GeneralState, ActionDecision, EditDetails, EditMessageDetails
# from ..utils.utils import render_system_prompt
from ..services.hitl_manager import get_hitl_manager


class EditMaterialNode(BaseWorkflowNode):
    """
    Node for editing synthesized material.
    Uses HITL pattern for iterative edits.
    """

    def __init__(self, logger: logging.Logger = None):
        super().__init__(logger)
        self.model = self.create_model()  # Initialize on first call

    def get_node_name(self) -> str:
        """Returns node name for configuration"""
        return "edit_material"
    
    def _build_context_from_state(self, state) -> dict:
        """Builds context for prompt from workflow state"""
        context = {}
        
        if hasattr(state, 'synthesized_material'):
            context['generated_material'] = state.synthesized_material
            
        return context

    def get_model(self):
        """Returns model for LLM access"""
        return self.model

    def fuzzy_find_and_replace(
        self, document: str, target: str, replacement: str, threshold: float = 0.85
    ) -> Tuple[str, bool, Optional[str], float]:
        """
        Fuzzy search and replace text in document.
        Direct port from Jupyter notebook.

        Returns: (new_document, success, found_text, similarity)
        """
        # Edge case: empty strings
        if not target or not document:
            return document, False, None, 0.0

        # For short strings - only exact match
        if len(target) < 10:
            if target in document:
                idx = document.index(target)
                new_doc = document[:idx] + replacement + document[idx + len(target) :]
                return new_doc, True, target, 1.0
            return document, False, None, 0.0

        # Calculate distance
        max_distance = max(1, int(len(target) * (1 - threshold)))

        # For very long strings, limit distance
        if len(target) > 100:
            max_distance = min(max_distance, 15)

        # Search
        try:
            matches = find_near_matches(target, document, max_l_dist=max_distance)
        except Exception as e:
            self.logger.error(f"Fuzzy search error: {e}")
            return document, False, None, 0.0

        if not matches:
            return document, False, None, 0.0

        # Take first match
        match = matches[0]

        # Calculate similarity
        if len(target) > 0:
            similarity = max(0.0, 1 - (match.dist / len(target)))
        else:
            similarity = 1.0 if match.dist == 0 else 0.0

        # Replace
        new_document = document[: match.start] + replacement + document[match.end :]

        return new_document, True, match.matched, similarity

    async def handle_edit_action(
        self, state: GeneralState, action: EditDetails, messages: list
    ) -> Command:
        """Handle edit action"""
        document = state.synthesized_material

        # Use fuzzy_find_and_replace
        new_document, success, found_text, similarity = self.fuzzy_find_and_replace(
            document, action.old_text, action.new_text
        )

        if not success:
            # Text not found
            error_msg = "Specified text not found in document. Please check the fragment and try again."
            self.logger.warning(
                f"Text not found: '{action.old_text[:50]}...' (similarity: {similarity:.2f})"
            )

            messages.append(SystemMessage(content=f"[EDIT ERROR]: {error_msg}"))

            return Command(
                goto="edit_material",
                update={
                    "feedback_messages": messages,
                    "needs_user_input": False,
                    "last_action": "edit_error",
                },
            )

        # Successful editing
        edit_count = state.edit_count + 1
        self.logger.info(f"Edit #{edit_count} applied (similarity: {similarity:.2f})")

        messages.append(
            SystemMessage(
                content=f"[EDIT SUCCESS #{edit_count}]: Replaced text (similarity: {similarity:.2f})"
            )
        )

        # Update state
        update_dict = {
            "synthesized_material": new_document,
            "feedback_messages": messages,
            "edit_count": edit_count,
            "needs_user_input": not action.continue_editing,
            "last_action": "edit",
        }

        # If not continuing autonomously, set message
        if not action.continue_editing:
            update_dict["agent_message"] = "Edit applied. What other changes are needed?"

        return Command(goto="edit_material", update=update_dict)

    async def handle_message_action(
        self, state: GeneralState, action: EditMessageDetails, messages: list
    ) -> Command:
        """Handle user message"""
        messages.append(AIMessage(content=action.content))

        return Command(
            goto="edit_material",
            update={
                "feedback_messages": messages,
                "needs_user_input": True,
                "agent_message": action.content,
                "last_action": "message",
            },
        )

    async def handle_complete_action(self, state: GeneralState) -> Command:
        """Complete editing"""
        self.logger.info("Edit session completed")

        return Command(
            goto="generating_questions",  # Move to next node
            update={
                "needs_user_input": True,  # Reset flag for next node
                "agent_message": None,
                "last_action": "complete",
                "feedback_messages": [],
            },
        )

    async def __call__(self, state: GeneralState, config: RunnableConfig) -> Command:
        """
        Main editing node logic.
        Handles cycle: input request -> analysis -> action -> repeat
        """
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        self.logger.debug(f"EditMaterialNode called for thread {thread_id}")

        # Check HITL settings
        hitl_manager = get_hitl_manager()
        hitl_enabled = hitl_manager.is_enabled("edit_material", thread_id)
        self.logger.info(f"HITL for edit_material: {hitl_enabled}")

        # Get message history
        messages = state.feedback_messages.copy() if state.feedback_messages else []

        # Check if there's material to edit
        if not state.synthesized_material:
            self.logger.warning("No synthesized material to edit")
            return Command(
                goto="generating_questions",
                update={"agent_message": "No material to edit"},
            )

        # If HITL disabled, skip this node
        if not hitl_enabled:
            self.logger.info("HITL disabled for edit_material, skipping to next node")
            return Command(
                goto="generating_questions",
                update={
                    "agent_message": "Material accepted without editing (autonomous mode)",
                    "last_action": "skip_hitl",
                },
            )

        # Request user input if needed
        if state.needs_user_input:
            msg_to_user = state.agent_message or "Which changes to make to the material? "

            # Use interrupt to get input
            interrupt_data = {"message": [msg_to_user]}
            user_feedback = interrupt(interrupt_data)

            if user_feedback:
                # Validate edit request in HITL cycle
                if self.security_guard:
                    user_feedback = await self.validate_input(user_feedback)

                messages.append(HumanMessage(content=user_feedback))

                # Reset flags and continue processing
                return Command(
                    goto="edit_material",
                    update={
                        "feedback_messages": messages,
                        "agent_message": None,
                        "needs_user_input": False,
                    },
                )

        # Get personalized prompt from service with additional context
        extra_context = {
            "template_variant": "initial",
            "generated_material": state.synthesized_material if hasattr(state, 'synthesized_material') else ""
        }
        system_prompt = await self.get_system_prompt(state, config, extra_context)

        # Step 1: Determine action type
        model = self.get_model()
        decision = await model.with_structured_output(ActionDecision).ainvoke(
            [SystemMessage(content=system_prompt)] + messages
        )

        self.logger.debug(f"Action decision: {decision.action_type}")
        messages.append(AIMessage(content=decision.model_dump_json()))

        # Step 2: Execute action based on type
        if decision.action_type == "edit":
            details = await model.with_structured_output(EditDetails).ainvoke(
                [SystemMessage(content=system_prompt)] + messages
            )

            self.logger.info(f"Edit details: {details.model_dump_json()}")

            return await self.handle_edit_action(state, details, messages)

        elif decision.action_type == "message":
            details = await model.with_structured_output(EditMessageDetails).ainvoke(
                [SystemMessage(content=system_prompt)] + messages
            )
            self.logger.info(f"Edit message details: {details.model_dump_json()}")
            return await self.handle_message_action(state, details, messages)

        elif decision.action_type == "complete":
            return await self.handle_complete_action(state)

        # Should not happen, but just in case
        self.logger.error(f"Unknown action type: {decision.action_type}")
        return Command(
            goto="edit_material",
            update={
                "needs_user_input": True,
                "agent_message": "An error occurred. Please try again.",
            },
        )
