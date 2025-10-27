"""
Educational notes processing node.
Simple logic without HITL: processes images if available, requests once if not.
"""

import base64
import logging
from typing import List
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.types import Command, interrupt

from ..core.state import GeneralState
from .base import BaseWorkflowNode


logger = logging.getLogger(__name__)


def load_images_as_base64(image_paths: List[str]) -> List[str]:
    """
    Loads images in base64 format.

    Args:
        image_paths: List of image paths

    Returns:
        List of base64 image strings
    """
    base64_images = []
    for image_path in image_paths:
        try:
            with open(image_path, "rb") as image_file:
                base64_string = base64.b64encode(image_file.read()).decode("utf-8")
                base64_images.append(base64_string)
                logger.info(f"Loaded image: {image_path}")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")

    return base64_images


class RecognitionNode(BaseWorkflowNode):
    """
    Educational notes processing node with support for:
    - Processing notes in any format (images or text)
    - Direct text notes input
    - Minimum text length validation
    - Skipping processing at user's request
    """
    
    MIN_TEXT_LENGTH = 50  # Minimum length for valid note text

    def __init__(self):
        super().__init__(logger)
        self.model = self.create_model()

    def get_node_name(self) -> str:
        """Returns node name for configuration lookup"""
        return "recognition_handwritten"
    
    def _build_context_from_state(self, state) -> dict:
        """Builds context for prompt from workflow state"""
        return {
            # Recognition node does not require context from state for prompt
        }

    async def __call__(self, state: GeneralState, config) -> Command:
        """
        Main recognition node logic.

        Args:
            state: Current state with potential image_paths
            config: LangGraph configuration

        Returns:
            Command with transition to next node
        """
        thread_id = config["configurable"]["thread_id"]
        logger.info(f"Starting recognition processing for thread {thread_id}")

        # Case 1: Images available - process them
        if state.image_paths:
            logger.info(
                f"Found {len(state.image_paths)} images, processing recognition"
            )

            try:
                # Process images
                recognized_text = await self._process_images(state.image_paths, state, config)

                if recognized_text:
                    logger.info(
                        f"Successfully recognized text from images for thread {thread_id}"
                    )
                    return Command(
                        goto="synthesis_material",
                        update={"recognized_notes": recognized_text},
                    )
                else:
                    logger.warning(
                        f"Failed to recognize text from images for thread {thread_id}"
                    )
                    # Skip synthesis on recognition error
                    return Command(
                        goto="generating_questions",
                        update={
                            "recognized_notes": "",
                            "synthesized_material": state.generated_material
                        }
                    )

            except Exception as e:
                logger.error(f"Error processing images for thread {thread_id}: {e}")
                # Skip synthesis in case of error
                return Command(
                    goto="generating_questions",
                    update={
                        "recognized_notes": "",
                        "synthesized_material": state.generated_material
                    }
                )

        # Case 2: No images - request notes from user
        logger.info(f"No images found for thread {thread_id}, requesting notes from user")

        # Request notes from user (images or text)
        message_content = (
            "📸 To improve material quality, you can add notes from classes.\n\n"
            "Action options:\n"
            "• Send photos of notes or paste text\n"
            "• Materials accepted in any format\n"
            "• Write 'skip' to continue without notes"
        )

        # Make interrupt to get user response
        interrupt_json = {"message": [message_content]}
        user_response = interrupt(interrupt_json)

        # Process user response
        # Check text length - less than 50 characters means skip
        cleaned_text = user_response.strip()
        if len(cleaned_text) < self.MIN_TEXT_LENGTH:
            logger.info(f"Text too short ({len(cleaned_text)} chars), user wants to skip notes for thread {thread_id}")
            # Text too short - user wants to skip
            return Command(
                goto="generating_questions",
                update={
                    "recognized_notes": "",
                    "synthesized_material": state.generated_material,  # Use generated_material as final
                },
            )
        
        # Text sufficient length - use as recognized notes
        logger.info(f"Received text notes ({len(cleaned_text)} chars) for thread {thread_id}, proceeding to synthesis")
        return Command(
            goto="synthesis_material",
            update={"recognized_notes": cleaned_text}
        )

    async def _process_images(self, image_paths: List[str], state: GeneralState, config) -> str:
        """
        Processes images using GPT-4-vision.

        Args:
            image_paths: List of image paths
            state: Workflow state
            config: LangGraph configuration

        Returns:
            Recognized text or empty string on error
        """
        import time
        start_time = time.time()
        try:
            # Load images in base64
            base64_images = load_images_as_base64(image_paths)
            if not base64_images:
                logger.error("Failed to load any images for recognition")
                return ""

            # Get personalized prompt from service
            system_content = await self.get_system_prompt(state, config)
            
            # Log prompt details
            logger.info(f"🔍 [RECOGNITION] System prompt details:")
            logger.info(f"🔍 [RECOGNITION] - Length: {len(system_content)} chars")
            logger.info(f"🔍 [RECOGNITION] - Full prompt: {system_content}")
            
            # Check if prompt contains vision-related instructions
            vision_keywords = ['image', 'vision', 'see', 'look', 'recognize', 'text', 'handwritten', 'notes']
            prompt_lower = system_content.lower()
            found_keywords = [kw for kw in vision_keywords if kw in prompt_lower]
            logger.info(f"🔍 [RECOGNITION] - Vision keywords found: {found_keywords}")

            # Create content with images for GPT-4-vision
            user_content = [
                {
                    "type": "text",
                    "text": "Here are handwritten notes images for recognition:",
                }
            ]

            for base64_img in base64_images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                    }
                )

            # Create messages for model
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=user_content),
            ]

            # Log request details
            logger.info(f"🔍 [RECOGNITION] Sending request to OpenAI API")
            logger.info(f"🔍 [RECOGNITION] Model: {self.model.model_name}")
            logger.info(f"🔍 [RECOGNITION] Temperature: {self.model.temperature}")
            logger.info(f"🔍 [RECOGNITION] Max tokens: {self.model.max_tokens}")
            logger.info(f"🔍 [RECOGNITION] System prompt length: {len(system_content)} chars")
            logger.info(f"🔍 [RECOGNITION] Number of images: {len(base64_images)}")
            logger.info(f"🔍 [RECOGNITION] System prompt preview: {system_content[:200]}...")
            
            # Log message structure
            logger.info(f"🔍 [RECOGNITION] Message structure:")
            logger.info(f"🔍 [RECOGNITION] - Number of messages: {len(messages)}")
            for i, msg in enumerate(messages):
                logger.info(f"🔍 [RECOGNITION] - Message {i}: {type(msg).__name__}")
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, list):
                        logger.info(f"🔍 [RECOGNITION]   - Content type: list with {len(msg.content)} items")
                        for j, item in enumerate(msg.content):
                            if isinstance(item, dict):
                                logger.info(f"🔍 [RECOGNITION]   - Item {j}: {item.get('type', 'unknown')}")
                                if item.get('type') == 'image_url':
                                    img_url = item.get('image_url', {}).get('url', '')
                                    logger.info(f"🔍 [RECOGNITION]   - Image URL length: {len(img_url)} chars")
                                    logger.info(f"🔍 [RECOGNITION]   - Image URL preview: {img_url[:100]}...")
                            else:
                                logger.info(f"🔍 [RECOGNITION]   - Item {j}: {str(item)[:100]}...")
                    else:
                        logger.info(f"🔍 [RECOGNITION]   - Content type: {type(msg.content)}")
                        logger.info(f"🔍 [RECOGNITION]   - Content length: {len(str(msg.content))} chars")
                        logger.info(f"🔍 [RECOGNITION]   - Content preview: {str(msg.content)[:200]}...")

            # Send request to model
            response = await self.model.ainvoke(messages)

            # Log response details
            logger.info(f"🔍 [RECOGNITION] Received response from OpenAI API")
            logger.info(f"🔍 [RECOGNITION] Response type: {type(response)}")
            logger.info(f"🔍 [RECOGNITION] Response attributes: {dir(response)}")
            
            # Check for different response formats
            if hasattr(response, 'content'):
                logger.info(f"🔍 [RECOGNITION] Response content length: {len(response.content)}")
                logger.info(f"🔍 [RECOGNITION] Response content preview: {response.content[:200]}...")
                logger.info(f"🔍 [RECOGNITION] Response content full: {response.content}")
            elif hasattr(response, 'text'):
                logger.info(f"🔍 [RECOGNITION] Response text length: {len(response.text)}")
                logger.info(f"🔍 [RECOGNITION] Response text preview: {response.text[:200]}...")
                logger.info(f"🔍 [RECOGNITION] Response text full: {response.text}")
            elif hasattr(response, 'message'):
                logger.info(f"🔍 [RECOGNITION] Response message: {response.message}")
                if hasattr(response.message, 'content'):
                    logger.info(f"🔍 [RECOGNITION] Response message content: {response.message.content}")
            else:
                logger.warning(f"🔍 [RECOGNITION] Unknown response format: {response}")
                logger.warning(f"🔍 [RECOGNITION] Response str: {str(response)}")
            
            # Check if response has expected attributes
            if not hasattr(response, 'content'):
                logger.error(f"🔍 [RECOGNITION] Response missing 'content' attribute. Available attributes: {dir(response)}")
                return ""

            # Process response (remove reasoning section if present)
            content = response.content
            logger.info(f"🔍 [RECOGNITION] Raw content length: {len(content)} chars")
            
            if "[END OF REASONING]" in content:
                # Remove the [END OF REASONING] marker and any text before it
                # since it's at the end according to logs
                content = content.replace("[END OF REASONING]", "").strip()
                logger.info(f"🔍 [RECOGNITION] Removed [END OF REASONING] marker, final length: {len(content)} chars")

            # Validation of recognized text from handwritten notes
            if self.security_guard and content:
                logger.info(f"🔍 [RECOGNITION] Running security validation on content")
                content = await self.validate_input(content)
                logger.info(f"🔍 [RECOGNITION] Security validation completed, final content length: {len(content)} chars")

            elapsed = time.time() - start_time
            if elapsed > 5.0:
                logger.warning(f"Image recognition completed in {elapsed:.2f}s (slow), text length: {len(content)} chars")
            else:
                logger.info(f"Image recognition completed in {elapsed:.2f}s, text length: {len(content)} chars")
            
            # Final validation
            if not content or len(content.strip()) == 0:
                logger.warning(f"🔍 [RECOGNITION] WARNING: Empty content after processing!")
                logger.warning(f"🔍 [RECOGNITION] Raw response was: {response.content[:500] if hasattr(response, 'content') else 'No content'}")
            else:
                logger.info(f"🔍 [RECOGNITION] SUCCESS: Recognized text with {len(content)} characters")
            
            return content

        except Exception as e:
            logger.error(f"Error in image processing: {e}")
            logger.error(f"🔍 [RECOGNITION] Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"🔍 [RECOGNITION] Traceback: {traceback.format_exc()}")
            return ""
