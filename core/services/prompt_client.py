"""
HTTP client for interacting with Prompt Configuration Service.
"""

import asyncio
import httpx
from typing import Optional, Dict, Any
import logging
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class WorkflowExecutionError(Exception):
    """Exception for critical workflow errors requiring stop."""
    pass


class PromptConfigClient:
    """
    MVP version of client for Prompt Configuration Service with retry mechanism.
    Throws WorkflowExecutionError without fallback when service is unavailable.
    """
    
    def __init__(self, base_url: str = None, timeout: int = None, retry_count: int = None):
        self.settings = get_settings()
        self.base_url = base_url or self.settings.prompt_service_url
        self.timeout = timeout or self.settings.prompt_service_timeout
        self.retry_count = retry_count or self.settings.prompt_service_retry_count
        self.retry_delay = 0.5  # seconds
        self.logger = logger
    
    async def generate_prompt(self, user_id, node_name: str, context: Dict[str, Any]) -> str:
        """
        Gets prompt from configuration service with retry mechanism.
        Throws error when service is unavailable (without fallback).
        
        Args:
            user_id: User ID (UUID or int)
            node_name: LangGraph node name (e.g., 'generating_content')
            context: Workflow context for template substitution
            
        Returns:
            str: Generated prompt
            
        Raises:
            WorkflowExecutionError: When service is unavailable after all retry attempts
        """
        import time
        import uuid
        start_time = time.time()
        last_error: Optional[Exception] = None
        
        # Convert user_id to UUID string
        if isinstance(user_id, uuid.UUID):
            user_id_str = str(user_id)
        elif isinstance(user_id, int):
            user_id_str = str(uuid.UUID(int=user_id))
        else:
            # If already string, use as is
            user_id_str = str(user_id)
        
        self.logger.info(f"Requesting prompt for user_id={user_id_str}, node={node_name}")
        
        for attempt in range(self.retry_count):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/api/v1/generate-prompt",
                        json={
                            "user_id": user_id_str,
                            "node_name": node_name,
                            "context": context
                        }
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    prompt = result.get("prompt")
                    
                    if not prompt:
                        raise ValueError("Empty prompt received from service")
                    
                    # Validate minimum prompt length
                    if len(prompt) < 50:
                        raise ValueError(f"Prompt too short ({len(prompt)} chars): {prompt[:100]}")
                    
                    
                    elapsed = time.time() - start_time
                    if elapsed > 2.0:
                        self.logger.warning(f"Successfully received prompt ({len(prompt)} chars) for {node_name} in {elapsed:.2f}s (slow)")
                    else:
                        self.logger.info(f"Successfully received prompt ({len(prompt)} chars) for {node_name} in {elapsed:.2f}s")
                    return prompt
                    
            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError, ValueError) as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1}/{self.retry_count} failed for {node_name}: {e}")
                
                if attempt < self.retry_count - 1:
                    # Exponential delay between attempts
                    delay = self.retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
        
        # All attempts exhausted - throw error
        error_msg = (
            f"Prompt configuration service is unavailable after {self.retry_count} attempts. "
            f"Please try again in a few minutes. Last error: {last_error}"
        )
        self.logger.error(error_msg)
        raise WorkflowExecutionError(error_msg)
    