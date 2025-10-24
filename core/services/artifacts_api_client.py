#!/usr/bin/env python3
"""
API client for artifacts-service
"""

import logging
import httpx
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ArtifactsAPIClient:
    """Client for working with artifacts-service via API"""
    
    def __init__(self, base_url: str = "http://artifacts-service:8001"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def create_file(
        self,
        thread_id: str,
        session_id: str,
        file_path: str,
        content: str,
        content_type: str = "text/markdown"
    ) -> Dict[str, Any]:
        """Create or update file"""
        url = f"{self.base_url}/threads/{thread_id}/sessions/{session_id}/files/{file_path}"
        
        data = {
            "content": content,
            "content_type": content_type
        }
        
        try:
            response = await self.client.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to create file {file_path}: {e}")
            raise
    
    async def get_file(
        self,
        thread_id: str,
        session_id: str,
        file_path: str
    ) -> str:
        """Get file content"""
        url = f"{self.base_url}/threads/{thread_id}/sessions/{session_id}/files/{file_path}"
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to get file {file_path}: {e}")
            raise
    
    async def list_session_files(
        self,
        thread_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Get list of files in session"""
        url = f"{self.base_url}/threads/{thread_id}/sessions/{session_id}"
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to list session files: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check service health"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
