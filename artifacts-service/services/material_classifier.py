"""Service for automatic classification of learning materials using AI.

This service analyzes material content and classifies it by:
- Subject (e.g., Mathematics, Physics, Chemistry, Web3)
- Grade level (e.g., 'Beginner', 'Intermediate', 'Advanced')
- Topic (e.g., 'Linear Equations', 'Blockchain Basics')
"""

import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import os

logger = logging.getLogger(__name__)


class MaterialClassification(BaseModel):
    """Classification result for a learning material."""
    
    subject: str = Field(
        description="Main subject area (e.g., Mathematics, Physics, Chemistry, Computer Science, Web3, Economics)"
    )
    grade: str = Field(
        description="Grade level (e.g., 'Beginner', 'Intermediate', 'Advanced')"
    )
    topic: str = Field(
        description="Specific topic (e.g., 'Linear Equations', 'Blockchain Fundamentals', 'Newton's Laws')"
    )
    confidence: str = Field(
        default="high",
        description="Confidence level: high, medium, low"
    )


class MaterialClassifierService:
    """AI-powered service for classifying learning materials."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize classifier with OpenAI API."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Classification will not work.")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def classify_material(
        self, 
        content: str, 
        input_query: Optional[str] = None
    ) -> MaterialClassification:
        """Classify learning material by analyzing its content.
        
        Args:
            content: Full material content (markdown)
            input_query: Optional original user query that generated this material
            
        Returns:
            MaterialClassification with subject, grade, and topic
        """
        if not self.client:
            logger.warning("OpenAI client not initialized. Returning default classification.")
            return MaterialClassification(
                subject="Unknown",
                grade="Unknown",
                topic="Unknown",
                confidence="low"
            )
        
        try:
            # Prepare content preview (first 3000 chars for efficiency)
            content_preview = content[:3000]
            
            # Build context for AI
            context_parts = [
                "Analyze the following learning material and classify it."
            ]
            if input_query:
                context_parts.append(f"\nOriginal user query: {input_query}")
            context_parts.append(f"\n\nMaterial content:\n{content_preview}")
            
            full_prompt = "\n".join(context_parts)
            
            # Call OpenAI with structured output
            logger.info("Calling OpenAI API for material classification...")
            response = await self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                response_format=MaterialClassification,
                temperature=0.1,
                max_tokens=500
            )
            
            classification = response.choices[0].message.parsed
            logger.info(
                f"Material classified: subject={classification.subject}, "
                f"grade={classification.grade}, topic={classification.topic}"
            )
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying material: {e}", exc_info=True)
            # Return fallback classification
            return MaterialClassification(
                subject="Unknown",
                grade="Unknown",
                topic="Unknown",
                confidence="low"
            )
    
    @staticmethod
    def _get_system_prompt() -> str:
        """Get system prompt for classification."""
        return """You are an expert educational content analyzer. Your task is to classify learning materials.

Analyze the provided material and determine:

1. **Subject**: Main academic or professional field
   Examples: Mathematics, Physics, Chemistry, Biology, Computer Science, Web3, Blockchain, 
   Economics, History, Language Arts, Engineering, Data Science, Machine Learning, etc.

2. **Grade Level**: Target audience level
   For general: "Beginner", "Intermediate", "Advanced"

3. **Topic**: Specific concept or area within the subject
   Examples: "Linear Equations", "Blockchain Fundamentals", "Newton's Laws of Motion", 
   "Photosynthesis", "React Components", "Supply and Demand"

4. **Confidence**: Your confidence in this classification
   - "high": Clear indicators of subject, grade, and topic
   - "medium": Some ambiguity in one category
   - "low": Unclear or very general content

Guidelines:
- Be specific but not overly narrow
- Consider the complexity level and terminology used
- If content spans multiple subjects, choose the primary one
- For Web3/Blockchain topics, use "Web3" or "Blockchain" as subject
- Make your best judgment even if some information is ambiguous
"""

    async def classify_from_preview(
        self, 
        title: Optional[str] = None,
        first_paragraph: Optional[str] = None,
        input_query: Optional[str] = None
    ) -> MaterialClassification:
        """Quick classification from minimal information (title + first paragraph).
        
        This is a lighter version for quick classification without reading full content.
        
        Args:
            title: Material title
            first_paragraph: First paragraph of material
            input_query: Original user query
            
        Returns:
            MaterialClassification
        """
        if not self.client:
            return MaterialClassification(
                subject="Unknown",
                grade="Unknown",
                topic="Unknown",
                confidence="low"
            )
        
        # Build minimal context
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if input_query:
            parts.append(f"User query: {input_query}")
        if first_paragraph:
            parts.append(f"First paragraph: {first_paragraph}")
        
        if not parts:
            return MaterialClassification(
                subject="Unknown",
                grade="Unknown", 
                topic="Unknown",
                confidence="low"
            )
        
        content = "\n\n".join(parts)
        return await self.classify_material(content=content, input_query=None)


# Global instance
_classifier_service: Optional[MaterialClassifierService] = None


def get_classifier_service() -> MaterialClassifierService:
    """Get global classifier service instance."""
    global _classifier_service
    if _classifier_service is None:
        _classifier_service = MaterialClassifierService()
    return _classifier_service



