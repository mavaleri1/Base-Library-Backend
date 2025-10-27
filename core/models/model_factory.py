"""
Factory for creating LLM models with OpenAI-compatible provider support.
"""

import logging
from typing import Optional, Dict

from langchain_openai import ChatOpenAI

from ..config.config_manager import get_config_manager
from ..config.config_models import ModelConfig, ProviderConfig


logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating LLM models with OpenAI-compatible provider support"""
    
    def __init__(self, providers_config: Dict[str, ProviderConfig]):
        """
        Initialize the model factory.
        
        Args:
            providers_config: Dictionary of provider configurations
        """
        self.providers_config = providers_config
    
    def create_model(self, config: ModelConfig) -> ChatOpenAI:
        """
        Create a ChatOpenAI model with provider-specific configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Configured ChatOpenAI instance
        """
        # Get provider configuration
        provider_config = self.providers_config.get(config.provider)
        if not provider_config:
            # Fallback to OpenAI if provider not found
            provider_config = ProviderConfig(name="openai")
            logger.warning(f"Provider '{config.provider}' not found, using OpenAI as fallback")
        
        # Check structured output support
        if config.requires_structured_output and not provider_config.supports_structured_output:
            raise ValueError(
                f"Provider '{config.provider}' does not support structured output required by this node. "
                f"Please use a provider with structured output support (e.g., 'openai', 'fireworks')"
            )
        
        # Check vision support
        if getattr(config, 'requires_vision', False) and not getattr(provider_config, 'supports_vision', False):
            raise ValueError(
                f"Provider '{config.provider}' does not support vision required by this node. "
                f"Please use a provider with vision support (e.g., 'openai')"
            )
        
        # Collect parameters for ChatOpenAI
        model_params = {
            "model": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        
        # Add API key if specified
        if provider_config.api_key:
            model_params["openai_api_key"] = provider_config.api_key
        
        # Add base_url if specified
        if provider_config.base_url:
            model_params["openai_api_base"] = provider_config.base_url
        
        # Add optional generation parameters
        if config.top_p is not None:
            model_params["top_p"] = config.top_p
        if config.frequency_penalty is not None:
            model_params["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty is not None:
            model_params["presence_penalty"] = config.presence_penalty
        
        logger.info(f"🔍 [MODEL_FACTORY] Creating model:")
        logger.info(f"🔍 [MODEL_FACTORY] - Provider: {config.provider}")
        logger.info(f"🔍 [MODEL_FACTORY] - Model name: {config.model_name}")
        logger.info(f"🔍 [MODEL_FACTORY] - Base URL: {provider_config.base_url}")
        logger.info(f"🔍 [MODEL_FACTORY] - Temperature: {config.temperature}")
        logger.info(f"🔍 [MODEL_FACTORY] - Max tokens: {config.max_tokens}")
        logger.info(f"🔍 [MODEL_FACTORY] - Supports structured output: {provider_config.supports_structured_output}")
        logger.info(f"🔍 [MODEL_FACTORY] - Supports vision: {getattr(provider_config, 'supports_vision', 'N/A')}")
        
        model = ChatOpenAI(**model_params)
        logger.info(f"🔍 [MODEL_FACTORY] Model created successfully with parameters: {model_params}")
        
        return model
    
    def create_model_for_node(self, node_name: str) -> ChatOpenAI:
        """
        Create a model for a specific workflow node.
        
        Args:
            node_name: Name of the workflow node
            
        Returns:
            Configured ChatOpenAI instance for the node
        """
        config_manager = get_config_manager()
        config = config_manager.get_model_config(node_name)
        
        if not config_manager.has_node_config(node_name):
            logger.warning(
                f"No specific configuration found for node '{node_name}', using default configuration"
            )
        
        return self.create_model(config)


# Global factory instance
_model_factory: Optional[ModelFactory] = None


def initialize_model_factory(api_key: str = None, config_manager=None) -> ModelFactory:
    """
    Initialize the global model factory with provider support.
    API keys are taken from provider configuration (already with substituted env variables).
    
    Args:
        api_key: Fallback OpenAI API key (for backward compatibility)
        config_manager: Optional config manager
        
    Returns:
        ModelFactory instance
    """
    global _model_factory
    
    # Load provider configuration (already with substituted API keys)
    config_manager = config_manager or get_config_manager()
    providers_config = config_manager.get_providers_config()
    
    # If providers are not configured, create default OpenAI provider
    if not providers_config:
        logger.warning("No providers configured, using default OpenAI provider")
        providers_config = {
            "openai": ProviderConfig(
                name="openai",
                api_key=api_key,
                supports_structured_output=True
            )
        }
    
    _model_factory = ModelFactory(providers_config)
    return _model_factory


def get_model_factory() -> ModelFactory:
    """Get the global model factory instance."""
    if _model_factory is None:
        raise RuntimeError("Model factory not initialized. Call initialize_model_factory() first.")
    return _model_factory


def create_model_for_node(node_name: str) -> ChatOpenAI:
    """
    Convenience function to create a model for a node using global factory.
    
    Args:
        node_name: Name of the workflow node
        
    Returns:
        Configured ChatOpenAI instance
    """
    factory = get_model_factory()
    return factory.create_model_for_node(node_name)