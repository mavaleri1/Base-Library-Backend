"""
Core service settings.
"""

from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Main core service settings"""

    # OpenAI settings (optional if using other providers)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    # PostgreSQL settings for AsyncPostgresSaver
    database_url: str = Field(
        description="PostgreSQL connection string for checkpointer"
    )

    # LangFuse settings
    langfuse_public_key: Optional[str] = Field(
        default=None, description="LangFuse public key"
    )
    langfuse_secret_key: Optional[str] = Field(
        default=None, description="LangFuse secret key"
    )
    langfuse_host: str = Field(
        default="https://localhost:3000",
        description="LangFuse host",
    )

    # Configuration file paths
    prompts_config_path: str = Field(
        default="./configs/prompts.yaml",
        description="Path to prompts",
    )
    graph_config_path: str = Field(
        default="./configs/graph.yaml",
        description="Path to graph configuration",
    )
    main_dir: str = Field(default="./data", description="Working directory")

    # Image processing settings
    max_image_size: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum image size in bytes (10MB)",
    )
    max_images_per_request: int = Field(
        default=10, description="Maximum number of images per request"
    )
    temp_storage_path: str = Field(
        default="/tmp/core", description="Temporary storage for images"
    )
    supported_image_formats: List[str] = Field(
        default=[".jpg", ".jpeg", ".png"],
        description="Supported image formats",
    )

    # Service settings
    host: str = Field(default="0.0.0.0", description="Host for FastAPI service")
    port: int = Field(default=8000, description="Port for FastAPI service")

    # Local artifacts storage
    artifacts_base_path: str = Field(
        default="data/artifacts", description="Base path for local artifacts"
    )
    artifacts_ensure_permissions: bool = Field(
        default=True, description="Ensure file access permissions"
    )
    artifacts_max_file_size: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum artifact file size (10MB)",
    )
    artifacts_atomic_writes: bool = Field(
        default=True, description="Use atomic file writes"
    )

    log_level: str = Field(default="DEBUG", description="Logging level")

    # Security settings
    security_enabled: bool = Field(
        default=True,
        description="Enable prompt injection protection system",
    )
    security_fuzzy_threshold: float = Field(
        default=0.85,
        description="Fuzzy matching threshold for cleaning",
    )
    security_min_content_length: int = Field(
        default=10,
        description="Minimum content length for validation",
    )

    # Prompt Configuration Service settings
    prompt_service_url: str = Field(
        default="http://localhost:8002",
        description="Prompt Configuration Service URL",
    )
    prompt_service_timeout: int = Field(
        default=5,
        description="Timeout for Prompt Service requests (seconds)",
    )
    prompt_service_retry_count: int = Field(
        default=3,
        description="Number of retry attempts for Prompt Service",
    )

    # Web UI settings
    web_ui_base_url: str = Field(
        default="http://127.0.0.1:",
        description="Base URL for Web UI interface",
    )

    def is_artifacts_configured(self) -> bool:
        """Check local artifacts storage configuration"""
        return bool(self.artifacts_base_path)

    def is_langfuse_configured(self) -> bool:
        """Check LangFuse integration configuration"""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    class Config:
        env_file = ["../.env.local", "../.env", ".env.local", ".env"]
        extra = "ignore"  # Ignore extra environment variables


# Global settings instance
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Singleton for getting settings"""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings
