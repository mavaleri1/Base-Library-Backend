"""Service for user settings operations.

Updated for Web3 wallet-based authentication.
"""

import logging
import uuid
from typing import Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from models.placeholder import PlaceholderValue
from models.user_settings import UserPlaceholderSetting, UserProfile
from repositories.placeholder_repo import PlaceholderRepository
from repositories.user_settings_repo import UserSettingsRepository

logger = logging.getLogger(__name__)


# Default placeholder values - map placeholder name to value name from yaml
DEFAULT_PLACEHOLDER_VALUES = {
    "expert_role": "industry_practitioner",
    "subject_name": "ai_llm",
    "subject_keywords": "ai_llm_keywords",
    "language": "english_tech",
    "style": "conceptual_focused",
    "target_audience_inline": "intermediate_specialists",
    "target_audience_block": "intermediate_specialists_block",
    "material_type_inline": "conceptual_overview",
    "material_type_block": "conceptual_overview_block",
    "explanation_depth": "intermediate_depth",
    "topic_coverage": "focused_essentials",
    "question_formats": "analytical_questions",
    "question_purpose": "explore_connections_block",
    "question_purpose_inline": "explore_connections",
    "question_quantity": "qty_3",
}


class UserService:
    """Service for managing user settings - Web3 wallet-based."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = UserSettingsRepository(db)
        self.placeholder_repo = PlaceholderRepository(db)
    
    async def get_or_create_user_by_wallet(
        self, wallet_address: str, username: Optional[str] = None
    ) -> UserProfile:
        """Get or create user by wallet address."""
        return await self.repo.get_or_create_user(wallet_address, username)
    
    async def get_user_by_wallet(self, wallet_address: str) -> Optional[UserProfile]:
        """Get user by wallet address."""
        return await self.repo.get_user_by_wallet(wallet_address)
    
    async def get_user_settings(self, user_id: uuid.UUID) -> Dict[str, PlaceholderValue]:
        """Get user settings as a dictionary of placeholder names to PlaceholderValue objects."""
        await self.ensure_user_has_settings(user_id)
        settings = await self.repo.get_user_settings(user_id)
        
        result = {}
        for setting in settings:
            if setting.placeholder and setting.placeholder_value:
                result[setting.placeholder.name] = setting.placeholder_value
        
        return result
    
    async def get_user_settings_by_wallet(self, wallet_address: str) -> Dict[str, PlaceholderValue]:
        """Get user settings by wallet address."""
        user = await self.repo.get_user_by_wallet(wallet_address)
        if not user:
            raise ValueError(f"User with wallet {wallet_address} not found")
        return await self.get_user_settings(user.id)
    
    async def get_user_settings_with_details(self, user_id: uuid.UUID):
        """Get user settings with full placeholder and value details."""
        return await self.repo.get_user_settings(user_id)
    
    async def get_user_placeholder_values(
        self, user_id: uuid.UUID, placeholder_names: List[str]
    ) -> Dict[str, str]:
        """Get placeholder values for specific placeholder names."""
        return await self.repo.get_user_settings_by_names(user_id, placeholder_names)
    
    async def set_user_placeholder(
        self, user_id: uuid.UUID, placeholder_id: uuid.UUID, value_id: uuid.UUID
    ) -> UserPlaceholderSetting:
        """Set a user placeholder value."""
        user = await self.repo.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        return await self.repo.upsert_setting(user_id, placeholder_id, value_id)
    
    async def apply_profile_to_user(self, user_id: uuid.UUID, profile_id: uuid.UUID) -> None:
        """Apply a profile's settings to a user."""
        from services.profile_service import ProfileService
        
        user = await self.repo.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        profile_service = ProfileService(self.db)
        profile = await profile_service.get_profile_by_id(profile_id)
        
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")
        
        # Get profile settings and apply them to user
        settings_list = []
        for setting in profile.placeholder_settings:
            settings_list.append({
                "placeholder_id": setting.placeholder_id,
                "placeholder_value_id": setting.placeholder_value_id
            })
        
        if settings_list:
            await self.repo.bulk_upsert(user_id, settings_list)
            logger.info(f"Successfully applied {len(settings_list)} settings from profile {profile.name} to user {user_id}")
        else:
            logger.warning(f"Profile {profile_id} contains no settings to apply")
    
    async def reset_to_defaults(self, user_id: uuid.UUID) -> None:
        """Reset user settings to defaults."""
        user = await self.repo.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        await self.apply_default_settings(user_id)
    
    async def ensure_user_has_settings(self, user_id: uuid.UUID) -> None:
        """Ensure user has settings, apply defaults if not."""
        settings = await self.repo.get_user_settings(user_id)
        if not settings:
            await self.apply_default_settings(user_id)
    
    async def apply_default_settings(self, user_id: uuid.UUID) -> None:
        """Apply default settings to user."""
        # Delete existing settings
        await self.repo.delete_user_settings(user_id)
        logger.info(f"Deleted existing settings for user {user_id}")
        
        # Apply default values
        settings_to_create = []
        
        for placeholder_name, default_value_name in DEFAULT_PLACEHOLDER_VALUES.items():
            # Find placeholder by name
            placeholder = await self.placeholder_repo.find_by_name(placeholder_name)
            if not placeholder:
                logger.warning(f"Placeholder '{placeholder_name}' not found in database")
                continue
                
            # Find value by name
            placeholder_value = None
            for value in placeholder.values:
                if value.name == default_value_name:
                    placeholder_value = value
                    break
            
            if placeholder_value:
                logger.debug(f"Found value '{default_value_name}' for placeholder '{placeholder_name}'")
                settings_to_create.append({
                    "placeholder_id": placeholder.id,
                    "placeholder_value_id": placeholder_value.id
                })
            else:
                logger.warning(f"Value '{default_value_name}' not found for placeholder '{placeholder_name}'. Available values: {[v.name for v in placeholder.values]}")
        
        logger.info(f"Prepared {len(settings_to_create)} settings for user {user_id}")
        if settings_to_create:
            await self.repo.bulk_upsert(user_id, settings_to_create)
            logger.info(f"Successfully inserted {len(settings_to_create)} settings for user {user_id}")
        else:
            logger.error(f"No settings to create for user {user_id}!")