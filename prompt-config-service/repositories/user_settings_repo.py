"""Repository for user settings operations.

Updated for Web3 wallet-based authentication.
"""

import uuid
from typing import Dict, List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.placeholder import Placeholder, PlaceholderValue
from models.user_settings import UserPlaceholderSetting, UserProfile


class UserSettingsRepository:
    """Repository for managing user settings - Web3 wallet-based."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_user_by_wallet(self, wallet_address: str) -> Optional[UserProfile]:
        """Get user profile by wallet address."""
        result = await self.db.execute(
            select(UserProfile).where(UserProfile.wallet_address == wallet_address.lower())
        )
        return result.scalar_one_or_none()
    
    async def get_user_by_id(self, user_id: uuid.UUID) -> Optional[UserProfile]:
        """Get user profile by internal ID."""
        result = await self.db.execute(
            select(UserProfile).where(UserProfile.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def create_user(self, wallet_address: str, username: Optional[str] = None) -> UserProfile:
        """Create new user profile from wallet address."""
        user_profile = UserProfile(
            wallet_address=wallet_address.lower(),
            username=username
        )
        self.db.add(user_profile)
        await self.db.flush()
        await self.db.refresh(user_profile)
        return user_profile
    
    async def get_or_create_user(self, wallet_address: str, username: Optional[str] = None) -> UserProfile:
        """Get existing user or create new one."""
        user_profile = await self.get_user_by_wallet(wallet_address)
        
        if not user_profile:
            user_profile = await self.create_user(wallet_address, username)
        
        return user_profile
    
    async def get_user_settings(self, user_id: uuid.UUID) -> List[UserPlaceholderSetting]:
        """Get all placeholder settings for a user."""
        result = await self.db.execute(
            select(UserPlaceholderSetting)
            .options(
                selectinload(UserPlaceholderSetting.placeholder),
                selectinload(UserPlaceholderSetting.placeholder_value)
            )
            .where(UserPlaceholderSetting.user_id == user_id)
        )
        return list(result.scalars().all())
    
    async def get_user_settings_by_names(
        self, user_id: uuid.UUID, placeholder_names: List[str]
    ) -> Dict[str, str]:
        """
        Get placeholder values by their names for a specific user.
        Returns a dictionary {placeholder_name: value}.
        """
        result = await self.db.execute(
            select(
                Placeholder.name,
                PlaceholderValue.value
            )
            .join(UserPlaceholderSetting, UserPlaceholderSetting.placeholder_id == Placeholder.id)
            .join(PlaceholderValue, UserPlaceholderSetting.placeholder_value_id == PlaceholderValue.id)
            .where(
                UserPlaceholderSetting.user_id == user_id,
                Placeholder.name.in_(placeholder_names)
            )
        )
        
        return {name: value for name, value in result.all()}
    
    async def upsert_setting(
        self, user_id: uuid.UUID, placeholder_id: uuid.UUID, value_id: uuid.UUID
    ) -> UserPlaceholderSetting:
        """Create or update a user placeholder setting."""
        # Check if setting already exists
        result = await self.db.execute(
            select(UserPlaceholderSetting)
            .where(
                UserPlaceholderSetting.user_id == user_id,
                UserPlaceholderSetting.placeholder_id == placeholder_id
            )
        )
        setting = result.scalar_one_or_none()
        
        if setting:
            # Update existing setting
            setting.placeholder_value_id = value_id
        else:
            # Create new setting
            setting = UserPlaceholderSetting(
                user_id=user_id,
                placeholder_id=placeholder_id,
                placeholder_value_id=value_id
            )
            self.db.add(setting)
        
        await self.db.flush()
        await self.db.refresh(setting)
        return setting
    
    async def delete_user_settings(self, user_id: uuid.UUID) -> None:
        """Delete all settings for a user."""
        await self.db.execute(
            delete(UserPlaceholderSetting)
            .where(UserPlaceholderSetting.user_id == user_id)
        )
        await self.db.flush()
    
    async def bulk_upsert(self, user_id: uuid.UUID, settings: List[Dict]) -> None:
        """Bulk create or update user settings."""
        for setting_data in settings:
            await self.upsert_setting(
                user_id=user_id,
                placeholder_id=setting_data["placeholder_id"],
                value_id=setting_data["placeholder_value_id"]
            )