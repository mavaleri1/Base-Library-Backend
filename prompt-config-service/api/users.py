"""API endpoints for user settings operations."""

import logging
import uuid
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.user_settings import UserProfile
from schemas.placeholder import PlaceholderValueSchema
from schemas.user_settings import SetPlaceholderRequest
from services.placeholder_service import PlaceholderService
from services.user_service import UserService
from utils.auth import get_current_user_from_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.get("/{user_id}/placeholders/{placeholder_name}")
async def get_user_placeholder_by_name(
    user_id: uuid.UUID,
    placeholder_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserProfile = Depends(get_current_user_from_token)
):
    """Get specific user placeholder by name."""
    # If authenticated user exists, use their ID
    # Otherwise, use the user_id from URL (for backward compatibility)
    effective_user_id = current_user.id if current_user else user_id
    
    # Ensure user exists in prompt-config-service
    if not current_user:
        from repositories.user_settings_repo import UserSettingsRepository
        repo = UserSettingsRepository(db)
        user = await repo.get_user_by_id(effective_user_id)
        if not user:
            # Create placeholder user for backward compatibility
            logger.warning(f"Creating user {effective_user_id} in prompt-config-service without wallet address")
            user = UserProfile(
                id=effective_user_id,
                wallet_address="unknown"
            )
            db.add(user)
            await db.flush()
            await db.refresh(user)
    
    service = UserService(db)
    
    try:
        settings_raw = await service.get_user_settings_with_details(effective_user_id)
        
        # Find the specific placeholder by name
        for setting in settings_raw:
            if (setting.placeholder and setting.placeholder_value and 
                setting.placeholder.name == placeholder_name):
                return {
                    "placeholder_id": str(setting.placeholder_id),
                    "placeholder_name": setting.placeholder.name,
                    "placeholder_display_name": setting.placeholder.display_name,
                    "value_id": str(setting.placeholder_value_id),
                    "value": setting.placeholder_value.value,
                    "display_name": setting.placeholder_value.display_name
                }
        
        # If placeholder not found, return 404
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Placeholder '{placeholder_name}' not found for user {effective_user_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user placeholder: {str(e)}"
        )


@router.get("/{user_id}/placeholders")
async def get_user_placeholders(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: UserProfile = Depends(get_current_user_from_token)
):
    """Get current user placeholder settings."""
    # If authenticated user exists, use their ID
    # Otherwise, use the user_id from URL (for backward compatibility)
    effective_user_id = current_user.id if current_user else user_id
    
    # Ensure user exists in prompt-config-service
    if not current_user:
        from repositories.user_settings_repo import UserSettingsRepository
        repo = UserSettingsRepository(db)
        user = await repo.get_user_by_id(effective_user_id)
        if not user:
            # Create placeholder user for backward compatibility
            logger.warning(f"Creating user {effective_user_id} in prompt-config-service without wallet address")
            user = UserProfile(
                id=effective_user_id,
                wallet_address="unknown"
            )
            db.add(user)
            await db.flush()
            await db.refresh(user)
    
    service = UserService(db)
    
    try:
        settings_raw = await service.get_user_settings_with_details(effective_user_id)
        
        # Convert to proper format with full placeholder info
        placeholders = {}
        for setting in settings_raw:
            if setting.placeholder and setting.placeholder_value:
                placeholders[setting.placeholder.name] = {
                    "placeholder_id": str(setting.placeholder_id),
                    "placeholder_name": setting.placeholder.name,
                    "placeholder_display_name": setting.placeholder.display_name,
                    "value_id": str(setting.placeholder_value_id),
                    "value": setting.placeholder_value.value,
                    "display_name": setting.placeholder_value.display_name
                }
        
        result = {
            "placeholders": placeholders,
            "active_profile_id": None,
            "active_profile_name": None
        }
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user settings: {str(e)}"
        )


@router.put("/{user_id}/placeholders/{placeholder_id}")
async def set_user_placeholder(
    user_id: uuid.UUID,
    placeholder_id: uuid.UUID,
    request: SetPlaceholderRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserProfile = Depends(get_current_user_from_token)
):
    """Set user placeholder value."""
    # If authenticated user exists, use their ID
    effective_user_id = current_user.id if current_user else user_id
    
    user_service = UserService(db)
    placeholder_service = PlaceholderService(db)
    
    # Verify placeholder exists
    placeholder = await placeholder_service.get_placeholder_by_id(placeholder_id)
    if not placeholder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Placeholder {placeholder_id} not found"
        )
    
    # Verify value exists and belongs to this placeholder
    value = await placeholder_service.repo.find_value_by_id(request.value_id)
    if not value:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Placeholder value {request.value_id} not found"
        )
    
    if value.placeholder_id != placeholder_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Value {request.value_id} does not belong to placeholder {placeholder_id}"
        )
    
    try:
        setting = await user_service.set_user_placeholder(
            effective_user_id, placeholder_id, request.value_id
        )
        await db.commit()
        return {"message": "Placeholder value updated successfully"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update placeholder: {str(e)}"
        )


@router.post("/{user_id}/apply-profile/{profile_id}")
async def apply_profile_to_user(
    user_id: uuid.UUID,
    profile_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: UserProfile = Depends(get_current_user_from_token)
):
    """Apply profile settings to user."""
    # If authenticated user exists, use their ID
    # If not, current_user will be None and we'll use user_id from URL
    effective_user_id = current_user.id if current_user else user_id
    
    service = UserService(db)
    
    # If user doesn't exist in prompt-config-service, try to create them
    # This handles the case where user exists in artifacts-service but not in prompt-config-service
    if not current_user:
        from repositories.user_settings_repo import UserSettingsRepository
        repo = UserSettingsRepository(db)
        user = await repo.get_user_by_id(effective_user_id)
        if not user:
            # Create a placeholder user record for backward compatibility
            # In production, this should be avoided by using JWT authentication
            logger.warning(f"Creating user {effective_user_id} in prompt-config-service without wallet address")
            user = UserProfile(
                id=effective_user_id,
                wallet_address="unknown"  # Placeholder value
            )
            db.add(user)
            await db.flush()
            await db.refresh(user)
    
    try:
        await service.apply_profile_to_user(effective_user_id, profile_id)
        await db.commit()
        
        # Return updated user settings with profile info
        settings_raw = await service.get_user_settings_with_details(effective_user_id)
        
        # Get profile name
        from services.profile_service import ProfileService
        profile_service = ProfileService(db)
        profile = await profile_service.get_profile_by_id(profile_id)
        
        placeholders = {}
        for setting in settings_raw:
            if setting.placeholder and setting.placeholder_value:
                placeholders[setting.placeholder.name] = {
                    "placeholder_id": str(setting.placeholder_id),
                    "placeholder_name": setting.placeholder.name,
                    "placeholder_display_name": setting.placeholder.display_name,
                    "value_id": str(setting.placeholder_value_id),
                    "value": setting.placeholder_value.value,
                    "display_name": setting.placeholder_value.display_name
                }
        
        return {
            "message": f"Profile applied successfully",
            "placeholders": placeholders,
            "active_profile_id": str(profile_id),
            "active_profile_name": profile.display_name if profile else None
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply profile: {str(e)}"
        )


@router.post("/{user_id}/reset")
async def reset_user_settings(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: UserProfile = Depends(get_current_user_from_token)
):
    """Reset user settings to defaults."""
    # If authenticated user exists, use their ID
    effective_user_id = current_user.id if current_user else user_id
    
    service = UserService(db)
    
    try:
        await service.reset_to_defaults(effective_user_id)
        await db.commit()
        
        # Return updated settings with details
        settings_raw = await service.get_user_settings_with_details(effective_user_id)
        
        placeholders = {}
        for setting in settings_raw:
            if setting.placeholder and setting.placeholder_value:
                placeholders[setting.placeholder.name] = {
                    "placeholder_id": str(setting.placeholder_id),
                    "placeholder_name": setting.placeholder.name,
                    "placeholder_display_name": setting.placeholder.display_name,
                    "value_id": str(setting.placeholder_value_id),
                    "value": setting.placeholder_value.value,
                    "display_name": setting.placeholder_value.display_name
                }
        
        return {
            "message": f"Settings reset to defaults",
            "placeholders": placeholders,
            "active_profile_id": None,
            "active_profile_name": None
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset user settings: {str(e)}"
        )


@router.get("/{user_id}/profile-debug")
async def get_user_profile_debug(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: UserProfile = Depends(get_current_user_from_token)
):
    """
    Detailed debug information about user profile settings.
    Shows all settings, their sources and values.
    """
    effective_user_id = current_user.id if current_user else user_id
    
    service = UserService(db)
    
    try:
        # Get all user settings
        user_settings = await service.get_user_settings_with_details(effective_user_id)
        
        # Get all available placeholders for comparison
        from services.placeholder_service import PlaceholderService
        placeholder_service = PlaceholderService(db)
        all_placeholders = await placeholder_service.get_all_placeholders()
        
        # Form detailed information
        debug_info = {
            "user_id": str(effective_user_id),
            "user_settings_count": len(user_settings),
            "total_available_placeholders": len(all_placeholders),
            "user_settings": [],
            "missing_placeholders": [],
            "key_profile_settings": {}
        }
        
        # Key profile settings for tracking
        key_settings = [
            "expert_role", "subject_name", "subject_keywords", 
            "target_audience_inline", "material_type_inline",
            "explanation_depth", "topic_coverage", "language", "style"
        ]
        
        # Process user settings
        user_setting_names = set()
        for setting in user_settings:
            if setting.placeholder and setting.placeholder_value:
                setting_info = {
                    "placeholder_name": setting.placeholder.name,
                    "placeholder_display_name": setting.placeholder.display_name,
                    "value_name": setting.placeholder_value.name,
                    "value": setting.placeholder_value.value,
                    "is_key_setting": setting.placeholder.name in key_settings
                }
                debug_info["user_settings"].append(setting_info)
                user_setting_names.add(setting.placeholder.name)
                
                # Add to key settings
                if setting.placeholder.name in key_settings:
                    debug_info["key_profile_settings"][setting.placeholder.name] = {
                        "value": setting.placeholder_value.value,
                        "display_name": setting.placeholder_value.display_name
                    }
        
        # Find missing settings
        for placeholder in all_placeholders:
            if placeholder.name not in user_setting_names:
                debug_info["missing_placeholders"].append({
                    "name": placeholder.name,
                    "display_name": placeholder.display_name,
                    "is_key_setting": placeholder.name in key_settings
                })
        
        # Statistics
        debug_info["statistics"] = {
            "key_settings_configured": len(debug_info["key_profile_settings"]),
            "key_settings_missing": len([p for p in debug_info["missing_placeholders"] if p["is_key_setting"]]),
            "total_key_settings": len(key_settings),
            "coverage_percentage": round((len(debug_info["key_profile_settings"]) / len(key_settings)) * 100, 1)
        }
        
        logger.info(f"🔍 Profile debug info for user {effective_user_id}: {debug_info['statistics']}")
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Failed to get profile debug info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get profile debug info: {str(e)}"
        )