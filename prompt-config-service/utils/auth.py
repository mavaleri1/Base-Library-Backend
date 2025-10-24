"""JWT authentication utilities for prompt-config-service."""

import logging
import uuid
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import PyJWTError
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import get_db
from models.user_settings import UserProfile
from repositories.user_settings_repo import UserSettingsRepository

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


def verify_jwt_token(token: str) -> Optional[dict]:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except PyJWTError as e:
        logger.error(f"JWT verification failed: {e}")
        return None


async def get_current_user_from_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Optional[UserProfile]:
    """Get or create user from JWT token.
    
    This dependency:
    1. Extracts and verifies JWT token
    2. Gets wallet_address and user_id from token
    3. Checks if user exists in prompt-config-service database
    4. If not exists, creates user with same UUID from artifacts-service
    5. Returns user profile
    """
    if not credentials:
        logger.warning("No authorization credentials provided")
        return None
    
    token = credentials.credentials
    payload = verify_jwt_token(token)
    
    if not payload:
        logger.warning("Invalid JWT token")
        return None
    
    wallet_address = payload.get("wallet_address")
    user_id_str = payload.get("user_id")
    
    if not wallet_address or not user_id_str:
        logger.warning("Missing wallet_address or user_id in token payload")
        return None
    
    try:
        user_id = uuid.UUID(user_id_str)
    except (ValueError, AttributeError):
        logger.error(f"Invalid user_id format in token: {user_id_str}")
        return None
    
    repo = UserSettingsRepository(db)
    
    # Try to get user by wallet address first
    user = await repo.get_user_by_wallet(wallet_address)
    
    if not user:
        # User doesn't exist in prompt-config-service, create with same UUID from artifacts-service
        logger.info(f"Creating user in prompt-config-service for wallet {wallet_address[:10]}... with ID {user_id}")
        
        # Check if user with this ID already exists (edge case)
        existing_user_by_id = await repo.get_user_by_id(user_id)
        if existing_user_by_id:
            logger.warning(f"User with ID {user_id} exists but has different wallet address")
            return existing_user_by_id
        
        # Create new user with same UUID as in artifacts-service
        user = UserProfile(
            id=user_id,  # Use the same UUID from artifacts-service
            wallet_address=wallet_address.lower()
        )
        db.add(user)
        await db.flush()
        await db.refresh(user)
        await db.commit()
        logger.info(f"User created successfully: {user.id}")
    
    return user


async def require_auth(
    user: Optional[UserProfile] = Depends(get_current_user_from_token)
) -> UserProfile:
    """Require authentication - raises 401 if user is not authenticated."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user

