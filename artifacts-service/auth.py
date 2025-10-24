"""Authentication and authorization middleware for Artifacts Service."""

from datetime import datetime, timedelta
from typing import Optional, Annotated
import logging

from fastapi import Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import asyncpg

from settings import settings

logger = logging.getLogger(__name__)

# Security schemes
security = HTTPBearer(auto_error=False)


class AuthService:
    """Handle authentication and authorization."""
    
    def __init__(self):
        """Initialize auth service."""
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Create database connection pool."""
        if settings.database_url:
            try:
                self.pool = await asyncpg.create_pool(
                    settings.database_url,
                    min_size=1,
                    max_size=5,
                    command_timeout=60
                )
                logger.info("Auth database connection pool created")
            except Exception as e:
                logger.error(f"Failed to create auth database pool: {e}")
                # Don't raise - auth is optional for MVP
    
    async def disconnect(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Auth database connection pool closed")
    
    def create_jwt_token(self, user_id: int, username: str) -> str:
        """Create JWT token for authenticated user."""
        if not settings.jwt_secret_key:
            raise ValueError("JWT secret key not configured")
        
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expiration_minutes)
        payload = {
            "sub": str(user_id),
            "username": username,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(
            payload,
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm
        )
        
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[dict]:
        """Verify JWT token and return payload."""
        if not settings.jwt_secret_key:
            return None
        
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm]
            )
            return payload
        except JWTError as e:
            logger.debug(f"JWT verification failed: {e}")
            return None
    
    async def verify_auth_code(self, username: str, code: str) -> Optional[int]:
        """Verify auth code from database."""
        if not self.pool:
            return None
        
        async with self.pool.acquire() as conn:
            try:
                # Check if code exists and is not expired
                result = await conn.fetchrow(
                    """
                    SELECT user_id FROM auth_codes
                    WHERE username = $1 
                    AND code = $2
                    AND created_at > $3
                    """,
                    username,
                    code,
                    datetime.utcnow() - timedelta(minutes=settings.auth_code_expiration_minutes)
                )
                
                if result:
                    # Delete used code
                    await conn.execute(
                        "DELETE FROM auth_codes WHERE username = $1 AND code = $2",
                        username,
                        code
                    )
                    logger.info(f"Verified and deleted auth code for user {username}")
                    return result['user_id']
                
                return None
                
            except Exception as e:
                logger.error(f"Failed to verify auth code: {e}")
                return None


# Global auth service instance
auth_service = AuthService()


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    x_api_key: Annotated[Optional[str], Header()] = None,
    x_user_id: Annotated[Optional[str], Header()] = None
) -> Optional[str]:
    """Extract current user from JWT token or API key headers."""
    
    logger.info(f"🔐 get_current_user called with credentials: {bool(credentials)}, api_key: {bool(x_api_key)}")
    if credentials:
        logger.info(f"🔑 Token received: {credentials.credentials[:50]}...")
    
    # Option 1: Check API key authentication (for bot)
    if x_api_key and x_user_id:
        if settings.bot_api_key and x_api_key == settings.bot_api_key:
            logger.debug(f"Authenticated bot request for user {x_user_id}")
            return x_user_id
        else:
            logger.warning("Invalid API key provided")
    
    # Option 2: Check JWT token authentication (for web UI)
    if credentials and credentials.credentials:
        # Try regular JWT token first
        payload = auth_service.verify_jwt_token(credentials.credentials)
        logger.info(f"📋 Regular JWT payload: {payload}")
        if payload:
            user_id = payload.get("sub")
            logger.info(f"✅ Regular JWT - user_id from 'sub': {user_id}")
            if user_id:
                logger.debug(f"Authenticated JWT request for user {user_id}")
                return user_id
            else:
                logger.warning(f"⚠️ Regular JWT has no 'sub' field, trying Web3")
        
        # Try Web3 JWT token
        try:
            # Use jose library (same as regular auth)
            from settings import get_settings
            settings = get_settings()
            SECRET_KEY = settings.jwt_secret_key
            ALGORITHM = "HS256"
            
            logger.info(f"🔍 Trying Web3 token verification with secret key: {SECRET_KEY[:20]}...")
            web3_payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
            logger.info(f"✅ Web3 token decoded successfully: {web3_payload}")
            if web3_payload:
                # Return user_id for Web3 users (not wallet_address)
                user_id = web3_payload.get("user_id")
                wallet_address = web3_payload.get("wallet_address")
                logger.info(f"🎯 Authenticated Web3 JWT request for user {user_id} (wallet: {wallet_address})")
                return user_id
        except JWTError as e:
            logger.error(f"❌ Web3 token verification failed: {e}")
        except Exception as e:
            logger.error(f"❌ Web3 token verification failed: {e}")
    
    # No valid authentication found
    return None


async def require_auth(
    user_id: Annotated[Optional[str], Depends(get_current_user)]
) -> str:
    """Require authentication for endpoint."""
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


async def verify_resource_owner(
    thread_id: str,
    user_id: Annotated[str, Depends(require_auth)]
) -> str:
    """Verify that user owns the requested resource."""
    # In our system, thread_id equals user_id (Telegram user ID)
    if thread_id != user_id:
        logger.warning(f"User {user_id} attempted to access thread {thread_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to this resource is forbidden"
        )
    return user_id