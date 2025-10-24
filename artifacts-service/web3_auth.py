"""Web3 authentication endpoints for core.

This module provides Web3 wallet-based authentication:
- Request nonce for signature
- Verify signature and create/login user
- JWT token generation
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from eth_account.messages import encode_defunct
from eth_account import Account
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from models_web3 import User, UserSession, Web3Nonce
from settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# JWT settings
SECRET_KEY = settings.jwt_secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Security
security = HTTPBearer()

# Database setup for core
# Convert sync postgresql:// URL to async postgresql+asyncpg:// URL
async_db_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
engine = create_async_engine(async_db_url)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    """Dependency to get database session."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# API Models
class NonceRequest(BaseModel):
    """Request for nonce generation."""
    wallet_address: str = Field(..., description="Ethereum wallet address (0x...)")


class NonceResponse(BaseModel):
    """Response containing nonce for signing."""
    nonce: str = Field(..., description="Random nonce to sign")
    message: str = Field(..., description="Full message to sign")
    expires_in: int = Field(..., description="Nonce expiration time in seconds")


class SignatureVerifyRequest(BaseModel):
    """Request to verify signature."""
    wallet_address: str = Field(..., description="Ethereum wallet address (0x...)")
    signature: str = Field(..., description="Signed message (0x...)")
    nonce: str = Field(..., description="Nonce that was signed")


class AuthResponse(BaseModel):
    """Successful authentication response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    user: dict = Field(..., description="User profile information")


# Router
router = APIRouter(prefix="/auth", tags=["Web3 Authentication"])


def create_jwt_token(wallet_address: str, user_id: str) -> str:
    """Create JWT token for authenticated user."""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "wallet_address": wallet_address.lower(),
        "user_id": user_id,
        "exp": expire
    }
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_jwt_token(token: str) -> Optional[dict]:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Dependency to get current authenticated user from JWT."""
    token = credentials.credentials
    payload = verify_jwt_token(token)
    
    wallet_address = payload.get("wallet_address")
    if not wallet_address:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    # Get user from database
    result = await db.execute(
        select(User).where(User.wallet_address == wallet_address)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user


@router.post("/request-nonce", response_model=NonceResponse)
async def request_nonce(
    request: NonceRequest,
    db: AsyncSession = Depends(get_db)
):
    """Request a nonce for Web3 signature.
    
    The frontend should:
    1. Call this endpoint with wallet address
    2. Get the nonce and message
    3. Ask user to sign the message with their wallet
    4. Send signature to /verify-signature
    """
    wallet_address = request.wallet_address.lower()
    
    # Validate wallet address format
    if not wallet_address.startswith("0x") or len(wallet_address) != 42:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Ethereum wallet address format"
        )
    
    # Generate random nonce
    nonce = secrets.token_hex(32)
    expires_at = datetime.utcnow() + timedelta(minutes=5)  # 5 minutes expiration
    
    # Delete old nonce for this wallet (if exists)
    await db.execute(
        delete(Web3Nonce).where(Web3Nonce.wallet_address == wallet_address)
    )
    
    # Save new nonce
    nonce_record = Web3Nonce(
        wallet_address=wallet_address,
        nonce=nonce,
        expires_at=expires_at
    )
    db.add(nonce_record)
    await db.commit()
    
    # Create message to sign
    message = f"Sign this message to authenticate with Base Library\n\nNonce: {nonce}\nWallet: {wallet_address}\n\nThis request will not trigger any blockchain transaction or cost any gas fees."
    
    logger.info(f"Generated nonce for wallet {wallet_address[:10]}...")
    
    return NonceResponse(
        nonce=nonce,
        message=message,
        expires_in=300  # 5 minutes
    )


@router.post("/verify-signature", response_model=AuthResponse)
async def verify_signature(
    request: SignatureVerifyRequest,
    db: AsyncSession = Depends(get_db)
):
    """Verify Web3 signature and authenticate user.
    
    This endpoint:
    1. Verifies the signature matches the wallet address
    2. Creates user if doesn't exist
    3. Returns JWT token for authenticated sessions
    """
    wallet_address = request.wallet_address.lower()
    logger.info(f"Verifying signature for wallet: {wallet_address}")
    logger.info(f"Request nonce: {request.nonce}")
    
    # Get nonce from database
    result = await db.execute(
        select(Web3Nonce).where(Web3Nonce.wallet_address == wallet_address)
    )
    nonce_record = result.scalar_one_or_none()
    
    if not nonce_record:
        logger.error(f"No nonce found for wallet: {wallet_address}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No nonce found for this wallet. Please request a nonce first."
        )
    
    logger.info(f"Found nonce record: {nonce_record.nonce}, expires: {nonce_record.expires_at}")
    
    # Check nonce hasn't expired
    if nonce_record.is_expired():
        logger.error(f"Nonce expired for wallet: {wallet_address}")
        await db.delete(nonce_record)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nonce has expired. Please request a new nonce."
        )
    
    # Check nonce matches
    logger.info(f"Comparing nonces - stored: {nonce_record.nonce}, received: {request.nonce}")
    if nonce_record.nonce != request.nonce:
        logger.error(f"Nonce mismatch for wallet: {wallet_address}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid nonce"
        )
    
    # Recreate the message that was signed
    message = f"Sign this message to authenticate with Base Library\n\nNonce: {request.nonce}\nWallet: {wallet_address}\n\nThis request will not trigger any blockchain transaction or cost any gas fees."
    
    # Verify signature
    try:
        message_hash = encode_defunct(text=message)
        recovered_address = Account.recover_message(message_hash, signature=request.signature)
        
        if recovered_address.lower() != wallet_address:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Signature verification failed"
            )
    except Exception as e:
        logger.error(f"Signature verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid signature"
        )
    
    # Delete used nonce
    await db.delete(nonce_record)
    
    # Get or create user
    result = await db.execute(
        select(User).where(User.wallet_address == wallet_address)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        # Create new user
        user = User(wallet_address=wallet_address)
        db.add(user)
        await db.flush()
        await db.refresh(user)
        logger.info(f"Created new user for wallet {wallet_address[:10]}...")
    else:
        # Update last login
        user.last_login = datetime.utcnow()
        logger.info(f"User logged in: {wallet_address[:10]}...")
    
    await db.commit()
    
    # Create JWT token
    access_token = create_jwt_token(wallet_address, str(user.id))
    
    return AuthResponse(
        access_token=access_token,
        token_type="bearer",
        user={
            "id": str(user.id),
            "wallet_address": user.wallet_address,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
    )


@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current authenticated user information."""
    return {
        "id": str(current_user.id),
        "wallet_address": current_user.wallet_address,
        "created_at": current_user.created_at.isoformat(),
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None
    }


@router.post("/logout")
async def logout():
    """Logout endpoint.
    
    Note: JWT tokens are stateless, so logout is handled client-side
    by deleting the token. This endpoint is here for API completeness.
    """
    return {"message": "Logged out successfully. Please delete the token from client."}

