"""Permission checking utilities for materials access control."""

import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models_web3 import Material, User

logger = logging.getLogger(__name__)


async def check_material_ownership(
    material_id: str,
    user_wallet: str,
    db: AsyncSession
) -> bool:
    """Check if user owns the material.
    
    Args:
        material_id: Material UUID
        user_wallet: User's wallet address
        db: Database session
        
    Returns:
        True if user owns the material, False otherwise
    """
    try:
        result = await db.execute(
            select(Material).where(Material.id == material_id)
        )
        material = result.scalar_one_or_none()
        
        if not material:
            return False
        
        # Get author's wallet address
        author_result = await db.execute(
            select(User).where(User.id == material.author_id)
        )
        author = author_result.scalar_one_or_none()
        
        if not author:
            return False
        
        # Compare wallet addresses (case-insensitive)
        return author.wallet_address.lower() == user_wallet.lower()
        
    except Exception as e:
        logger.error(f"Error checking material ownership: {e}", exc_info=True)
        return False


async def can_view_material(
    material: Material,
    user_wallet: Optional[str] = None,
    db: Optional[AsyncSession] = None
) -> bool:
    """Check if user can view the material.
    
    Args:
        material: Material object
        user_wallet: User's wallet address (None for anonymous)
        db: Database session (required if user_wallet is provided)
        
    Returns:
        True if user can view the material, False otherwise
    """
    # Published materials are accessible to everyone
    if material.status == "published":
        return True
    
    # Draft and archived materials only accessible to owner
    if user_wallet and db:
        # Get author's wallet address
        author_result = await db.execute(
            select(User).where(User.id == material.author_id)
        )
        author = author_result.scalar_one_or_none()
        
        if author:
            return author.wallet_address.lower() == user_wallet.lower()
    
    return False


async def can_edit_material(
    material: Material,
    user_wallet: str,
    db: AsyncSession
) -> bool:
    """Check if user can edit the material.
    
    Only material owner can edit it.
    
    Args:
        material: Material object
        user_wallet: User's wallet address
        db: Database session
        
    Returns:
        True if user can edit the material, False otherwise
    """
    return await check_material_ownership(str(material.id), user_wallet, db)


async def get_material_with_permissions(
    material: Material,
    current_user_wallet: Optional[str],
    db: AsyncSession
) -> dict:
    """Get material data with permission flags.
    
    Args:
        material: Material object
        current_user_wallet: Current user's wallet address (None if not authenticated)
        db: Database session
        
    Returns:
        Dictionary with material data and permission flags
    """
    # Get author's wallet address
    author_result = await db.execute(
        select(User).where(User.id == material.author_id)
    )
    author = author_result.scalar_one_or_none()
    author_wallet = author.wallet_address if author else "unknown"
    
    # Check permissions
    can_edit = False
    if current_user_wallet:
        can_edit = author_wallet.lower() == current_user_wallet.lower()
    
    return {
        "id": str(material.id),
        "author_id": str(material.author_id),
        "author_wallet": author_wallet,
        "thread_id": material.thread_id,
        "session_id": material.session_id,
        "file_path": material.file_path,
        "subject": material.subject,
        "grade": material.grade,
        "topic": material.topic,
        "content_hash": material.content_hash,
        "ipfs_cid": material.ipfs_cid,
        "title": material.title,
        "word_count": material.word_count,
        "status": material.status,
        "created_at": material.created_at.isoformat(),
        "updated_at": material.updated_at.isoformat(),
        "can_edit": can_edit,
        # NFT fields
        "nft_minted": material.nft_minted or False,
        "nft_token_id": material.nft_token_id,
        "is_owner": can_edit
    }


