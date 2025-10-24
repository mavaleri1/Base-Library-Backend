"""Service for checking material ownership and NFT status."""

import uuid
from typing import Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from models_web3 import Material, User
from .nft_validation import validate_blockchain_sync_data, verify_content_hash_match


async def check_material_ownership(
    material_id: str, 
    current_user_wallet: str,
    db: AsyncSession
) -> Dict:
    """
    Check if current user owns the material and can mint NFT.
    
    Args:
        material_id: UUID of the material
        current_user_wallet: Wallet address of current user
        db: Database session
        
    Returns:
        Dict with ownership and NFT status information
    """
    try:
        # Convert string to UUID
        material_uuid = uuid.UUID(material_id)
    except ValueError:
        return {
            "isOwner": False,
            "nftMinted": False,
            "tokenId": None,
            "canMint": False,
            "error": "Invalid material ID format"
        }
    
    # Get material with author information
    result = await db.execute(
        select(Material)
        .options(selectinload(Material.author))
        .where(Material.id == material_uuid)
    )
    material = result.scalar_one_or_none()
    
    if not material:
        return {
            "isOwner": False,
            "nftMinted": False,
            "tokenId": None,
            "canMint": False,
            "error": "Material not found"
        }
    
    # Check ownership by comparing wallet addresses
    is_owner = material.author.wallet_address.lower() == current_user_wallet.lower()
    
    # Get NFT status
    nft_minted = material.nft_minted or False
    token_id = material.nft_token_id
    
    # Can mint only if user is owner AND NFT is not minted yet
    can_mint = is_owner and not nft_minted
    
    return {
        "isOwner": is_owner,
        "nftMinted": nft_minted,
        "tokenId": token_id,
        "canMint": can_mint
    }


async def get_nft_status(
    material_id: str,
    db: AsyncSession
) -> Dict:
    """
    Get NFT status for a material.
    
    Args:
        material_id: UUID of the material
        db: Database session
        
    Returns:
        Dict with NFT status information
    """
    try:
        # Convert string to UUID
        material_uuid = uuid.UUID(material_id)
    except ValueError:
        return {
            "nftMinted": False,
            "tokenId": None,
            "txHash": None,
            "ipfsCid": None,
            "error": "Invalid material ID format"
        }
    
    # Get material
    result = await db.execute(
        select(Material).where(Material.id == material_uuid)
    )
    material = result.scalar_one_or_none()
    
    if not material:
        return {
            "nftMinted": False,
            "tokenId": None,
            "txHash": None,
            "ipfsCid": None,
            "error": "Material not found"
        }
    
    return {
        "nftMinted": material.nft_minted or False,
        "tokenId": material.nft_token_id,
        "txHash": material.nft_tx_hash,
        "ipfsCid": material.nft_ipfs_cid
    }


async def sync_material_with_blockchain(
    material_id: str,
    current_user_wallet: str,
    ipfs_cid: str,
    content_hash: str,
    tx_hash: str,
    token_id: int,
    db: AsyncSession
) -> Dict:
    """
    Sync material with blockchain after NFT minting.
    
    Args:
        material_id: UUID of the material
        current_user_wallet: Wallet address of current user
        ipfs_cid: IPFS CID used for NFT metadata
        content_hash: Content hash used for NFT verification
        tx_hash: Transaction hash of NFT minting
        token_id: Token ID in blockchain contract
        db: Database session
        
    Returns:
        Dict with updated material information
    """
    try:
        # Convert string to UUID
        material_uuid = uuid.UUID(material_id)
    except ValueError:
        return {
            "error": "Invalid material ID format"
        }
    
    # Validate blockchain sync data
    validation_result = validate_blockchain_sync_data(
        ipfs_cid=ipfs_cid,
        content_hash=content_hash,
        tx_hash=tx_hash,
        token_id=token_id
    )
    
    if not validation_result["valid"]:
        return {
            "error": f"Validation failed: {validation_result['error']}"
        }
    
    # Get material with author information
    result = await db.execute(
        select(Material)
        .options(selectinload(Material.author))
        .where(Material.id == material_uuid)
    )
    material = result.scalar_one_or_none()
    
    if not material:
        return {
            "error": "Material not found"
        }
    
    # Check ownership
    is_owner = material.author.wallet_address.lower() == current_user_wallet.lower()
    if not is_owner:
        return {
            "error": "Only material author can sync with blockchain"
        }
    
    # Check if NFT is already minted
    if material.nft_minted:
        return {
            "error": "NFT is already minted for this material"
        }
    
    # Verify content hash matches actual material content
    if not verify_content_hash_match(material.content, content_hash):
        return {
            "error": "Content hash does not match material content"
        }
    
    # Check for duplicate content hash (prevent duplicate NFTs)
    existing_material_result = await db.execute(
        select(Material).where(
            Material.nft_content_hash == content_hash,
            Material.id != material_uuid
        )
    )
    existing_material = existing_material_result.scalar_one_or_none()
    
    if existing_material:
        return {
            "error": f"Material with this content already exists (Token ID: {existing_material.nft_token_id})"
        }
    
    # Update material with NFT information
    from datetime import datetime
    
    material.nft_minted = True
    material.nft_token_id = token_id
    material.nft_tx_hash = tx_hash
    material.nft_ipfs_cid = ipfs_cid
    material.nft_content_hash = content_hash
    material.nft_created_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(material)
    
    return {
        "id": str(material.id),
        "blockchain": {
            "tokenId": material.nft_token_id,
            "txHash": material.nft_tx_hash,
            "ipfsCid": material.nft_ipfs_cid,
            "contentHash": material.nft_content_hash,
            "isPublished": False,  # Can be extended later
            "createdAt": int(material.nft_created_at.timestamp()) if material.nft_created_at else None,
            "updatedAt": int(material.updated_at.timestamp())
        }
    }
