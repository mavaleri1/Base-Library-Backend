"""Validation utilities for NFT data."""

import re
import hashlib
from typing import Dict, Optional


def validate_ipfs_cid(cid: str) -> Dict[str, bool]:
    """
    Validate IPFS CID format.
    
    Args:
        cid: IPFS Content Identifier
        
    Returns:
        Dict with validation result
    """
    if not cid:
        return {"valid": False, "error": "IPFS CID is required"}
    
    # Basic IPFS CID validation (starts with Qm or bafy)
    if not re.match(r'^(Qm[1-9A-HJ-NP-Za-km-z]{44}|bafy[a-z0-9]{50,})$', cid):
        return {"valid": False, "error": "Invalid IPFS CID format"}
    
    return {"valid": True}


def validate_content_hash(content_hash: str) -> Dict[str, bool]:
    """
    Validate content hash format (SHA-256).
    
    Args:
        content_hash: SHA-256 hash string
        
    Returns:
        Dict with validation result
    """
    if not content_hash:
        return {"valid": False, "error": "Content hash is required"}
    
    # SHA-256 should be 64 characters long and contain only hex characters
    if not re.match(r'^[a-fA-F0-9]{64}$', content_hash):
        return {"valid": False, "error": "Invalid content hash format (must be SHA-256)"}
    
    return {"valid": True}


def validate_tx_hash(tx_hash: str) -> Dict[str, bool]:
    """
    Validate Ethereum transaction hash format.
    
    Args:
        tx_hash: Transaction hash string
        
    Returns:
        Dict with validation result
    """
    if not tx_hash:
        return {"valid": False, "error": "Transaction hash is required"}
    
    # Ethereum tx hash should be 66 characters long (0x + 64 hex chars)
    if not re.match(r'^0x[a-fA-F0-9]{64}$', tx_hash):
        return {"valid": False, "error": "Invalid transaction hash format"}
    
    return {"valid": True}


def validate_token_id(token_id: int) -> Dict[str, bool]:
    """
    Validate NFT token ID.
    
    Args:
        token_id: Token ID integer
        
    Returns:
        Dict with validation result
    """
    if token_id is None:
        return {"valid": False, "error": "Token ID is required"}
    
    if not isinstance(token_id, int):
        return {"valid": False, "error": "Token ID must be an integer"}
    
    if token_id < 0:
        return {"valid": False, "error": "Token ID must be non-negative"}
    
    # Reasonable upper limit for token ID
    if token_id > 2**256 - 1:
        return {"valid": False, "error": "Token ID is too large"}
    
    return {"valid": True}


def validate_blockchain_sync_data(
    ipfs_cid: str,
    content_hash: str,
    tx_hash: str,
    token_id: int
) -> Dict[str, bool]:
    """
    Validate all blockchain sync data.
    
    Args:
        ipfs_cid: IPFS Content Identifier
        content_hash: Content hash string
        tx_hash: Transaction hash string
        token_id: Token ID integer
        
    Returns:
        Dict with validation result
    """
    # Validate IPFS CID
    ipfs_result = validate_ipfs_cid(ipfs_cid)
    if not ipfs_result["valid"]:
        return ipfs_result
    
    # Validate content hash
    hash_result = validate_content_hash(content_hash)
    if not hash_result["valid"]:
        return hash_result
    
    # Validate transaction hash
    tx_result = validate_tx_hash(tx_hash)
    if not tx_result["valid"]:
        return tx_result
    
    # Validate token ID
    token_result = validate_token_id(token_id)
    if not token_result["valid"]:
        return token_result
    
    return {"valid": True}


def verify_content_hash_match(content: str, provided_hash: str) -> bool:
    """
    Verify that the provided hash matches the actual content.
    
    Args:
        content: Material content string
        provided_hash: Hash provided by client
        
    Returns:
        True if hash matches, False otherwise
    """
    try:
        # Calculate SHA-256 hash of content
        actual_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return actual_hash.lower() == provided_hash.lower()
    except Exception:
        return False
