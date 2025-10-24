"""Content hashing and IPFS CID preparation utilities.

This module provides functions for:
- SHA-256 hashing of material content
- Preparing content for IPFS (CIDv1 calculation)
- Future integration with IPFS/Arweave for permanent storage
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_content_hash(content: str, encoding: str = "utf-8") -> str:
    """Calculate SHA-256 hash of material content.
    
    This hash will be stored in blockchain for content verification.
    
    Args:
        content: Material content (markdown text)
        encoding: Text encoding (default: utf-8)
        
    Returns:
        Hexadecimal SHA-256 hash string (64 characters)
    """
    try:
        content_bytes = content.encode(encoding)
        hash_object = hashlib.sha256(content_bytes)
        hash_hex = hash_object.hexdigest()
        
        logger.debug(f"Calculated content hash: {hash_hex}")
        return hash_hex
        
    except Exception as e:
        logger.error(f"Error calculating content hash: {e}", exc_info=True)
        raise


def verify_content_hash(content: str, expected_hash: str, encoding: str = "utf-8") -> bool:
    """Verify that content matches the expected hash.
    
    Args:
        content: Material content
        expected_hash: Expected SHA-256 hash
        encoding: Text encoding
        
    Returns:
        True if content matches hash, False otherwise
    """
    try:
        actual_hash = calculate_content_hash(content, encoding)
        return actual_hash == expected_hash
    except Exception as e:
        logger.error(f"Error verifying content hash: {e}", exc_info=True)
        return False


def prepare_ipfs_cid_placeholder(content_hash: str, material_id: str) -> str:
    """Prepare a placeholder CID for future IPFS upload.
    
    NOTE: This is NOT a real IPFS CID. It's a placeholder that follows
    a predictable format for future replacement when content is uploaded to IPFS.
    
    Real IPFS CID will be generated when content is actually uploaded to IPFS.
    Format: "ipfs://placeholder/{material_id}/{first_16_chars_of_hash}"
    
    Args:
        content_hash: SHA-256 hash of content
        material_id: UUID of material
        
    Returns:
        Placeholder CID string
    """
    short_hash = content_hash[:16]
    placeholder = f"ipfs://placeholder/{material_id}/{short_hash}"
    logger.debug(f"Generated IPFS CID placeholder: {placeholder}")
    return placeholder


def calculate_multihash_sha256(content: str, encoding: str = "utf-8") -> bytes:
    """Calculate multihash SHA-256 for IPFS CID generation.
    
    This function prepares content for real IPFS CIDv1 generation.
    The multihash format is: <hash-function-code><digest-length><hash-digest>
    
    For SHA-256:
    - Hash function code: 0x12 (18 in decimal)
    - Digest length: 0x20 (32 bytes)
    - Hash digest: 32 bytes of SHA-256 hash
    
    NOTE: This is preparation for future IPFS integration. 
    Real CID generation requires IPFS libraries (e.g., py-cid, py-multihash).
    
    Args:
        content: Material content
        encoding: Text encoding
        
    Returns:
        Multihash bytes (34 bytes total: 2 byte header + 32 byte hash)
    """
    try:
        # Calculate SHA-256 hash
        content_bytes = content.encode(encoding)
        hash_digest = hashlib.sha256(content_bytes).digest()
        
        # Construct multihash: function_code (0x12) + length (0x20) + digest
        multihash = bytes([0x12, 0x20]) + hash_digest
        
        logger.debug(f"Calculated multihash (length: {len(multihash)} bytes)")
        return multihash
        
    except Exception as e:
        logger.error(f"Error calculating multihash: {e}", exc_info=True)
        raise


class ContentHashManager:
    """Manager for content hashing and IPFS preparation."""
    
    @staticmethod
    def create_blockchain_metadata(
        content: str,
        material_id: str,
        subject: Optional[str] = None,
        grade: Optional[str] = None,
        topic: Optional[str] = None,
        author_wallet: Optional[str] = None
    ) -> dict:
        """Create complete metadata package for blockchain storage.
        
        This is the metadata that will be stored on-chain (small footprint).
        The actual content will be in IPFS/Arweave.
        
        Args:
            content: Material content
            material_id: Material UUID
            subject: Subject classification
            grade: Grade level
            topic: Topic
            author_wallet: Author's wallet address (future NFT owner)
            
        Returns:
            Dictionary with blockchain-ready metadata
        """
        content_hash = calculate_content_hash(content)
        ipfs_cid_placeholder = prepare_ipfs_cid_placeholder(content_hash, material_id)
        
        metadata = {
            "material_id": material_id,
            "content_hash": content_hash,
            "ipfs_cid": ipfs_cid_placeholder,
            "subject": subject or "Unknown",
            "grade": grade or "Unknown",
            "topic": topic or "Unknown",
            "author_wallet": author_wallet,
            "version": "1.0",
            "storage_type": "ipfs_placeholder"
        }
        
        logger.info(
            f"Created blockchain metadata for material {material_id}: "
            f"subject={subject}, grade={grade}, topic={topic}"
        )
        
        return metadata
    
    @staticmethod
    def calculate_word_count(content: str) -> int:
        """Calculate approximate word count of material.
        
        Args:
            content: Material content (markdown)
            
        Returns:
            Approximate word count
        """
        # Simple word count (split by whitespace)
        # Could be improved with markdown parsing to exclude code blocks, etc.
        words = content.split()
        return len(words)


# Global instance
_content_hash_manager: Optional[ContentHashManager] = None


def get_content_hash_manager() -> ContentHashManager:
    """Get global content hash manager instance."""
    global _content_hash_manager
    if _content_hash_manager is None:
        _content_hash_manager = ContentHashManager()
    return _content_hash_manager



