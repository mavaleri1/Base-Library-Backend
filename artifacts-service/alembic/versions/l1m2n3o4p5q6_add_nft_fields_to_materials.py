"""Add NFT fields to materials table

This migration adds NFT-specific fields to the existing materials table
to track NFT minting status and blockchain transaction details.

The materials table was already designed for blockchain integration with:
- author_id (future NFT owner)
- content_hash (for blockchain verification) 
- ipfs_cid (for decentralized storage)
- subject/grade/topic (blockchain metadata)

This migration adds the missing NFT tracking fields:
- nft_minted: whether NFT has been minted
- nft_token_id: token ID in blockchain
- nft_tx_hash: minting transaction hash
- nft_ipfs_cid: IPFS CID used for NFT metadata
- nft_content_hash: content hash used for NFT
- nft_created_at: when NFT was minted

Revision ID: l1m2n3o4p5q6
Revises: k5l6m7n8o9p0
Create Date: 2025-01-27 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'l1m2n3o4p5q6'
down_revision: Union[str, Sequence[str], None] = 'k5l6m7n8o9p0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add NFT fields to materials table."""
    
    print("Adding NFT fields to materials table...")
    
    # Add NFT tracking fields
    op.add_column('materials', sa.Column('nft_minted', sa.Boolean(), nullable=False, server_default='false', comment='Whether NFT has been minted for this material'))
    op.add_column('materials', sa.Column('nft_token_id', sa.Integer(), nullable=True, comment='Token ID in blockchain contract'))
    op.add_column('materials', sa.Column('nft_tx_hash', sa.String(length=255), nullable=True, comment='Transaction hash of NFT minting'))
    op.add_column('materials', sa.Column('nft_ipfs_cid', sa.String(length=255), nullable=True, comment='IPFS CID used for NFT metadata'))
    op.add_column('materials', sa.Column('nft_content_hash', sa.String(length=255), nullable=True, comment='Content hash used for NFT verification'))
    op.add_column('materials', sa.Column('nft_created_at', sa.DateTime(), nullable=True, comment='When NFT was minted'))
    
    # Add indexes for efficient NFT queries
    print("Creating indexes for NFT fields...")
    op.create_index('idx_materials_nft_minted', 'materials', ['nft_minted'])
    op.create_index('idx_materials_nft_token_id', 'materials', ['nft_token_id'])
    
    # Update existing materials to have nft_minted = false
    print("Setting default NFT status for existing materials...")
    op.execute("UPDATE materials SET nft_minted = false WHERE nft_minted IS NULL")
    
    print("NFT fields migration completed successfully!")


def downgrade() -> None:
    """Remove NFT fields from materials table."""
    
    print("Removing NFT fields from materials table...")
    
    # Drop indexes
    op.drop_index('idx_materials_nft_token_id', table_name='materials')
    op.drop_index('idx_materials_nft_minted', table_name='materials')
    
    # Drop columns
    op.drop_column('materials', 'nft_created_at')
    op.drop_column('materials', 'nft_content_hash')
    op.drop_column('materials', 'nft_ipfs_cid')
    op.drop_column('materials', 'nft_tx_hash')
    op.drop_column('materials', 'nft_token_id')
    op.drop_column('materials', 'nft_minted')
    
    print("NFT fields downgrade completed!")
