"""Add index for nft_content_hash field

This migration adds an index on the nft_content_hash field for efficient
duplicate detection queries.

Revision ID: m7n8o9p0q1r2
Revises: l1m2n3o4p5q6
Create Date: 2025-01-27 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'm7n8o9p0q1r2'
down_revision: Union[str, Sequence[str], None] = 'l1m2n3o4p5q6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add index for nft_content_hash field."""
    
    print("Adding index for nft_content_hash field...")
    
    # Add index for efficient duplicate detection
    op.create_index(
        'idx_materials_nft_content_hash', 
        'materials', 
        ['nft_content_hash']
    )
    
    print("Index for nft_content_hash added successfully!")


def downgrade() -> None:
    """Remove index for nft_content_hash field."""
    
    print("Removing index for nft_content_hash field...")
    
    # Drop index
    op.drop_index('idx_materials_nft_content_hash', table_name='materials')
    
    print("Index for nft_content_hash removed successfully!")
