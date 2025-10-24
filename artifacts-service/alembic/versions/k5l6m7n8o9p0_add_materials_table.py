"""Add materials table for learning content with blockchain metadata

This migration adds the materials table to store educational content
with blockchain-ready metadata (subject, grade, topic, hash, CID).

Changes:
- CREATE TABLE materials (learning materials with metadata)
- Add indexes for efficient querying
- Add foreign key to users (author/future NFT owner)

Revision ID: k5l6m7n8o9p0
Revises: e6g9b2h3i4j5
Create Date: 2025-10-13 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'k5l6m7n8o9p0'
down_revision: Union[str, Sequence[str], None] = 'e6g9b2h3i4j5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add materials table for learning content."""
    
    print("Creating materials table...")
    op.create_table(
        'materials',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, comment='Material UUID'),
        sa.Column('author_id', postgresql.UUID(as_uuid=True), nullable=False, comment='Reference to material author (future NFT owner)'),
        sa.Column('thread_id', sa.String(length=255), nullable=False, comment='LangGraph thread identifier'),
        sa.Column('session_id', sa.String(length=255), nullable=False, comment='Session identifier (e.g. session-20251012_194704)'),
        sa.Column('subject', sa.String(length=100), nullable=True, comment='Subject area (e.g., Mathematics, Physics, Chemistry)'),
        sa.Column('grade', sa.String(length=50), nullable=True, comment="Grade level (e.g., 'Beginner', 'Intermediate', 'Advanced')"),
        sa.Column('topic', sa.String(length=255), nullable=True, comment="Specific topic (e.g., 'Linear Equations', 'Blockchain')"),
        sa.Column('content', sa.Text(), nullable=False, comment='Full material content (markdown)'),
        sa.Column('input_query', sa.Text(), nullable=True, comment='Original user query that generated this material'),
        sa.Column('content_hash', sa.String(length=64), nullable=False, comment='SHA-256 hash of content (for blockchain verification)'),
        sa.Column('ipfs_cid', sa.String(length=100), nullable=True, comment='IPFS CID (Content Identifier) for decentralized storage'),
        sa.Column('file_path', sa.String(length=512), nullable=False, comment='Relative path to material file on localhost'),
        sa.Column('title', sa.String(length=255), nullable=True, comment='Material title (auto-generated or user-provided)'),
        sa.Column('word_count', sa.Integer(), nullable=True, comment='Approximate word count'),
        sa.Column('status', sa.String(length=20), nullable=False, comment='Material status: draft, published, archived'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='Material creation timestamp'),
        sa.Column('updated_at', sa.DateTime(), nullable=False, comment='Last update timestamp'),
        sa.ForeignKeyConstraint(['author_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('content_hash'),
        sa.UniqueConstraint('ipfs_cid'),
        comment='Learning materials with blockchain-ready metadata'
    )
    
    print("Creating indexes for materials table...")
    op.create_index('idx_materials_author', 'materials', ['author_id'])
    op.create_index('idx_materials_thread', 'materials', ['thread_id'])
    op.create_index('idx_materials_session', 'materials', ['session_id'])
    op.create_index('idx_materials_subject_grade', 'materials', ['subject', 'grade'])
    op.create_index('idx_materials_hash', 'materials', ['content_hash'], unique=True)
    op.create_index('idx_materials_cid', 'materials', ['ipfs_cid'], unique=True)
    op.create_index('idx_materials_status', 'materials', ['status'])
    op.create_index('idx_materials_created', 'materials', ['created_at'])
    
    print("Materials table migration completed successfully!")


def downgrade() -> None:
    """Remove materials table."""
    
    print("Dropping materials table...")
    op.drop_index('idx_materials_created', table_name='materials')
    op.drop_index('idx_materials_status', table_name='materials')
    op.drop_index('idx_materials_cid', table_name='materials')
    op.drop_index('idx_materials_hash', table_name='materials')
    op.drop_index('idx_materials_subject_grade', table_name='materials')
    op.drop_index('idx_materials_session', table_name='materials')
    op.drop_index('idx_materials_thread', table_name='materials')
    op.drop_index('idx_materials_author', table_name='materials')
    op.drop_table('materials')
    
    print("Materials table downgrade completed!")



