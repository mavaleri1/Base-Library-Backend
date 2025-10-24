"""Migrate to Web3 authentication
IMPORTANT: This migration will DELETE all existing user data!

Revision ID: d5f8a1b2c3d4
Revises: 954ed3a47a38
Create Date: 2025-10-09 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd5f8a1b2c3d4'
down_revision: Union[str, Sequence[str], None] = '954ed3a47a38'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade to Web3 authentication.
    
    WARNING: This will delete all existing user data!
    """
    
    # Step 1: Drop dependent table first (due to FK constraint)
    print("Dropping user_placeholder_settings table...")
    op.drop_table('user_placeholder_settings')
    
    # Step 2: Drop old user_profiles table
    print("Dropping old user_profiles table...")
    op.drop_table('user_profiles')
    
    # Step 3: Create new user_profiles table with Web3 structure
    print("Creating new user_profiles table with Web3 support...")
    op.create_table(
        'user_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, comment='Internal UUID for database relationships'),
        sa.Column('wallet_address', sa.String(length=42), nullable=False, comment='Ethereum wallet address (0x...)'),
        sa.Column('username', sa.String(length=100), nullable=True, comment='Optional display name for user'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False, comment='User profile creation timestamp'),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False, comment='Last update timestamp'),
        sa.Column('last_login', sa.DateTime(), nullable=True, comment='Last successful login timestamp'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('wallet_address'),
        comment='User profiles for Web3 wallet-based authentication'
    )
    
    # Create indexes
    op.create_index('ix_user_profiles_wallet', 'user_profiles', ['wallet_address'], unique=True)
    
    # Step 4: Create new user_placeholder_settings table
    print("Creating new user_placeholder_settings table...")
    op.create_table(
        'user_placeholder_settings',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False, comment='Reference to user profile'),
        sa.Column('placeholder_id', postgresql.UUID(as_uuid=True), nullable=False, comment='Reference to placeholder'),
        sa.Column('placeholder_value_id', postgresql.UUID(as_uuid=True), nullable=False, comment='Selected value for this placeholder'),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False, comment='Last update timestamp'),
        sa.ForeignKeyConstraint(['placeholder_id'], ['placeholders.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['placeholder_value_id'], ['placeholder_values.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['user_id'], ['user_profiles.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id', 'placeholder_id'),
        comment='User-specific placeholder settings'
    )
    
    print("Migration to Web3 authentication completed successfully!")
    print("Note: All previous Telegram user data has been removed.")


def downgrade() -> None:
    """Downgrade back to Telegram authentication.
    
    WARNING: This will delete all Web3 user data!
    """
    
    # Drop new tables
    print("Dropping Web3 user_placeholder_settings table...")
    op.drop_index('ix_user_profiles_wallet', table_name='user_profiles')
    op.drop_table('user_placeholder_settings')
    
    print("Dropping Web3 user_profiles table...")
    op.drop_table('user_profiles')
    
    # Recreate old Telegram-based tables
    print("Recreating Telegram user_profiles table...")
    op.create_table(
        'user_profiles',
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('user_id')
    )
    
    print("Recreating Telegram user_placeholder_settings table...")
    op.create_table(
        'user_placeholder_settings',
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('placeholder_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('placeholder_value_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['placeholder_id'], ['placeholders.id']),
        sa.ForeignKeyConstraint(['placeholder_value_id'], ['placeholder_values.id']),
        sa.ForeignKeyConstraint(['user_id'], ['user_profiles.user_id']),
        sa.PrimaryKeyConstraint('user_id', 'placeholder_id')
    )
    
    print("Downgrade to Telegram authentication completed!")
    print("Note: All Web3 user data has been removed.")

