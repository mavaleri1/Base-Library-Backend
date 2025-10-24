"""Add Web3 authentication support

This migration adds Web3 wallet-based authentication.

Changes:
- DROP TABLE auth_codes (Telegram auth, no longer needed)
- CREATE TABLE users (Web3 wallet-based user profiles)
- CREATE TABLE user_sessions (mapping users to thread_ids)
- CREATE TABLE web3_nonces (for signature verification)

Revision ID: e6g9b2h3i4j5
Revises: f0cfe9e964b9
Create Date: 2025-10-09 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e6g9b2h3i4j5'
down_revision: Union[str, Sequence[str], None] = 'f0cfe9e964b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add Web3 authentication support."""
    
    # Step 1: Drop old Telegram auth_codes table
    print("Dropping auth_codes table (Telegram auth)...")
    op.drop_index('idx_auth_codes_created_at', table_name='auth_codes')
    op.drop_table('auth_codes')
    
    # Step 2: Create users table
    print("Creating users table...")
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, comment='Internal UUID for database relationships'),
        sa.Column('wallet_address', sa.String(length=42), nullable=False, comment='Ethereum wallet address (0x...)'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='User creation timestamp'),
        sa.Column('last_login', sa.DateTime(), nullable=True, comment='Last successful login timestamp'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('wallet_address'),
        comment='User profiles for Web3 wallet-based authentication'
    )
    op.create_index('idx_users_wallet', 'users', ['wallet_address'], unique=True)
    
    # Step 3: Create user_sessions table
    print("Creating user_sessions table...")
    op.create_table(
        'user_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, comment='Session UUID'),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False, comment='Reference to user'),
        sa.Column('thread_id', sa.String(length=255), nullable=False, comment='LangGraph thread identifier'),
        sa.Column('session_name', sa.String(length=255), nullable=True, comment='Optional session name/description'),
        sa.Column('status', sa.String(length=20), nullable=False, comment='Session status: active, completed, archived'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='Session creation timestamp'),
        sa.Column('last_activity', sa.DateTime(), nullable=False, comment='Last activity timestamp'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('thread_id'),
        comment='User sessions mapping to LangGraph thread_ids'
    )
    op.create_index('idx_sessions_user', 'user_sessions', ['user_id'])
    op.create_index('idx_sessions_thread', 'user_sessions', ['thread_id'], unique=True)
    op.create_index('idx_sessions_status', 'user_sessions', ['status'])
    
    # Step 4: Create web3_nonces table
    print("Creating web3_nonces table...")
    op.create_table(
        'web3_nonces',
        sa.Column('wallet_address', sa.String(length=42), nullable=False, comment='Ethereum wallet address (0x...)'),
        sa.Column('nonce', sa.String(length=64), nullable=False, comment='Random nonce for signature verification'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='Nonce creation timestamp'),
        sa.Column('expires_at', sa.DateTime(), nullable=False, comment='Nonce expiration timestamp'),
        sa.PrimaryKeyConstraint('wallet_address'),
        comment='Temporary nonces for Web3 authentication'
    )
    op.create_index('idx_nonces_expires', 'web3_nonces', ['expires_at'])
    
    print("Web3 authentication migration completed successfully!")


def downgrade() -> None:
    """Remove Web3 authentication and restore Telegram auth."""
    
    # Drop Web3 tables
    print("Dropping Web3 tables...")
    op.drop_index('idx_nonces_expires', table_name='web3_nonces')
    op.drop_table('web3_nonces')
    
    op.drop_index('idx_sessions_status', table_name='user_sessions')
    op.drop_index('idx_sessions_thread', table_name='user_sessions')
    op.drop_index('idx_sessions_user', table_name='user_sessions')
    op.drop_table('user_sessions')
    
    op.drop_index('idx_users_wallet', table_name='users')
    op.drop_table('users')
    
    # Recreate Telegram auth_codes table
    print("Recreating auth_codes table (Telegram auth)...")
    op.create_table(
        'auth_codes',
        sa.Column('username', sa.String(length=255), nullable=False),
        sa.Column('code', sa.String(length=10), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('username', 'code')
    )
    op.create_index('idx_auth_codes_created_at', 'auth_codes', ['created_at'])
    
    print("Downgrade to Telegram authentication completed!")

