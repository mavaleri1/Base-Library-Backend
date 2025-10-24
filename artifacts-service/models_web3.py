"""Database models for Web3 authentication and user management in core.

This module contains models for:
- User profiles (Web3 wallet-based)
- User sessions (thread_id mapping)
- Web3 nonces (for signature verification)
- Learning materials (with blockchain-ready metadata)

Telegram support has been completely removed.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, ForeignKey, Index, String, Uuid, Text, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship

Base = declarative_base()


class User(Base):
    """User model - Web3 wallet-based authentication.
    
    Each user is identified by their Ethereum wallet address.
    This is the master table for all user-related data in core.
    """
    
    __tablename__ = "users"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        comment="Internal UUID for database relationships"
    )
    
    # Web3 wallet address - unique identifier
    wallet_address: Mapped[str] = mapped_column(
        String(42),
        unique=True,
        nullable=False,
        index=True,
        comment="Ethereum wallet address (0x...)"
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        comment="User creation timestamp"
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        comment="Last successful login timestamp"
    )
    
    # Relationships
    sessions: Mapped[list["UserSession"]] = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    materials: Mapped[list["Material"]] = relationship(
        "Material",
        back_populates="author",
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index("idx_users_wallet", "wallet_address"),
        {"comment": "User profiles for Web3 wallet-based authentication"}
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, wallet={self.wallet_address[:10]}...)>"


class UserSession(Base):
    """User session model - maps users to their LangGraph thread_ids.
    
    Each user can have multiple sessions (threads) with the AI system.
    This table maintains the mapping between wallet addresses and thread_ids.
    """
    
    __tablename__ = "user_sessions"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        comment="Session UUID"
    )
    
    # Foreign key to users
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to user"
    )
    
    # LangGraph thread ID
    thread_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="LangGraph thread identifier"
    )
    
    # Session metadata
    session_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Optional session name/description"
    )
    
    # Status
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        nullable=False,
        comment="Session status: active, completed, archived"
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        comment="Session creation timestamp"
    )
    last_activity: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        comment="Last activity timestamp"
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="sessions"
    )
    
    __table_args__ = (
        Index("idx_sessions_user", "user_id"),
        Index("idx_sessions_thread", "thread_id"),
        Index("idx_sessions_status", "status"),
        {"comment": "User sessions mapping to LangGraph thread_ids"}
    )
    
    def __repr__(self) -> str:
        return f"<UserSession(id={self.id}, thread={self.thread_id[:20]}...)>"


class Web3Nonce(Base):
    """Web3 nonce model - for signature verification during authentication.
    
    Stores temporary nonces for each wallet address to prevent replay attacks.
    Nonces should be short-lived and deleted after successful authentication.
    """
    
    __tablename__ = "web3_nonces"
    
    # Primary key - wallet address
    wallet_address: Mapped[str] = mapped_column(
        String(42),
        primary_key=True,
        comment="Ethereum wallet address (0x...)"
    )
    
    # Nonce value
    nonce: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Random nonce for signature verification"
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        comment="Nonce creation timestamp"
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        comment="Nonce expiration timestamp"
    )
    
    __table_args__ = (
        Index("idx_nonces_expires", "expires_at"),
        {"comment": "Temporary nonces for Web3 authentication"}
    )
    
    def __repr__(self) -> str:
        return f"<Web3Nonce(wallet={self.wallet_address[:10]}..., expires={self.expires_at})>"
    
    def is_expired(self) -> bool:
        """Check if nonce has expired."""
        return datetime.utcnow() > self.expires_at


class Material(Base):
    """Learning material model - stores educational content with blockchain-ready metadata.
    
    This model is prepared for future blockchain integration where:
    - Author gets NFT (right to edit)
    - Blockchain stores only metadata: subject, grade, topic, hash, CID, NFT owner
    - Actual content stored in IPFS/Arweave (permanent storage)
    - Material stored both in DB and localhost for redundancy
    """
    
    __tablename__ = "materials"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        comment="Material UUID"
    )
    
    # Author (NFT owner in future blockchain implementation)
    author_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to material author (future NFT owner)"
    )
    
    # Session and thread linkage
    thread_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="LangGraph thread identifier"
    )
    session_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Session identifier (e.g. session-20251012_194704)"
    )
    
    # Material classification (for blockchain metadata)
    subject: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="Subject area (e.g., Mathematics, Physics, Chemistry)"
    )
    grade: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Grade level (e.g., 'Beginner', 'Intermediate', 'Advanced')"
    )
    topic: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Specific topic (e.g., 'Linear Equations', 'Blockchain')"
    )
    
    # Content storage
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Full material content (markdown)"
    )
    input_query: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Original user query that generated this material"
    )
    
    # Blockchain-ready fields
    content_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="SHA-256 hash of content (for blockchain verification)"
    )
    ipfs_cid: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        unique=True,
        index=True,
        comment="IPFS CID (Content Identifier) for decentralized storage"
    )
    
    # File system path
    file_path: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Relative path to material file on localhost"
    )
    
    # Material metadata
    title: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Material title (auto-generated or user-provided)"
    )
    word_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Approximate word count"
    )
    
    # Status and lifecycle
    status: Mapped[str] = mapped_column(
        String(20),
        default="draft",
        nullable=False,
        index=True,
        comment="Material status: draft, published, archived"
    )
    
    # NFT tracking fields
    nft_minted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Whether NFT has been minted for this material"
    )
    nft_token_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        index=True,
        comment="Token ID in blockchain contract"
    )
    nft_tx_hash: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Transaction hash of NFT minting"
    )
    nft_ipfs_cid: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="IPFS CID used for NFT metadata"
    )
    nft_content_hash: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Content hash used for NFT verification"
    )
    nft_created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        comment="When NFT was minted"
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
        comment="Material creation timestamp"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        comment="Last update timestamp"
    )
    
    # Relationships
    author: Mapped["User"] = relationship(
        "User",
        back_populates="materials"
    )
    
    __table_args__ = (
        Index("idx_materials_author", "author_id"),
        Index("idx_materials_thread", "thread_id"),
        Index("idx_materials_session", "session_id"),
        Index("idx_materials_subject_grade", "subject", "grade"),
        Index("idx_materials_hash", "content_hash"),
        Index("idx_materials_cid", "ipfs_cid"),
        Index("idx_materials_status", "status"),
        Index("idx_materials_created", "created_at"),
        Index("idx_materials_nft_minted", "nft_minted"),
        Index("idx_materials_nft_token_id", "nft_token_id"),
        {"comment": "Learning materials with blockchain-ready metadata and NFT tracking"}
    )
    
    def __repr__(self) -> str:
        return f"<Material(id={self.id}, subject={self.subject}, grade={self.grade}, topic={self.topic[:30] if self.topic else None}...)>"

