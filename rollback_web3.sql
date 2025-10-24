-- ============================================================================
-- Web3 Migration ROLLBACK SQL Scripts  
-- ============================================================================
-- This file contains SQL scripts to rollback Web3 migration
-- and restore Telegram-based authentication
--
-- WARNING: This will DELETE all Web3 user data!
--
-- Usage:
--   psql -h localhost -p 5431 -U postgres -d prompt_config -f rollback_web3.sql
--   psql -h localhost -p 5431 -U postgres -d core -f rollback_web3.sql
-- ============================================================================

\echo '============================================================================'
\echo 'Web3 Migration ROLLBACK Script'
\echo '============================================================================'
\echo ''
\echo 'WARNING: This will delete all Web3 user data!'
\echo 'Press Ctrl+C to cancel or Enter to continue...'
\prompt 'Continue? (yes/no): ' confirm

\echo ''
\echo 'Detecting database...'

-- Check which database we are connected to
DO $$
DECLARE
    db_name text;
BEGIN
    SELECT current_database() INTO db_name;
    RAISE NOTICE 'Connected to database: %', db_name;
END $$;

\echo ''
\echo '============================================================================'
\echo 'PROMPT_CONFIG DATABASE ROLLBACK'
\echo '============================================================================'

-- Only run if connected to prompt_config
DO $$
DECLARE
    db_name text;
BEGIN
    SELECT current_database() INTO db_name;
    
    IF db_name = 'prompt_config' THEN
        
        RAISE NOTICE 'Step 1: Dropping Web3 user_placeholder_settings table...';
        DROP TABLE IF EXISTS user_placeholder_settings CASCADE;
        
        RAISE NOTICE 'Step 2: Dropping Web3 user_profiles table...';
        DROP INDEX IF EXISTS ix_user_profiles_wallet;
        DROP TABLE IF EXISTS user_profiles CASCADE;
        
        RAISE NOTICE 'Step 3: Recreating Telegram user_profiles table...';
        CREATE TABLE user_profiles (
            user_id BIGINT PRIMARY KEY,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        
        RAISE NOTICE 'Step 4: Recreating Telegram user_placeholder_settings table...';
        CREATE TABLE user_placeholder_settings (
            user_id BIGINT NOT NULL REFERENCES user_profiles(user_id),
            placeholder_id UUID NOT NULL REFERENCES placeholders(id),
            placeholder_value_id UUID NOT NULL REFERENCES placeholder_values(id),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, placeholder_id)
        );
        
        RAISE NOTICE '✓ prompt_config rollback completed!';
        RAISE NOTICE 'Telegram authentication restored.';
    END IF;
END $$;

\echo ''
\echo '============================================================================'
\echo 'core DATABASE ROLLBACK'
\echo '============================================================================'

-- Only run if connected to core
DO $$
DECLARE
    db_name text;
BEGIN
    SELECT current_database() INTO db_name;
    
    IF db_name = 'core' THEN
        
        RAISE NOTICE 'Step 1: Dropping Web3 tables...';
        DROP INDEX IF EXISTS idx_nonces_expires;
        DROP TABLE IF EXISTS web3_nonces CASCADE;
        
        DROP INDEX IF EXISTS idx_sessions_status;
        DROP INDEX IF EXISTS idx_sessions_thread;
        DROP INDEX IF EXISTS idx_sessions_user;
        DROP TABLE IF EXISTS user_sessions CASCADE;
        
        DROP INDEX IF EXISTS idx_users_wallet;
        DROP TABLE IF EXISTS users CASCADE;
        
        RAISE NOTICE 'Step 2: Recreating Telegram auth_codes table...';
        CREATE TABLE auth_codes (
            username VARCHAR(255) NOT NULL,
            code VARCHAR(10) NOT NULL,
            user_id BIGINT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            PRIMARY KEY (username, code)
        );
        
        CREATE INDEX idx_auth_codes_created_at ON auth_codes(created_at);
        
        RAISE NOTICE '✓ core rollback completed!';
        RAISE NOTICE 'Telegram authentication restored.';
    END IF;
END $$;

\echo ''
\echo '============================================================================'
\echo 'Rollback Complete!'
\echo '============================================================================'
\echo ''
\echo 'Telegram-based authentication has been restored.'
\echo 'All Web3 user data has been deleted.'
\echo '============================================================================'

