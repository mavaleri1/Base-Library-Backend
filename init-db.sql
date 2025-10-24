-- Note: core database is automatically created via POSTGRES_DB env variable

-- Create prompt_config database if it doesn't exist
SELECT 'CREATE DATABASE prompt_config'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'prompt_config')\gexec