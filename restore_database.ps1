# PowerShell script for PostgreSQL database restore from backup
# Usage: .\restore_database.ps1 -Database learnflow -BackupFile "backup_learnflow_20241201_120000.dump"

param(
    [Parameter(Mandatory=$true)]
    [string]$Database,
    
    [Parameter(Mandatory=$true)]
    [string]$BackupFile,
    
    [switch]$DropExisting = $false
)

# Database connection settings
$DB_HOST = "localhost"
$DB_PORT = "5431"
$DB_USER = "postgres"
$DB_PASSWORD = "postgres"

# Check if backup file exists
if (!(Test-Path $BackupFile)) {
    Write-Host "ERROR: Backup file not found: $BackupFile" -ForegroundColor Red
    exit 1
}

Write-Host "Restoring database: $Database" -ForegroundColor Green
Write-Host "Backup file: $BackupFile" -ForegroundColor Cyan

# Set password environment variable
$env:PGPASSWORD = $DB_PASSWORD

try {
    # If DropExisting flag is set, drop existing database
    if ($DropExisting) {
        Write-Host "WARNING: Dropping existing database..." -ForegroundColor Yellow
        
        # Terminate all active connections to database
        & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$Database' AND pid <> pg_backend_pid();"
        
        # Drop database
        & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "DROP DATABASE IF EXISTS $Database;"
        
        # Create new database
        & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "CREATE DATABASE $Database;"
        
        Write-Host "SUCCESS: Database recreated" -ForegroundColor Green
    }
    
    # Restore from backup
    Write-Host "Restoring data from backup..." -ForegroundColor Yellow
    
    & pg_restore -h $DB_HOST -p $DB_PORT -U $DB_USER -d $Database -v $BackupFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: Database restored successfully!" -ForegroundColor Green
    } else {
        throw "Error restoring database"
    }
}
catch {
    Write-Host "ERROR: Failed to restore database" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
finally {
    # Clean up environment variable
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
}

Write-Host "Restore completed successfully!" -ForegroundColor Green
