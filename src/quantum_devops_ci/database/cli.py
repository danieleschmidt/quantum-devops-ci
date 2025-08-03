#!/usr/bin/env python3
"""
Database CLI utility for quantum DevOps CI/CD.

This utility provides command-line access to database operations
including migrations, data management, and maintenance.
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from .connection import get_connection, DatabaseConfig
from .migrations import MigrationManager, run_migrations
from .models import create_all_tables
from .repositories import (
    BuildRepository, HardwareUsageRepository, TestResultRepository,
    CostRepository, JobRepository
)


def cmd_init(args):
    """Initialize database with schema."""
    print("🗄️ Initializing quantum DevOps database...")
    
    try:
        # Run migrations
        success = run_migrations()
        
        if success:
            print("✅ Database initialized successfully!")
            
            # Show database info
            conn = get_connection()
            print(f"📊 Database type: {conn.config.database_type}")
            if conn.config.database_type == 'sqlite':
                print(f"📁 Database file: {conn.config.database_name}")
        else:
            print("❌ Database initialization failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)


def cmd_migrate(args):
    """Run database migrations."""
    print("🔄 Running database migrations...")
    
    try:
        manager = MigrationManager()
        status = manager.get_migration_status()
        
        print(f"📊 Current status:")
        print(f"  Applied migrations: {status['applied_count']}")
        print(f"  Latest version: {status['latest_version']}")
        
        # Run migrations
        success = run_migrations()
        
        if success:
            new_status = manager.get_migration_status()
            print(f"✅ Migrations completed!")
            print(f"  Applied migrations: {new_status['applied_count']}")
            print(f"  Latest version: {new_status['latest_version']}")
        else:
            print("❌ Some migrations failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Migration error: {e}")
        sys.exit(1)


def cmd_status(args):
    """Show database status."""
    print("📊 Database Status")
    print("=" * 40)
    
    try:
        conn = get_connection()
        
        # Connection info
        print(f"Database type: {conn.config.database_type}")
        if conn.config.database_type == 'sqlite':
            db_path = Path(conn.config.database_name)
            if db_path.exists():
                size_mb = db_path.stat().st_size / (1024 * 1024)
                print(f"Database file: {conn.config.database_name}")
                print(f"Database size: {size_mb:.2f} MB")
            else:
                print("Database file: Not found")
        
        # Migration status
        manager = MigrationManager()
        migration_status = manager.get_migration_status()
        print(f"\nMigration status:")
        print(f"  Applied migrations: {migration_status['applied_count']}")
        print(f"  Latest version: {migration_status['latest_version']}")
        
        # Table status
        print(f"\nTable status:")
        tables = ['build_records', 'hardware_usage_records', 'test_results', 
                 'cost_records', 'job_records']
        
        for table in tables:
            exists = conn.table_exists(table)
            status = "✅ Exists" if exists else "❌ Missing"
            print(f"  {table}: {status}")
            
            if exists:
                # Get row count
                try:
                    count_result = conn.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                    count = count_result[0]['count'] if count_result else 0
                    print(f"    Rows: {count:,}")
                except:
                    print(f"    Rows: Unable to count")
        
    except Exception as e:
        print(f"❌ Error getting database status: {e}")
        sys.exit(1)


def cmd_stats(args):
    """Show database statistics."""
    print("📈 Database Statistics")
    print("=" * 40)
    
    try:
        # Repository instances
        build_repo = BuildRepository()
        hardware_repo = HardwareUsageRepository()
        test_repo = TestResultRepository()
        cost_repo = CostRepository()
        job_repo = JobRepository()
        
        # Build statistics
        recent_builds = build_repo.get_recent_builds(30)
        total_builds = len(recent_builds)
        success_rate = build_repo.get_success_rate(30)
        
        print(f"Builds (last 30 days):")
        print(f"  Total builds: {total_builds}")
        print(f"  Success rate: {success_rate:.1%}")
        
        # Hardware usage statistics
        cost_summary = hardware_repo.get_cost_summary(30)
        print(f"\nHardware Usage (last 30 days):")
        print(f"  Total cost: ${cost_summary['total_cost']:.2f}")
        print(f"  Total jobs: {cost_summary['total_jobs']}")
        print(f"  Daily average: ${cost_summary['daily_average']:.2f}")
        
        # Test statistics
        test_failure_rate = test_repo.get_failure_rate(30)
        print(f"\nTests (last 30 days):")
        print(f"  Failure rate: {test_failure_rate:.1%}")
        
        # Job performance
        job_metrics = job_repo.get_performance_metrics(30)
        print(f"\nJob Performance (last 30 days):")
        print(f"  Total jobs: {job_metrics['total_jobs']}")
        print(f"  Success rate: {job_metrics['success_rate']:.1%}")
        print(f"  Avg queue time: {job_metrics['avg_queue_time_minutes']:.1f} minutes")
        print(f"  Avg execution time: {job_metrics['avg_execution_time_minutes']:.1f} minutes")
        print(f"  Avg cost: ${job_metrics['avg_cost']:.2f}")
        
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
        sys.exit(1)


def cmd_cleanup(args):
    """Clean up old database records."""
    print(f"🧹 Cleaning up records older than {args.days} days...")
    
    try:
        conn = get_connection()
        cutoff_date = datetime.now() - timedelta(days=args.days)
        
        tables_cleaned = 0
        total_deleted = 0
        
        # Clean up old records from each table
        tables = [
            'build_records',
            'hardware_usage_records', 
            'test_results',
            'cost_records',
            'job_records'
        ]
        
        for table in tables:
            if conn.table_exists(table):
                # Count records to delete
                count_sql = f"SELECT COUNT(*) as count FROM {table} WHERE created_at < ?"
                count_result = conn.execute_query(count_sql, (cutoff_date.isoformat(),))
                count_to_delete = count_result[0]['count'] if count_result else 0
                
                if count_to_delete > 0:
                    if args.dry_run:
                        print(f"  {table}: Would delete {count_to_delete} records")
                    else:
                        # Delete old records
                        delete_sql = f"DELETE FROM {table} WHERE created_at < ?"
                        deleted = conn.execute_command(delete_sql, (cutoff_date.isoformat(),))
                        print(f"  {table}: Deleted {deleted} records")
                        total_deleted += deleted
                        tables_cleaned += 1
                else:
                    print(f"  {table}: No old records to delete")
        
        if args.dry_run:
            print(f"🔍 Dry run completed - no records actually deleted")
        else:
            print(f"✅ Cleanup completed: {total_deleted} records deleted from {tables_cleaned} tables")
            
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        sys.exit(1)


def cmd_backup(args):
    """Backup database."""
    print(f"💾 Backing up database to {args.output}...")
    
    try:
        conn = get_connection()
        
        if conn.config.database_type == 'sqlite':
            # For SQLite, copy the database file
            import shutil
            source = Path(conn.config.database_name)
            destination = Path(args.output)
            
            if source.exists():
                shutil.copy2(source, destination)
                print(f"✅ Database backed up to {destination}")
                
                # Show backup size
                size_mb = destination.stat().st_size / (1024 * 1024)
                print(f"📊 Backup size: {size_mb:.2f} MB")
            else:
                print(f"❌ Source database not found: {source}")
                sys.exit(1)
        else:
            print(f"❌ Backup not implemented for {conn.config.database_type}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Backup error: {e}")
        sys.exit(1)


def cmd_export(args):
    """Export data to JSON."""
    print(f"📤 Exporting data to {args.output}...")
    
    try:
        import json
        
        # Get repositories
        build_repo = BuildRepository()
        hardware_repo = HardwareUsageRepository()
        test_repo = TestResultRepository()
        cost_repo = CostRepository()
        job_repo = JobRepository()
        
        # Export data
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'build_records': [record.to_dict() for record in build_repo.get_all(limit=args.limit)],
            'hardware_usage_records': [record.to_dict() for record in hardware_repo.get_all(limit=args.limit)],
            'test_results': [record.to_dict() for record in test_repo.get_all(limit=args.limit)],
            'cost_records': [record.to_dict() for record in cost_repo.get_all(limit=args.limit)],
            'job_records': [record.to_dict() for record in job_repo.get_all(limit=args.limit)]
        }
        
        # Write to file
        with open(args.output, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        # Show export stats
        total_records = sum(len(records) for records in export_data.values() if isinstance(records, list))
        print(f"✅ Exported {total_records} records to {args.output}")
        
    except Exception as e:
        print(f"❌ Export error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='quantum-db',
        description='Database utility for quantum DevOps CI/CD'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize database')
    init_parser.set_defaults(func=cmd_init)
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Run migrations')
    migrate_parser.set_defaults(func=cmd_migrate)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show database status')
    status_parser.set_defaults(func=cmd_status)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    stats_parser.set_defaults(func=cmd_stats)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old records')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Days to keep (default: 30)')
    cleanup_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
    cleanup_parser.set_defaults(func=cmd_cleanup)
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup database')
    backup_parser.add_argument('output', help='Backup file path')
    backup_parser.set_defaults(func=cmd_backup)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export data to JSON')
    export_parser.add_argument('output', help='Output JSON file')
    export_parser.add_argument('--limit', type=int, help='Limit records per table')
    export_parser.set_defaults(func=cmd_export)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()