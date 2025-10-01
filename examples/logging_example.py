#!/usr/bin/env python3
"""
Example demonstrating the comprehensive logging and error handling systems.

This example shows how to use the new logging and error handling infrastructure
in the Reddit deleted comment dataset processing pipeline.
"""

import sys
import time
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from logger import initialize_logging, get_logger, ComponentType
from error_handler import initialize_error_handling, get_error_manager


def main():
    """Demonstrate logging and error handling systems."""
    
    # Configuration for logging
    logging_config = {
        'level': 'INFO',
        'file_path': './logs/example.log',
        'error_log_path': './logs/example_errors.log',
        'performance_log_path': './logs/example_performance.log',
        'max_size_mb': 10,
        'backup_count': 3,
        'console_output': True,
        'log_memory_usage': True
    }
    
    # Configuration for error handling
    error_config = {
        'retry_attempts': 3,
        'retry_delay': 1.0,
        'max_retry_delay': 10.0,
        'exponential_backoff': True,
        'continue_on_error': True,
        'checkpoint_enabled': True,
        'checkpoint_interval': 1000,
        'checkpoint_path': './data/example_checkpoint.json'
    }
    
    # Initialize systems
    print("Initializing logging and error handling systems...")
    logger_system = initialize_logging(logging_config)
    error_manager = initialize_error_handling(error_config)
    
    # Example 1: Basic logging
    print("\n=== Example 1: Basic Logging ===")
    logger_system.log_info(
        ComponentType.MAIN_PIPELINE,
        "Starting example demonstration",
        example_param="demo_value"
    )
    
    logger_system.log_warning(
        ComponentType.DATA_DOWNLOADER,
        "This is a warning message",
        warning_code="W001"
    )
    
    # Example 2: Performance tracking
    print("\n=== Example 2: Performance Tracking ===")
    operation_context = logger_system.start_operation(
        ComponentType.REDDIT_PARSER,
        "parse_example_data"
    )
    
    # Simulate some work
    print("Simulating data parsing...")
    time.sleep(2)
    
    # End operation with statistics
    stats = logger_system.end_operation(
        operation_context,
        records_processed=1000,
        success=True
    )
    
    print(f"Operation completed: {stats.records_processed} records in {stats.duration_seconds:.2f}s")
    print(f"Processing rate: {stats.records_per_second:.2f} records/second")
    
    # Example 3: Error handling with recovery
    print("\n=== Example 3: Error Handling ===")
    
    with error_manager.error_context(
        ComponentType.COMMENT_CLASSIFIER,
        "classify_comments"
    ):
        try:
            # Simulate an operation that might fail
            print("Attempting operation that might fail...")
            
            # Simulate a recoverable error (this would normally be real processing)
            # For demo purposes, we'll just log what would happen
            logger_system.log_info(
                ComponentType.COMMENT_CLASSIFIER,
                "Processing comments batch",
                batch_size=100
            )
            
            # Simulate successful completion
            logger_system.log_info(
                ComponentType.COMMENT_CLASSIFIER,
                "Comment classification completed successfully"
            )
            
        except Exception as e:
            # This would be handled automatically by the error context manager
            logger_system.log_error(
                ComponentType.COMMENT_CLASSIFIER,
                f"Error during classification: {e}",
                error=e,
                recovery_action="retry"
            )
    
    # Example 4: Memory usage logging
    print("\n=== Example 4: System Resource Monitoring ===")
    logger_system.log_memory_usage(
        ComponentType.PARQUET_WRITER,
        "Before writing large dataset"
    )
    
    logger_system.log_system_resources(ComponentType.PARQUET_WRITER)
    
    # Example 5: Checkpoint system
    print("\n=== Example 5: Checkpoint System ===")
    checkpoint_manager = error_manager.checkpoint_manager
    
    # Create example checkpoint data
    from error_handler import CheckpointData
    checkpoint_data = CheckpointData(
        checkpoint_id="example_checkpoint_1",
        timestamp="2024-01-01T12:00:00Z",
        component="example_component",
        operation="example_operation",
        progress={"stage": "processing", "file_index": 3},
        processed_files=["file1.json", "file2.json"],
        current_file="file3.json",
        current_position=5000,
        total_records=20000,
        processed_records=8000,
        error_count=1,
        metadata={"batch_size": 1000}
    )
    
    # Save checkpoint
    success = checkpoint_manager.save_checkpoint(checkpoint_data)
    if success:
        print("Checkpoint saved successfully")
        
        # Load checkpoint
        loaded_checkpoint = checkpoint_manager.load_checkpoint()
        if loaded_checkpoint:
            print(f"Checkpoint loaded: {loaded_checkpoint.processed_records}/{loaded_checkpoint.total_records} records processed")
    
    # Example 6: Get summaries
    print("\n=== Example 6: System Summaries ===")
    
    # Error summary
    error_summary = logger_system.get_error_summary()
    print(f"Error Summary: {error_summary['total_errors']} total errors")
    
    # Performance summary
    performance_summary = logger_system.get_performance_summary()
    print(f"Performance Summary: {performance_summary.get('total_records_processed', 0)} total records processed")
    
    # Error statistics
    error_stats = error_manager.get_error_statistics()
    print(f"Error Statistics: {error_stats['total_errors']} errors, {error_stats['recovered_errors']} recovered")
    
    # Final logging
    logger_system.log_info(
        ComponentType.MAIN_PIPELINE,
        "Example demonstration completed successfully"
    )
    
    print("\n=== Example Complete ===")
    print("Check the following log files for detailed output:")
    print(f"- Main log: {logging_config['file_path']}")
    print(f"- Error log: {logging_config['error_log_path']}")
    print(f"- Performance log: {logging_config['performance_log_path']}")
    
    # Shutdown logging system
    logger_system.shutdown()


if __name__ == "__main__":
    main()