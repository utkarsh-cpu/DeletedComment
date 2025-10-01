# Logging and Error Handling Infrastructure

This document describes the comprehensive logging and error handling systems implemented for the Reddit deleted comment dataset processing pipeline.

## Overview

The system provides:
- **Structured JSON logging** with multiple log levels and handlers
- **Performance tracking** with detailed metrics
- **Error recovery** with multiple strategies
- **Checkpoint system** for resuming interrupted processing
- **System resource monitoring**

## Components

### 1. Logging System (`src/logger.py`)

#### Features
- Structured JSON logging with timestamps and component context
- Multiple log handlers (file, console, error-specific)
- Performance statistics tracking
- Memory usage monitoring
- Automatic log rotation

#### Usage

```python
from logger import initialize_logging, get_logger, ComponentType

# Initialize logging system
config = {
    'level': 'INFO',
    'file_path': './logs/processing.log',
    'error_log_path': './logs/errors.log',
    'console_output': True
}
logger_system = initialize_logging(config)

# Log messages with context
logger_system.log_info(
    ComponentType.DATA_DOWNLOADER,
    "Starting download",
    file_size_mb=150.5
)

# Track operation performance
context = logger_system.start_operation(
    ComponentType.REDDIT_PARSER,
    "parse_comments"
)
# ... do work ...
stats = logger_system.end_operation(
    context,
    records_processed=10000,
    success=True
)
```

### 2. Error Handling System (`src/error_handler.py`)

#### Features
- Automatic error categorization and severity assessment
- Multiple recovery strategies (retry, skip, fallback, abort)
- Exponential backoff for retries
- Checkpoint system for recovery
- Detailed error context tracking

#### Usage

```python
from error_handler import initialize_error_handling, get_error_manager

# Initialize error handling
config = {
    'retry_attempts': 3,
    'retry_delay': 1.0,
    'exponential_backoff': True,
    'checkpoint_enabled': True
}
error_manager = initialize_error_handling(config)

# Use error context manager for automatic handling
with error_manager.error_context(
    ComponentType.DATA_DOWNLOADER,
    "download_file",
    file_path="data.json"
):
    # Your code here - errors are automatically handled
    download_file("data.json")
```

### 3. Checkpoint System

#### Features
- Automatic checkpoint creation at configurable intervals
- Resume processing from last checkpoint
- Metadata tracking for recovery context

#### Usage

```python
# Checkpoints are automatically managed by the error handler
# Manual checkpoint operations:
checkpoint_manager = error_manager.checkpoint_manager

# Save checkpoint
checkpoint_data = CheckpointData(
    checkpoint_id="processing_batch_1",
    component="reddit_parser",
    processed_records=5000,
    total_records=20000,
    # ... other fields
)
checkpoint_manager.save_checkpoint(checkpoint_data)

# Load checkpoint
checkpoint = checkpoint_manager.load_checkpoint()
if checkpoint:
    # Resume from checkpoint
    start_position = checkpoint.current_position
```

## Configuration

### Logging Configuration

```yaml
logging:
  level: "INFO"                         # Log level
  file_path: "./logs/processing.log"    # Main log file
  error_log_path: "./logs/errors.log"   # Error-specific log
  max_size_mb: 100                      # Max log file size
  backup_count: 5                       # Number of backup files
  console_output: true                  # Enable console logging
  log_memory_usage: true                # Log memory statistics
```

### Error Handling Configuration

```yaml
error_handling:
  retry_attempts: 3                     # Max retry attempts
  retry_delay: 1.0                      # Initial retry delay (seconds)
  exponential_backoff: true             # Use exponential backoff
  max_retry_delay: 60.0                 # Maximum retry delay
  continue_on_error: true               # Continue processing on errors
  checkpoint_enabled: true              # Enable checkpoint system
  checkpoint_interval: 10000            # Save checkpoint every N records
  checkpoint_path: "./data/checkpoint.json"  # Checkpoint file path
```

## Error Recovery Strategies

The system automatically determines the appropriate recovery strategy based on error type:

| Error Type | Strategy | Description |
|------------|----------|-------------|
| Network errors | Retry | Exponential backoff with jitter |
| I/O errors | Retry | Short delays, limited attempts |
| Parsing errors | Skip | Skip malformed data, continue processing |
| Memory errors | Checkpoint restore | Free memory, restore from checkpoint |
| Critical errors | Abort | Stop processing immediately |

## Log Output Formats

### Structured JSON Logs
```json
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "level": "INFO",
  "component": "reddit_parser",
  "message": "Processing comments batch",
  "module": "reddit_parser",
  "function": "parse_comments_file",
  "line": 145,
  "batch_size": 1000,
  "file_path": "comments_2023.json"
}
```

### Performance Logs
```json
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "level": "INFO",
  "component": "performance",
  "message": "Performance: reddit_parser.parse_comments - 10000 records in 5.23s (1912.05 rec/s)",
  "operation": "parse_comments",
  "duration_seconds": 5.23,
  "records_processed": 10000,
  "records_per_second": 1912.05,
  "memory_usage_mb": 245.7
}
```

### Error Logs
```json
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "level": "ERROR",
  "component": "data_downloader",
  "message": "Download failed: Connection timeout",
  "error_type": "ConnectionError",
  "recovery_action": "retry",
  "retry_count": 1,
  "file_path": "reddit_data.torrent",
  "exception": {
    "type": "ConnectionError",
    "message": "Connection timeout after 30 seconds",
    "traceback": ["...", "..."]
  }
}
```

## Integration with Main Pipeline

The main pipeline (`main.py`) has been updated to use both systems:

```python
# Initialize systems
setup_logging_and_error_handling(config)
logger_system = get_logger()
error_manager = get_error_manager()

# Use in pipeline operations
with error_manager.error_context(ComponentType.MAIN_PIPELINE, "complete_pipeline"):
    # Pipeline operations with automatic error handling
    success = run_processing_stages()
```

## Monitoring and Statistics

### Error Statistics
```python
# Get error summary
error_summary = logger_system.get_error_summary()
print(f"Total errors: {error_summary['total_errors']}")
print(f"Error types: {error_summary['error_types']}")
print(f"Recovery actions: {error_summary['recovery_actions']}")

# Get error recovery statistics
error_stats = error_manager.get_error_statistics()
print(f"Recovery rate: {error_stats['recovery_rate']:.2%}")
```

### Performance Statistics
```python
# Get performance summary
perf_summary = logger_system.get_performance_summary()
print(f"Total records: {perf_summary['total_records_processed']}")
print(f"Processing time: {perf_summary['total_processing_time']:.2f}s")
print(f"Overall rate: {perf_summary['overall_rate']:.2f} rec/s")
```

## Best Practices

1. **Use component-specific logging**: Always specify the correct ComponentType
2. **Include context**: Add relevant parameters to log messages
3. **Monitor memory usage**: Use `log_memory_usage()` for memory-intensive operations
4. **Handle errors gracefully**: Use error context managers for automatic recovery
5. **Save checkpoints regularly**: Configure appropriate checkpoint intervals
6. **Review logs regularly**: Monitor error patterns and performance trends

## Example Usage

See `examples/logging_example.py` for a complete demonstration of all features.

## Requirements Satisfied

This implementation satisfies the following requirements:

- **5.3**: Detailed error information logging and recovery action tracking
- **5.4**: Comprehensive error handling with graceful degradation and recovery suggestions
- **Structured logging**: JSON format with component context and performance metrics
- **Checkpoint system**: Resume interrupted processing from saved state
- **Performance tracking**: Detailed statistics and resource monitoring