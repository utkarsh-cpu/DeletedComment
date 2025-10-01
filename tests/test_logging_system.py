"""
Tests for the comprehensive logging and error handling systems.
"""

import pytest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from logger import (
    RedditDatasetLogger, ComponentType, ProcessingStats, 
    initialize_logging, get_logger
)
from error_handler import (
    ErrorRecoveryManager, CheckpointManager, ErrorContext, 
    ErrorSeverity, RecoveryAction, initialize_error_handling, get_error_manager
)


class TestLoggingSystem:
    """Test cases for the logging system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'level': 'DEBUG',
            'file_path': f'{self.temp_dir}/test.log',
            'error_log_path': f'{self.temp_dir}/errors.log',
            'performance_log_path': f'{self.temp_dir}/performance.log',
            'max_size_mb': 10,
            'backup_count': 3,
            'console_output': False,
            'log_memory_usage': True
        }
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger_system = RedditDatasetLogger(self.config)
        
        assert logger_system is not None
        assert 'main' in logger_system.loggers
        assert 'error' in logger_system.loggers
        assert 'performance' in logger_system.loggers
    
    def test_component_logging(self):
        """Test logging for different components."""
        logger_system = RedditDatasetLogger(self.config)
        
        # Test info logging
        logger_system.log_info(
            ComponentType.DATA_DOWNLOADER,
            "Test info message",
            test_param="test_value"
        )
        
        # Test warning logging
        logger_system.log_warning(
            ComponentType.REDDIT_PARSER,
            "Test warning message"
        )
        
        # Test error logging
        test_error = ValueError("Test error")
        logger_system.log_error(
            ComponentType.COMMENT_CLASSIFIER,
            "Test error message",
            error=test_error,
            recovery_action="retry"
        )
        
        # Verify log files were created
        assert Path(self.config['file_path']).exists()
        assert Path(self.config['error_log_path']).exists()
    
    def test_performance_logging(self):
        """Test performance statistics logging."""
        logger_system = RedditDatasetLogger(self.config)
        
        # Create test performance stats
        stats = ProcessingStats(
            component="test_component",
            operation="test_operation",
            start_time=time.time() - 10,
            end_time=time.time(),
            records_processed=1000,
            memory_usage_mb=100.5,
            cpu_usage_percent=25.0,
            success=True
        )
        
        logger_system.log_performance(stats)
        
        # Verify performance stats were stored
        assert len(logger_system.performance_stats) == 1
        assert logger_system.performance_stats[0].records_processed == 1000
    
    def test_operation_tracking(self):
        """Test operation start/end tracking."""
        logger_system = RedditDatasetLogger(self.config)
        
        # Start operation
        context = logger_system.start_operation(
            ComponentType.PARQUET_WRITER,
            "write_dataset"
        )
        
        assert 'component' in context
        assert 'operation' in context
        assert 'start_time' in context
        
        # End operation
        stats = logger_system.end_operation(
            context,
            records_processed=500,
            success=True
        )
        
        assert stats.component == ComponentType.PARQUET_WRITER.value
        assert stats.operation == "write_dataset"
        assert stats.records_processed == 500
        assert stats.success is True
    
    def test_error_summary(self):
        """Test error summary generation."""
        logger_system = RedditDatasetLogger(self.config)
        
        # Log some errors
        logger_system.log_error(
            ComponentType.DATA_DOWNLOADER,
            "Network error",
            error=ConnectionError("Connection failed"),
            recovery_action="retry"
        )
        
        logger_system.log_error(
            ComponentType.REDDIT_PARSER,
            "Parse error",
            error=ValueError("Invalid JSON"),
            recovery_action="skip"
        )
        
        # Get error summary
        summary = logger_system.get_error_summary()
        
        assert summary['total_errors'] == 2
        assert 'ConnectionError' in summary['error_types']
        assert 'ValueError' in summary['error_types']
        assert 'retry' in summary['recovery_actions']
        assert 'skip' in summary['recovery_actions']
    
    def test_global_logger_initialization(self):
        """Test global logger initialization and access."""
        # Initialize global logger
        logger_system = initialize_logging(self.config)
        
        # Get global logger instance
        global_logger = get_logger()
        
        assert logger_system is global_logger
        
        # Test logging through global instance
        global_logger.log_info(
            ComponentType.MAIN_PIPELINE,
            "Global logger test"
        )


class TestErrorHandlingSystem:
    """Test cases for the error handling system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'retry_attempts': 3,
            'retry_delay': 0.1,
            'max_retry_delay': 1.0,
            'exponential_backoff': True,
            'continue_on_error': True,
            'checkpoint_enabled': True,
            'checkpoint_interval': 100,
            'checkpoint_path': f'{self.temp_dir}/checkpoint.json'
        }
        
        # Initialize logging for error handler
        logging_config = {
            'level': 'DEBUG',
            'file_path': f'{self.temp_dir}/test.log',
            'console_output': False
        }
        initialize_logging(logging_config)
    
    def test_checkpoint_manager(self):
        """Test checkpoint creation and restoration."""
        checkpoint_manager = CheckpointManager(
            self.config['checkpoint_path'],
            interval=100
        )
        
        # Create test checkpoint data
        from error_handler import CheckpointData
        checkpoint_data = CheckpointData(
            checkpoint_id="test_checkpoint_1",
            timestamp="2024-01-01T12:00:00Z",
            component="test_component",
            operation="test_operation",
            progress={"stage": "parsing", "file_index": 5},
            processed_files=["file1.json", "file2.json"],
            current_file="file3.json",
            current_position=1000,
            total_records=10000,
            processed_records=3000,
            error_count=2,
            metadata={"batch_size": 1000}
        )
        
        # Save checkpoint
        success = checkpoint_manager.save_checkpoint(checkpoint_data)
        assert success is True
        
        # Load checkpoint
        loaded_checkpoint = checkpoint_manager.load_checkpoint()
        assert loaded_checkpoint is not None
        assert loaded_checkpoint.checkpoint_id == "test_checkpoint_1"
        assert loaded_checkpoint.processed_records == 3000
        assert loaded_checkpoint.total_records == 10000
    
    def test_error_recovery_manager(self):
        """Test error recovery manager initialization."""
        error_manager = ErrorRecoveryManager(self.config)
        
        assert error_manager is not None
        assert error_manager.config == self.config
        assert 'network' in error_manager.retry_strategies
        assert 'io' in error_manager.retry_strategies
        assert 'parsing' in error_manager.retry_strategies
    
    def test_error_categorization(self):
        """Test error categorization logic."""
        error_manager = ErrorRecoveryManager(self.config)
        
        # Test network error
        network_error = ConnectionError("Connection timeout")
        category = error_manager._categorize_error(network_error)
        assert category == 'network'
        
        # Test I/O error
        io_error = FileNotFoundError("File not found")
        category = error_manager._categorize_error(io_error)
        assert category == 'io'
        
        # Test parsing error
        parse_error = json.JSONDecodeError("Invalid JSON", "", 0)
        category = error_manager._categorize_error(parse_error)
        assert category == 'parsing'
        
        # Test memory error
        memory_error = MemoryError("Out of memory")
        category = error_manager._categorize_error(memory_error)
        assert category == 'memory'
    
    def test_error_severity_assessment(self):
        """Test error severity assessment."""
        error_manager = ErrorRecoveryManager(self.config)
        
        # Create test error context
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            file_path=None,
            record_index=None,
            batch_id=None,
            timestamp="2024-01-01T12:00:00Z",
            error_type="ValueError",
            error_message="Test error",
            stack_trace="",
            severity=ErrorSeverity.MEDIUM,
            retry_count=0,
            max_retries=3,
            recovery_suggestions=[]
        )
        
        # Test different error severities
        critical_error = KeyboardInterrupt("User interrupt")
        severity = error_manager._assess_severity(critical_error, context)
        assert severity == ErrorSeverity.CRITICAL
        
        network_error = ConnectionError("Network timeout")
        severity = error_manager._assess_severity(network_error, context)
        assert severity == ErrorSeverity.MEDIUM
        
        parse_error = ValueError("Invalid value")
        severity = error_manager._assess_severity(parse_error, context)
        assert severity == ErrorSeverity.LOW
    
    def test_recovery_action_determination(self):
        """Test recovery action determination logic."""
        error_manager = ErrorRecoveryManager(self.config)
        
        # Create test error context
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            file_path=None,
            record_index=None,
            batch_id=None,
            timestamp="2024-01-01T12:00:00Z",
            error_type="ConnectionError",
            error_message="Network timeout",
            stack_trace="",
            severity=ErrorSeverity.MEDIUM,
            retry_count=0,
            max_retries=3,
            recovery_suggestions=[]
        )
        
        # Test retry for network error
        network_error = ConnectionError("Network timeout")
        action = error_manager._determine_recovery_action(network_error, context, 'network')
        assert action == RecoveryAction.RETRY
        
        # Test skip for parsing error
        context.error_type = "JSONDecodeError"
        parse_error = json.JSONDecodeError("Invalid JSON", "", 0)
        action = error_manager._determine_recovery_action(parse_error, context, 'parsing')
        assert action == RecoveryAction.SKIP
        
        # Test abort for critical error
        context.severity = ErrorSeverity.CRITICAL
        critical_error = KeyboardInterrupt("User interrupt")
        action = error_manager._determine_recovery_action(critical_error, context, 'general')
        assert action == RecoveryAction.ABORT
    
    def test_error_context_manager(self):
        """Test error context manager functionality."""
        error_manager = ErrorRecoveryManager(self.config)
        
        # Test successful operation
        with error_manager.error_context(
            ComponentType.DATA_DOWNLOADER,
            "test_operation"
        ):
            # This should complete without error
            pass
        
        # Test operation with recoverable error
        retry_count = 0
        with error_manager.error_context(
            ComponentType.DATA_DOWNLOADER,
            "test_operation_with_retry"
        ):
            if retry_count < 2:
                retry_count += 1
                # This would normally raise an error that gets retried
                pass
    
    def test_global_error_manager_initialization(self):
        """Test global error manager initialization and access."""
        # Initialize global error manager
        error_manager = initialize_error_handling(self.config)
        
        # Get global error manager instance
        global_error_manager = get_error_manager()
        
        assert error_manager is global_error_manager
    
    def test_error_statistics(self):
        """Test error statistics tracking."""
        error_manager = ErrorRecoveryManager(self.config)
        
        # Create test error context
        context = ErrorContext(
            component=ComponentType.DATA_DOWNLOADER.value,
            operation="test_operation",
            file_path=None,
            record_index=None,
            batch_id=None,
            timestamp="2024-01-01T12:00:00Z",
            error_type="ValueError",
            error_message="Test error",
            stack_trace="",
            severity=ErrorSeverity.MEDIUM,
            retry_count=0,
            max_retries=3,
            recovery_suggestions=[]
        )
        
        # Handle some errors
        test_error = ValueError("Test error")
        error_manager.handle_error(test_error, context)
        
        # Get statistics
        stats = error_manager.get_error_statistics()
        
        assert stats['total_errors'] > 0
        assert 'ValueError' in stats['error_types']


class TestIntegration:
    """Integration tests for logging and error handling systems."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.logging_config = {
            'level': 'DEBUG',
            'file_path': f'{self.temp_dir}/test.log',
            'error_log_path': f'{self.temp_dir}/errors.log',
            'console_output': False
        }
        
        self.error_config = {
            'retry_attempts': 2,
            'retry_delay': 0.1,
            'continue_on_error': True,
            'checkpoint_enabled': True,
            'checkpoint_path': f'{self.temp_dir}/checkpoint.json'
        }
    
    def test_integrated_error_handling_and_logging(self):
        """Test integrated error handling and logging."""
        # Initialize both systems
        logger_system = initialize_logging(self.logging_config)
        error_manager = initialize_error_handling(self.error_config)
        
        # Test operation with error handling and logging
        operation_context = logger_system.start_operation(
            ComponentType.REDDIT_PARSER,
            "test_operation"
        )
        
        try:
            # Simulate some work
            time.sleep(0.1)
            
            # Simulate an error
            test_error = ValueError("Simulated error")
            
            error_context = ErrorContext(
                component=ComponentType.REDDIT_PARSER.value,
                operation="test_operation",
                file_path="test_file.json",
                record_index=100,
                batch_id="batch_1",
                timestamp="2024-01-01T12:00:00Z",
                error_type="ValueError",
                error_message="Simulated error",
                stack_trace="",
                severity=ErrorSeverity.MEDIUM,
                retry_count=0,
                max_retries=2,
                recovery_suggestions=[]
            )
            
            recovery_action = error_manager.handle_error(test_error, error_context)
            
            # Complete operation
            stats = logger_system.end_operation(
                operation_context,
                records_processed=100,
                success=recovery_action != RecoveryAction.ABORT
            )
            
            # Verify integration worked
            assert stats is not None
            assert len(logger_system.error_records) > 0
            assert error_manager.error_stats['total_errors'] > 0
            
        except Exception as e:
            logger_system.end_operation(
                operation_context,
                success=False,
                error_message=str(e)
            )
            raise


if __name__ == "__main__":
    pytest.main([__file__])