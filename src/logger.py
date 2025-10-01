"""
Comprehensive logging system for Reddit deleted comment dataset processing.

This module provides structured logging with multiple handlers, performance tracking,
and error recovery action logging as specified in requirements 5.3 and 5.4.
"""

import logging
import logging.handlers
import json
import time
import traceback
import psutil
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ComponentType(Enum):
    """Component type enumeration for structured logging."""
    DATA_DOWNLOADER = "data_downloader"
    REDDIT_PARSER = "reddit_parser"
    COMMENT_CLASSIFIER = "comment_classifier"
    METADATA_EXTRACTOR = "metadata_extractor"
    PARQUET_WRITER = "parquet_writer"
    DRIVE_UPLOADER = "drive_uploader"
    PROGRESS_MONITOR = "progress_monitor"
    CLEANUP_MANAGER = "cleanup_manager"
    MAIN_PIPELINE = "main_pipeline"
    ERROR_HANDLER = "error_handler"


@dataclass
class ProcessingStats:
    """Processing statistics for performance logging."""
    component: str
    operation: str
    start_time: float
    end_time: float
    records_processed: int
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    
    @property
    def duration_seconds(self) -> float:
        """Calculate operation duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def records_per_second(self) -> float:
        """Calculate processing rate."""
        if self.duration_seconds > 0:
            return self.records_processed / self.duration_seconds
        return 0.0


@dataclass
class ErrorRecord:
    """Error record for detailed error tracking."""
    timestamp: str
    component: str
    error_type: str
    error_message: str
    file_path: Optional[str]
    line_number: Optional[int]
    recovery_action: str
    stack_trace: Optional[str]
    context: Dict[str, Any]


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "component": getattr(record, 'component', 'unknown'),
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, default=str)


class RedditDatasetLogger:
    """
    Comprehensive logging system for Reddit dataset processing.
    
    Provides structured logging, performance tracking, and error recovery logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the logging system.
        
        Args:
            config: Configuration dictionary with logging settings
        """
        self.config = config
        self.loggers: Dict[str, logging.Logger] = {}
        self.performance_stats: List[ProcessingStats] = []
        self.error_records: List[ErrorRecord] = []
        
        # Create log directories
        self._create_log_directories()
        
        # Setup loggers
        self._setup_main_logger()
        self._setup_error_logger()
        self._setup_performance_logger()
        
        # Initialize system monitoring
        self.process = psutil.Process()
        self.start_time = time.time()
    
    def _create_log_directories(self) -> None:
        """Create necessary log directories."""
        log_file_path = Path(self.config.get('file_path', './logs/processing.log'))
        error_log_path = Path(self.config.get('error_log_path', './logs/errors.log'))
        
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _setup_main_logger(self) -> None:
        """Setup main application logger."""
        logger = logging.getLogger('reddit_dataset')
        logger.setLevel(getattr(logging, self.config.get('level', 'INFO')))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.config.get('file_path', './logs/processing.log'),
            maxBytes=self.config.get('max_size_mb', 100) * 1024 * 1024,
            backupCount=self.config.get('backup_count', 5),
            encoding='utf-8'
        )
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)
        
        # Console handler if enabled
        if self.config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        self.loggers['main'] = logger
    
    def _setup_error_logger(self) -> None:
        """Setup dedicated error logger."""
        error_logger = logging.getLogger('reddit_dataset.errors')
        error_logger.setLevel(logging.ERROR)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            filename=self.config.get('error_log_path', './logs/errors.log'),
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setFormatter(StructuredFormatter())
        error_logger.addHandler(error_handler)
        
        self.loggers['error'] = error_logger
    
    def _setup_performance_logger(self) -> None:
        """Setup performance metrics logger."""
        perf_logger = logging.getLogger('reddit_dataset.performance')
        perf_logger.setLevel(logging.INFO)
        
        # Performance file handler
        perf_handler = logging.handlers.RotatingFileHandler(
            filename=self.config.get('performance_log_path', './logs/performance.log'),
            maxBytes=25 * 1024 * 1024,  # 25MB
            backupCount=5,
            encoding='utf-8'
        )
        perf_handler.setFormatter(StructuredFormatter())
        perf_logger.addHandler(perf_handler)
        
        self.loggers['performance'] = perf_logger
    
    def get_logger(self, component: ComponentType) -> logging.Logger:
        """
        Get logger for specific component.
        
        Args:
            component: Component type
            
        Returns:
            Logger instance for the component
        """
        logger_name = f'reddit_dataset.{component.value}'
        
        if logger_name not in self.loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, self.config.get('level', 'INFO')))
            
            # Use main logger's handlers
            main_logger = self.loggers['main']
            for handler in main_logger.handlers:
                logger.addHandler(handler)
            
            self.loggers[logger_name] = logger
        
        return self.loggers[logger_name]
    
    def log_info(self, component: ComponentType, message: str, **kwargs) -> None:
        """Log info message with component context."""
        logger = self.get_logger(component)
        extra_fields = {'component': component.value, 'extra_fields': kwargs}
        logger.info(message, extra=extra_fields)
    
    def log_warning(self, component: ComponentType, message: str, **kwargs) -> None:
        """Log warning message with component context."""
        logger = self.get_logger(component)
        extra_fields = {'component': component.value, 'extra_fields': kwargs}
        logger.warning(message, extra=extra_fields)
    
    def log_error(self, component: ComponentType, message: str, 
                  error: Optional[Exception] = None, recovery_action: str = "none",
                  file_path: Optional[str] = None, **kwargs) -> None:
        """
        Log error with detailed context and recovery action.
        
        Args:
            component: Component where error occurred
            message: Error message
            error: Exception object if available
            recovery_action: Action taken to recover from error
            file_path: File path related to error
            **kwargs: Additional context
        """
        logger = self.get_logger(component)
        error_logger = self.loggers['error']
        
        # Create error record
        error_record = ErrorRecord(
            timestamp=datetime.now().isoformat(),
            component=component.value,
            error_type=type(error).__name__ if error else "Unknown",
            error_message=str(error) if error else message,
            file_path=file_path,
            line_number=None,
            recovery_action=recovery_action,
            stack_trace=traceback.format_exc() if error else None,
            context=kwargs
        )
        
        self.error_records.append(error_record)
        
        # Log to main logger
        extra_fields = {
            'component': component.value,
            'recovery_action': recovery_action,
            'extra_fields': kwargs
        }
        
        if error:
            logger.error(message, exc_info=True, extra=extra_fields)
            error_logger.error(
                f"Error in {component.value}: {message}",
                exc_info=True,
                extra={'component': component.value, 'extra_fields': asdict(error_record)}
            )
        else:
            logger.error(message, extra=extra_fields)
            error_logger.error(
                f"Error in {component.value}: {message}",
                extra={'component': component.value, 'extra_fields': asdict(error_record)}
            )
    
    def log_performance(self, stats: ProcessingStats) -> None:
        """
        Log performance statistics.
        
        Args:
            stats: Processing statistics
        """
        perf_logger = self.loggers['performance']
        self.performance_stats.append(stats)
        
        message = (
            f"Performance: {stats.component}.{stats.operation} - "
            f"{stats.records_processed} records in {stats.duration_seconds:.2f}s "
            f"({stats.records_per_second:.2f} rec/s)"
        )
        
        extra_fields = {
            'component': stats.component,
            'extra_fields': asdict(stats)
        }
        
        perf_logger.info(message, extra=extra_fields)
    
    def start_operation(self, component: ComponentType, operation: str) -> Dict[str, Any]:
        """
        Start tracking an operation for performance logging.
        
        Args:
            component: Component performing the operation
            operation: Operation name
            
        Returns:
            Context dictionary for tracking
        """
        context = {
            'component': component.value,
            'operation': operation,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
            'start_cpu': self._get_cpu_usage()
        }
        
        self.log_info(component, f"Starting operation: {operation}")
        return context
    
    def end_operation(self, context: Dict[str, Any], records_processed: int = 0,
                     success: bool = True, error_message: Optional[str] = None) -> ProcessingStats:
        """
        End tracking an operation and log performance statistics.
        
        Args:
            context: Context from start_operation
            records_processed: Number of records processed
            success: Whether operation succeeded
            error_message: Error message if operation failed
            
        Returns:
            Processing statistics
        """
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        stats = ProcessingStats(
            component=context['component'],
            operation=context['operation'],
            start_time=context['start_time'],
            end_time=end_time,
            records_processed=records_processed,
            memory_usage_mb=end_memory,
            cpu_usage_percent=self._get_cpu_usage(),
            success=success,
            error_message=error_message
        )
        
        self.log_performance(stats)
        
        component = ComponentType(context['component'])
        if success:
            self.log_info(
                component,
                f"Completed operation: {context['operation']} - "
                f"{records_processed} records in {stats.duration_seconds:.2f}s"
            )
        else:
            self.log_error(
                component,
                f"Failed operation: {context['operation']} - {error_message}",
                recovery_action="operation_failed"
            )
        
        return stats
    
    def log_memory_usage(self, component: ComponentType, context: str = "") -> None:
        """Log current memory usage."""
        if not self.config.get('log_memory_usage', True):
            return
        
        memory_mb = self._get_memory_usage()
        memory_percent = self.process.memory_percent()
        
        self.log_info(
            component,
            f"Memory usage: {memory_mb:.1f} MB ({memory_percent:.1f}%)",
            context=context,
            memory_mb=memory_mb,
            memory_percent=memory_percent
        )
    
    def log_system_resources(self, component: ComponentType) -> None:
        """Log comprehensive system resource usage."""
        try:
            # Memory information
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # CPU information
            cpu_percent = self.process.cpu_percent()
            
            # Disk usage for current directory
            disk_usage = psutil.disk_usage('.')
            
            self.log_info(
                component,
                "System resources",
                memory_rss_mb=memory_info.rss / 1024 / 1024,
                memory_vms_mb=memory_info.vms / 1024 / 1024,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                disk_free_gb=disk_usage.free / 1024 / 1024 / 1024,
                disk_used_gb=disk_usage.used / 1024 / 1024 / 1024,
                disk_total_gb=disk_usage.total / 1024 / 1024 / 1024
            )
        except Exception as e:
            self.log_error(
                component,
                f"Failed to log system resources: {e}",
                error=e,
                recovery_action="continue_without_resource_logging"
            )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self.process.cpu_percent()
        except Exception:
            return 0.0
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        error_counts = {}
        recovery_actions = {}
        
        for error in self.error_records:
            # Count errors by type
            error_type = error.error_type
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            # Count recovery actions
            action = error.recovery_action
            recovery_actions[action] = recovery_actions.get(action, 0) + 1
        
        return {
            'total_errors': len(self.error_records),
            'error_types': error_counts,
            'recovery_actions': recovery_actions,
            'recent_errors': [asdict(error) for error in self.error_records[-10:]]
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance statistics."""
        if not self.performance_stats:
            return {'message': 'No performance data available'}
        
        total_records = sum(stat.records_processed for stat in self.performance_stats)
        total_time = sum(stat.duration_seconds for stat in self.performance_stats)
        avg_rate = total_records / total_time if total_time > 0 else 0
        
        component_stats = {}
        for stat in self.performance_stats:
            if stat.component not in component_stats:
                component_stats[stat.component] = {
                    'operations': 0,
                    'total_records': 0,
                    'total_time': 0,
                    'avg_rate': 0
                }
            
            comp_stat = component_stats[stat.component]
            comp_stat['operations'] += 1
            comp_stat['total_records'] += stat.records_processed
            comp_stat['total_time'] += stat.duration_seconds
            comp_stat['avg_rate'] = (
                comp_stat['total_records'] / comp_stat['total_time']
                if comp_stat['total_time'] > 0 else 0
            )
        
        return {
            'total_records_processed': total_records,
            'total_processing_time': total_time,
            'overall_rate': avg_rate,
            'component_breakdown': component_stats,
            'uptime_seconds': time.time() - self.start_time
        }
    
    def shutdown(self) -> None:
        """Shutdown logging system and flush all handlers."""
        # Log final summaries
        main_logger = self.loggers.get('main')
        if main_logger:
            main_logger.info("Shutting down logging system")
            main_logger.info(f"Error summary: {self.get_error_summary()}")
            main_logger.info(f"Performance summary: {self.get_performance_summary()}")
        
        # Close all handlers
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.close()


# Global logger instance
_logger_instance: Optional[RedditDatasetLogger] = None


def initialize_logging(config: Dict[str, Any]) -> RedditDatasetLogger:
    """
    Initialize global logging system.
    
    Args:
        config: Logging configuration
        
    Returns:
        Logger instance
    """
    global _logger_instance
    _logger_instance = RedditDatasetLogger(config)
    return _logger_instance


def get_logger() -> RedditDatasetLogger:
    """
    Get global logger instance.
    
    Returns:
        Logger instance
        
    Raises:
        RuntimeError: If logging not initialized
    """
    if _logger_instance is None:
        raise RuntimeError("Logging system not initialized. Call initialize_logging() first.")
    return _logger_instance