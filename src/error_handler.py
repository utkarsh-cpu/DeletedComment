"""
Error recovery and checkpoint system for Reddit deleted comment dataset processing.

This module provides comprehensive error handling, recovery mechanisms, and checkpoint
functionality to resume interrupted processing as specified in requirements 5.3 and 5.4.
"""

import json
import time
import traceback
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager

# Import ComponentType directly to avoid circular dependency
from logger import ComponentType


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Available recovery actions."""
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    ABORT = "abort"
    CONTINUE = "continue"
    RESTART_COMPONENT = "restart_component"
    CHECKPOINT_RESTORE = "checkpoint_restore"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    component: str
    operation: str
    file_path: Optional[str]
    record_index: Optional[int]
    batch_id: Optional[str]
    timestamp: str
    error_type: str
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    retry_count: int
    max_retries: int
    recovery_suggestions: List[str]


@dataclass
class CheckpointData:
    """Checkpoint data structure."""
    checkpoint_id: str
    timestamp: str
    component: str
    operation: str
    progress: Dict[str, Any]
    processed_files: List[str]
    current_file: Optional[str]
    current_position: int
    total_records: int
    processed_records: int
    error_count: int
    metadata: Dict[str, Any]


class RetryStrategy:
    """Retry strategy configuration."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_backoff: bool = True,
                 jitter: bool = True):
        """
        Initialize retry strategy.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_backoff: Use exponential backoff
            jitter: Add random jitter to delays
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** attempt)
        else:
            delay = self.base_delay
        
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
        
        return delay


class CheckpointManager:
    """
    Manages checkpoint creation and restoration for resuming interrupted processing.
    """
    
    def __init__(self, checkpoint_path: str, interval: int = 10000):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_path: Path to checkpoint file
            interval: Save checkpoint every N records
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.interval = interval
        self.last_checkpoint_time = time.time()
        self.checkpoint_lock = threading.Lock()
        
        # Ensure checkpoint directory exists
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = None  # Will be initialized when needed
    
    def _get_logger(self):
        """Get logger instance, initializing if needed."""
        if self.logger is None:
            try:
                from logger import get_logger
                self.logger = get_logger()
            except Exception:
                # Fallback to basic logging if advanced logger not available
                import logging
                self.logger = logging.getLogger('checkpoint_manager')
        return self.logger
    
    def save_checkpoint(self, checkpoint_data: CheckpointData) -> bool:
        """
        Save checkpoint data to file.
        
        Args:
            checkpoint_data: Checkpoint data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.checkpoint_lock:
                # Create backup of existing checkpoint
                if self.checkpoint_path.exists():
                    backup_path = self.checkpoint_path.with_suffix('.bak')
                    self.checkpoint_path.rename(backup_path)
                
                # Save new checkpoint
                with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(checkpoint_data), f, indent=2, default=str)
                
                self.last_checkpoint_time = time.time()
                
                self._get_logger().log_info(
                    ComponentType.ERROR_HANDLER,
                    f"Checkpoint saved: {checkpoint_data.checkpoint_id}",
                    processed_records=checkpoint_data.processed_records,
                    total_records=checkpoint_data.total_records,
                    progress_percent=(checkpoint_data.processed_records / 
                                    checkpoint_data.total_records * 100 
                                    if checkpoint_data.total_records > 0 else 0)
                )
                
                return True
                
        except Exception as e:
            self._get_logger().log_error(
                ComponentType.ERROR_HANDLER,
                f"Failed to save checkpoint: {e}",
                error=e,
                recovery_action="continue_without_checkpoint"
            )
            return False
    
    def load_checkpoint(self) -> Optional[CheckpointData]:
        """
        Load checkpoint data from file.
        
        Returns:
            Checkpoint data if available, None otherwise
        """
        try:
            if not self.checkpoint_path.exists():
                return None
            
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            checkpoint = CheckpointData(**data)
            
            self._get_logger().log_info(
                ComponentType.ERROR_HANDLER,
                f"Checkpoint loaded: {checkpoint.checkpoint_id}",
                processed_records=checkpoint.processed_records,
                total_records=checkpoint.total_records
            )
            
            return checkpoint
            
        except Exception as e:
            self._get_logger().log_error(
                ComponentType.ERROR_HANDLER,
                f"Failed to load checkpoint: {e}",
                error=e,
                recovery_action="start_from_beginning"
            )
            return None
    
    def should_save_checkpoint(self, records_processed: int) -> bool:
        """
        Check if checkpoint should be saved based on interval.
        
        Args:
            records_processed: Number of records processed
            
        Returns:
            True if checkpoint should be saved
        """
        return records_processed % self.interval == 0
    
    def clear_checkpoint(self) -> None:
        """Clear checkpoint file after successful completion."""
        try:
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
                self._get_logger().log_info(
                    ComponentType.ERROR_HANDLER,
                    "Checkpoint cleared after successful completion"
                )
        except Exception as e:
            self._get_logger().log_error(
                ComponentType.ERROR_HANDLER,
                f"Failed to clear checkpoint: {e}",
                error=e,
                recovery_action="manual_cleanup_required"
            )


class ErrorRecoveryManager:
    """
    Comprehensive error recovery manager with multiple recovery strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize error recovery manager.
        
        Args:
            config: Error handling configuration
        """
        self.config = config
        self.logger = None  # Will be initialized when needed
        
        # Initialize retry strategies for different error types
        self.retry_strategies = {
            'network': RetryStrategy(
                max_attempts=config.get('retry_attempts', 3),
                base_delay=config.get('retry_delay', 1.0),
                max_delay=config.get('max_retry_delay', 60.0),
                exponential_backoff=config.get('exponential_backoff', True)
            ),
            'io': RetryStrategy(
                max_attempts=config.get('retry_attempts', 3),
                base_delay=0.5,
                max_delay=10.0
            ),
            'parsing': RetryStrategy(
                max_attempts=1,  # Don't retry parsing errors
                base_delay=0.0
            ),
            'memory': RetryStrategy(
                max_attempts=2,
                base_delay=5.0,
                max_delay=30.0
            )
        }
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config.get('checkpoint_path', './data/checkpoint.json'),
            config.get('checkpoint_interval', 10000)
        )
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'fatal_errors': 0,
            'error_types': {},
            'recovery_actions': {}
        }
    
    def _get_logger(self):
        """Get logger instance, initializing if needed."""
        if self.logger is None:
            try:
                from logger import get_logger
                self.logger = get_logger()
            except Exception:
                # Fallback to basic logging if advanced logger not available
                import logging
                self.logger = logging.getLogger('error_recovery_manager')
        return self.logger
    
    def handle_error(self, error: Exception, context: ErrorContext,
                    recovery_callback: Optional[Callable] = None) -> RecoveryAction:
        """
        Handle error with appropriate recovery strategy.
        
        Args:
            error: Exception that occurred
            context: Error context information
            recovery_callback: Optional callback for custom recovery
            
        Returns:
            Recovery action to take
        """
        self.error_stats['total_errors'] += 1
        self.error_stats['error_types'][context.error_type] = (
            self.error_stats['error_types'].get(context.error_type, 0) + 1
        )
        
        # Determine error category and severity
        error_category = self._categorize_error(error)
        severity = self._assess_severity(error, context)
        context.severity = severity
        
        # Log error with full context
        self._get_logger().log_error(
            ComponentType(context.component),
            f"Error in {context.operation}: {context.error_message}",
            error=error,
            recovery_action="determining",
            file_path=context.file_path,
            **{
                'record_index': context.record_index,
                'batch_id': context.batch_id,
                'severity': severity.value,
                'retry_count': context.retry_count,
                'error_category': error_category
            }
        )
        
        # Determine recovery action
        recovery_action = self._determine_recovery_action(error, context, error_category)
        
        # Execute recovery action
        success = self._execute_recovery_action(recovery_action, context, recovery_callback)
        
        if success:
            self.error_stats['recovered_errors'] += 1
        else:
            self.error_stats['fatal_errors'] += 1
        
        self.error_stats['recovery_actions'][recovery_action.value] = (
            self.error_stats['recovery_actions'].get(recovery_action.value, 0) + 1
        )
        
        return recovery_action
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize error type for appropriate handling strategy."""
        error_type = type(error).__name__
        
        # Network-related errors
        if any(keyword in error_type.lower() for keyword in 
               ['connection', 'timeout', 'network', 'http', 'url']):
            return 'network'
        
        # I/O related errors
        if any(keyword in error_type.lower() for keyword in 
               ['file', 'io', 'permission', 'disk', 'space']):
            return 'io'
        
        # Memory related errors
        if any(keyword in error_type.lower() for keyword in 
               ['memory', 'allocation', 'outofmemory']):
            return 'memory'
        
        # Parsing related errors
        if any(keyword in error_type.lower() for keyword in 
               ['json', 'parse', 'decode', 'format', 'syntax']):
            return 'parsing'
        
        return 'general'
    
    def _assess_severity(self, error: Exception, context: ErrorContext) -> ErrorSeverity:
        """Assess error severity based on error type and context."""
        error_type = type(error).__name__.lower()
        
        # Critical errors that should stop processing
        if any(keyword in error_type for keyword in 
               ['systemexit', 'keyboardinterrupt', 'outofmemory']):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if any(keyword in error_type for keyword in 
               ['permission', 'authentication', 'authorization']):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if any(keyword in error_type for keyword in 
               ['connection', 'timeout', 'network']):
            return ErrorSeverity.MEDIUM
        
        # Low severity errors (data-related)
        if any(keyword in error_type for keyword in 
               ['json', 'parse', 'decode', 'value']):
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _determine_recovery_action(self, error: Exception, context: ErrorContext,
                                 error_category: str) -> RecoveryAction:
        """Determine appropriate recovery action based on error and context."""
        
        # Critical errors should abort
        if context.severity == ErrorSeverity.CRITICAL:
            return RecoveryAction.ABORT
        
        # Check if we've exceeded retry limits
        retry_strategy = self.retry_strategies.get(error_category, 
                                                 self.retry_strategies['network'])
        
        if context.retry_count >= retry_strategy.max_attempts:
            if self.config.get('continue_on_error', True):
                return RecoveryAction.SKIP
            else:
                return RecoveryAction.ABORT
        
        # Determine action based on error category
        if error_category == 'network':
            return RecoveryAction.RETRY
        elif error_category == 'io':
            return RecoveryAction.RETRY
        elif error_category == 'memory':
            return RecoveryAction.CHECKPOINT_RESTORE
        elif error_category == 'parsing':
            return RecoveryAction.SKIP  # Skip malformed data
        else:
            return RecoveryAction.RETRY if context.retry_count < 2 else RecoveryAction.SKIP
    
    def _execute_recovery_action(self, action: RecoveryAction, context: ErrorContext,
                               recovery_callback: Optional[Callable] = None) -> bool:
        """Execute the determined recovery action."""
        
        try:
            if action == RecoveryAction.RETRY:
                return self._handle_retry(context)
            
            elif action == RecoveryAction.SKIP:
                return self._handle_skip(context)
            
            elif action == RecoveryAction.FALLBACK:
                return self._handle_fallback(context, recovery_callback)
            
            elif action == RecoveryAction.CHECKPOINT_RESTORE:
                return self._handle_checkpoint_restore(context)
            
            elif action == RecoveryAction.CONTINUE:
                return self._handle_continue(context)
            
            elif action == RecoveryAction.ABORT:
                return self._handle_abort(context)
            
            else:
                self._get_logger().log_warning(
                    ComponentType.ERROR_HANDLER,
                    f"Unknown recovery action: {action}",
                    context=asdict(context)
                )
                return False
                
        except Exception as e:
            self._get_logger().log_error(
                ComponentType.ERROR_HANDLER,
                f"Failed to execute recovery action {action}: {e}",
                error=e,
                recovery_action="recovery_failed"
            )
            return False
    
    def _handle_retry(self, context: ErrorContext) -> bool:
        """Handle retry recovery action."""
        error_category = self._categorize_error(Exception(context.error_message))
        retry_strategy = self.retry_strategies.get(error_category, 
                                                 self.retry_strategies['network'])
        
        delay = retry_strategy.get_delay(context.retry_count)
        
        self._get_logger().log_info(
            ComponentType.ERROR_HANDLER,
            f"Retrying operation after {delay:.2f}s delay (attempt {context.retry_count + 1})",
            operation=context.operation,
            delay_seconds=delay
        )
        
        time.sleep(delay)
        return True
    
    def _handle_skip(self, context: ErrorContext) -> bool:
        """Handle skip recovery action."""
        self._get_logger().log_warning(
            ComponentType.ERROR_HANDLER,
            f"Skipping failed operation: {context.operation}",
            file_path=context.file_path,
            record_index=context.record_index
        )
        return True
    
    def _handle_fallback(self, context: ErrorContext, 
                        recovery_callback: Optional[Callable]) -> bool:
        """Handle fallback recovery action."""
        if recovery_callback:
            try:
                recovery_callback(context)
                self._get_logger().log_info(
                    ComponentType.ERROR_HANDLER,
                    f"Fallback recovery successful for: {context.operation}"
                )
                return True
            except Exception as e:
                self._get_logger().log_error(
                    ComponentType.ERROR_HANDLER,
                    f"Fallback recovery failed: {e}",
                    error=e,
                    recovery_action="fallback_failed"
                )
        
        return False
    
    def _handle_checkpoint_restore(self, context: ErrorContext) -> bool:
        """Handle checkpoint restore recovery action."""
        checkpoint = self.checkpoint_manager.load_checkpoint()
        if checkpoint:
            self._get_logger().log_info(
                ComponentType.ERROR_HANDLER,
                f"Restoring from checkpoint: {checkpoint.checkpoint_id}",
                processed_records=checkpoint.processed_records
            )
            return True
        else:
            self._get_logger().log_warning(
                ComponentType.ERROR_HANDLER,
                "No checkpoint available for restoration"
            )
            return False
    
    def _handle_continue(self, context: ErrorContext) -> bool:
        """Handle continue recovery action."""
        self._get_logger().log_info(
            ComponentType.ERROR_HANDLER,
            f"Continuing processing despite error in: {context.operation}"
        )
        return True
    
    def _handle_abort(self, context: ErrorContext) -> bool:
        """Handle abort recovery action."""
        self._get_logger().log_error(
            ComponentType.ERROR_HANDLER,
            f"Aborting processing due to critical error in: {context.operation}",
            recovery_action="processing_aborted"
        )
        return False
    
    @contextmanager
    def error_context(self, component: ComponentType, operation: str,
                     file_path: Optional[str] = None, record_index: Optional[int] = None,
                     batch_id: Optional[str] = None):
        """
        Context manager for automatic error handling.
        
        Args:
            component: Component performing the operation
            operation: Operation name
            file_path: File being processed
            record_index: Record index being processed
            batch_id: Batch identifier
        """
        retry_count = 0
        max_retries = self.config.get('retry_attempts', 3)
        
        while retry_count <= max_retries:
            try:
                yield
                return  # Success, exit context manager
                
            except Exception as e:
                context = ErrorContext(
                    component=component.value,
                    operation=operation,
                    file_path=file_path,
                    record_index=record_index,
                    batch_id=batch_id,
                    timestamp=datetime.now().isoformat(),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                    severity=ErrorSeverity.MEDIUM,
                    retry_count=retry_count,
                    max_retries=max_retries,
                    recovery_suggestions=self._generate_recovery_suggestions(e)
                )
                
                recovery_action = self.handle_error(e, context)
                
                if recovery_action == RecoveryAction.ABORT:
                    raise
                elif recovery_action == RecoveryAction.SKIP:
                    return  # Skip and exit context manager
                elif recovery_action == RecoveryAction.RETRY:
                    retry_count += 1
                    if retry_count > max_retries:
                        raise
                    # Continue the while loop for retry
                else:
                    return  # Default: exit context manager
    
    def _generate_recovery_suggestions(self, error: Exception) -> List[str]:
        """Generate recovery suggestions based on error type."""
        error_type = type(error).__name__.lower()
        suggestions = []
        
        if 'connection' in error_type or 'network' in error_type:
            suggestions.extend([
                "Check network connectivity",
                "Verify API endpoints are accessible",
                "Consider increasing timeout values"
            ])
        
        elif 'permission' in error_type or 'access' in error_type:
            suggestions.extend([
                "Check file permissions",
                "Verify authentication credentials",
                "Ensure sufficient disk space"
            ])
        
        elif 'memory' in error_type:
            suggestions.extend([
                "Reduce batch size",
                "Enable memory optimization",
                "Close unused resources"
            ])
        
        elif 'json' in error_type or 'parse' in error_type:
            suggestions.extend([
                "Validate input data format",
                "Check for corrupted files",
                "Enable data validation"
            ])
        
        return suggestions
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        return {
            **self.error_stats,
            'error_rate': (
                self.error_stats['total_errors'] / 
                max(1, self.error_stats['total_errors'] + self.error_stats['recovered_errors'])
            ),
            'recovery_rate': (
                self.error_stats['recovered_errors'] / 
                max(1, self.error_stats['total_errors'])
            )
        }


# Global error recovery manager instance
_error_manager_instance: Optional[ErrorRecoveryManager] = None


def initialize_error_handling(config: Dict[str, Any]) -> ErrorRecoveryManager:
    """
    Initialize global error handling system.
    
    Args:
        config: Error handling configuration
        
    Returns:
        Error recovery manager instance
    """
    global _error_manager_instance
    _error_manager_instance = ErrorRecoveryManager(config)
    return _error_manager_instance


def get_error_manager() -> ErrorRecoveryManager:
    """
    Get global error recovery manager instance.
    
    Returns:
        Error recovery manager instance
        
    Raises:
        RuntimeError: If error handling not initialized
    """
    if _error_manager_instance is None:
        raise RuntimeError("Error handling system not initialized. Call initialize_error_handling() first.")
    return _error_manager_instance