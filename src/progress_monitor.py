"""
Progress monitoring and system resource tracking for Reddit dataset processing.

This module provides comprehensive tracking of processing stages, memory usage,
and performance metrics to help monitor the health and progress of the data
processing pipeline.
"""

import time
import psutil
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ProcessingStats:
    """Statistics for processing operations."""
    stage: str
    start_time: float
    current_progress: int
    total_items: int
    items_per_second: float
    memory_usage_mb: float
    peak_memory_mb: float
    errors_count: int
    warnings_count: int


class ProgressMonitor:
    """
    Monitors processing progress and system resources.
    
    Provides stage-based progress tracking, memory monitoring, and performance
    metrics logging for the Reddit dataset processing pipeline.
    """
    
    def __init__(self, log_interval: int = 10):
        """
        Initialize the progress monitor.
        
        Args:
            log_interval: Seconds between automatic progress logs
        """
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        
        # Current processing state
        self.current_stage: Optional[str] = None
        self.stage_start_time: Optional[float] = None
        self.current_progress: int = 0
        self.total_items: int = 0
        
        # Performance tracking
        self.peak_memory_mb: float = 0.0
        self.errors_count: int = 0
        self.warnings_count: int = 0
        self.last_log_time: float = 0.0
        
        # Stage history
        self.stage_history: Dict[str, ProcessingStats] = {}
        
        # Initialize process monitoring
        self.process = psutil.Process()
        
    def start_stage(self, stage_name: str, total_items: int = 0) -> None:
        """
        Start tracking a new processing stage.
        
        Args:
            stage_name: Name of the processing stage
            total_items: Total number of items to process in this stage
        """
        # Finalize previous stage if exists
        if self.current_stage:
            self._finalize_current_stage()
        
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        self.current_progress = 0
        self.total_items = total_items
        self.last_log_time = time.time()
        
        self.logger.info(f"Started stage: {stage_name} (total items: {total_items})")
        
    def update_progress(self, current: int, total: Optional[int] = None) -> None:
        """
        Update progress for the current stage.
        
        Args:
            current: Current number of processed items
            total: Total items (updates the total if provided)
        """
        if not self.current_stage:
            self.logger.warning("Progress update called without active stage")
            return
            
        self.current_progress = current
        if total is not None:
            self.total_items = total
            
        # Log progress at intervals
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self._log_current_progress()
            self.last_log_time = current_time
            
    def monitor_memory_usage(self) -> Dict[str, float]:
        """
        Monitor current memory usage and update peak values.
        
        Returns:
            Dictionary with memory usage statistics in MB
        """
        try:
            # Get memory info
            memory_info = self.process.memory_info()
            virtual_memory = psutil.virtual_memory()
            
            # Convert to MB
            current_memory_mb = memory_info.rss / 1024 / 1024
            available_memory_mb = virtual_memory.available / 1024 / 1024
            total_memory_mb = virtual_memory.total / 1024 / 1024
            memory_percent = virtual_memory.percent
            
            # Update peak memory
            if current_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = current_memory_mb
                
            memory_stats = {
                'current_mb': current_memory_mb,
                'peak_mb': self.peak_memory_mb,
                'available_mb': available_memory_mb,
                'total_mb': total_memory_mb,
                'usage_percent': memory_percent
            }
            
            # Log warning if memory usage is high
            if memory_percent > 85:
                self.logger.warning(
                    f"High memory usage: {memory_percent:.1f}% "
                    f"({current_memory_mb:.1f} MB)"
                )
                
            return memory_stats
            
        except Exception as e:
            self.logger.error(f"Error monitoring memory usage: {e}")
            return {
                'current_mb': 0.0,
                'peak_mb': self.peak_memory_mb,
                'available_mb': 0.0,
                'total_mb': 0.0,
                'usage_percent': 0.0
            }
            
    def log_processing_stats(self, additional_stats: Optional[Dict[str, Any]] = None) -> None:
        """
        Log comprehensive processing statistics.
        
        Args:
            additional_stats: Optional additional statistics to include
        """
        if not self.current_stage:
            self.logger.warning("Cannot log stats without active stage")
            return
            
        # Get current memory stats
        memory_stats = self.monitor_memory_usage()
        
        # Calculate performance metrics
        elapsed_time = time.time() - (self.stage_start_time or time.time())
        items_per_second = self.current_progress / elapsed_time if elapsed_time > 0 else 0.0
        
        # Create stats dictionary
        stats = {
            'stage': self.current_stage,
            'progress': f"{self.current_progress}/{self.total_items}",
            'completion_percent': (self.current_progress / self.total_items * 100) if self.total_items > 0 else 0.0,
            'elapsed_time_seconds': elapsed_time,
            'items_per_second': items_per_second,
            'memory_current_mb': memory_stats['current_mb'],
            'memory_peak_mb': memory_stats['peak_mb'],
            'memory_usage_percent': memory_stats['usage_percent'],
            'errors_count': self.errors_count,
            'warnings_count': self.warnings_count,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add additional stats if provided
        if additional_stats:
            stats.update(additional_stats)
            
        # Log the statistics
        self.logger.info(f"Processing Stats: {stats}")
        
        # Estimate completion time
        if self.total_items > 0 and items_per_second > 0:
            remaining_items = self.total_items - self.current_progress
            eta_seconds = remaining_items / items_per_second
            eta_minutes = eta_seconds / 60
            
            if eta_minutes > 1:
                self.logger.info(f"Estimated time remaining: {eta_minutes:.1f} minutes")
            else:
                self.logger.info(f"Estimated time remaining: {eta_seconds:.0f} seconds")
                
    def increment_error_count(self) -> None:
        """Increment the error counter for the current stage."""
        self.errors_count += 1
        
    def increment_warning_count(self) -> None:
        """Increment the warning counter for the current stage."""
        self.warnings_count += 1
        
    def get_stage_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current stage progress.
        
        Returns:
            Dictionary with current stage summary
        """
        if not self.current_stage:
            return {}
            
        elapsed_time = time.time() - (self.stage_start_time or time.time())
        items_per_second = self.current_progress / elapsed_time if elapsed_time > 0 else 0.0
        memory_stats = self.monitor_memory_usage()
        
        return {
            'stage': self.current_stage,
            'progress': self.current_progress,
            'total': self.total_items,
            'completion_percent': (self.current_progress / self.total_items * 100) if self.total_items > 0 else 0.0,
            'elapsed_time': elapsed_time,
            'items_per_second': items_per_second,
            'memory_mb': memory_stats['current_mb'],
            'peak_memory_mb': memory_stats['peak_mb'],
            'errors': self.errors_count,
            'warnings': self.warnings_count
        }
        
    def finalize_stage(self) -> ProcessingStats:
        """
        Finalize the current stage and return its statistics.
        
        Returns:
            ProcessingStats object with final stage metrics
        """
        if not self.current_stage:
            raise ValueError("No active stage to finalize")
            
        return self._finalize_current_stage()
        
    def get_all_stage_stats(self) -> Dict[str, ProcessingStats]:
        """
        Get statistics for all completed stages.
        
        Returns:
            Dictionary mapping stage names to their ProcessingStats
        """
        return self.stage_history.copy()
        
    def _finalize_current_stage(self) -> ProcessingStats:
        """Finalize the current stage and store its statistics."""
        if not self.current_stage or not self.stage_start_time:
            raise ValueError("No active stage to finalize")
            
        elapsed_time = time.time() - self.stage_start_time
        items_per_second = self.current_progress / elapsed_time if elapsed_time > 0 else 0.0
        memory_stats = self.monitor_memory_usage()
        
        # Create final stats
        final_stats = ProcessingStats(
            stage=self.current_stage,
            start_time=self.stage_start_time,
            current_progress=self.current_progress,
            total_items=self.total_items,
            items_per_second=items_per_second,
            memory_usage_mb=memory_stats['current_mb'],
            peak_memory_mb=self.peak_memory_mb,
            errors_count=self.errors_count,
            warnings_count=self.warnings_count
        )
        
        # Store in history
        self.stage_history[self.current_stage] = final_stats
        
        # Log final stats
        self.logger.info(
            f"Completed stage '{self.current_stage}': "
            f"{self.current_progress}/{self.total_items} items "
            f"in {elapsed_time:.1f}s ({items_per_second:.1f} items/s)"
        )
        
        # Reset current stage
        self.current_stage = None
        self.stage_start_time = None
        self.current_progress = 0
        self.total_items = 0
        self.errors_count = 0
        self.warnings_count = 0
        
        return final_stats
        
    def _log_current_progress(self) -> None:
        """Log current progress information."""
        if not self.current_stage:
            return
            
        elapsed_time = time.time() - (self.stage_start_time or time.time())
        items_per_second = self.current_progress / elapsed_time if elapsed_time > 0 else 0.0
        completion_percent = (self.current_progress / self.total_items * 100) if self.total_items > 0 else 0.0
        
        self.logger.info(
            f"Stage '{self.current_stage}': "
            f"{self.current_progress}/{self.total_items} "
            f"({completion_percent:.1f}%) - "
            f"{items_per_second:.1f} items/s"
        )