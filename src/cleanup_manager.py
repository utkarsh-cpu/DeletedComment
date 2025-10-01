"""
Storage management and cleanup utilities for Reddit dataset processing.

This module provides functionality to manage local storage, monitor disk space,
and clean up files after successful processing to prevent storage issues.
"""

import os
import shutil
import logging
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class StorageInfo:
    """Information about storage usage and availability."""
    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    files_deleted: List[str]
    space_freed_mb: float
    errors: List[str]
    success: bool


class CleanupManager:
    """
    Manages local storage and file cleanup operations.
    
    Provides methods to monitor disk space, clean up processed files,
    and manage storage to prevent disk space issues during processing.
    """
    
    def __init__(self, min_free_space_gb: float = 5.0, auto_cleanup: bool = True):
        """
        Initialize the cleanup manager.
        
        Args:
            min_free_space_gb: Minimum free space to maintain (in GB)
            auto_cleanup: Whether to automatically clean up files
        """
        self.min_free_space_gb = min_free_space_gb
        self.auto_cleanup = auto_cleanup
        self.logger = logging.getLogger(__name__)
        
        # Track files for cleanup
        self.pending_cleanup: List[str] = []
        self.protected_files: List[str] = []
        
    def check_disk_space(self, path: str = ".") -> StorageInfo:
        """
        Check available disk space for the given path.
        
        Args:
            path: Path to check disk space for (defaults to current directory)
            
        Returns:
            StorageInfo object with disk space details
        """
        try:
            # Get disk usage statistics
            total, used, free = shutil.disk_usage(path)
            
            # Convert to GB
            total_gb = total / (1024**3)
            used_gb = used / (1024**3)
            available_gb = free / (1024**3)
            usage_percent = (used / total) * 100
            
            storage_info = StorageInfo(
                total_gb=total_gb,
                used_gb=used_gb,
                available_gb=available_gb,
                usage_percent=usage_percent
            )
            
            # Log warning if space is low
            if available_gb < self.min_free_space_gb:
                self.logger.warning(
                    f"Low disk space: {available_gb:.1f} GB available "
                    f"(minimum: {self.min_free_space_gb} GB)"
                )
            
            return storage_info
            
        except Exception as e:
            self.logger.error(f"Error checking disk space: {e}")
            return StorageInfo(0.0, 0.0, 0.0, 0.0)
            
    def add_file_for_cleanup(self, file_path: str) -> None:
        """
        Add a file to the cleanup queue.
        
        Args:
            file_path: Path to file that can be cleaned up later
        """
        if os.path.exists(file_path) and file_path not in self.protected_files:
            self.pending_cleanup.append(file_path)
            self.logger.debug(f"Added file to cleanup queue: {file_path}")
            
    def protect_file(self, file_path: str) -> None:
        """
        Protect a file from automatic cleanup.
        
        Args:
            file_path: Path to file that should not be cleaned up
        """
        self.protected_files.append(file_path)
        # Remove from cleanup queue if present
        if file_path in self.pending_cleanup:
            self.pending_cleanup.remove(file_path)
        self.logger.debug(f"Protected file from cleanup: {file_path}")
        
    def cleanup_local_files(self, file_paths: Optional[List[str]] = None, 
                          force: bool = False) -> CleanupResult:
        """
        Clean up local files to free disk space.
        
        Args:
            file_paths: Specific files to clean up (uses pending_cleanup if None)
            force: Force cleanup even if auto_cleanup is disabled
            
        Returns:
            CleanupResult with details of the cleanup operation
        """
        if not self.auto_cleanup and not force:
            self.logger.info("Auto cleanup disabled, skipping cleanup")
            return CleanupResult([], 0.0, [], False)
            
        # Determine which files to clean up
        if file_paths is None:
            file_paths = self.pending_cleanup.copy()
        
        deleted_files = []
        total_space_freed = 0.0
        errors = []
        
        for file_path in file_paths:
            try:
                # Skip if file is protected
                if file_path in self.protected_files:
                    self.logger.debug(f"Skipping protected file: {file_path}")
                    continue
                    
                # Check if file exists
                if not os.path.exists(file_path):
                    self.logger.debug(f"File not found, skipping: {file_path}")
                    continue
                    
                # Get file size before deletion
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                # Delete the file
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    self.logger.warning(f"Unknown file type, skipping: {file_path}")
                    continue
                    
                deleted_files.append(file_path)
                total_space_freed += file_size_mb
                
                # Remove from pending cleanup
                if file_path in self.pending_cleanup:
                    self.pending_cleanup.remove(file_path)
                    
                self.logger.info(f"Deleted file: {file_path} ({file_size_mb:.1f} MB)")
                
            except Exception as e:
                error_msg = f"Error deleting {file_path}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                
        # Log summary
        if deleted_files:
            self.logger.info(
                f"Cleanup completed: {len(deleted_files)} files deleted, "
                f"{total_space_freed:.1f} MB freed"
            )
        
        return CleanupResult(
            files_deleted=deleted_files,
            space_freed_mb=total_space_freed,
            errors=errors,
            success=len(errors) == 0
        )
        
    def cleanup_by_pattern(self, directory: str, pattern: str, 
                          max_age_hours: Optional[float] = None) -> CleanupResult:
        """
        Clean up files matching a pattern in a directory.
        
        Args:
            directory: Directory to search for files
            pattern: File pattern to match (e.g., "*.tmp", "temp_*")
            max_age_hours: Only delete files older than this many hours
            
        Returns:
            CleanupResult with details of the cleanup operation
        """
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                return CleanupResult([], 0.0, [f"Directory not found: {directory}"], False)
                
            # Find matching files
            matching_files = list(directory_path.glob(pattern))
            files_to_delete = []
            
            # Filter by age if specified
            if max_age_hours is not None:
                current_time = time.time()
                max_age_seconds = max_age_hours * 3600
                
                for file_path in matching_files:
                    if file_path.is_file():
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > max_age_seconds:
                            files_to_delete.append(str(file_path))
            else:
                files_to_delete = [str(f) for f in matching_files if f.is_file()]
                
            # Clean up the files
            return self.cleanup_local_files(files_to_delete, force=True)
            
        except Exception as e:
            error_msg = f"Error in pattern cleanup: {e}"
            self.logger.error(error_msg)
            return CleanupResult([], 0.0, [error_msg], False)
            
    def ensure_free_space(self, required_gb: float, cleanup_dirs: Optional[List[str]] = None) -> bool:
        """
        Ensure minimum free space is available, cleaning up if necessary.
        
        Args:
            required_gb: Required free space in GB
            cleanup_dirs: Directories to clean up temporary files from
            
        Returns:
            True if sufficient space is available, False otherwise
        """
        storage_info = self.check_disk_space()
        
        if storage_info.available_gb >= required_gb:
            self.logger.debug(f"Sufficient space available: {storage_info.available_gb:.1f} GB")
            return True
            
        self.logger.warning(
            f"Insufficient space: {storage_info.available_gb:.1f} GB available, "
            f"{required_gb:.1f} GB required"
        )
        
        # Try cleanup pending files first
        if self.pending_cleanup:
            self.logger.info("Attempting cleanup of pending files...")
            cleanup_result = self.cleanup_local_files(force=True)
            
            # Check space again
            storage_info = self.check_disk_space()
            if storage_info.available_gb >= required_gb:
                self.logger.info(f"Cleanup successful, {storage_info.available_gb:.1f} GB available")
                return True
                
        # Try cleaning up temporary files in specified directories
        if cleanup_dirs:
            for cleanup_dir in cleanup_dirs:
                self.logger.info(f"Cleaning temporary files in {cleanup_dir}...")
                
                # Clean up common temporary file patterns
                temp_patterns = ["*.tmp", "*.temp", "temp_*", "*.partial", "*.download"]
                for pattern in temp_patterns:
                    self.cleanup_by_pattern(cleanup_dir, pattern, max_age_hours=24)
                    
                # Check space again
                storage_info = self.check_disk_space()
                if storage_info.available_gb >= required_gb:
                    self.logger.info(f"Cleanup successful, {storage_info.available_gb:.1f} GB available")
                    return True
                    
        # Final space check
        storage_info = self.check_disk_space()
        if storage_info.available_gb < required_gb:
            self.logger.error(
                f"Unable to free sufficient space: {storage_info.available_gb:.1f} GB available, "
                f"{required_gb:.1f} GB required"
            )
            return False
            
        return True
        
    def get_directory_size(self, directory: str) -> float:
        """
        Get the total size of a directory in MB.
        
        Args:
            directory: Path to directory
            
        Returns:
            Directory size in MB
        """
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
                        
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            self.logger.error(f"Error calculating directory size for {directory}: {e}")
            return 0.0
            
    def get_largest_files(self, directory: str, count: int = 10) -> List[Tuple[str, float]]:
        """
        Get the largest files in a directory.
        
        Args:
            directory: Directory to search
            count: Number of largest files to return
            
        Returns:
            List of tuples (file_path, size_mb) sorted by size descending
        """
        try:
            file_sizes = []
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        file_sizes.append((file_path, size_mb))
                        
            # Sort by size descending and return top N
            file_sizes.sort(key=lambda x: x[1], reverse=True)
            return file_sizes[:count]
            
        except Exception as e:
            self.logger.error(f"Error finding largest files in {directory}: {e}")
            return []
            
    def get_cleanup_summary(self) -> Dict[str, any]:
        """
        Get a summary of cleanup status and pending operations.
        
        Returns:
            Dictionary with cleanup summary information
        """
        storage_info = self.check_disk_space()
        
        return {
            'storage': {
                'total_gb': storage_info.total_gb,
                'used_gb': storage_info.used_gb,
                'available_gb': storage_info.available_gb,
                'usage_percent': storage_info.usage_percent
            },
            'cleanup': {
                'auto_cleanup_enabled': self.auto_cleanup,
                'min_free_space_gb': self.min_free_space_gb,
                'pending_cleanup_count': len(self.pending_cleanup),
                'protected_files_count': len(self.protected_files),
                'space_warning': storage_info.available_gb < self.min_free_space_gb
            },
            'pending_files': self.pending_cleanup.copy(),
            'protected_files': self.protected_files.copy()
        }