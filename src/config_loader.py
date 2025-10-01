"""
Configuration loader for Reddit deleted comment dataset processing.
Handles loading from config.yaml and environment variables.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Loads and manages configuration from YAML file and environment variables."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._apply_env_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Override sensitive data with environment variables
        env_mappings = {
            # Google Drive API credentials
            'GOOGLE_DRIVE_CREDENTIALS_PATH': ['google_drive', 'credentials_path'],
            'GOOGLE_DRIVE_TOKEN_PATH': ['google_drive', 'token_path'],
            'GOOGLE_DRIVE_FOLDER_ID': ['google_drive', 'folder_id'],
            
            # Data source configuration
            'REDDIT_DATA_URL': ['data_source', 'torrent_url'],
            'DOWNLOAD_PATH': ['data_source', 'download_path'],
            'EXTRACT_PATH': ['data_source', 'extract_path'],
            
            # Storage paths
            'OUTPUT_DIR': ['storage', 'output_path'],
            'TEMP_DIR': ['storage', 'temp_path'],
            
            # Processing configuration
            'CHUNK_SIZE': ['processing', 'chunk_size'],
            'MAX_WORKERS': ['processing', 'max_workers'],
            'MEMORY_LIMIT_GB': ['processing', 'memory_limit_gb'],
            
            # Logging configuration
            'LOG_LEVEL': ['logging', 'level'],
            'LOG_FILE': ['logging', 'file_path'],
            
            # Resource limits
            'DISK_SPACE_THRESHOLD_GB': ['resources', 'disk_space_threshold_gb'],
            'MAX_FILE_SIZE_GB': ['resources', 'max_file_size_gb'],
            
            # Development settings
            'DEBUG_MODE': ['development', 'debug_mode'],
            'SAMPLE_SIZE': ['development', 'sample_size']
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(env_value, config_path)
                self._set_nested_config(config_path, converted_value)
    
    def _set_nested_config(self, path: list, value: str) -> None:
        """Set a nested configuration value."""
        current = self.config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'google_drive.credentials_path')."""
        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_required(self, key_path: str) -> Any:
        """Get required configuration value, raise error if missing."""
        value = self.get(key_path)
        if value is None:
            raise ValueError(f"Required configuration missing: {key_path}")
        return value
    
    def _convert_env_value(self, value: str, config_path: list) -> Any:
        """Convert environment variable string to appropriate type based on config path."""
        # Boolean conversions
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversions for known numeric fields
        numeric_fields = {
            'chunk_size', 'max_workers', 'memory_limit_gb', 'batch_size',
            'upload_timeout', 'retry_attempts', 'chunk_upload_size',
            'max_size_mb', 'backup_count', 'checkpoint_interval',
            'disk_space_threshold_gb', 'max_file_size_gb', 'temp_cleanup_interval',
            'sample_size', 'max_concurrent_downloads', 'io_buffer_size',
            'pandas_chunksize', 'max_comment_length', 'min_comment_length'
        }
        
        if len(config_path) > 1 and config_path[-1] in numeric_fields:
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    pass
        
        # Float conversions for threshold fields
        float_fields = {
            'retry_delay', 'max_retry_delay', 'memory_warning_threshold',
            'memory_critical_threshold', 'confidence_threshold'
        }
        
        if len(config_path) > 1 and config_path[-1] in float_fields:
            try:
                return float(value)
            except ValueError:
                pass
        
        # Return as string if no conversion applies
        return value
    
    def validate_config(self) -> None:
        """Validate that all required configuration is present and valid."""
        # Required configuration keys
        required_keys = [
            # 'data_source.torrent_url',
            'data_source.magnet_link',
            'storage.output_path',
            'processing.chunk_size',
            'google_drive.credentials_path'
        ]
        
        # Check for missing required keys
        missing_keys = []
        for key in required_keys:
            value = self.get(key)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")
        
        # Validate data types and ranges
        self._validate_data_types()
        self._validate_ranges()
        self._validate_paths()
        self._validate_file_existence()
    
    def _validate_data_types(self) -> None:
        """Validate configuration data types."""
        type_validations = {
            'processing.chunk_size': int,
            'processing.max_workers': int,
            'processing.memory_limit_gb': (int, float),
            'storage.auto_cleanup': bool,
            'google_drive.retry_attempts': int,
            'logging.console_output': bool,
            'error_handling.checkpoint_enabled': bool,
            'resources.disk_space_threshold_gb': (int, float),
            'performance.use_multiprocessing': bool
        }
        
        for key, expected_type in type_validations.items():
            value = self.get(key)
            if value is not None and not isinstance(value, expected_type):
                raise ValueError(f"Configuration key '{key}' must be of type {expected_type.__name__}, got {type(value).__name__}")
    
    def _validate_ranges(self) -> None:
        """Validate configuration value ranges."""
        range_validations = {
            'processing.chunk_size': (1, 1000000),
            'processing.max_workers': (1, 32),
            'processing.memory_limit_gb': (1, 128),
            'google_drive.retry_attempts': (0, 10),
            'error_handling.retry_attempts': (0, 10),
            'resources.memory_warning_threshold': (0.0, 1.0),
            'resources.memory_critical_threshold': (0.0, 1.0),
            'classification.confidence_threshold': (0.0, 1.0)
        }
        
        for key, (min_val, max_val) in range_validations.items():
            value = self.get(key)
            if value is not None and not (min_val <= value <= max_val):
                raise ValueError(f"Configuration key '{key}' must be between {min_val} and {max_val}, got {value}")
    
    def _validate_paths(self) -> None:
        """Validate path configurations."""
        path_keys = [
            'data_source.download_path',
            'data_source.extract_path',
            'storage.output_path',
            'storage.temp_path',
            'logging.file_path'
        ]
        
        for key in path_keys:
            path_value = self.get(key)
            if path_value:
                path_obj = Path(path_value)
                # Create parent directories if they don't exist
                try:
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                except (OSError, PermissionError) as e:
                    raise ValueError(f"Cannot create directory for '{key}': {e}")
    
    def _validate_file_existence(self) -> None:
        """Validate that required files exist."""
        # Only validate credentials file if it's not empty
        credentials_path = self.get('google_drive.credentials_path')
        if credentials_path and credentials_path.strip():
            if not Path(credentials_path).exists():
                raise ValueError(f"Google Drive credentials file not found: {credentials_path}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of configuration validation status."""
        summary = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'required_keys_present': True,
            'file_paths_valid': True,
            'data_types_valid': True
        }
        
        try:
            self.validate_config()
        except ValueError as e:
            summary['valid'] = False
            summary['errors'].append(str(e))
        
        # Check for potential issues
        warnings = []
        
        # Check memory settings
        memory_limit = self.get('processing.memory_limit_gb', 8)
        if memory_limit > 16:
            warnings.append("Memory limit is set quite high - ensure system has sufficient RAM")
        
        # Check chunk size
        chunk_size = self.get('processing.chunk_size', 100000)
        if chunk_size > 500000:
            warnings.append("Large chunk size may cause memory issues")
        
        # Check disk space threshold
        disk_threshold = self.get('resources.disk_space_threshold_gb', 5)
        if disk_threshold < 2:
            warnings.append("Disk space threshold is very low - may cause processing failures")
        
        summary['warnings'] = warnings
        
        return summary
    
    def get_all(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self.config.copy()
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self.config.update(updates)
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to YAML file."""
        save_path = output_path or self.config_path
        
        try:
            with open(save_path, 'w') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to save configuration to {save_path}: {e}")
    
    def create_default_config(self, output_path: str = "config.yaml") -> None:
        """Create a default configuration file with all available options."""
        default_config = {
            'processing': {
                'chunk_size': 100000,
                'max_workers': 4,
                'memory_limit_gb': 8,
                'batch_size': 10000,
                'strict_validation': False,
                'parallel_processing': True
            },
            'data_source': {
                'torrent_url': '',
                'magnet_link': '',
                'download_path': './data/raw',
                'extract_path': './data/extracted',
                'verify_checksums': True,
                'checksum_algorithm': 'sha256',
                'download_timeout': 3600,
                'max_file_size_gb': 50
            },
            'storage': {
                'compression': 'snappy',
                'parquet_version': '2.6',
                'auto_cleanup': True,
                'backup_enabled': False,
                'output_path': './data/processed',
                'temp_path': './data/temp',
                'chunk_size_mb': 256,
                'row_group_size': 50000
            },
            'google_drive': {
                'credentials_path': './credentials.json',
                'token_path': './token.json',
                'folder_name': 'Reddit_Deleted_Comments',
                'upload_timeout': 300,
                'retry_attempts': 3,
                'chunk_upload_size': 10485760,
                'create_folder_structure': True,
                'share_files': False,
                'delete_after_upload': False
            },
            'classification': {
                'deleted_markers': ['[deleted]'],
                'removed_markers': ['[removed]'],
                'deleted_authors': ['[deleted]'],
                'confidence_threshold': 0.8,
                'include_context': True,
                'extract_removal_reason': True
            },
            'logging': {
                'level': 'INFO',
                'file_path': './logs/processing.log',
                'max_size_mb': 100,
                'backup_count': 5,
                'console_output': True,
                'progress_interval': 10,
                'detailed_errors': True,
                'log_memory_usage': True
            },
            'error_handling': {
                'retry_attempts': 3,
                'retry_delay': 1.0,
                'exponential_backoff': True,
                'max_retry_delay': 60.0,
                'continue_on_error': True,
                'checkpoint_enabled': True,
                'checkpoint_interval': 10000,
                'checkpoint_path': './data/checkpoint.json',
                'error_log_path': './logs/errors.log'
            },
            'resources': {
                'disk_space_threshold_gb': 5,
                'memory_warning_threshold': 0.8,
                'memory_critical_threshold': 0.95,
                'max_file_size_gb': 2,
                'temp_cleanup_interval': 3600,
                'monitor_system_resources': True,
                'gc_interval': 1000
            }
        }
        
        try:
            with open(output_path, 'w') as f:
                yaml.safe_dump(default_config, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to create default configuration at {output_path}: {e}")
    
    def get_environment_template(self) -> str:
        """Get a template for environment variables."""
        template = """# Reddit Deleted Comment Dataset Processing - Environment Variables
# Copy this to .env and fill in your values

# Google Drive API Configuration
GOOGLE_DRIVE_CREDENTIALS_PATH=./credentials.json
GOOGLE_DRIVE_TOKEN_PATH=./token.json
GOOGLE_DRIVE_FOLDER_ID=

# Data Source Configuration
REDDIT_DATA_URL=
DOWNLOAD_PATH=./data/raw
EXTRACT_PATH=./data/extracted

# Storage Configuration
OUTPUT_DIR=./data/processed
TEMP_DIR=./data/temp

# Processing Configuration
CHUNK_SIZE=100000
MAX_WORKERS=4
MEMORY_LIMIT_GB=8

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/processing.log

# Resource Management
DISK_SPACE_THRESHOLD_GB=5
MAX_FILE_SIZE_GB=2

# Development Settings (optional)
DEBUG_MODE=false
SAMPLE_SIZE=0
"""
        return template
    
    def export_env_template(self, output_path: str = ".env.example") -> None:
        """Export environment variable template to file."""
        template = self.get_environment_template()
        
        try:
            with open(output_path, 'w') as f:
                f.write(template)
        except Exception as e:
            raise ValueError(f"Failed to export environment template to {output_path}: {e}")
    
    def get_config_documentation(self) -> Dict[str, str]:
        """Get documentation for configuration options."""
        return {
            'processing.chunk_size': 'Number of records to process in each chunk (affects memory usage)',
            'processing.max_workers': 'Maximum number of worker threads for parallel processing',
            'processing.memory_limit_gb': 'Memory limit for processing operations in GB',
            'data_source.torrent_url': 'URL to Reddit dataset from Academic Torrents (required)',
            'data_source.verify_checksums': 'Whether to verify file integrity using checksums',
            'storage.compression': 'Parquet compression algorithm (snappy, gzip, lz4, brotli, zstd)',
            'storage.auto_cleanup': 'Automatically clean up temporary files after processing',
            'google_drive.credentials_path': 'Path to Google API credentials JSON file',
            'google_drive.folder_name': 'Name of folder to create on Google Drive',
            'classification.confidence_threshold': 'Minimum confidence score for comment classification',
            'logging.level': 'Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)',
            'error_handling.checkpoint_enabled': 'Enable checkpoint system for recovery from failures',
            'resources.disk_space_threshold_gb': 'Minimum free disk space required (GB)',
            'resources.memory_warning_threshold': 'Memory usage threshold for warnings (0.0-1.0)'
        }