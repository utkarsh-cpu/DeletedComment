#!/usr/bin/env python3
"""
Main orchestration script for Reddit deleted comment dataset processing.
This script coordinates all components in the processing pipeline.
"""

import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from config_loader import ConfigLoader
from data_downloader import DataDownloader
from reddit_parser import RedditParser
from comment_classifier import CommentClassifier
from metadata_extractor import MetadataExtractor
from parquet_writer import ParquetWriter
from drive_uploader import DriveUploader
from progress_monitor import ProgressMonitor
from cleanup_manager import CleanupManager
from logger import initialize_logging, get_logger, ComponentType
from error_handler import initialize_error_handling, get_error_manager


class PipelineOrchestrator:
    """Orchestrates the entire Reddit deleted comment dataset processing pipeline."""
    
    def __init__(self, config: ConfigLoader):
        """Initialize the pipeline orchestrator with configuration."""
        self.config = config
        
        # Initialize logging and error handling systems
        self.logger_system = get_logger()
        self.error_manager = get_error_manager()
        
        # Initialize components
        self.downloader = DataDownloader(
            download_dir=config.get('data_source.download_path', './data/raw'),
            max_retries=config.get('error_handling.retry_attempts', 3)
        )
        
        self.parser = RedditParser(
            chunk_size=config.get('processing.chunk_size', 100000),
            strict_validation=False
        )
        
        self.classifier = CommentClassifier(config.get_all())
        self.metadata_extractor = MetadataExtractor(config.get_all())
        self.parquet_writer = ParquetWriter(config.get_all())
        self.drive_uploader = DriveUploader(
            credentials_path=config.get('google_drive.credentials_path'),
            token_path=config.get('google_drive.token_path', 'token.json')
        )
        
        self.progress_monitor = ProgressMonitor(
            log_interval=config.get('logging.progress_interval', 10)
        )
        
        self.cleanup_manager = CleanupManager(
            min_free_space_gb=config.get('resources.disk_space_threshold_gb', 5.0),
            auto_cleanup=config.get('storage.auto_cleanup', True)
        )
        
        # Pipeline state
        self.checkpoint_data = {}
        self.processing_stats = {}
    
    def run_pipeline(self, resume_from_checkpoint: bool = False) -> bool:
        """
        Run the complete processing pipeline.
        
        Args:
            resume_from_checkpoint: Whether to resume from a previous checkpoint
            
        Returns:
            bool: True if pipeline completed successfully, False otherwise
        """
        with self.error_manager.error_context(
            ComponentType.MAIN_PIPELINE, 
            "complete_pipeline"
        ):
            self.logger_system.log_info(
                ComponentType.MAIN_PIPELINE,
                "Starting Reddit deleted comment dataset processing pipeline"
            )
            
            # Log system resources at start
            self.logger_system.log_system_resources(ComponentType.MAIN_PIPELINE)
            
            pipeline_context = self.logger_system.start_operation(
                ComponentType.MAIN_PIPELINE, 
                "complete_pipeline"
            )
            
            try:
                # Stage 1: Download and extract data
                if not resume_from_checkpoint or 'download_complete' not in self.checkpoint_data:
                    if not self._download_and_extract_data():
                        return False
                    self.checkpoint_data['download_complete'] = True
                    self._save_checkpoint()
                
                # Stage 2: Parse and classify comments
                if not resume_from_checkpoint or 'classification_complete' not in self.checkpoint_data:
                    if not self._parse_and_classify_comments():
                        return False
                    self.checkpoint_data['classification_complete'] = True
                    self._save_checkpoint()
                
                # Stage 3: Create Parquet datasets
                if not resume_from_checkpoint or 'parquet_complete' not in self.checkpoint_data:
                    if not self._create_parquet_datasets():
                        return False
                    self.checkpoint_data['parquet_complete'] = True
                    self._save_checkpoint()
                
                # Stage 4: Upload to Google Drive
                if not resume_from_checkpoint or 'upload_complete' not in self.checkpoint_data:
                    if not self._upload_to_drive():
                        return False
                    self.checkpoint_data['upload_complete'] = True
                    self._save_checkpoint()
                
                # Stage 5: Cleanup
                if self.config.get('storage.auto_cleanup', True):
                    self._cleanup_files()
                
                # Pipeline completion
                total_records = sum(len(comments) for comments in self.processing_stats.values())
                stats = self.logger_system.end_operation(
                    pipeline_context, 
                    records_processed=total_records,
                    success=True
                )
                
                self.logger_system.log_info(
                    ComponentType.MAIN_PIPELINE,
                    f"Pipeline completed successfully in {stats.duration_seconds:.2f} seconds"
                )
                self._log_final_stats()
                
                return True
                
            except Exception as e:
                self.logger_system.end_operation(
                    pipeline_context,
                    success=False,
                    error_message=str(e)
                )
                self.logger_system.log_error(
                    ComponentType.MAIN_PIPELINE,
                    f"Pipeline failed: {e}",
                    error=e,
                    recovery_action="pipeline_aborted"
                )
                return False
    
    def _download_and_extract_data(self) -> bool:
        """Download and extract Reddit data."""
        with self.error_manager.error_context(
            ComponentType.DATA_DOWNLOADER,
            "download_and_extract"
        ):
            self.progress_monitor.start_stage("download", 1)
            self.progress_monitor.update_progress(0, 1)
            self.logger_system.log_info(
                ComponentType.DATA_DOWNLOADER,
                "Starting data download and extraction"
            )
            
            # torrent_url = self.config.get_required('data_source.torrent_url')
            magnet_link = self.config.get_required('data_source.magnet_link')
            
            # Download dataset
            # downloaded_file = self.downloader.download_dataset(torrent_url)
            downloaded_file = self.downloader.download_from_torrent(magnet_link)
            
            # Extract files
            extracted_files = self.downloader.extract_files(downloaded_file)
            self.checkpoint_data['extracted_files'] = extracted_files
            
            self.progress_monitor.update_progress("download", 1, 1)
            self.logger_system.log_info(
                ComponentType.DATA_DOWNLOADER,
                f"Downloaded and extracted {len(extracted_files)} files",
                file_count=len(extracted_files)
            )
            
            return True
    
    def _parse_and_classify_comments(self) -> bool:
        """Parse Reddit data and classify comments."""
        with self.error_manager.error_context(
            ComponentType.REDDIT_PARSER,
            "parse_and_classify"
        ):
            self.logger_system.log_info(
                ComponentType.REDDIT_PARSER,
                "Starting comment parsing and classification"
            )
            
            extracted_files = self.checkpoint_data.get('extracted_files', [])
            
            if not extracted_files:
                self.logger_system.log_error(
                    ComponentType.REDDIT_PARSER,
                    "No extracted files found for processing",
                    recovery_action="abort_processing"
                )
                return False
            
            # Process each file
            total_files = len(extracted_files)
            total_comments_processed = 0
            
            for i, file_path in enumerate(extracted_files):
                self.progress_monitor.update_progress("parsing", i, total_files)
                
                file_context = self.logger_system.start_operation(
                    ComponentType.REDDIT_PARSER,
                    f"parse_file_{i}"
                )
                
                try:
                    # Parse comments from file
                    file_comments = 0
                    for comment_batch in self.parser.parse_comments_file(file_path):
                        # Classify each comment in the batch
                        for comment in comment_batch:
                            classification = self.classifier.classify_comment(comment)
                            
                            # Store classified comments for later processing
                            if classification.removal_type.value in ['user_deleted', 'moderator_removed']:
                                self._store_classified_comment(comment, classification)
                            
                            file_comments += 1
                            total_comments_processed += 1
                    
                    self.logger_system.end_operation(
                        file_context,
                        records_processed=file_comments,
                        success=True
                    )
                    
                    # Log memory usage periodically
                    if i % 5 == 0:
                        self.logger_system.log_memory_usage(
                            ComponentType.REDDIT_PARSER,
                            f"After processing file {i+1}/{total_files}"
                        )
                
                except Exception as e:
                    self.logger_system.end_operation(
                        file_context,
                        success=False,
                        error_message=str(e)
                    )
                    # Continue with next file if continue_on_error is enabled
                    if self.config.get('error_handling.continue_on_error', True):
                        continue
                    else:
                        raise
            
            self.progress_monitor.update_progress("parsing", total_files, total_files)
            self.logger_system.log_info(
                ComponentType.REDDIT_PARSER,
                "Comment parsing and classification completed",
                total_comments=total_comments_processed,
                files_processed=total_files
            )
            
            return True
    
    def _create_parquet_datasets(self) -> bool:
        """Create compressed Parquet datasets."""
        try:
            self.logger.info("Creating Parquet datasets")
            
            # Create datasets for each removal type
            removal_types = ['user_deleted', 'moderator_removed']
            
            for removal_type in removal_types:
                self.logger.info(f"Creating {removal_type} dataset")
                
                # Get classified comments for this removal type
                comments = self._get_classified_comments(removal_type)
                
                if not comments:
                    self.logger.warning(f"No {removal_type} comments found")
                    continue
                
                # Extract metadata and create training records
                training_records = []
                for comment in comments:
                    record = self.metadata_extractor.build_training_record(comment)
                    training_records.append(record)
                
                # Write to Parquet
                output_path = Path(self.config.get('storage.output_path', './data/processed'))
                parquet_file = output_path / f"{removal_type}_train.parquet"
                
                self.parquet_writer.write_dataset(
                    training_records,
                    str(parquet_file),
                    compression=self.config.get('storage.compression', 'snappy')
                )
                
                self.logger.info(f"Created {parquet_file} with {len(training_records)} records")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Parquet creation failed: {e}")
            return False
    
    def _upload_to_drive(self) -> bool:
        """Upload Parquet files to Google Drive."""
        try:
            self.logger.info("Uploading files to Google Drive")
            
            # Authenticate with Google Drive
            if not self.drive_uploader.authenticate():
                self.logger.error("Google Drive authentication failed")
                return False
            
            # Find Parquet files to upload
            output_path = Path(self.config.get('storage.output_path', './data/processed'))
            parquet_files = list(output_path.glob("*.parquet"))
            
            if not parquet_files:
                self.logger.warning("No Parquet files found for upload")
                return True
            
            # Create folder on Drive
            folder_name = self.config.get('google_drive.folder_name', 'Reddit_Deleted_Comments')
            folder_id = self.drive_uploader.create_folder(folder_name)
            
            # Upload each file
            for i, file_path in enumerate(parquet_files):
                self.progress_monitor.update_progress("upload", i, len(parquet_files))
                
                upload_result = self.drive_uploader.upload_file(str(file_path), folder_id)
                if upload_result:
                    self.logger.info(f"Uploaded {file_path.name} to Google Drive")
                else:
                    self.logger.error(f"Failed to upload {file_path.name}")
                    return False
            
            self.progress_monitor.update_progress("upload", len(parquet_files), len(parquet_files))
            self.logger.info("All files uploaded successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Upload failed: {e}")
            return False
    
    def _cleanup_files(self) -> None:
        """Clean up temporary and processed files."""
        try:
            self.logger.info("Starting cleanup process")
            
            # Get files to clean up
            cleanup_paths = []
            
            # Add extracted files
            if 'extracted_files' in self.checkpoint_data:
                cleanup_paths.extend(self.checkpoint_data['extracted_files'])
            
            # Add processed Parquet files if upload was successful
            if 'upload_complete' in self.checkpoint_data:
                output_path = Path(self.config.get('storage.output_path', './data/processed'))
                cleanup_paths.extend(str(f) for f in output_path.glob("*.parquet"))
            
            # Perform cleanup
            if cleanup_paths:
                result = self.cleanup_manager.cleanup_local_files(cleanup_paths)
                if result.success:
                    self.logger.info(f"Cleanup completed: freed {result.space_freed_mb:.2f} MB")
                else:
                    self.logger.warning(f"Cleanup had errors: {result.errors}")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def _store_classified_comment(self, comment: Dict[str, Any], classification) -> None:
        """Store classified comment for later processing."""
        # This is a simplified implementation - in practice, you might use a database
        # or temporary files to store large amounts of classified comments
        removal_type = classification.removal_type.value
        
        if removal_type not in self.processing_stats:
            self.processing_stats[removal_type] = []
        
        comment['classification'] = classification
        self.processing_stats[removal_type].append(comment)
    
    def _get_classified_comments(self, removal_type: str) -> list:
        """Get classified comments for a specific removal type."""
        return self.processing_stats.get(removal_type, [])
    
    def _save_checkpoint(self) -> None:
        """Save checkpoint data for recovery."""
        if self.config.get('error_handling.checkpoint_enabled', True):
            checkpoint_file = Path('./data/checkpoint.json')
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(checkpoint_file, 'w') as f:
                json.dump(self.checkpoint_data, f, indent=2)
    
    def _load_checkpoint(self) -> bool:
        """Load checkpoint data for recovery."""
        checkpoint_file = Path('./data/checkpoint.json')
        if checkpoint_file.exists():
            try:
                import json
                with open(checkpoint_file, 'r') as f:
                    self.checkpoint_data = json.load(f)
                return True
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
        return False
    
    def _log_final_stats(self) -> None:
        """Log final processing statistics."""
        stats = {
            'user_deleted_count': len(self.processing_stats.get('user_deleted', [])),
            'moderator_removed_count': len(self.processing_stats.get('moderator_removed', [])),
            'memory_stats': self.progress_monitor.get_memory_stats(),
            'processing_time': self.progress_monitor.get_total_processing_time()
        }
        
        self.logger.info(f"Final processing statistics: {stats}")


def setup_logging_and_error_handling(config: ConfigLoader) -> None:
    """Set up comprehensive logging and error handling systems."""
    # Initialize logging system
    logging_config = config.get('logging', {})
    initialize_logging(logging_config)
    
    # Initialize error handling system
    error_config = config.get('error_handling', {})
    initialize_error_handling(error_config)
    
    # Also set up basic Python logging for compatibility
    log_level = config.get('logging.level', 'INFO')
    log_file = config.get('logging.file_path', './logs/processing.log')
    
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure basic logging for any components not using the new system
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if config.get('logging.console_output', True) else logging.NullHandler()
        ]
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Reddit Deleted Comment Dataset Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config.yaml
  python main.py --resume --config custom_config.yaml
  python main.py --validate-config
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume from previous checkpoint if available'
    )
    
    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration and exit'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Override log level from configuration'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without actual processing'
    )
    
    return parser


def main():
    """Main entry point for the Reddit deleted comment dataset processor."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = ConfigLoader(args.config)
        
        # Override log level if specified
        if args.log_level:
            config.update({'logging': {'level': args.log_level}})
        
        # Validate configuration
        config.validate_config()
        
        if args.validate_config:
            print("Configuration is valid!")
            return 0
        
        # Set up logging and error handling
        setup_logging_and_error_handling(config)
        logger_system = get_logger()
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Reddit deleted comment dataset processing...")
        logger.info(f"Configuration loaded from: {config.config_path}")
        
        if args.dry_run:
            logger.info("Dry run mode - no actual processing will be performed")
            return 0
        
        # Create and run pipeline orchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Load checkpoint if resuming
        if args.resume:
            if orchestrator._load_checkpoint():
                logger.info("Resuming from previous checkpoint")
            else:
                logger.info("No checkpoint found, starting fresh")
        
        # Run the pipeline
        success = orchestrator.run_pipeline(resume_from_checkpoint=args.resume)
        
        if success:
            logger.info("Pipeline completed successfully!")
            return 0
        else:
            logger.error("Pipeline failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if '--log-level' in sys.argv and 'DEBUG' in sys.argv:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())