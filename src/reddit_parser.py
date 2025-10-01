"""
Reddit data parser component for processing Reddit JSON data files efficiently.
Handles streaming large JSON files with memory-efficient chunk processing.
Includes robust error handling and data validation.
"""

import json
import logging
import gzip
import bz2
import re
from pathlib import Path
from typing import Dict, Iterator, List, Any, Optional, Union, Set
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ParsingStats:
    """Statistics for parsing operations."""
    total_lines: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    json_errors: int = 0
    validation_errors: int = 0
    processing_time: float = 0.0
    
    
class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class RedditParser:
    """Efficient parser for Reddit JSON data files with streaming capabilities."""
    
    def __init__(self, chunk_size: int = 10000, strict_validation: bool = False):
        """
        Initialize Reddit parser.
        
        Args:
            chunk_size: Number of records to process in each chunk
            strict_validation: If True, raises exceptions on validation errors
        """
        self.chunk_size = chunk_size
        self.strict_validation = strict_validation
        self.logger = logging.getLogger(__name__)
        self.stats = ParsingStats()
        
        # Validation patterns
        self.id_pattern = re.compile(r'^[a-z0-9]+$')
        self.subreddit_pattern = re.compile(r'^[A-Za-z0-9_]+$')
        
        # Required fields for validation
        self.required_comment_fields = {'id', 'body', 'author', 'subreddit', 'created_utc'}
        self.required_submission_fields = {'id', 'title', 'author', 'subreddit', 'created_utc'}
        
    def parse_comments_file(self, file_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """
        Stream comment data from Reddit JSON file efficiently with robust error handling.
        
        Args:
            file_path: Path to the Reddit comments JSON file
            
        Yields:
            Dict: Individual comment record
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValidationError: If strict_validation is True and validation fails
            
        Requirements: 5.2, 5.3, 5.4
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Comments file not found: {file_path}")
            
        self.logger.info(f"Starting to parse comments file: {file_path}")
        self._reset_stats()
        
        try:
            with self._open_file(file_path) as file:
                for line_num, line in enumerate(file, 1):
                    self.stats.total_lines += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                        
                    try:
                        comment = self._parse_json_line(line, line_num, file_path)
                        if comment is None:
                            continue
                            
                        # Validate comment structure and data quality
                        validation_result = self._validate_comment(comment, line_num, file_path)
                        if not validation_result.is_valid:
                            if self.strict_validation:
                                raise ValidationError(f"Comment validation failed at line {line_num}: {validation_result.error}")
                            self.stats.validation_errors += 1
                            self.logger.debug(f"Skipping invalid comment at line {line_num}: {validation_result.error}")
                            continue
                            
                        # Normalize and yield valid comment
                        normalized_comment = self._normalize_comment(comment)
                        self.stats.valid_records += 1
                        yield normalized_comment
                        
                    except json.JSONDecodeError as e:
                        self.stats.json_errors += 1
                        self._log_parsing_error("JSON decode error", line_num, file_path, str(e))
                        if self.strict_validation:
                            raise
                        continue
                    except Exception as e:
                        self._log_parsing_error("Unexpected error", line_num, file_path, str(e))
                        if self.strict_validation:
                            raise
                        continue
                        
                    # Progress logging
                    if line_num % 10000 == 0:
                        self.logger.info(f"Processed {line_num} lines, {self.stats.valid_records} valid comments")
                        
        except Exception as e:
            self.logger.error(f"Critical error parsing comments file {file_path}: {e}")
            raise
        finally:
            self._log_final_stats(file_path)
            
    def parse_submissions_file(self, file_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """
        Stream submission data from Reddit JSON file efficiently with robust error handling.
        
        Args:
            file_path: Path to the Reddit submissions JSON file
            
        Yields:
            Dict: Individual submission record
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValidationError: If strict_validation is True and validation fails
            
        Requirements: 5.2, 5.3, 5.4
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Submissions file not found: {file_path}")
            
        self.logger.info(f"Starting to parse submissions file: {file_path}")
        self._reset_stats()
        
        try:
            with self._open_file(file_path) as file:
                for line_num, line in enumerate(file, 1):
                    self.stats.total_lines += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                        
                    try:
                        submission = self._parse_json_line(line, line_num, file_path)
                        if submission is None:
                            continue
                            
                        # Validate submission structure and data quality
                        validation_result = self._validate_submission(submission, line_num, file_path)
                        if not validation_result.is_valid:
                            if self.strict_validation:
                                raise ValidationError(f"Submission validation failed at line {line_num}: {validation_result.error}")
                            self.stats.validation_errors += 1
                            self.logger.debug(f"Skipping invalid submission at line {line_num}: {validation_result.error}")
                            continue
                            
                        # Normalize and yield valid submission
                        normalized_submission = self._normalize_submission(submission)
                        self.stats.valid_records += 1
                        yield normalized_submission
                        
                    except json.JSONDecodeError as e:
                        self.stats.json_errors += 1
                        self._log_parsing_error("JSON decode error", line_num, file_path, str(e))
                        if self.strict_validation:
                            raise
                        continue
                    except Exception as e:
                        self._log_parsing_error("Unexpected error", line_num, file_path, str(e))
                        if self.strict_validation:
                            raise
                        continue
                        
                    # Progress logging
                    if line_num % 10000 == 0:
                        self.logger.info(f"Processed {line_num} lines, {self.stats.valid_records} valid submissions")
                        
        except Exception as e:
            self.logger.error(f"Critical error parsing submissions file {file_path}: {e}")
            raise
        finally:
            self._log_final_stats(file_path)
            
    def chunk_processor(self, file_path: Union[str, Path], chunk_size: Optional[int] = None) -> Iterator[List[Dict[str, Any]]]:
        """
        Process Reddit data in memory-efficient chunks to handle large files.
        
        Args:
            file_path: Path to the Reddit JSON file
            chunk_size: Number of records per chunk (uses instance default if None)
            
        Yields:
            List[Dict]: Chunk of records
            
        Requirements: 5.2, 5.3
        """
        chunk_size = chunk_size or self.chunk_size
        file_path = Path(file_path)
        
        self.logger.info(f"Processing file in chunks of {chunk_size}: {file_path}")
        
        # Determine file type and use appropriate parser
        if self._is_comments_file(file_path):
            parser = self.parse_comments_file
        elif self._is_submissions_file(file_path):
            parser = self.parse_submissions_file
        else:
            # Default to comments parser
            parser = self.parse_comments_file
            
        chunk = []
        total_processed = 0
        
        try:
            for record in parser(file_path):
                chunk.append(record)
                
                if len(chunk) >= chunk_size:
                    total_processed += len(chunk)
                    self.logger.debug(f"Yielding chunk of {len(chunk)} records (total: {total_processed})")
                    yield chunk
                    chunk = []
                    
            # Yield remaining records
            if chunk:
                total_processed += len(chunk)
                self.logger.debug(f"Yielding final chunk of {len(chunk)} records (total: {total_processed})")
                yield chunk
                
            self.logger.info(f"Completed processing {total_processed} records from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error in chunk processing for {file_path}: {e}")
            raise
            
    def _open_file(self, file_path: Path):
        """Open file with appropriate decompression based on extension."""
        if file_path.suffix == '.gz':
            return gzip.open(file_path, 'rt', encoding='utf-8')
        elif file_path.suffix == '.bz2':
            return bz2.open(file_path, 'rt', encoding='utf-8')
        else:
            return open(file_path, 'r', encoding='utf-8')
            
    def _is_valid_comment(self, record: Dict[str, Any]) -> bool:
        """Basic validation that a record is a comment (legacy method)."""
        return self.required_comment_fields.issubset(set(record.keys()))
        
    def _is_valid_submission(self, record: Dict[str, Any]) -> bool:
        """Basic validation that a record is a submission (legacy method)."""
        return self.required_submission_fields.issubset(set(record.keys()))
        
    def _normalize_comment(self, comment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize comment data to consistent format with proper null handling.
        
        Requirements: 5.3, 5.4
        """
        return {
            'id': self._safe_str(comment.get('id', '')),
            'body': self._safe_str(comment.get('body', '')),
            'author': self._safe_str(comment.get('author', '')),
            'subreddit': self._safe_str(comment.get('subreddit', '')),
            'created_utc': self._safe_int(comment.get('created_utc', 0)),
            'parent_id': self._safe_str(comment.get('parent_id', '')),
            'link_id': self._safe_str(comment.get('link_id', '')),
            'score': self._safe_int(comment.get('score', 0)),
            'controversiality': self._safe_int(comment.get('controversiality', 0)),
            'gilded': self._safe_int(comment.get('gilded', 0)),
            'distinguished': comment.get('distinguished'),  # Can be None
            'stickied': self._safe_bool(comment.get('stickied', False)),
            'archived': self._safe_bool(comment.get('archived', False)),
            'edited': comment.get('edited', False)  # Can be False or timestamp
        }
        
    def _normalize_submission(self, submission: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize submission data to consistent format with proper null handling.
        
        Requirements: 5.3, 5.4
        """
        return {
            'id': self._safe_str(submission.get('id', '')),
            'title': self._safe_str(submission.get('title', '')),
            'selftext': self._safe_str(submission.get('selftext', '')),
            'author': self._safe_str(submission.get('author', '')),
            'subreddit': self._safe_str(submission.get('subreddit', '')),
            'created_utc': self._safe_int(submission.get('created_utc', 0)),
            'score': self._safe_int(submission.get('score', 0)),
            'num_comments': self._safe_int(submission.get('num_comments', 0)),
            'url': self._safe_str(submission.get('url', '')),
            'domain': self._safe_str(submission.get('domain', '')),
            'is_self': self._safe_bool(submission.get('is_self', False)),
            'over_18': self._safe_bool(submission.get('over_18', False)),
            'spoiler': self._safe_bool(submission.get('spoiler', False)),
            'locked': self._safe_bool(submission.get('locked', False)),
            'stickied': self._safe_bool(submission.get('stickied', False)),
            'archived': self._safe_bool(submission.get('archived', False))
        }
        
    def _is_comments_file(self, file_path: Path) -> bool:
        """Determine if file contains comments based on filename."""
        filename = file_path.name.lower()
        return 'comment' in filename or 'rc_' in filename
        
    def _is_submissions_file(self, file_path: Path) -> bool:
        """Determine if file contains submissions based on filename."""
        filename = file_path.name.lower()
        return 'submission' in filename or 'rs_' in filename
        
    def _reset_stats(self) -> None:
        """Reset parsing statistics for new file."""
        self.stats = ParsingStats()
        
    def _parse_json_line(self, line: str, line_num: int, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse a single JSON line with error handling.
        
        Args:
            line: JSON line to parse
            line_num: Line number for error reporting
            file_path: File path for error reporting
            
        Returns:
            Parsed JSON object or None if parsing fails
        """
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            self.stats.json_errors += 1
            self._log_parsing_error("JSON decode error", line_num, file_path, str(e))
            return None
            
    def _validate_comment(self, comment: Dict[str, Any], line_num: int, file_path: Path) -> 'ValidationResult':
        """
        Comprehensive validation for comment records.
        
        Args:
            comment: Comment record to validate
            line_num: Line number for error reporting
            file_path: File path for error reporting
            
        Returns:
            ValidationResult with validation status and error details
        """
        # Check required fields
        missing_fields = self.required_comment_fields - set(comment.keys())
        if missing_fields:
            return ValidationResult(False, f"Missing required fields: {missing_fields}")
            
        # Validate field types and values
        try:
            # ID validation
            comment_id = str(comment.get('id', ''))
            if not comment_id or not self.id_pattern.match(comment_id):
                return ValidationResult(False, f"Invalid comment ID: {comment_id}")
                
            # Subreddit validation
            subreddit = str(comment.get('subreddit', ''))
            if not subreddit:
                return ValidationResult(False, "Empty subreddit")
                
            # Timestamp validation
            created_utc = comment.get('created_utc')
            if not isinstance(created_utc, (int, float)) or created_utc <= 0:
                return ValidationResult(False, f"Invalid timestamp: {created_utc}")
                
            # Author validation (allow [deleted] and [removed])
            author = str(comment.get('author', ''))
            if not author:
                return ValidationResult(False, "Empty author")
                
            # Body validation (allow empty for deleted comments)
            body = comment.get('body')
            if body is None:
                return ValidationResult(False, "Missing body field")
                
            return ValidationResult(True, "")
            
        except Exception as e:
            return ValidationResult(False, f"Validation error: {str(e)}")
            
    def _validate_submission(self, submission: Dict[str, Any], line_num: int, file_path: Path) -> 'ValidationResult':
        """
        Comprehensive validation for submission records.
        
        Args:
            submission: Submission record to validate
            line_num: Line number for error reporting
            file_path: File path for error reporting
            
        Returns:
            ValidationResult with validation status and error details
        """
        # Check required fields
        missing_fields = self.required_submission_fields - set(submission.keys())
        if missing_fields:
            return ValidationResult(False, f"Missing required fields: {missing_fields}")
            
        # Validate field types and values
        try:
            # ID validation
            submission_id = str(submission.get('id', ''))
            if not submission_id or not self.id_pattern.match(submission_id):
                return ValidationResult(False, f"Invalid submission ID: {submission_id}")
                
            # Subreddit validation
            subreddit = str(submission.get('subreddit', ''))
            if not subreddit:
                return ValidationResult(False, "Empty subreddit")
                
            # Timestamp validation
            created_utc = submission.get('created_utc')
            if not isinstance(created_utc, (int, float)) or created_utc <= 0:
                return ValidationResult(False, f"Invalid timestamp: {created_utc}")
                
            # Author validation
            author = str(submission.get('author', ''))
            if not author:
                return ValidationResult(False, "Empty author")
                
            # Title validation
            title = submission.get('title')
            if not title or not isinstance(title, str):
                return ValidationResult(False, "Invalid or missing title")
                
            return ValidationResult(True, "")
            
        except Exception as e:
            return ValidationResult(False, f"Validation error: {str(e)}")
            
    def _log_parsing_error(self, error_type: str, line_num: int, file_path: Path, error_msg: str) -> None:
        """Log parsing errors with structured format."""
        self.logger.warning(
            f"{error_type} at line {line_num} in {file_path.name}: {error_msg}"
        )
        
    def _log_final_stats(self, file_path: Path) -> None:
        """Log final parsing statistics."""
        self.logger.info(
            f"Parsing complete for {file_path.name}: "
            f"{self.stats.valid_records} valid records, "
            f"{self.stats.json_errors} JSON errors, "
            f"{self.stats.validation_errors} validation errors, "
            f"total lines: {self.stats.total_lines}"
        )
        
    def get_parsing_stats(self) -> ParsingStats:
        """Get current parsing statistics."""
        return self.stats
        
    def _safe_str(self, value: Any) -> str:
        """Safely convert value to string, handling None and other types."""
        if value is None:
            return ''
        return str(value)
        
    def _safe_int(self, value: Any) -> int:
        """Safely convert value to int, handling None and invalid types."""
        if value is None:
            return 0
        try:
            return int(float(value))  # Handle string numbers
        except (ValueError, TypeError):
            return 0
            
    def _safe_bool(self, value: Any) -> bool:
        """Safely convert value to bool, handling None and various types."""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        try:
            return bool(int(value))
        except (ValueError, TypeError):
            return False


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    error: str