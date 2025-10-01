"""
Metadata extractor component for extracting relevant metadata for ML training datasets.
Handles formatting data for train.parquet structure and ML label assignment.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingRecord:
    """Standardized training record structure."""
    id: str
    comment_text: str
    subreddit: str
    timestamp: datetime
    removal_type: str
    target_label: str
    parent_id: Optional[str] = None
    thread_id: Optional[str] = None
    score: Optional[int] = None
    author: Optional[str] = None
    controversiality: Optional[int] = None
    gilded: Optional[int] = None
    parent_context: Optional[str] = None


class MetadataExtractor:
    """Extracts and formats metadata for ML training datasets."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metadata extractor.
        
        Args:
            config: Configuration dictionary with extraction settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Load subreddit-specific rules
        self.subreddit_rules = self._load_subreddit_rules()
        
        # Common toxic/spam patterns for labeling
        self.toxic_patterns = self._load_toxic_patterns()
        
        # Statistics tracking
        self.stats = {
            'total_extracted': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'missing_metadata': 0,
            'label_assignments': {}
        }
        
    def extract_comment_metadata(self, comment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract core metadata from a Reddit comment for training purposes.
        
        Args:
            comment: Reddit comment dictionary
            
        Returns:
            Dict: Extracted metadata
            
        Requirements: 3.1, 3.2, 3.3
        """
        try:
            self.stats['total_extracted'] += 1
            
            # Extract basic metadata
            metadata = {
                'id': self._safe_extract(comment, 'id', str, ''),
                'body': self._safe_extract(comment, 'body', str, ''),
                'author': self._safe_extract(comment, 'author', str, ''),
                'subreddit': self._safe_extract(comment, 'subreddit', str, ''),
                'created_utc': self._safe_extract(comment, 'created_utc', (int, float), 0),
                'score': self._safe_extract(comment, 'score', int, 0),
                'parent_id': self._safe_extract(comment, 'parent_id', str, ''),
                'link_id': self._safe_extract(comment, 'link_id', str, ''),
                'controversiality': self._safe_extract(comment, 'controversiality', int, 0),
                'gilded': self._safe_extract(comment, 'gilded', int, 0),
                'distinguished': comment.get('distinguished'),
                'stickied': self._safe_extract(comment, 'stickied', bool, False),
                'archived': self._safe_extract(comment, 'archived', bool, False),
                'edited': comment.get('edited', False)
            }
            
            # Convert timestamp to datetime
            metadata['timestamp'] = self._convert_timestamp(metadata['created_utc'])
            
            # Extract thread ID from link_id
            metadata['thread_id'] = self._extract_thread_id(metadata['link_id'])
            
            # Clean parent ID
            metadata['parent_id'] = self._clean_parent_id(metadata['parent_id'])
            
            # Normalize subreddit name
            metadata['subreddit'] = self._normalize_subreddit(metadata['subreddit'])
            
            # Add derived metadata
            metadata['comment_length'] = len(metadata['body']) if metadata['body'] else 0
            metadata['has_parent'] = bool(metadata['parent_id'])
            metadata['is_top_level'] = metadata['parent_id'] == metadata['thread_id']
            
            self.stats['successful_extractions'] += 1
            return metadata
            
        except Exception as e:
            self.stats['failed_extractions'] += 1
            self.logger.error(f"Failed to extract metadata for comment {comment.get('id', 'unknown')}: {e}")
            return self._create_fallback_metadata(comment)
            
    def generate_training_labels(self, comment: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Generate ML training labels based on comment and removal context.
        
        Args:
            comment: Reddit comment dictionary
            context: Removal context from classifier
            
        Returns:
            str: Training label for ML
            
        Requirements: 3.2, 3.3
        """
        removal_type = context.get('removal_type', 'unknown')
        subreddit = comment.get('subreddit', '').lower()
        score = comment.get('score', 0)
        controversiality = comment.get('controversiality', 0)
        body = comment.get('body', '')
        
        # Generate label based on removal type and context
        if removal_type == 'moderator_removed':
            label = self._generate_moderation_label(comment, context, subreddit, score, controversiality, body)
        elif removal_type == 'user_deleted':
            label = self._generate_deletion_label(comment, context, subreddit, score, controversiality, body)
        else:
            label = 'unknown_removal'
            
        # Track label statistics
        if label not in self.stats['label_assignments']:
            self.stats['label_assignments'][label] = 0
        self.stats['label_assignments'][label] += 1
        
        return label
        
    def build_training_record(self, comment: Dict[str, Any], removal_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build a complete training record in the standardized format for train.parquet.
        
        Args:
            comment: Reddit comment dictionary
            removal_context: Optional removal context from classifier
            
        Returns:
            Dict: Complete training record
            
        Requirements: 3.1, 3.2, 3.3
        """
        # Extract core metadata
        metadata = self.extract_comment_metadata(comment)
        
        # Use provided context or create basic context
        context = removal_context or {'removal_type': 'active'}
        
        # Generate training label
        target_label = self.generate_training_labels(comment, context)
        
        # Build standardized training record
        training_record = {
            'id': metadata['id'],
            'comment_text': self._prepare_comment_text(metadata['body'], context),
            'subreddit': metadata['subreddit'],
            'timestamp': metadata['timestamp'],
            'removal_type': context.get('removal_type', 'unknown'),
            'target_label': target_label,
            'parent_id': metadata['parent_id'],
            'thread_id': metadata['thread_id'],
            'score': metadata['score'],
            'author': metadata['author'],
            'controversiality': metadata['controversiality'],
            'gilded': metadata['gilded'],
            'comment_length': metadata['comment_length'],
            'has_parent': metadata['has_parent'],
            'is_top_level': metadata['is_top_level']
        }
        
        return training_record
        
    def _safe_extract(self, data: Dict[str, Any], key: str, expected_type: Union[type, tuple], default: Any) -> Any:
        """Safely extract and convert data with type checking."""
        value = data.get(key, default)
        
        if value is None:
            return default
            
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                try:
                    # Try to convert to first type in tuple
                    return expected_type[0](value)
                except (ValueError, TypeError):
                    return default
        else:
            if not isinstance(value, expected_type):
                try:
                    return expected_type(value)
                except (ValueError, TypeError):
                    return default
                    
        return value
        
    def _convert_timestamp(self, created_utc: Union[int, float]) -> datetime:
        """Convert Unix timestamp to datetime object."""
        try:
            return datetime.fromtimestamp(created_utc, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            return datetime.fromtimestamp(0, tz=timezone.utc)
            
    def _extract_thread_id(self, link_id: str) -> str:
        """Extract clean thread ID from Reddit link_id format."""
        if isinstance(link_id, str) and link_id.startswith('t3_'):
            return link_id[3:]
        return link_id if isinstance(link_id, str) else ''
        
    def _clean_parent_id(self, parent_id: str) -> str:
        """Clean parent ID by removing Reddit prefixes."""
        if isinstance(parent_id, str):
            if parent_id.startswith('t1_'):
                return parent_id[3:]  # Comment parent
            elif parent_id.startswith('t3_'):
                return parent_id[3:]  # Post parent
        return parent_id if isinstance(parent_id, str) else ''
        
    def _normalize_subreddit(self, subreddit: str) -> str:
        """Normalize subreddit name to lowercase."""
        if isinstance(subreddit, str):
            return subreddit.lower().strip()
        return ''
        
    def _prepare_comment_text(self, body: str, context: Dict[str, Any]) -> str:
        """Prepare comment text for training, handling deletion markers."""
        if not isinstance(body, str):
            return '[CONTENT_UNAVAILABLE]'
            
        # Check for deletion markers
        deletion_markers = {'[deleted]', '[removed]'}
        if body.strip() in deletion_markers:
            return '[CONTENT_REMOVED]'
            
        # Return original text if available
        return body.strip()
        
    def _create_fallback_metadata(self, comment: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback metadata when extraction fails."""
        self.stats['missing_metadata'] += 1
        
        return {
            'id': comment.get('id', ''),
            'body': '',
            'author': '',
            'subreddit': '',
            'created_utc': 0,
            'timestamp': datetime.fromtimestamp(0, tz=timezone.utc),
            'score': 0,
            'parent_id': '',
            'thread_id': '',
            'link_id': '',
            'controversiality': 0,
            'gilded': 0,
            'distinguished': None,
            'stickied': False,
            'archived': False,
            'edited': False,
            'comment_length': 0,
            'has_parent': False,
            'is_top_level': False
        } 
       
    def _generate_moderation_label(self, comment: Dict[str, Any], context: Dict[str, Any], 
                                 subreddit: str, score: int, controversiality: int, body: str) -> str:
        """Generate labels for moderator-removed comments."""
        # Check subreddit-specific rules
        if subreddit in self.subreddit_rules:
            rule_label = self._apply_subreddit_rules(subreddit, comment, context)
            if rule_label:
                return rule_label
                
        # Check for toxic patterns
        if self._contains_toxic_patterns(body):
            return 'toxic_content'
            
        # Score-based labeling
        if score < -10:
            return 'heavily_downvoted'
        elif controversiality > 0:
            return 'controversial_content'
            
        # Subreddit category-based labeling
        if subreddit in ['askreddit', 'iama', 'explainlikeimfive']:
            return 'rule_violation'
        elif subreddit in ['science', 'askscience', 'history']:
            return 'misinformation'
        elif subreddit in ['news', 'worldnews', 'politics']:
            return 'policy_violation'
        else:
            return 'moderation_removal'
            
    def _generate_deletion_label(self, comment: Dict[str, Any], context: Dict[str, Any],
                               subreddit: str, score: int, controversiality: int, body: str) -> str:
        """Generate labels for user-deleted comments."""
        # Score-based patterns
        if score < -5:
            return 'regret_deletion'
        elif score > 10 and controversiality > 0:
            return 'controversial_deletion'
        elif score > 50:
            return 'high_visibility_deletion'
            
        # Time-based patterns (if we had edit history)
        # For now, use general categories
        if controversiality > 0:
            return 'controversial_deletion'
        else:
            return 'voluntary_deletion'
            
    def _load_subreddit_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load subreddit-specific labeling rules."""
        return {
            'askreddit': {
                'common_violations': ['off_topic', 'personal_story', 'yes_no_question'],
                'default_label': 'rule_violation'
            },
            'science': {
                'common_violations': ['pseudoscience', 'anecdotal', 'off_topic'],
                'default_label': 'misinformation'
            },
            'news': {
                'common_violations': ['editorialized', 'duplicate', 'opinion'],
                'default_label': 'policy_violation'
            },
            'politics': {
                'common_violations': ['incivility', 'off_topic', 'spam'],
                'default_label': 'policy_violation'
            }
        }
        
    def _load_toxic_patterns(self) -> List[str]:
        """Load patterns that indicate toxic content."""
        return [
            r'\b(hate|kill|die|stupid|idiot)\b',
            r'\b(f[*u]ck|sh[*i]t|damn)\b',
            r'\b(racist|sexist|homophobic)\b',
            r'[A-Z]{3,}',  # Excessive caps
            r'!{3,}',      # Excessive exclamation
        ]
        
    def _apply_subreddit_rules(self, subreddit: str, comment: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """Apply subreddit-specific labeling rules."""
        rules = self.subreddit_rules.get(subreddit, {})
        
        # For now, return default label
        # In a full implementation, this would analyze comment content against specific rules
        return rules.get('default_label')
        
    def _contains_toxic_patterns(self, text: str) -> bool:
        """Check if text contains toxic patterns."""
        if not isinstance(text, str):
            return False
            
        text_lower = text.lower()
        for pattern in self.toxic_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
        
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get current extraction statistics."""
        return {
            'total_extracted': self.stats['total_extracted'],
            'successful_extractions': self.stats['successful_extractions'],
            'failed_extractions': self.stats['failed_extractions'],
            'missing_metadata': self.stats['missing_metadata'],
            'success_rate': (self.stats['successful_extractions'] / max(1, self.stats['total_extracted'])) * 100,
            'label_distribution': self.stats['label_assignments'].copy()
        }
        
    def reset_stats(self) -> None:
        """Reset extraction statistics."""
        self.stats = {
            'total_extracted': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'missing_metadata': 0,
            'label_assignments': {}
        }
        
    def extract_parent_context(self, comment: Dict[str, Any], parent_comments: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[str]:
        """
        Extract parent comment context for better training data quality.
        
        Args:
            comment: Current comment dictionary
            parent_comments: Optional dictionary of parent comments keyed by ID
            
        Returns:
            Optional[str]: Parent context text or None if not available
            
        Requirements: 3.4, 3.5
        """
        parent_id = self._clean_parent_id(comment.get('parent_id', ''))
        
        if not parent_id:
            return None
            
        # If parent comments dictionary is provided, look up parent
        if parent_comments and parent_id in parent_comments:
            parent_comment = parent_comments[parent_id]
            parent_body = parent_comment.get('body', '')
            
            # Return parent context if available and not deleted
            if parent_body and parent_body not in {'[deleted]', '[removed]'}:
                return self._truncate_context(parent_body, max_length=200)
                
        return None
        
    def build_enhanced_training_record(self, comment: Dict[str, Any], 
                                     removal_context: Optional[Dict[str, Any]] = None,
                                     parent_comments: Optional[Dict[str, Dict[str, Any]]] = None,
                                     thread_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build enhanced training record with parent context and comprehensive missing data handling.
        
        Args:
            comment: Reddit comment dictionary
            removal_context: Optional removal context from classifier
            parent_comments: Optional dictionary of parent comments
            thread_context: Optional thread/submission context
            
        Returns:
            Dict: Enhanced training record with context
            
        Requirements: 3.4, 3.5
        """
        # Build base training record
        training_record = self.build_training_record(comment, removal_context)
        
        # Add parent context
        parent_context = self.extract_parent_context(comment, parent_comments)
        training_record['parent_context'] = parent_context
        
        # Add thread context if available
        if thread_context:
            training_record['thread_title'] = thread_context.get('title', '')
            training_record['thread_selftext'] = self._truncate_context(
                thread_context.get('selftext', ''), max_length=300
            )
            training_record['thread_score'] = thread_context.get('score', 0)
            training_record['thread_num_comments'] = thread_context.get('num_comments', 0)
        else:
            training_record['thread_title'] = ''
            training_record['thread_selftext'] = ''
            training_record['thread_score'] = 0
            training_record['thread_num_comments'] = 0
            
        # Handle missing values with appropriate placeholders
        training_record = self._handle_missing_values(training_record)
        
        # Add data quality indicators
        training_record['data_quality'] = self._assess_data_quality(training_record)
        
        return training_record
        
    def _handle_missing_values(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle missing or null values with appropriate placeholders.
        
        Args:
            record: Training record dictionary
            
        Returns:
            Dict: Record with missing values handled
            
        Requirements: 3.4, 3.5
        """
        # Define default values for different field types
        string_defaults = {
            'comment_text': '[CONTENT_UNAVAILABLE]',
            'subreddit': 'unknown',
            'target_label': 'unknown',
            'parent_id': '',
            'thread_id': '',
            'author': '[UNKNOWN_USER]',
            'parent_context': '',
            'thread_title': '',
            'thread_selftext': ''
        }
        
        numeric_defaults = {
            'score': 0,
            'controversiality': 0,
            'gilded': 0,
            'comment_length': 0,
            'thread_score': 0,
            'thread_num_comments': 0
        }
        
        boolean_defaults = {
            'has_parent': False,
            'is_top_level': False
        }
        
        # Apply string defaults
        for field, default in string_defaults.items():
            if field in record and (record[field] is None or record[field] == ''):
                record[field] = default
                
        # Apply numeric defaults
        for field, default in numeric_defaults.items():
            if field in record and (record[field] is None or not isinstance(record[field], (int, float))):
                record[field] = default
                
        # Apply boolean defaults
        for field, default in boolean_defaults.items():
            if field in record and record[field] is None:
                record[field] = default
                
        # Handle timestamp specially
        if 'timestamp' in record and record['timestamp'] is None:
            record['timestamp'] = datetime.fromtimestamp(0, tz=timezone.utc)
            
        # Handle removal_type specially
        if 'removal_type' in record and record['removal_type'] is None:
            record['removal_type'] = 'unknown'
            
        return record
        
    def _assess_data_quality(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the quality of extracted data for training purposes.
        
        Args:
            record: Training record dictionary
            
        Returns:
            Dict: Data quality assessment
        """
        quality = {
            'completeness_score': 0.0,
            'has_content': False,
            'has_context': False,
            'has_metadata': False,
            'training_ready': False
        }
        
        # Check content availability
        content_available = (
            record.get('comment_text', '') not in {'[CONTENT_UNAVAILABLE]', '[CONTENT_REMOVED]', ''}
        )
        quality['has_content'] = content_available
        
        # Check context availability
        context_available = bool(record.get('parent_context', '')) or bool(record.get('thread_title', ''))
        quality['has_context'] = context_available
        
        # Check metadata completeness
        required_metadata = ['id', 'subreddit', 'timestamp', 'removal_type', 'target_label']
        metadata_complete = all(
            record.get(field) not in {None, '', 'unknown'} for field in required_metadata
        )
        quality['has_metadata'] = metadata_complete
        
        # Calculate completeness score
        completeness_factors = [
            1.0 if quality['has_content'] else 0.0,
            0.5 if quality['has_context'] else 0.0,
            1.0 if quality['has_metadata'] else 0.0,
            0.5 if record.get('score', 0) != 0 else 0.0,  # Has engagement data
            0.3 if record.get('parent_id', '') != '' else 0.0,  # Has parent relationship
        ]
        quality['completeness_score'] = sum(completeness_factors) / len(completeness_factors)
        
        # Determine if ready for training
        quality['training_ready'] = (
            quality['has_metadata'] and 
            quality['completeness_score'] >= 0.6 and
            record.get('removal_type', 'unknown') != 'unknown'
        )
        
        return quality
        
    def _truncate_context(self, text: str, max_length: int = 200) -> str:
        """
        Truncate context text to reasonable length for training.
        
        Args:
            text: Text to truncate
            max_length: Maximum length to keep
            
        Returns:
            str: Truncated text
        """
        if not isinstance(text, str):
            return ''
            
        text = text.strip()
        if len(text) <= max_length:
            return text
            
        # Try to truncate at word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we can keep most of the text
            return truncated[:last_space] + '...'
        else:
            return truncated + '...'
            
    def batch_extract_with_context(self, comments: List[Dict[str, Any]], 
                                 submissions: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Extract metadata for a batch of comments with context resolution.
        
        Args:
            comments: List of comment dictionaries
            submissions: Optional dictionary of submissions keyed by ID
            
        Returns:
            List[Dict]: List of enhanced training records
            
        Requirements: 3.4, 3.5
        """
        # Build parent comment lookup
        parent_lookup = {comment.get('id', ''): comment for comment in comments}
        
        enhanced_records = []
        
        for comment in comments:
            try:
                # Get thread context if available
                thread_id = self._extract_thread_id(comment.get('link_id', ''))
                thread_context = submissions.get(thread_id) if submissions else None
                
                # Build enhanced record
                enhanced_record = self.build_enhanced_training_record(
                    comment=comment,
                    parent_comments=parent_lookup,
                    thread_context=thread_context
                )
                
                enhanced_records.append(enhanced_record)
                
            except Exception as e:
                self.logger.error(f"Failed to process comment {comment.get('id', 'unknown')}: {e}")
                # Add fallback record
                fallback_record = self.build_training_record(comment)
                fallback_record = self._handle_missing_values(fallback_record)
                enhanced_records.append(fallback_record)
                
        self.logger.info(f"Processed {len(enhanced_records)} comments with context")
        return enhanced_records