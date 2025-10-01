"""
Comment classifier component for identifying and categorizing deleted/removed comments.
Distinguishes between user-deleted and moderator-removed comments based on Reddit markers.
"""

import logging
from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass
from enum import Enum


class RemovalType(Enum):
    """Enumeration of comment removal types."""
    ACTIVE = "active"
    USER_DELETED = "user_deleted"
    MODERATOR_REMOVED = "moderator_removed"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of comment classification."""
    removal_type: RemovalType
    confidence: float
    reason: str
    metadata: Dict[str, Any]


class CommentClassifier:
    """Classifier for identifying deleted and removed Reddit comments."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize comment classifier.
        
        Args:
            config: Configuration dictionary with classification settings
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration or use defaults
        config = config or {}
        classification_config = config.get('classification', {})
        
        # Deletion markers from configuration
        self.deleted_markers = set(classification_config.get('deleted_markers', ['[deleted]']))
        self.removed_markers = set(classification_config.get('removed_markers', ['[removed]']))
        self.deleted_authors = set(classification_config.get('deleted_authors', ['[deleted]']))
        
        # Additional patterns for detection
        self.suspicious_patterns = {
            'empty_body': '',
            'null_body': None,
            'whitespace_only': ' \t\n\r'
        }
        
        # Statistics tracking
        self.stats = {
            'total_classified': 0,
            'user_deleted': 0,
            'moderator_removed': 0,
            'active': 0,
            'unknown': 0
        }
        
    def classify_comment(self, comment: Dict[str, Any]) -> str:
        """
        Classify a comment as deleted, removed, or active.
        
        Args:
            comment: Reddit comment dictionary
            
        Returns:
            str: Classification result ("user_deleted", "moderator_removed", or "active")
            
        Requirements: 2.1, 2.2, 2.3, 2.4
        """
        result = self._classify_comment_detailed(comment)
        self._update_stats(result.removal_type)
        
        return result.removal_type.value
        
    def _classify_comment_detailed(self, comment: Dict[str, Any]) -> ClassificationResult:
        """
        Perform detailed classification with confidence and metadata.
        
        Args:
            comment: Reddit comment dictionary
            
        Returns:
            ClassificationResult: Detailed classification result
        """
        body = comment.get('body', '')
        author = comment.get('author', '')
        
        # Check for explicit deletion markers
        if self._is_user_deleted_by_body(body):
            return ClassificationResult(
                removal_type=RemovalType.USER_DELETED,
                confidence=1.0,
                reason="Body contains user deletion marker",
                metadata={'marker': body, 'detection_method': 'body_marker'}
            )
            
        # Check for explicit removal markers
        if self._is_moderator_removed_by_body(body):
            return ClassificationResult(
                removal_type=RemovalType.MODERATOR_REMOVED,
                confidence=1.0,
                reason="Body contains moderator removal marker",
                metadata={'marker': body, 'detection_method': 'body_marker'}
            )
            
        # Check for deleted author (account-level deletion)
        if self._is_user_deleted_by_author(author):
            return ClassificationResult(
                removal_type=RemovalType.USER_DELETED,
                confidence=0.9,
                reason="Author is marked as deleted",
                metadata={'author': author, 'detection_method': 'author_marker'}
            )
            
        # Check for suspicious patterns that might indicate removal
        suspicious_result = self._check_suspicious_patterns(comment)
        if suspicious_result:
            return suspicious_result
            
        # Default to active comment
        return ClassificationResult(
            removal_type=RemovalType.ACTIVE,
            confidence=1.0,
            reason="No deletion or removal markers found",
            metadata={'detection_method': 'default'}
        )
        
    def is_deleted_comment(self, comment: Dict[str, Any]) -> bool:
        """
        Check if a comment is deleted (either user-deleted or moderator-removed).
        
        Args:
            comment: Reddit comment dictionary
            
        Returns:
            bool: True if comment is deleted/removed, False otherwise
            
        Requirements: 2.1
        """
        classification = self.classify_comment(comment)
        return classification in [RemovalType.USER_DELETED.value, RemovalType.MODERATOR_REMOVED.value]
        
    def extract_removal_context(self, comment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata and context about comment removal.
        
        Args:
            comment: Reddit comment dictionary
            
        Returns:
            Dict: Removal context metadata
            
        Requirements: 2.5
        """
        result = self._classify_comment_detailed(comment)
        
        context = {
            'removal_type': result.removal_type.value,
            'confidence': result.confidence,
            'reason': result.reason,
            'detection_method': result.metadata.get('detection_method', 'unknown'),
            'original_body': comment.get('body', ''),
            'original_author': comment.get('author', ''),
            'subreddit': comment.get('subreddit', ''),
            'created_utc': comment.get('created_utc', 0),
            'score': comment.get('score', 0),
            'parent_id': comment.get('parent_id', ''),
            'link_id': comment.get('link_id', ''),
            'controversiality': comment.get('controversiality', 0)
        }
        
        # Add specific markers if detected
        if 'marker' in result.metadata:
            context['deletion_marker'] = result.metadata['marker']
            
        return context
        
    def _is_user_deleted_by_body(self, body: str) -> bool:
        """Check if comment body indicates user deletion."""
        if not isinstance(body, str):
            return False
        return body.strip() in self.deleted_markers
        
    def _is_moderator_removed_by_body(self, body: str) -> bool:
        """Check if comment body indicates moderator removal."""
        if not isinstance(body, str):
            return False
        return body.strip() in self.removed_markers
        
    def _is_user_deleted_by_author(self, author: str) -> bool:
        """Check if author indicates user deletion."""
        if not isinstance(author, str):
            return False
        return author.strip() in self.deleted_authors
        
    def _check_suspicious_patterns(self, comment: Dict[str, Any]) -> Optional[ClassificationResult]:
        """
        Check for suspicious patterns that might indicate removal.
        
        Args:
            comment: Reddit comment dictionary
            
        Returns:
            ClassificationResult or None if no suspicious patterns found
        """
        body = comment.get('body')
        author = comment.get('author', '')
        
        # Check for empty or null body with normal author
        if (body is None or body == '') and author not in self.deleted_authors:
            return ClassificationResult(
                removal_type=RemovalType.UNKNOWN,
                confidence=0.3,
                reason="Empty body with active author (possible removal)",
                metadata={'detection_method': 'suspicious_pattern', 'pattern': 'empty_body'}
            )
            
        # Check for whitespace-only body
        if isinstance(body, str) and body.strip() == '':
            return ClassificationResult(
                removal_type=RemovalType.UNKNOWN,
                confidence=0.2,
                reason="Whitespace-only body (possible removal)",
                metadata={'detection_method': 'suspicious_pattern', 'pattern': 'whitespace_only'}
            )
            
        return None
        
    def _update_stats(self, removal_type: RemovalType) -> None:
        """Update classification statistics."""
        self.stats['total_classified'] += 1
        
        if removal_type == RemovalType.USER_DELETED:
            self.stats['user_deleted'] += 1
        elif removal_type == RemovalType.MODERATOR_REMOVED:
            self.stats['moderator_removed'] += 1
        elif removal_type == RemovalType.ACTIVE:
            self.stats['active'] += 1
        else:
            self.stats['unknown'] += 1
            
    def get_classification_stats(self) -> Dict[str, int]:
        """Get current classification statistics."""
        return self.stats.copy()
        
    def reset_stats(self) -> None:
        """Reset classification statistics."""
        for key in self.stats:
            self.stats[key] = 0
            
    def separate_by_removal_type(self, comments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Separate comments into different categories based on removal type.
        
        Args:
            comments: List of Reddit comment dictionaries
            
        Returns:
            Dict: Comments separated by removal type
            
        Requirements: 2.5
        """
        separated = {
            'user_deleted': [],
            'moderator_removed': [],
            'active': [],
            'unknown': []
        }
        
        for comment in comments:
            classification = self.classify_comment(comment)
            separated[classification].append(comment)
            
        self.logger.info(
            f"Separated {len(comments)} comments: "
            f"{len(separated['user_deleted'])} user-deleted, "
            f"{len(separated['moderator_removed'])} moderator-removed, "
            f"{len(separated['active'])} active, "
            f"{len(separated['unknown'])} unknown"
        )
        
        return separated
        
    def create_training_datasets(self, comments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create separate training datasets for user-deleted and moderator-removed comments.
        
        Args:
            comments: List of Reddit comment dictionaries
            
        Returns:
            Dict: Training datasets separated by removal type
            
        Requirements: 2.5
        """
        separated = self.separate_by_removal_type(comments)
        
        # Create training datasets with enhanced metadata
        training_datasets = {
            'user_deleted_train': [],
            'moderator_removed_train': []
        }
        
        # Process user-deleted comments
        for comment in separated['user_deleted']:
            context = self.extract_removal_context(comment)
            training_record = self._create_training_record(comment, context, 'user_deleted')
            training_datasets['user_deleted_train'].append(training_record)
            
        # Process moderator-removed comments
        for comment in separated['moderator_removed']:
            context = self.extract_removal_context(comment)
            training_record = self._create_training_record(comment, context, 'moderator_removed')
            training_datasets['moderator_removed_train'].append(training_record)
            
        self.logger.info(
            f"Created training datasets: "
            f"{len(training_datasets['user_deleted_train'])} user-deleted records, "
            f"{len(training_datasets['moderator_removed_train'])} moderator-removed records"
        )
        
        return training_datasets
        
    def _create_training_record(self, comment: Dict[str, Any], context: Dict[str, Any], removal_type: str) -> Dict[str, Any]:
        """
        Create a training record with standardized format.
        
        Args:
            comment: Original comment dictionary
            context: Removal context metadata
            removal_type: Type of removal (user_deleted or moderator_removed)
            
        Returns:
            Dict: Standardized training record
        """
        return {
            'id': comment.get('id', ''),
            'comment_text': self._get_original_text(comment, context),
            'subreddit': comment.get('subreddit', ''),
            'timestamp': comment.get('created_utc', 0),
            'removal_type': removal_type,
            'target_label': self._generate_target_label(comment, context, removal_type),
            'parent_id': comment.get('parent_id', ''),
            'thread_id': self._extract_thread_id(comment.get('link_id', '')),
            'score': comment.get('score', 0),
            'author': comment.get('author', ''),
            'confidence': context.get('confidence', 0.0),
            'detection_method': context.get('detection_method', 'unknown'),
            'controversiality': comment.get('controversiality', 0)
        }
        
    def _get_original_text(self, comment: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Extract original comment text, handling deletion markers.
        
        Args:
            comment: Original comment dictionary
            context: Removal context metadata
            
        Returns:
            str: Original text or placeholder for deleted content
        """
        body = comment.get('body', '')
        
        # If body contains deletion markers, we don't have original text
        if body in self.deleted_markers or body in self.removed_markers:
            return '[CONTENT_REMOVED]'
            
        # Return original body if available
        return body if isinstance(body, str) else '[CONTENT_UNAVAILABLE]'
        
    def _generate_target_label(self, comment: Dict[str, Any], context: Dict[str, Any], removal_type: str) -> str:
        """
        Generate target labels for machine learning based on removal context.
        
        Args:
            comment: Original comment dictionary
            context: Removal context metadata
            removal_type: Type of removal
            
        Returns:
            str: Target label for ML training
        """
        subreddit = comment.get('subreddit', '').lower()
        score = comment.get('score', 0)
        controversiality = comment.get('controversiality', 0)
        
        # Basic labeling logic based on removal type and context
        if removal_type == 'moderator_removed':
            # Moderator removals are likely policy violations
            if controversiality > 0:
                return 'controversial_content'
            elif score < -5:
                return 'heavily_downvoted'
            elif subreddit in ['askreddit', 'iama', 'science']:
                return 'rule_violation'
            else:
                return 'policy_violation'
                
        elif removal_type == 'user_deleted':
            # User deletions might be regret, privacy, or other reasons
            if score < 0:
                return 'regret_deletion'
            elif controversiality > 0:
                return 'controversial_deletion'
            else:
                return 'voluntary_deletion'
                
        return 'unknown_removal'
        
    def _extract_thread_id(self, link_id: str) -> str:
        """
        Extract thread ID from Reddit link_id format.
        
        Args:
            link_id: Reddit link_id (format: t3_xxxxxx)
            
        Returns:
            str: Clean thread ID
        """
        if isinstance(link_id, str) and link_id.startswith('t3_'):
            return link_id[3:]  # Remove 't3_' prefix
        return link_id if isinstance(link_id, str) else ''
        
    def validate_classification(self, comment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate classification results and provide detailed analysis.
        
        Args:
            comment: Reddit comment dictionary
            
        Returns:
            Dict: Validation results with detailed analysis
            
        Requirements: 2.5
        """
        result = self._classify_comment_detailed(comment)
        context = self.extract_removal_context(comment)
        
        validation = {
            'comment_id': comment.get('id', ''),
            'classification': result.removal_type.value,
            'confidence': result.confidence,
            'reason': result.reason,
            'validation_checks': {
                'has_body': 'body' in comment and comment['body'] is not None,
                'has_author': 'author' in comment and comment['author'] is not None,
                'has_timestamp': 'created_utc' in comment,
                'has_subreddit': 'subreddit' in comment and comment['subreddit'] != '',
                'body_is_deletion_marker': comment.get('body', '') in self.deleted_markers,
                'body_is_removal_marker': comment.get('body', '') in self.removed_markers,
                'author_is_deleted': comment.get('author', '') in self.deleted_authors
            },
            'metadata': context,
            'training_ready': self._is_training_ready(comment, result)
        }
        
        return validation
        
    def _is_training_ready(self, comment: Dict[str, Any], result: ClassificationResult) -> bool:
        """
        Check if a comment is ready for training dataset inclusion.
        
        Args:
            comment: Reddit comment dictionary
            result: Classification result
            
        Returns:
            bool: True if ready for training
        """
        # Must have essential fields
        required_fields = ['id', 'subreddit', 'created_utc', 'author']
        if not all(field in comment for field in required_fields):
            return False
            
        # Must be classified as deleted or removed
        if result.removal_type not in [RemovalType.USER_DELETED, RemovalType.MODERATOR_REMOVED]:
            return False
            
        # Must have reasonable confidence
        if result.confidence < 0.5:
            return False
            
        return True