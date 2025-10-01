"""
Unit tests for comment_classifier module.
Tests comment classification with known deletion patterns and edge cases.
"""

import pytest
from unittest.mock import Mock, patch

from src.comment_classifier import CommentClassifier, RemovalType, ClassificationResult


class TestCommentClassifier:
    """Test suite for CommentClassifier class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.classifier = CommentClassifier()
        
    def test_init_default_config(self):
        """Test classifier initialization with default configuration."""
        classifier = CommentClassifier()
        
        assert '[deleted]' in classifier.deleted_markers
        assert '[removed]' in classifier.removed_markers
        assert '[deleted]' in classifier.deleted_authors
        assert classifier.stats['total_classified'] == 0
        
    def test_init_custom_config(self):
        """Test classifier initialization with custom configuration."""
        config = {
            'classification': {
                'deleted_markers': ['[user_deleted]', '[gone]'],
                'removed_markers': ['[mod_removed]', '[banned]'],
                'deleted_authors': ['[deleted_user]']
            }
        }
        
        classifier = CommentClassifier(config)
        
        assert '[user_deleted]' in classifier.deleted_markers
        assert '[gone]' in classifier.deleted_markers
        assert '[mod_removed]' in classifier.removed_markers
        assert '[banned]' in classifier.removed_markers
        assert '[deleted_user]' in classifier.deleted_authors
        
    def test_classify_comment_user_deleted_by_body(self):
        """Test classification of user-deleted comment by body marker."""
        comment = {
            'id': 'test123',
            'body': '[deleted]',
            'author': 'some_user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        result = self.classifier.classify_comment(comment)
        
        assert result == RemovalType.USER_DELETED.value
        
        # Check stats
        stats = self.classifier.get_classification_stats()
        assert stats['user_deleted'] == 1
        assert stats['total_classified'] == 1
        
    def test_classify_comment_moderator_removed_by_body(self):
        """Test classification of moderator-removed comment by body marker."""
        comment = {
            'id': 'test123',
            'body': '[removed]',
            'author': 'some_user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        result = self.classifier.classify_comment(comment)
        
        assert result == RemovalType.MODERATOR_REMOVED.value
        
        # Check stats
        stats = self.classifier.get_classification_stats()
        assert stats['moderator_removed'] == 1
        
    def test_classify_comment_user_deleted_by_author(self):
        """Test classification of user-deleted comment by deleted author."""
        comment = {
            'id': 'test123',
            'body': 'This was a normal comment',
            'author': '[deleted]',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        result = self.classifier.classify_comment(comment)
        
        assert result == RemovalType.USER_DELETED.value
        
    def test_classify_comment_active(self):
        """Test classification of active (non-deleted) comment."""
        comment = {
            'id': 'test123',
            'body': 'This is a normal active comment',
            'author': 'active_user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        result = self.classifier.classify_comment(comment)
        
        assert result == RemovalType.ACTIVE.value
        
        # Check stats
        stats = self.classifier.get_classification_stats()
        assert stats['active'] == 1
        
    def test_classify_comment_suspicious_empty_body(self):
        """Test classification of comment with suspicious empty body."""
        comment = {
            'id': 'test123',
            'body': '',
            'author': 'normal_user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        result = self.classifier.classify_comment(comment)
        
        # Should be classified as unknown due to suspicious pattern
        assert result == RemovalType.UNKNOWN.value
        
    def test_classify_comment_suspicious_null_body(self):
        """Test classification of comment with null body."""
        comment = {
            'id': 'test123',
            'body': None,
            'author': 'normal_user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        result = self.classifier.classify_comment(comment)
        
        assert result == RemovalType.UNKNOWN.value
        
    def test_classify_comment_whitespace_only_body(self):
        """Test classification of comment with whitespace-only body."""
        comment = {
            'id': 'test123',
            'body': '   \t\n  ',
            'author': 'normal_user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        result = self.classifier.classify_comment(comment)
        
        assert result == RemovalType.UNKNOWN.value
        
    def test_is_deleted_comment_true_cases(self):
        """Test is_deleted_comment method for deleted/removed comments."""
        deleted_comment = {
            'id': 'test1',
            'body': '[deleted]',
            'author': 'user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        removed_comment = {
            'id': 'test2',
            'body': '[removed]',
            'author': 'user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        assert self.classifier.is_deleted_comment(deleted_comment) is True
        assert self.classifier.is_deleted_comment(removed_comment) is True
        
    def test_is_deleted_comment_false_case(self):
        """Test is_deleted_comment method for active comments."""
        active_comment = {
            'id': 'test1',
            'body': 'This is an active comment',
            'author': 'user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        assert self.classifier.is_deleted_comment(active_comment) is False
        
    def test_extract_removal_context_user_deleted(self):
        """Test removal context extraction for user-deleted comment."""
        comment = {
            'id': 'test123',
            'body': '[deleted]',
            'author': 'some_user',
            'subreddit': 'testsubreddit',
            'created_utc': 1640995200,
            'score': 5,
            'parent_id': 't1_parent123',
            'link_id': 't3_link123',
            'controversiality': 0
        }
        
        context = self.classifier.extract_removal_context(comment)
        
        assert context['removal_type'] == RemovalType.USER_DELETED.value
        assert context['confidence'] == 1.0
        assert context['reason'] == "Body contains user deletion marker"
        assert context['detection_method'] == 'body_marker'
        assert context['original_body'] == '[deleted]'
        assert context['original_author'] == 'some_user'
        assert context['subreddit'] == 'testsubreddit'
        assert context['created_utc'] == 1640995200
        assert context['score'] == 5
        assert context['parent_id'] == 't1_parent123'
        assert context['link_id'] == 't3_link123'
        assert context['controversiality'] == 0
        assert context['deletion_marker'] == '[deleted]'
        
    def test_extract_removal_context_moderator_removed(self):
        """Test removal context extraction for moderator-removed comment."""
        comment = {
            'id': 'test123',
            'body': '[removed]',
            'author': 'some_user',
            'subreddit': 'testsubreddit',
            'created_utc': 1640995200,
            'score': -10
        }
        
        context = self.classifier.extract_removal_context(comment)
        
        assert context['removal_type'] == RemovalType.MODERATOR_REMOVED.value
        assert context['confidence'] == 1.0
        assert context['reason'] == "Body contains moderator removal marker"
        assert context['detection_method'] == 'body_marker'
        assert context['deletion_marker'] == '[removed]'
        
    def test_separate_by_removal_type(self):
        """Test separation of comments by removal type."""
        comments = [
            {
                'id': 'active1',
                'body': 'Active comment 1',
                'author': 'user1',
                'subreddit': 'test',
                'created_utc': 1640995200
            },
            {
                'id': 'deleted1',
                'body': '[deleted]',
                'author': 'user2',
                'subreddit': 'test',
                'created_utc': 1640995300
            },
            {
                'id': 'removed1',
                'body': '[removed]',
                'author': 'user3',
                'subreddit': 'test',
                'created_utc': 1640995400
            },
            {
                'id': 'active2',
                'body': 'Active comment 2',
                'author': 'user4',
                'subreddit': 'test',
                'created_utc': 1640995500
            },
            {
                'id': 'deleted2',
                'body': 'Normal text',
                'author': '[deleted]',
                'subreddit': 'test',
                'created_utc': 1640995600
            }
        ]
        
        separated = self.classifier.separate_by_removal_type(comments)
        
        assert len(separated['active']) == 2
        assert len(separated['user_deleted']) == 2
        assert len(separated['moderator_removed']) == 1
        assert len(separated['unknown']) == 0
        
        # Verify specific comments
        assert separated['active'][0]['id'] == 'active1'
        assert separated['active'][1]['id'] == 'active2'
        assert separated['user_deleted'][0]['id'] == 'deleted1'
        assert separated['user_deleted'][1]['id'] == 'deleted2'
        assert separated['moderator_removed'][0]['id'] == 'removed1'
        
    def test_create_training_datasets(self):
        """Test creation of training datasets."""
        comments = [
            {
                'id': 'deleted1',
                'body': '[deleted]',
                'author': 'user1',
                'subreddit': 'askreddit',
                'created_utc': 1640995200,
                'score': 5,
                'parent_id': 't1_parent1',
                'link_id': 't3_link1',
                'controversiality': 0
            },
            {
                'id': 'removed1',
                'body': '[removed]',
                'author': 'user2',
                'subreddit': 'science',
                'created_utc': 1640995300,
                'score': -5,
                'parent_id': 't1_parent2',
                'link_id': 't3_link2',
                'controversiality': 1
            }
        ]
        
        datasets = self.classifier.create_training_datasets(comments)
        
        assert 'user_deleted_train' in datasets
        assert 'moderator_removed_train' in datasets
        assert len(datasets['user_deleted_train']) == 1
        assert len(datasets['moderator_removed_train']) == 1
        
        # Check user-deleted record structure
        user_deleted_record = datasets['user_deleted_train'][0]
        assert user_deleted_record['id'] == 'deleted1'
        assert user_deleted_record['comment_text'] == '[CONTENT_REMOVED]'
        assert user_deleted_record['subreddit'] == 'askreddit'
        assert user_deleted_record['timestamp'] == 1640995200
        assert user_deleted_record['removal_type'] == 'user_deleted'
        assert user_deleted_record['parent_id'] == 't1_parent1'
        assert user_deleted_record['thread_id'] == 'link1'
        assert user_deleted_record['score'] == 5
        assert user_deleted_record['author'] == 'user1'
        assert user_deleted_record['confidence'] == 1.0
        assert user_deleted_record['detection_method'] == 'body_marker'
        
        # Check moderator-removed record structure
        mod_removed_record = datasets['moderator_removed_train'][0]
        assert mod_removed_record['id'] == 'removed1'
        assert mod_removed_record['comment_text'] == '[CONTENT_REMOVED]'
        assert mod_removed_record['removal_type'] == 'moderator_removed'
        assert mod_removed_record['target_label'] == 'controversial_content'  # Due to controversiality > 0
        
    def test_get_original_text_with_deletion_markers(self):
        """Test original text extraction with deletion markers."""
        comment_deleted = {'body': '[deleted]'}
        comment_removed = {'body': '[removed]'}
        comment_normal = {'body': 'Normal comment text'}
        comment_null = {'body': None}
        
        context = {}  # Empty context for this test
        
        assert self.classifier._get_original_text(comment_deleted, context) == '[CONTENT_REMOVED]'
        assert self.classifier._get_original_text(comment_removed, context) == '[CONTENT_REMOVED]'
        assert self.classifier._get_original_text(comment_normal, context) == 'Normal comment text'
        assert self.classifier._get_original_text(comment_null, context) == '[CONTENT_UNAVAILABLE]'
        
    def test_generate_target_label_moderator_removed(self):
        """Test target label generation for moderator-removed comments."""
        # Controversial content
        comment1 = {
            'subreddit': 'test',
            'score': 0,
            'controversiality': 1
        }
        context1 = {}
        label1 = self.classifier._generate_target_label(comment1, context1, 'moderator_removed')
        assert label1 == 'controversial_content'
        
        # Heavily downvoted
        comment2 = {
            'subreddit': 'test',
            'score': -10,
            'controversiality': 0
        }
        context2 = {}
        label2 = self.classifier._generate_target_label(comment2, context2, 'moderator_removed')
        assert label2 == 'heavily_downvoted'
        
        # Rule violation in specific subreddit
        comment3 = {
            'subreddit': 'askreddit',
            'score': 5,
            'controversiality': 0
        }
        context3 = {}
        label3 = self.classifier._generate_target_label(comment3, context3, 'moderator_removed')
        assert label3 == 'rule_violation'
        
        # General policy violation
        comment4 = {
            'subreddit': 'other',
            'score': 0,
            'controversiality': 0
        }
        context4 = {}
        label4 = self.classifier._generate_target_label(comment4, context4, 'moderator_removed')
        assert label4 == 'policy_violation'
        
    def test_generate_target_label_user_deleted(self):
        """Test target label generation for user-deleted comments."""
        # Regret deletion (negative score)
        comment1 = {
            'subreddit': 'test',
            'score': -5,
            'controversiality': 0
        }
        context1 = {}
        label1 = self.classifier._generate_target_label(comment1, context1, 'user_deleted')
        assert label1 == 'regret_deletion'
        
        # Controversial deletion
        comment2 = {
            'subreddit': 'test',
            'score': 5,
            'controversiality': 1
        }
        context2 = {}
        label2 = self.classifier._generate_target_label(comment2, context2, 'user_deleted')
        assert label2 == 'controversial_deletion'
        
        # Voluntary deletion
        comment3 = {
            'subreddit': 'test',
            'score': 5,
            'controversiality': 0
        }
        context3 = {}
        label3 = self.classifier._generate_target_label(comment3, context3, 'user_deleted')
        assert label3 == 'voluntary_deletion'
        
    def test_extract_thread_id(self):
        """Test thread ID extraction from Reddit link_id format."""
        assert self.classifier._extract_thread_id('t3_abc123') == 'abc123'
        assert self.classifier._extract_thread_id('abc123') == 'abc123'
        assert self.classifier._extract_thread_id('') == ''
        assert self.classifier._extract_thread_id(None) == ''
        
    def test_validate_classification(self):
        """Test classification validation with detailed analysis."""
        comment = {
            'id': 'test123',
            'body': '[deleted]',
            'author': 'user',
            'subreddit': 'test',
            'created_utc': 1640995200,
            'score': 5
        }
        
        validation = self.classifier.validate_classification(comment)
        
        assert validation['comment_id'] == 'test123'
        assert validation['classification'] == RemovalType.USER_DELETED.value
        assert validation['confidence'] == 1.0
        assert validation['reason'] == "Body contains user deletion marker"
        
        # Check validation checks
        checks = validation['validation_checks']
        assert checks['has_body'] is True
        assert checks['has_author'] is True
        assert checks['has_timestamp'] is True
        assert checks['has_subreddit'] is True
        assert checks['body_is_deletion_marker'] is True
        assert checks['body_is_removal_marker'] is False
        assert checks['author_is_deleted'] is False
        
        assert validation['training_ready'] is True
        
    def test_validate_classification_not_training_ready(self):
        """Test classification validation for comment not ready for training."""
        # Missing required fields
        comment = {
            'id': 'test123',
            'body': 'Normal comment'
            # Missing author, subreddit, created_utc
        }
        
        validation = self.classifier.validate_classification(comment)
        
        assert validation['training_ready'] is False
        
    def test_is_training_ready_missing_fields(self):
        """Test training readiness check with missing fields."""
        comment = {
            'id': 'test123',
            'body': '[deleted]'
            # Missing required fields
        }
        
        result = ClassificationResult(
            removal_type=RemovalType.USER_DELETED,
            confidence=1.0,
            reason="Test",
            metadata={}
        )
        
        assert self.classifier._is_training_ready(comment, result) is False
        
    def test_is_training_ready_low_confidence(self):
        """Test training readiness check with low confidence."""
        comment = {
            'id': 'test123',
            'subreddit': 'test',
            'created_utc': 1640995200,
            'author': 'user',
            'body': '[deleted]'
        }
        
        result = ClassificationResult(
            removal_type=RemovalType.USER_DELETED,
            confidence=0.3,  # Low confidence
            reason="Test",
            metadata={}
        )
        
        assert self.classifier._is_training_ready(comment, result) is False
        
    def test_is_training_ready_active_comment(self):
        """Test training readiness check with active comment."""
        comment = {
            'id': 'test123',
            'subreddit': 'test',
            'created_utc': 1640995200,
            'author': 'user',
            'body': 'Active comment'
        }
        
        result = ClassificationResult(
            removal_type=RemovalType.ACTIVE,
            confidence=1.0,
            reason="Test",
            metadata={}
        )
        
        assert self.classifier._is_training_ready(comment, result) is False
        
    def test_reset_stats(self):
        """Test statistics reset functionality."""
        # Classify some comments to generate stats
        comment = {
            'id': 'test',
            'body': '[deleted]',
            'author': 'user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        self.classifier.classify_comment(comment)
        
        # Verify stats exist
        stats = self.classifier.get_classification_stats()
        assert stats['total_classified'] > 0
        
        # Reset and verify
        self.classifier.reset_stats()
        new_stats = self.classifier.get_classification_stats()
        assert new_stats['total_classified'] == 0
        assert new_stats['user_deleted'] == 0
        assert new_stats['moderator_removed'] == 0
        assert new_stats['active'] == 0
        assert new_stats['unknown'] == 0
        
    def test_classification_with_edge_cases(self):
        """Test classification with various edge cases."""
        # Empty string body
        comment1 = {
            'id': 'edge1',
            'body': '',
            'author': 'user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        result1 = self.classifier.classify_comment(comment1)
        assert result1 == RemovalType.UNKNOWN.value
        
        # Body with extra whitespace around marker
        comment2 = {
            'id': 'edge2',
            'body': '  [deleted]  ',
            'author': 'user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        result2 = self.classifier.classify_comment(comment2)
        assert result2 == RemovalType.USER_DELETED.value
        
        # Non-string body
        comment3 = {
            'id': 'edge3',
            'body': 123,  # Non-string
            'author': 'user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        result3 = self.classifier.classify_comment(comment3)
        assert result3 == RemovalType.ACTIVE.value  # Should not match deletion markers


if __name__ == "__main__":
    pytest.main([__file__])