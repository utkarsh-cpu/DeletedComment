"""
Unit tests for metadata_extractor module.
Tests metadata extraction and training label generation.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.metadata_extractor import MetadataExtractor, TrainingRecord


class TestMetadataExtractor:
    """Test suite for MetadataExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.extractor = MetadataExtractor()
        
    def test_init_default_config(self):
        """Test extractor initialization with default configuration."""
        extractor = MetadataExtractor()
        
        assert extractor.config == {}
        assert isinstance(extractor.subreddit_rules, dict)
        assert isinstance(extractor.toxic_patterns, list)
        assert extractor.stats['total_extracted'] == 0
        
    def test_init_custom_config(self):
        """Test extractor initialization with custom configuration."""
        config = {
            'custom_setting': 'test_value'
        }
        
        extractor = MetadataExtractor(config)
        
        assert extractor.config == config
        
    def test_extract_comment_metadata_complete(self):
        """Test metadata extraction from complete comment."""
        comment = {
            'id': 'test123',
            'body': 'This is a test comment',
            'author': 'test_user',
            'subreddit': 'TestSubreddit',
            'created_utc': 1640995200,
            'score': 15,
            'parent_id': 't1_parent123',
            'link_id': 't3_link123',
            'controversiality': 1,
            'gilded': 2,
            'distinguished': 'moderator',
            'stickied': True,
            'archived': False,
            'edited': 1640995300
        }
        
        metadata = self.extractor.extract_comment_metadata(comment)
        
        assert metadata['id'] == 'test123'
        assert metadata['body'] == 'This is a test comment'
        assert metadata['author'] == 'test_user'
        assert metadata['subreddit'] == 'testsubreddit'  # Should be normalized to lowercase
        assert metadata['created_utc'] == 1640995200
        assert isinstance(metadata['timestamp'], datetime)
        assert metadata['score'] == 15
        assert metadata['parent_id'] == 'parent123'  # Should remove 't1_' prefix
        assert metadata['thread_id'] == 'link123'  # Should remove 't3_' prefix
        assert metadata['controversiality'] == 1
        assert metadata['gilded'] == 2
        assert metadata['distinguished'] == 'moderator'
        assert metadata['stickied'] is True
        assert metadata['archived'] is False
        assert metadata['edited'] == 1640995300
        assert metadata['comment_length'] == len('This is a test comment')
        assert metadata['has_parent'] is True
        assert metadata['is_top_level'] is False
        
    def test_extract_comment_metadata_minimal(self):
        """Test metadata extraction from minimal comment."""
        comment = {
            'id': 'minimal123',
            'body': 'Short',
            'author': 'user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        metadata = self.extractor.extract_comment_metadata(comment)
        
        assert metadata['id'] == 'minimal123'
        assert metadata['body'] == 'Short'
        assert metadata['author'] == 'user'
        assert metadata['subreddit'] == 'test'
        assert metadata['score'] == 0  # Default value
        assert metadata['parent_id'] == ''  # Default value
        assert metadata['controversiality'] == 0  # Default value
        assert metadata['gilded'] == 0  # Default value
        assert metadata['distinguished'] is None  # Default value
        assert metadata['stickied'] is False  # Default value
        assert metadata['archived'] is False  # Default value
        assert metadata['comment_length'] == 5
        assert metadata['has_parent'] is False
        
    def test_extract_comment_metadata_with_errors(self):
        """Test metadata extraction with malformed data."""
        comment = {
            'id': 'error123',
            'body': None,  # Invalid type
            'author': 123,  # Invalid type
            'subreddit': '',
            'created_utc': 'invalid_timestamp',  # Invalid type
            'score': 'not_a_number'  # Invalid type
        }
        
        metadata = self.extractor.extract_comment_metadata(comment)
        
        # Should handle errors gracefully with fallback values
        assert metadata['id'] == 'error123'
        assert metadata['body'] == ''  # Converted from None
        assert metadata['author'] == '123'  # Converted to string
        assert metadata['subreddit'] == ''
        assert metadata['created_utc'] == 0  # Fallback for invalid timestamp
        assert metadata['score'] == 0  # Fallback for invalid score
        
    def test_safe_extract_string(self):
        """Test safe extraction of string values."""
        data = {'key': 'value'}
        
        # Valid string
        result = self.extractor._safe_extract(data, 'key', str, 'default')
        assert result == 'value'
        
        # Missing key
        result = self.extractor._safe_extract(data, 'missing', str, 'default')
        assert result == 'default'
        
        # None value
        data_none = {'key': None}
        result = self.extractor._safe_extract(data_none, 'key', str, 'default')
        assert result == 'default'
        
        # Type conversion
        data_int = {'key': 123}
        result = self.extractor._safe_extract(data_int, 'key', str, 'default')
        assert result == '123'
        
    def test_safe_extract_int(self):
        """Test safe extraction of integer values."""
        data = {'key': 42}
        
        # Valid int
        result = self.extractor._safe_extract(data, 'key', int, 0)
        assert result == 42
        
        # String number
        data_str = {'key': '123'}
        result = self.extractor._safe_extract(data_str, 'key', int, 0)
        assert result == 123
        
        # Invalid conversion
        data_invalid = {'key': 'not_a_number'}
        result = self.extractor._safe_extract(data_invalid, 'key', int, 0)
        assert result == 0
        
    def test_convert_timestamp(self):
        """Test timestamp conversion."""
        # Valid timestamp
        timestamp = self.extractor._convert_timestamp(1640995200)
        assert isinstance(timestamp, datetime)
        assert timestamp.tzinfo == timezone.utc
        
        # Invalid timestamp
        invalid_timestamp = self.extractor._convert_timestamp('invalid')
        assert isinstance(invalid_timestamp, datetime)
        assert invalid_timestamp == datetime.fromtimestamp(0, tz=timezone.utc)
        
    def test_extract_thread_id(self):
        """Test thread ID extraction."""
        assert self.extractor._extract_thread_id('t3_abc123') == 'abc123'
        assert self.extractor._extract_thread_id('abc123') == 'abc123'
        assert self.extractor._extract_thread_id('') == ''
        assert self.extractor._extract_thread_id(None) == ''
        
    def test_clean_parent_id(self):
        """Test parent ID cleaning."""
        assert self.extractor._clean_parent_id('t1_comment123') == 'comment123'
        assert self.extractor._clean_parent_id('t3_post123') == 'post123'
        assert self.extractor._clean_parent_id('plain_id') == 'plain_id'
        assert self.extractor._clean_parent_id('') == ''
        assert self.extractor._clean_parent_id(None) == ''
        
    def test_normalize_subreddit(self):
        """Test subreddit name normalization."""
        assert self.extractor._normalize_subreddit('AskReddit') == 'askreddit'
        assert self.extractor._normalize_subreddit('  Science  ') == 'science'
        assert self.extractor._normalize_subreddit('') == ''
        assert self.extractor._normalize_subreddit(None) == ''
        
    def test_prepare_comment_text(self):
        """Test comment text preparation."""
        context = {}
        
        # Normal text
        assert self.extractor._prepare_comment_text('Normal comment', context) == 'Normal comment'
        
        # Deletion markers
        assert self.extractor._prepare_comment_text('[deleted]', context) == '[CONTENT_REMOVED]'
        assert self.extractor._prepare_comment_text('[removed]', context) == '[CONTENT_REMOVED]'
        assert self.extractor._prepare_comment_text('  [deleted]  ', context) == '[CONTENT_REMOVED]'
        
        # Invalid types
        assert self.extractor._prepare_comment_text(None, context) == '[CONTENT_UNAVAILABLE]'
        assert self.extractor._prepare_comment_text(123, context) == '[CONTENT_UNAVAILABLE]'
        
    def test_generate_training_labels_moderator_removed(self):
        """Test training label generation for moderator-removed comments."""
        context = {'removal_type': 'moderator_removed'}
        
        # Controversial content
        comment1 = {
            'subreddit': 'test',
            'score': 5,
            'controversiality': 1,
            'body': 'Test comment'
        }
        label1 = self.extractor.generate_training_labels(comment1, context)
        assert label1 == 'controversial_content'
        
        # Heavily downvoted
        comment2 = {
            'subreddit': 'test',
            'score': -15,
            'controversiality': 0,
            'body': 'Downvoted comment'
        }
        label2 = self.extractor.generate_training_labels(comment2, context)
        assert label2 == 'heavily_downvoted'
        
        # Rule violation in specific subreddit
        comment3 = {
            'subreddit': 'askreddit',
            'score': 5,
            'controversiality': 0,
            'body': 'Off-topic comment'
        }
        label3 = self.extractor.generate_training_labels(comment3, context)
        assert label3 == 'rule_violation'
        
        # Misinformation in science subreddit
        comment4 = {
            'subreddit': 'science',
            'score': 0,
            'controversiality': 0,
            'body': 'Pseudoscience claim'
        }
        label4 = self.extractor.generate_training_labels(comment4, context)
        assert label4 == 'misinformation'
        
    def test_generate_training_labels_user_deleted(self):
        """Test training label generation for user-deleted comments."""
        context = {'removal_type': 'user_deleted'}
        
        # Regret deletion
        comment1 = {
            'subreddit': 'test',
            'score': -8,
            'controversiality': 0,
            'body': '[deleted]'
        }
        label1 = self.extractor.generate_training_labels(comment1, context)
        assert label1 == 'regret_deletion'
        
        # Controversial deletion
        comment2 = {
            'subreddit': 'test',
            'score': 15,
            'controversiality': 1,
            'body': '[deleted]'
        }
        label2 = self.extractor.generate_training_labels(comment2, context)
        assert label2 == 'controversial_deletion'
        
        # High visibility deletion
        comment3 = {
            'subreddit': 'test',
            'score': 75,
            'controversiality': 0,
            'body': '[deleted]'
        }
        label3 = self.extractor.generate_training_labels(comment3, context)
        assert label3 == 'high_visibility_deletion'
        
        # Voluntary deletion
        comment4 = {
            'subreddit': 'test',
            'score': 5,
            'controversiality': 0,
            'body': '[deleted]'
        }
        label4 = self.extractor.generate_training_labels(comment4, context)
        assert label4 == 'voluntary_deletion'
        
    def test_build_training_record(self):
        """Test building complete training record."""
        comment = {
            'id': 'train123',
            'body': 'Training comment',
            'author': 'trainer',
            'subreddit': 'MachineLearning',
            'created_utc': 1640995200,
            'score': 10,
            'parent_id': 't1_parent123',
            'link_id': 't3_thread123',
            'controversiality': 0,
            'gilded': 1
        }
        
        removal_context = {
            'removal_type': 'user_deleted',
            'confidence': 0.95
        }
        
        record = self.extractor.build_training_record(comment, removal_context)
        
        assert record['id'] == 'train123'
        assert record['comment_text'] == 'Training comment'
        assert record['subreddit'] == 'machinelearning'
        assert isinstance(record['timestamp'], datetime)
        assert record['removal_type'] == 'user_deleted'
        assert record['target_label'] == 'voluntary_deletion'
        assert record['parent_id'] == 'parent123'
        assert record['thread_id'] == 'thread123'
        assert record['score'] == 10
        assert record['author'] == 'trainer'
        assert record['controversiality'] == 0
        assert record['gilded'] == 1
        assert record['comment_length'] == len('Training comment')
        assert record['has_parent'] is True
        assert record['is_top_level'] is False
        
    def test_extract_parent_context(self):
        """Test parent context extraction."""
        comment = {
            'parent_id': 't1_parent123'
        }
        
        parent_comments = {
            'parent123': {
                'body': 'This is the parent comment with some context'
            }
        }
        
        context = self.extractor.extract_parent_context(comment, parent_comments)
        
        assert context == 'This is the parent comment with some context'
        
    def test_extract_parent_context_deleted_parent(self):
        """Test parent context extraction with deleted parent."""
        comment = {
            'parent_id': 't1_parent123'
        }
        
        parent_comments = {
            'parent123': {
                'body': '[deleted]'
            }
        }
        
        context = self.extractor.extract_parent_context(comment, parent_comments)
        
        assert context is None
        
    def test_extract_parent_context_no_parent(self):
        """Test parent context extraction with no parent."""
        comment = {
            'parent_id': ''
        }
        
        context = self.extractor.extract_parent_context(comment, {})
        
        assert context is None
        
    def test_truncate_context(self):
        """Test context text truncation."""
        # Short text (no truncation needed)
        short_text = "Short comment"
        result = self.extractor._truncate_context(short_text, 100)
        assert result == "Short comment"
        
        # Long text (needs truncation)
        long_text = "This is a very long comment that needs to be truncated because it exceeds the maximum length limit that we have set for context preservation in our training data."
        result = self.extractor._truncate_context(long_text, 50)
        assert len(result) <= 53  # 50 + "..."
        assert result.endswith('...')
        
        # Non-string input
        result = self.extractor._truncate_context(None, 50)
        assert result == ''
        
    def test_handle_missing_values(self):
        """Test missing value handling."""
        record = {
            'id': 'test123',
            'comment_text': None,
            'subreddit': '',
            'score': None,
            'has_parent': None,
            'timestamp': None
        }
        
        handled_record = self.extractor._handle_missing_values(record)
        
        assert handled_record['comment_text'] == '[CONTENT_UNAVAILABLE]'
        assert handled_record['subreddit'] == 'unknown'
        assert handled_record['score'] == 0
        assert handled_record['has_parent'] is False
        assert isinstance(handled_record['timestamp'], datetime)
        
    def test_assess_data_quality(self):
        """Test data quality assessment."""
        # High quality record
        high_quality_record = {
            'id': 'hq123',
            'comment_text': 'Good quality comment',
            'subreddit': 'test',
            'timestamp': datetime.now(timezone.utc),
            'removal_type': 'user_deleted',
            'target_label': 'voluntary_deletion',
            'score': 10,
            'parent_id': 'parent123',
            'parent_context': 'Parent comment context'
        }
        
        quality = self.extractor._assess_data_quality(high_quality_record)
        
        assert quality['has_content'] is True
        assert quality['has_context'] is True
        assert quality['has_metadata'] is True
        assert quality['training_ready'] is True
        assert quality['completeness_score'] > 0.8
        
        # Low quality record
        low_quality_record = {
            'id': 'lq123',
            'comment_text': '[CONTENT_UNAVAILABLE]',
            'subreddit': 'unknown',
            'removal_type': 'unknown',
            'score': 0
        }
        
        quality = self.extractor._assess_data_quality(low_quality_record)
        
        assert quality['has_content'] is False
        assert quality['has_metadata'] is False
        assert quality['training_ready'] is False
        assert quality['completeness_score'] < 0.5
        
    def test_batch_extract_with_context(self):
        """Test batch extraction with context resolution."""
        comments = [
            {
                'id': 'comment1',
                'body': 'First comment',
                'author': 'user1',
                'subreddit': 'test',
                'created_utc': 1640995200,
                'link_id': 't3_thread1'
            },
            {
                'id': 'comment2',
                'body': 'Second comment',
                'author': 'user2',
                'subreddit': 'test',
                'created_utc': 1640995300,
                'parent_id': 't1_comment1',
                'link_id': 't3_thread1'
            }
        ]
        
        submissions = {
            'thread1': {
                'title': 'Test Thread',
                'selftext': 'Thread description',
                'score': 100,
                'num_comments': 50
            }
        }
        
        records = self.extractor.batch_extract_with_context(comments, submissions)
        
        assert len(records) == 2
        
        # Check first record
        record1 = records[0]
        assert record1['id'] == 'comment1'
        assert record1['thread_title'] == 'Test Thread'
        assert record1['thread_selftext'] == 'Thread description'
        assert record1['thread_score'] == 100
        
        # Check second record (should have parent context)
        record2 = records[1]
        assert record2['id'] == 'comment2'
        assert record2['parent_context'] == 'First comment'  # From parent lookup
        
    def test_get_extraction_stats(self):
        """Test extraction statistics tracking."""
        # Extract some metadata to generate stats
        comment = {
            'id': 'stats123',
            'body': 'Stats test',
            'author': 'user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        self.extractor.extract_comment_metadata(comment)
        
        stats = self.extractor.get_extraction_stats()
        
        assert stats['total_extracted'] == 1
        assert stats['successful_extractions'] == 1
        assert stats['failed_extractions'] == 0
        assert stats['success_rate'] == 100.0
        assert isinstance(stats['label_distribution'], dict)
        
    def test_reset_stats(self):
        """Test statistics reset."""
        # Generate some stats
        comment = {
            'id': 'reset123',
            'body': 'Reset test',
            'author': 'user',
            'subreddit': 'test',
            'created_utc': 1640995200
        }
        
        self.extractor.extract_comment_metadata(comment)
        
        # Verify stats exist
        stats = self.extractor.get_extraction_stats()
        assert stats['total_extracted'] > 0
        
        # Reset and verify
        self.extractor.reset_stats()
        new_stats = self.extractor.get_extraction_stats()
        assert new_stats['total_extracted'] == 0
        assert new_stats['successful_extractions'] == 0
        
    def test_contains_toxic_patterns(self):
        """Test toxic pattern detection."""
        # Toxic content
        assert self.extractor._contains_toxic_patterns('I hate this stupid thing') is True
        assert self.extractor._contains_toxic_patterns('What the f*ck is this') is True
        assert self.extractor._contains_toxic_patterns('THIS IS TERRIBLE!!!') is True
        
        # Clean content
        assert self.extractor._contains_toxic_patterns('This is a nice comment') is False
        assert self.extractor._contains_toxic_patterns('I disagree with your opinion') is False
        
        # Edge cases
        assert self.extractor._contains_toxic_patterns('') is False
        assert self.extractor._contains_toxic_patterns(None) is False


if __name__ == "__main__":
    pytest.main([__file__])