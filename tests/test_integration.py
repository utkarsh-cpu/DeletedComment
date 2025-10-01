"""
Integration tests for the deleted comment dataset pipeline.
Tests end-to-end processing with sample Reddit data.
"""

import os
import tempfile
import shutil
import json
import pytest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import pandas as pd

from src.reddit_parser import RedditParser
from src.comment_classifier import CommentClassifier
from src.metadata_extractor import MetadataExtractor
from src.parquet_writer import ParquetWriter


class TestIntegration:
    """Integration test suite for the complete pipeline."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize components
        self.parser = RedditParser(chunk_size=10, strict_validation=False)
        self.classifier = CommentClassifier()
        self.extractor = MetadataExtractor()
        self.writer = ParquetWriter()
        
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def create_sample_reddit_data(self, num_comments=20):
        """Create sample Reddit data for testing."""
        comments = []
        
        for i in range(num_comments):
            # Create mix of active, deleted, and removed comments
            if i % 4 == 0:
                # User deleted comment
                comment = {
                    'id': f'deleted_{i}',
                    'body': '[deleted]',
                    'author': 'user_' + str(i),
                    'subreddit': 'test_subreddit',
                    'created_utc': 1640995200 + i * 3600,
                    'score': i - 5,  # Mix of positive and negative scores
                    'parent_id': f't1_parent_{i-1}' if i > 0 else f't3_thread_{i//5}',
                    'link_id': f't3_thread_{i//5}',
                    'controversiality': 1 if i % 10 == 0 else 0,
                    'gilded': 1 if i % 15 == 0 else 0
                }
            elif i % 4 == 1:
                # Moderator removed comment
                comment = {
                    'id': f'removed_{i}',
                    'body': '[removed]',
                    'author': 'user_' + str(i),
                    'subreddit': 'test_subreddit',
                    'created_utc': 1640995200 + i * 3600,
                    'score': i - 10,  # Generally more negative
                    'parent_id': f't1_parent_{i-1}' if i > 0 else f't3_thread_{i//5}',
                    'link_id': f't3_thread_{i//5}',
                    'controversiality': 1 if i % 7 == 0 else 0,
                    'gilded': 0
                }
            elif i % 4 == 2:
                # Deleted by author (account deleted)
                comment = {
                    'id': f'account_deleted_{i}',
                    'body': 'This was a normal comment before account deletion',
                    'author': '[deleted]',
                    'subreddit': 'test_subreddit',
                    'created_utc': 1640995200 + i * 3600,
                    'score': i,
                    'parent_id': f't1_parent_{i-1}' if i > 0 else f't3_thread_{i//5}',
                    'link_id': f't3_thread_{i//5}',
                    'controversiality': 0,
                    'gilded': 0
                }
            else:
                # Active comment
                comment = {
                    'id': f'active_{i}',
                    'body': f'This is an active comment number {i} with some content',
                    'author': f'active_user_{i}',
                    'subreddit': 'test_subreddit',
                    'created_utc': 1640995200 + i * 3600,
                    'score': i + 5,  # Generally positive
                    'parent_id': f't1_parent_{i-1}' if i > 0 else f't3_thread_{i//5}',
                    'link_id': f't3_thread_{i//5}',
                    'controversiality': 0,
                    'gilded': 1 if i % 20 == 0 else 0
                }
                
            comments.append(comment)
            
        return comments
        
    def create_sample_submissions(self, num_submissions=5):
        """Create sample Reddit submissions for context."""
        submissions = {}
        
        for i in range(num_submissions):
            submission = {
                'id': f'thread_{i}',
                'title': f'Test Thread {i}: Discussion Topic',
                'selftext': f'This is the description for test thread {i}. It contains some context.',
                'author': f'thread_author_{i}',
                'subreddit': 'test_subreddit',
                'created_utc': 1640995000 + i * 7200,
                'score': (i + 1) * 50,
                'num_comments': 20 + i * 10,
                'url': f'https://reddit.com/r/test_subreddit/thread_{i}',
                'domain': 'reddit.com'
            }
            submissions[f'thread_{i}'] = submission
            
        return submissions
        
    def create_test_data_file(self, comments, filename="test_comments.json"):
        """Create a test data file with Reddit comments."""
        file_path = Path(self.temp_dir) / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for comment in comments:
                f.write(json.dumps(comment) + '\n')
                
        return file_path
        
    def test_end_to_end_pipeline_small_dataset(self):
        """Test complete pipeline with small dataset."""
        # Create sample data
        comments = self.create_sample_reddit_data(12)
        submissions = self.create_sample_submissions(3)
        
        # Create test file
        comments_file = self.create_test_data_file(comments)
        
        # Step 1: Parse comments
        parsed_comments = list(self.parser.parse_comments_file(comments_file))
        
        assert len(parsed_comments) == 12
        
        # Step 2: Classify comments
        classified_comments = []
        for comment in parsed_comments:
            classification = self.classifier.classify_comment(comment)
            removal_context = self.classifier.extract_removal_context(comment)
            
            classified_comment = comment.copy()
            classified_comment['classification'] = classification
            classified_comment['removal_context'] = removal_context
            classified_comments.append(classified_comment)
            
        # Verify classifications
        user_deleted = [c for c in classified_comments if c['classification'] == 'user_deleted']
        moderator_removed = [c for c in classified_comments if c['classification'] == 'moderator_removed']
        active = [c for c in classified_comments if c['classification'] == 'active']
        
        assert len(user_deleted) > 0  # Should have some user-deleted comments
        assert len(moderator_removed) > 0  # Should have some moderator-removed comments
        assert len(active) > 0  # Should have some active comments
        
        # Step 3: Extract metadata and build training records
        training_records = []
        for comment in classified_comments:
            if comment['classification'] in ['user_deleted', 'moderator_removed']:
                record = self.extractor.build_enhanced_training_record(
                    comment=comment,
                    removal_context=comment['removal_context'],
                    parent_comments={c['id']: c for c in parsed_comments},
                    thread_context=submissions.get(
                        self.extractor._extract_thread_id(comment.get('link_id', ''))
                    )
                )
                training_records.append(record)
                
        assert len(training_records) > 0
        
        # Step 4: Write to Parquet
        output_dir = Path(self.temp_dir) / "output"
        
        # Separate by removal type
        user_deleted_records = [r for r in training_records if r['removal_type'] == 'user_deleted']
        moderator_removed_records = [r for r in training_records if r['removal_type'] == 'moderator_removed']
        
        dataset_paths = self.writer.export_training_datasets(
            iter(user_deleted_records),
            iter(moderator_removed_records),
            output_dir
        )
        
        # Verify output files
        if user_deleted_records:
            assert 'user_deleted' in dataset_paths
            user_deleted_path = Path(dataset_paths['user_deleted'])
            assert user_deleted_path.exists()
            
            # Verify file contents
            df_user = pd.read_parquet(user_deleted_path)
            assert len(df_user) == len(user_deleted_records)
            assert all(df_user['removal_type'] == 'user_deleted')
            
        if moderator_removed_records:
            assert 'moderator_removed' in dataset_paths
            moderator_removed_path = Path(dataset_paths['moderator_removed'])
            assert moderator_removed_path.exists()
            
            # Verify file contents
            df_mod = pd.read_parquet(moderator_removed_path)
            assert len(df_mod) == len(moderator_removed_records)
            assert all(df_mod['removal_type'] == 'moderator_removed')
            
        # Step 5: Verify data quality
        for record in training_records:
            quality = self.extractor._assess_data_quality(record)
            
            # All records should have basic metadata
            assert quality['has_metadata'] is True
            
            # Records with original content should have content
            if record['comment_text'] not in ['[CONTENT_REMOVED]', '[CONTENT_UNAVAILABLE]']:
                assert quality['has_content'] is True
                
    def test_pipeline_with_chunked_processing(self):
        """Test pipeline with chunked processing for memory efficiency."""
        # Create larger dataset
        comments = self.create_sample_reddit_data(50)
        comments_file = self.create_test_data_file(comments)
        
        # Process in chunks
        chunk_size = 10
        all_training_records = []
        
        for chunk in self.parser.chunk_processor(comments_file, chunk_size):
            # Process each chunk
            chunk_training_records = []
            
            for comment in chunk:
                classification = self.classifier.classify_comment(comment)
                
                if classification in ['user_deleted', 'moderator_removed']:
                    removal_context = self.classifier.extract_removal_context(comment)
                    record = self.extractor.build_training_record(comment, removal_context)
                    chunk_training_records.append(record)
                    
            all_training_records.extend(chunk_training_records)
            
        # Verify chunked processing worked
        assert len(all_training_records) > 0
        
        # Write chunked Parquet files
        output_dir = Path(self.temp_dir) / "chunked_output"
        
        chunk_files = self.writer.create_chunked_parquet(
            iter(all_training_records),
            output_dir,
            "chunked_dataset",
            chunk_size=15
        )
        
        assert len(chunk_files) > 1  # Should create multiple chunks
        
        # Verify all chunks can be read
        total_records = 0
        for chunk_file in chunk_files:
            df = pd.read_parquet(chunk_file)
            total_records += len(df)
            
        assert total_records == len(all_training_records)
        
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with malformed data."""
        # Create data with various issues
        problematic_comments = [
            # Valid comment
            {
                'id': 'valid1',
                'body': 'Valid comment',
                'author': 'user1',
                'subreddit': 'test',
                'created_utc': 1640995200
            },
            # Missing required fields
            {
                'id': 'invalid1',
                'body': 'Missing fields'
                # Missing author, subreddit, created_utc
            },
            # Invalid data types
            {
                'id': 'invalid2',
                'body': None,
                'author': 123,
                'subreddit': '',
                'created_utc': 'not_a_timestamp'
            },
            # Valid deleted comment
            {
                'id': 'deleted1',
                'body': '[deleted]',
                'author': 'user2',
                'subreddit': 'test',
                'created_utc': 1640995300
            }
        ]
        
        # Add some invalid JSON lines
        comments_file = Path(self.temp_dir) / "problematic_comments.json"
        with open(comments_file, 'w') as f:
            for comment in problematic_comments:
                f.write(json.dumps(comment) + '\n')
            f.write('invalid json line\n')  # Invalid JSON
            f.write('{"incomplete": "json"\n')  # Incomplete JSON
            
        # Parse with error handling (non-strict mode)
        parsed_comments = list(self.parser.parse_comments_file(comments_file))
        
        # Should get valid comments despite errors
        assert len(parsed_comments) >= 2  # At least the valid ones
        
        # Check parsing stats
        stats = self.parser.get_parsing_stats()
        assert stats.json_errors > 0  # Should have detected JSON errors
        
        # Continue with classification despite parsing errors
        training_records = []
        for comment in parsed_comments:
            try:
                classification = self.classifier.classify_comment(comment)
                if classification in ['user_deleted', 'moderator_removed']:
                    removal_context = self.classifier.extract_removal_context(comment)
                    record = self.extractor.build_training_record(comment, removal_context)
                    training_records.append(record)
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing comment {comment.get('id', 'unknown')}: {e}")
                continue
                
        # Should have at least some valid training records
        assert len(training_records) > 0
        
    def test_pipeline_with_context_resolution(self):
        """Test pipeline with parent comment and thread context resolution."""
        # Create hierarchical comment structure
        comments = [
            # Thread starter (top-level comment)
            {
                'id': 'top_comment',
                'body': 'This is the top-level comment that starts the discussion',
                'author': 'thread_starter',
                'subreddit': 'test',
                'created_utc': 1640995200,
                'parent_id': 't3_thread_1',  # Points to submission
                'link_id': 't3_thread_1',
                'score': 50
            },
            # Reply to top comment
            {
                'id': 'reply_1',
                'body': 'This is a reply to the top comment',
                'author': 'replier_1',
                'subreddit': 'test',
                'created_utc': 1640995300,
                'parent_id': 't1_top_comment',  # Points to top comment
                'link_id': 't3_thread_1',
                'score': 20
            },
            # Deleted reply
            {
                'id': 'deleted_reply',
                'body': '[deleted]',
                'author': 'deleted_user',
                'subreddit': 'test',
                'created_utc': 1640995400,
                'parent_id': 't1_reply_1',  # Points to previous reply
                'link_id': 't3_thread_1',
                'score': -5
            }
        ]
        
        submissions = {
            'thread_1': {
                'title': 'Interesting Discussion Topic',
                'selftext': 'What are your thoughts on this topic?',
                'score': 100,
                'num_comments': 25
            }
        }
        
        # Process with context resolution
        enhanced_records = self.extractor.batch_extract_with_context(comments, submissions)
        
        assert len(enhanced_records) == 3
        
        # Check context resolution
        reply_record = next(r for r in enhanced_records if r['id'] == 'reply_1')
        assert reply_record['parent_context'] == 'This is the top-level comment that starts the discussion'
        assert reply_record['thread_title'] == 'Interesting Discussion Topic'
        
        deleted_record = next(r for r in enhanced_records if r['id'] == 'deleted_reply')
        assert deleted_record['parent_context'] == 'This is a reply to the top comment'
        assert deleted_record['thread_title'] == 'Interesting Discussion Topic'
        
    def test_pipeline_data_quality_assessment(self):
        """Test data quality assessment throughout the pipeline."""
        # Create comments with varying quality levels
        comments = [
            # High quality: complete data, original content
            {
                'id': 'high_quality',
                'body': 'This is a high-quality comment with good content',
                'author': 'quality_user',
                'subreddit': 'test',
                'created_utc': 1640995200,
                'score': 15,
                'parent_id': 't1_parent',
                'link_id': 't3_thread',
                'controversiality': 0,
                'gilded': 1
            },
            # Medium quality: deleted content but good metadata
            {
                'id': 'medium_quality',
                'body': '[deleted]',
                'author': 'deleted_user',
                'subreddit': 'test',
                'created_utc': 1640995300,
                'score': 5,
                'parent_id': 't1_parent',
                'link_id': 't3_thread'
            },
            # Low quality: minimal data
            {
                'id': 'low_quality',
                'body': '[removed]',
                'author': '',
                'subreddit': '',
                'created_utc': 0
            }
        ]
        
        # Process and assess quality
        quality_assessments = []
        
        for comment in comments:
            classification = self.classifier.classify_comment(comment)
            
            if classification in ['user_deleted', 'moderator_removed']:
                removal_context = self.classifier.extract_removal_context(comment)
                record = self.extractor.build_enhanced_training_record(comment, removal_context)
                
                quality = self.extractor._assess_data_quality(record)
                quality_assessments.append({
                    'id': record['id'],
                    'quality': quality
                })
                
        # Verify quality assessments
        high_quality = next(q for q in quality_assessments if q['id'] == 'high_quality')
        assert high_quality['quality']['training_ready'] is True
        assert high_quality['quality']['completeness_score'] > 0.8
        
        medium_quality = next(q for q in quality_assessments if q['id'] == 'medium_quality')
        assert medium_quality['quality']['has_metadata'] is True
        
        low_quality = next(q for q in quality_assessments if q['id'] == 'low_quality')
        assert low_quality['quality']['training_ready'] is False
        assert low_quality['quality']['completeness_score'] < 0.5
        
    def test_pipeline_statistics_tracking(self):
        """Test statistics tracking throughout the pipeline."""
        # Create sample data
        comments = self.create_sample_reddit_data(30)
        comments_file = self.create_test_data_file(comments)
        
        # Reset all component stats
        self.parser._reset_stats()
        self.classifier.reset_stats()
        self.extractor.reset_stats()
        self.writer.reset_stats()
        
        # Process through pipeline
        parsed_comments = list(self.parser.parse_comments_file(comments_file))
        
        training_records = []
        for comment in parsed_comments:
            classification = self.classifier.classify_comment(comment)
            
            if classification in ['user_deleted', 'moderator_removed']:
                removal_context = self.classifier.extract_removal_context(comment)
                record = self.extractor.build_training_record(comment, removal_context)
                training_records.append(record)
                
        # Write to Parquet
        if training_records:
            output_path = Path(self.temp_dir) / "stats_test.parquet"
            self.writer.write_dataset(iter(training_records), output_path)
            
        # Check statistics
        parser_stats = self.parser.get_parsing_stats()
        assert parser_stats.total_lines == 30
        assert parser_stats.valid_records == 30
        
        classifier_stats = self.classifier.get_classification_stats()
        assert classifier_stats['total_classified'] == 30
        assert classifier_stats['user_deleted'] > 0
        assert classifier_stats['moderator_removed'] > 0
        
        extractor_stats = self.extractor.get_extraction_stats()
        assert extractor_stats['total_extracted'] >= len(training_records)
        assert extractor_stats['success_rate'] > 0
        
        if training_records:
            writer_stats = self.writer.get_compression_stats()
            assert writer_stats['files_created'] == 1
            assert writer_stats['total_records'] == len(training_records)
            
    def test_pipeline_schema_consistency(self):
        """Test that pipeline produces consistent schema across different data."""
        # Create different types of data
        datasets = [
            self.create_sample_reddit_data(10),  # Standard data
            [  # Minimal data
                {
                    'id': 'minimal1',
                    'body': '[deleted]',
                    'author': 'user',
                    'subreddit': 'test',
                    'created_utc': 1640995200
                }
            ],
            [  # Rich data with all fields
                {
                    'id': 'rich1',
                    'body': '[removed]',
                    'author': 'rich_user',
                    'subreddit': 'rich_test',
                    'created_utc': 1640995200,
                    'score': 100,
                    'parent_id': 't1_parent',
                    'link_id': 't3_thread',
                    'controversiality': 1,
                    'gilded': 5,
                    'distinguished': 'moderator',
                    'stickied': True,
                    'archived': False
                }
            ]
        ]
        
        all_schemas = []
        
        for i, dataset in enumerate(datasets):
            # Process dataset
            training_records = []
            
            for comment in dataset:
                classification = self.classifier.classify_comment(comment)
                
                if classification in ['user_deleted', 'moderator_removed']:
                    removal_context = self.classifier.extract_removal_context(comment)
                    record = self.extractor.build_training_record(comment, removal_context)
                    training_records.append(record)
                    
            if training_records:
                # Write to Parquet and check schema
                output_path = Path(self.temp_dir) / f"schema_test_{i}.parquet"
                self.writer.write_dataset(iter(training_records), output_path)
                
                # Read back and check schema
                df = pd.read_parquet(output_path)
                schema = set(df.columns)
                all_schemas.append(schema)
                
        # Verify all schemas are consistent
        if len(all_schemas) > 1:
            base_schema = all_schemas[0]
            for schema in all_schemas[1:]:
                # All schemas should have the same core columns
                core_columns = {
                    'id', 'comment_text', 'subreddit', 'timestamp', 
                    'removal_type', 'target_label'
                }
                assert core_columns.issubset(schema)
                assert core_columns.issubset(base_schema)


if __name__ == "__main__":
    pytest.main([__file__])