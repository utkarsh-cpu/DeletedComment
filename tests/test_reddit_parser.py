"""
Unit tests for reddit_parser module.
Tests JSON parsing with sample Reddit data and error handling.
"""

import json
import tempfile
import os
import gzip
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from src.reddit_parser import RedditParser, ParsingStats, ValidationError, ValidationResult


class TestRedditParser:
    """Test suite for RedditParser class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.parser = RedditParser(chunk_size=5, strict_validation=False)
        
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def create_test_comments_file(self, comments_data, filename="test_comments.json", compressed=False):
        """Helper to create test comments file."""
        file_path = os.path.join(self.temp_dir, filename)
        
        if compressed:
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                for comment in comments_data:
                    f.write(json.dumps(comment) + '\n')
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                for comment in comments_data:
                    f.write(json.dumps(comment) + '\n')
                    
        return file_path
        
    def create_test_submissions_file(self, submissions_data, filename="test_submissions.json"):
        """Helper to create test submissions file."""
        file_path = os.path.join(self.temp_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for submission in submissions_data:
                f.write(json.dumps(submission) + '\n')
                
        return file_path
        
    def test_init_default_values(self):
        """Test parser initialization with default values."""
        parser = RedditParser()
        
        assert parser.chunk_size == 10000
        assert parser.strict_validation is False
        assert isinstance(parser.stats, ParsingStats)
        
    def test_init_custom_values(self):
        """Test parser initialization with custom values."""
        parser = RedditParser(chunk_size=5000, strict_validation=True)
        
        assert parser.chunk_size == 5000
        assert parser.strict_validation is True
        
    def test_parse_comments_file_valid_data(self):
        """Test parsing valid comments data."""
        # Create test comments
        test_comments = [
            {
                "id": "comment1",
                "body": "This is a test comment",
                "author": "test_user",
                "subreddit": "test",
                "created_utc": 1640995200,
                "parent_id": "t1_parent1",
                "link_id": "t3_link1",
                "score": 5
            },
            {
                "id": "comment2", 
                "body": "[deleted]",
                "author": "deleted_user",
                "subreddit": "test",
                "created_utc": 1640995300,
                "parent_id": "t1_parent2",
                "link_id": "t3_link2",
                "score": -2
            }
        ]
        
        file_path = self.create_test_comments_file(test_comments)
        
        # Parse comments
        parsed_comments = list(self.parser.parse_comments_file(file_path))
        
        # Verify results
        assert len(parsed_comments) == 2
        assert parsed_comments[0]["id"] == "comment1"
        assert parsed_comments[0]["body"] == "This is a test comment"
        assert parsed_comments[1]["id"] == "comment2"
        assert parsed_comments[1]["body"] == "[deleted]"
        
        # Check stats
        stats = self.parser.get_parsing_stats()
        assert stats.valid_records == 2
        assert stats.json_errors == 0
        
    def test_parse_comments_file_with_invalid_json(self):
        """Test parsing comments file with invalid JSON lines."""
        # Create file with mixed valid and invalid JSON
        file_path = os.path.join(self.temp_dir, "invalid_comments.json")
        with open(file_path, 'w') as f:
            f.write('{"id": "valid1", "body": "test", "author": "user", "subreddit": "test", "created_utc": 1640995200}\n')
            f.write('invalid json line\n')
            f.write('{"id": "valid2", "body": "test2", "author": "user2", "subreddit": "test", "created_utc": 1640995300}\n')
            f.write('{"incomplete": "json"\n')  # Missing closing brace
            
        # Parse with non-strict validation
        parsed_comments = list(self.parser.parse_comments_file(file_path))
        
        # Should get only valid comments
        assert len(parsed_comments) == 2
        assert parsed_comments[0]["id"] == "valid1"
        assert parsed_comments[1]["id"] == "valid2"
        
        # Check error stats
        stats = self.parser.get_parsing_stats()
        assert stats.json_errors == 2
        assert stats.valid_records == 2
        
    def test_parse_comments_file_strict_validation(self):
        """Test parsing with strict validation enabled."""
        parser = RedditParser(strict_validation=True)
        
        # Create file with invalid JSON
        file_path = os.path.join(self.temp_dir, "invalid.json")
        with open(file_path, 'w') as f:
            f.write('invalid json\n')
            
        # Should raise exception in strict mode
        with pytest.raises(json.JSONDecodeError):
            list(parser.parse_comments_file(file_path))
            
    def test_parse_comments_file_missing_file(self):
        """Test parsing non-existent file."""
        missing_file = os.path.join(self.temp_dir, "missing.json")
        
        with pytest.raises(FileNotFoundError):
            list(self.parser.parse_comments_file(missing_file))
            
    def test_parse_comments_file_compressed(self):
        """Test parsing compressed (gzip) comments file."""
        test_comments = [
            {
                "id": "compressed1",
                "body": "Compressed comment",
                "author": "user",
                "subreddit": "test",
                "created_utc": 1640995200
            }
        ]
        
        file_path = self.create_test_comments_file(test_comments, "compressed.json.gz", compressed=True)
        
        parsed_comments = list(self.parser.parse_comments_file(file_path))
        
        assert len(parsed_comments) == 1
        assert parsed_comments[0]["id"] == "compressed1"
        
    def test_parse_submissions_file_valid_data(self):
        """Test parsing valid submissions data."""
        test_submissions = [
            {
                "id": "submission1",
                "title": "Test Submission",
                "selftext": "This is a test submission",
                "author": "test_user",
                "subreddit": "test",
                "created_utc": 1640995200,
                "score": 10,
                "num_comments": 5,
                "url": "https://reddit.com/r/test/submission1"
            },
            {
                "id": "submission2",
                "title": "Another Test",
                "selftext": "",
                "author": "user2",
                "subreddit": "test",
                "created_utc": 1640995300,
                "score": 0,
                "num_comments": 0,
                "url": "https://example.com"
            }
        ]
        
        file_path = self.create_test_submissions_file(test_submissions)
        
        parsed_submissions = list(self.parser.parse_submissions_file(file_path))
        
        assert len(parsed_submissions) == 2
        assert parsed_submissions[0]["id"] == "submission1"
        assert parsed_submissions[0]["title"] == "Test Submission"
        assert parsed_submissions[1]["id"] == "submission2"
        
    def test_chunk_processor_comments(self):
        """Test chunk processing for comments."""
        # Create test data with more records than chunk size
        test_comments = []
        for i in range(12):  # More than chunk_size (5)
            test_comments.append({
                "id": f"comment{i}",
                "body": f"Comment {i}",
                "author": f"user{i}",
                "subreddit": "test",
                "created_utc": 1640995200 + i
            })
            
        file_path = self.create_test_comments_file(test_comments, "RC_test_comments.json")
        
        chunks = list(self.parser.chunk_processor(file_path))
        
        # Should have 3 chunks: 5, 5, 2
        assert len(chunks) == 3
        assert len(chunks[0]) == 5
        assert len(chunks[1]) == 5
        assert len(chunks[2]) == 2
        
        # Verify content
        assert chunks[0][0]["id"] == "comment0"
        assert chunks[2][1]["id"] == "comment11"
        
    def test_chunk_processor_submissions(self):
        """Test chunk processing for submissions."""
        test_submissions = []
        for i in range(7):
            test_submissions.append({
                "id": f"submission{i}",
                "title": f"Submission {i}",
                "author": f"user{i}",
                "subreddit": "test",
                "created_utc": 1640995200 + i
            })
            
        file_path = self.create_test_submissions_file(test_submissions, "RS_test_submissions.json")
        
        chunks = list(self.parser.chunk_processor(file_path))
        
        # Should have 2 chunks: 5, 2
        assert len(chunks) == 2
        assert len(chunks[0]) == 5
        assert len(chunks[1]) == 2
        
    def test_normalize_comment(self):
        """Test comment normalization."""
        raw_comment = {
            "id": "test123",
            "body": "Test comment body",
            "author": "test_user",
            "subreddit": "testsubreddit",
            "created_utc": 1640995200,
            "parent_id": "t1_parent123",
            "link_id": "t3_link123",
            "score": 5,
            "controversiality": 0,
            "gilded": 1,
            "distinguished": "moderator",
            "stickied": True,
            "archived": False,
            "edited": 1640995300
        }
        
        normalized = self.parser._normalize_comment(raw_comment)
        
        assert normalized["id"] == "test123"
        assert normalized["body"] == "Test comment body"
        assert normalized["author"] == "test_user"
        assert normalized["subreddit"] == "testsubreddit"
        assert normalized["created_utc"] == 1640995200
        assert normalized["parent_id"] == "t1_parent123"
        assert normalized["link_id"] == "t3_link123"
        assert normalized["score"] == 5
        assert normalized["controversiality"] == 0
        assert normalized["gilded"] == 1
        assert normalized["distinguished"] == "moderator"
        assert normalized["stickied"] is True
        assert normalized["archived"] is False
        assert normalized["edited"] == 1640995300
        
    def test_normalize_comment_missing_fields(self):
        """Test comment normalization with missing fields."""
        minimal_comment = {
            "id": "minimal123",
            "body": "Minimal comment",
            "author": "user",
            "subreddit": "test",
            "created_utc": 1640995200
        }
        
        normalized = self.parser._normalize_comment(minimal_comment)
        
        # Should have defaults for missing fields
        assert normalized["id"] == "minimal123"
        assert normalized["parent_id"] == ""
        assert normalized["score"] == 0
        assert normalized["controversiality"] == 0
        assert normalized["gilded"] == 0
        assert normalized["distinguished"] is None
        assert normalized["stickied"] is False
        
    def test_normalize_submission(self):
        """Test submission normalization."""
        raw_submission = {
            "id": "sub123",
            "title": "Test Submission Title",
            "selftext": "Submission body text",
            "author": "submission_author",
            "subreddit": "testsubreddit",
            "created_utc": 1640995200,
            "score": 15,
            "num_comments": 10,
            "url": "https://example.com",
            "domain": "example.com",
            "is_self": True,
            "over_18": False,
            "spoiler": False,
            "locked": False,
            "stickied": True,
            "archived": False
        }
        
        normalized = self.parser._normalize_submission(raw_submission)
        
        assert normalized["id"] == "sub123"
        assert normalized["title"] == "Test Submission Title"
        assert normalized["selftext"] == "Submission body text"
        assert normalized["author"] == "submission_author"
        assert normalized["subreddit"] == "testsubreddit"
        assert normalized["created_utc"] == 1640995200
        assert normalized["score"] == 15
        assert normalized["num_comments"] == 10
        assert normalized["url"] == "https://example.com"
        assert normalized["domain"] == "example.com"
        assert normalized["is_self"] is True
        assert normalized["over_18"] is False
        assert normalized["stickied"] is True
        
    def test_validate_comment_valid(self):
        """Test comment validation with valid data."""
        valid_comment = {
            "id": "valid123",
            "body": "Valid comment",
            "author": "valid_user",
            "subreddit": "validsubreddit",
            "created_utc": 1640995200
        }
        
        result = self.parser._validate_comment(valid_comment, 1, Path("test.json"))
        
        assert result.is_valid is True
        assert result.error == ""
        
    def test_validate_comment_missing_fields(self):
        """Test comment validation with missing required fields."""
        invalid_comment = {
            "id": "invalid123",
            "body": "Missing fields"
            # Missing author, subreddit, created_utc
        }
        
        result = self.parser._validate_comment(invalid_comment, 1, Path("test.json"))
        
        assert result.is_valid is False
        assert "Missing required fields" in result.error
        
    def test_validate_comment_invalid_timestamp(self):
        """Test comment validation with invalid timestamp."""
        invalid_comment = {
            "id": "invalid123",
            "body": "Invalid timestamp",
            "author": "user",
            "subreddit": "test",
            "created_utc": "not_a_number"
        }
        
        result = self.parser._validate_comment(invalid_comment, 1, Path("test.json"))
        
        assert result.is_valid is False
        assert "Invalid timestamp" in result.error
        
    def test_validate_submission_valid(self):
        """Test submission validation with valid data."""
        valid_submission = {
            "id": "valid123",
            "title": "Valid Title",
            "author": "valid_user",
            "subreddit": "validsubreddit",
            "created_utc": 1640995200
        }
        
        result = self.parser._validate_submission(valid_submission, 1, Path("test.json"))
        
        assert result.is_valid is True
        assert result.error == ""
        
    def test_validate_submission_missing_title(self):
        """Test submission validation with missing title."""
        invalid_submission = {
            "id": "invalid123",
            "author": "user",
            "subreddit": "test",
            "created_utc": 1640995200
            # Missing title
        }
        
        result = self.parser._validate_submission(invalid_submission, 1, Path("test.json"))
        
        assert result.is_valid is False
        assert "Missing required fields" in result.error
        
    def test_is_comments_file(self):
        """Test comment file detection by filename."""
        assert self.parser._is_comments_file(Path("RC_2023-01.json")) is True
        assert self.parser._is_comments_file(Path("comments_data.json")) is True
        assert self.parser._is_comments_file(Path("RS_2023-01.json")) is False
        assert self.parser._is_comments_file(Path("submissions.json")) is False
        
    def test_is_submissions_file(self):
        """Test submission file detection by filename."""
        assert self.parser._is_submissions_file(Path("RS_2023-01.json")) is True
        assert self.parser._is_submissions_file(Path("submissions_data.json")) is True
        assert self.parser._is_submissions_file(Path("RC_2023-01.json")) is False
        assert self.parser._is_submissions_file(Path("comments.json")) is False
        
    def test_safe_str(self):
        """Test safe string conversion."""
        assert self.parser._safe_str("test") == "test"
        assert self.parser._safe_str(123) == "123"
        assert self.parser._safe_str(None) == ""
        assert self.parser._safe_str(True) == "True"
        
    def test_safe_int(self):
        """Test safe integer conversion."""
        assert self.parser._safe_int(123) == 123
        assert self.parser._safe_int("456") == 456
        assert self.parser._safe_int("123.45") == 123
        assert self.parser._safe_int(None) == 0
        assert self.parser._safe_int("invalid") == 0
        
    def test_safe_bool(self):
        """Test safe boolean conversion."""
        assert self.parser._safe_bool(True) is True
        assert self.parser._safe_bool(False) is False
        assert self.parser._safe_bool("true") is True
        assert self.parser._safe_bool("false") is False
        assert self.parser._safe_bool("1") is True
        assert self.parser._safe_bool("0") is False
        assert self.parser._safe_bool(1) is True
        assert self.parser._safe_bool(0) is False
        assert self.parser._safe_bool(None) is False
        assert self.parser._safe_bool("invalid") is False
        
    def test_parsing_stats_tracking(self):
        """Test parsing statistics tracking."""
        test_comments = [
            {
                "id": "valid1",
                "body": "Valid comment",
                "author": "user",
                "subreddit": "test",
                "created_utc": 1640995200
            }
        ]
        
        # Add invalid JSON line
        file_path = os.path.join(self.temp_dir, "stats_test.json")
        with open(file_path, 'w') as f:
            f.write(json.dumps(test_comments[0]) + '\n')
            f.write('invalid json line\n')
            f.write('\n')  # Empty line
            
        list(self.parser.parse_comments_file(file_path))
        
        stats = self.parser.get_parsing_stats()
        assert stats.total_lines == 3
        assert stats.valid_records == 1
        assert stats.json_errors == 1
        
    def test_reset_stats(self):
        """Test statistics reset functionality."""
        # Parse some data to generate stats
        test_comments = [{"id": "test", "body": "test", "author": "user", "subreddit": "test", "created_utc": 1640995200}]
        file_path = self.create_test_comments_file(test_comments)
        list(self.parser.parse_comments_file(file_path))
        
        # Verify stats exist
        stats = self.parser.get_parsing_stats()
        assert stats.valid_records > 0
        
        # Reset and verify
        self.parser._reset_stats()
        new_stats = self.parser.get_parsing_stats()
        assert new_stats.valid_records == 0
        assert new_stats.total_lines == 0
        assert new_stats.json_errors == 0


if __name__ == "__main__":
    pytest.main([__file__])