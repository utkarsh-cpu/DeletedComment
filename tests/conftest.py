"""
Pytest configuration and fixtures for the deleted comment dataset tests.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for test session."""
    temp_dir = tempfile.mkdtemp(prefix="reddit_dataset_tests_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_comment():
    """Provide a sample Reddit comment for testing."""
    return {
        'id': 'test_comment_123',
        'body': 'This is a sample Reddit comment for testing purposes.',
        'author': 'test_user',
        'subreddit': 'test_subreddit',
        'created_utc': 1640995200,
        'score': 15,
        'parent_id': 't1_parent_comment',
        'link_id': 't3_submission_123',
        'controversiality': 0,
        'gilded': 1,
        'distinguished': None,
        'stickied': False,
        'archived': False,
        'edited': False
    }


@pytest.fixture
def sample_deleted_comment():
    """Provide a sample deleted Reddit comment for testing."""
    return {
        'id': 'deleted_comment_456',
        'body': '[deleted]',
        'author': 'deleted_user',
        'subreddit': 'test_subreddit',
        'created_utc': 1640995300,
        'score': -5,
        'parent_id': 't1_parent_comment',
        'link_id': 't3_submission_123',
        'controversiality': 1,
        'gilded': 0
    }


@pytest.fixture
def sample_removed_comment():
    """Provide a sample moderator-removed Reddit comment for testing."""
    return {
        'id': 'removed_comment_789',
        'body': '[removed]',
        'author': 'rule_violator',
        'subreddit': 'test_subreddit',
        'created_utc': 1640995400,
        'score': -15,
        'parent_id': 't1_parent_comment',
        'link_id': 't3_submission_123',
        'controversiality': 1,
        'gilded': 0
    }


@pytest.fixture
def sample_submission():
    """Provide a sample Reddit submission for testing."""
    return {
        'id': 'submission_123',
        'title': 'Test Submission Title',
        'selftext': 'This is the body text of a test submission.',
        'author': 'submission_author',
        'subreddit': 'test_subreddit',
        'created_utc': 1640995000,
        'score': 100,
        'num_comments': 25,
        'url': 'https://reddit.com/r/test_subreddit/submission_123',
        'domain': 'reddit.com',
        'is_self': True,
        'over_18': False,
        'spoiler': False,
        'locked': False,
        'stickied': False,
        'archived': False
    }


@pytest.fixture
def sample_training_record():
    """Provide a sample training record for testing."""
    from datetime import datetime, timezone
    
    return {
        'id': 'training_record_123',
        'comment_text': 'Sample training comment text',
        'subreddit': 'test_subreddit',
        'timestamp': datetime.now(timezone.utc),
        'removal_type': 'user_deleted',
        'target_label': 'voluntary_deletion',
        'parent_id': 'parent_123',
        'thread_id': 'thread_123',
        'score': 10,
        'author': 'test_user',
        'controversiality': 0,
        'gilded': 1,
        'comment_length': 25,
        'has_parent': True,
        'is_top_level': False
    }


@pytest.fixture
def mock_credentials_file(tmp_path):
    """Create a mock Google Drive credentials file for testing."""
    credentials_content = {
        "installed": {
            "client_id": "test_client_id",
            "project_id": "test_project",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_secret": "test_client_secret",
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
        }
    }
    
    credentials_file = tmp_path / "credentials.json"
    import json
    with open(credentials_file, 'w') as f:
        json.dump(credentials_content, f)
        
    return str(credentials_file)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add performance marker to performance tests
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            
        # Add slow marker to tests that might be slow
        if any(keyword in item.nodeid for keyword in ["large", "stress", "end_to_end"]):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Set up logging for tests."""
    import logging
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress some noisy loggers during tests
    logging.getLogger('googleapiclient').setLevel(logging.ERROR)
    logging.getLogger('google_auth_oauthlib').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)


# Performance test configuration
@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        'small_dataset_size': 1000,
        'medium_dataset_size': 10000,
        'large_dataset_size': 50000,
        'stress_dataset_size': 100000,
        'memory_limit_mb': 1000,
        'min_throughput_comments_per_sec': 500,
        'min_compression_ratio': 2.0
    }


# Skip performance tests by default unless explicitly requested
def pytest_runtest_setup(item):
    """Skip performance tests unless explicitly requested."""
    if "performance" in item.keywords:
        if not item.config.getoption("--run-performance", default=False):
            pytest.skip("Performance tests skipped (use --run-performance to run)")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run performance tests"
    )
    parser.addoption(
        "--run-integration",
        action="store_true", 
        default=False,
        help="Run integration tests"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )