"""
Unit tests for data_downloader module.
Tests download functionality with mock torrent data and error handling.
"""

import os
import tempfile
import shutil
import hashlib
import zipfile
import tarfile
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from src.data_downloader import DataDownloader


class TestDataDownloader:
    """Test suite for DataDownloader class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = DataDownloader(download_dir=self.temp_dir, max_retries=2)
        
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_init_creates_download_directory(self):
        """Test that initialization creates download directory."""
        new_dir = os.path.join(self.temp_dir, "new_download_dir")
        downloader = DataDownloader(download_dir=new_dir)
        
        assert os.path.exists(new_dir)
        assert downloader.download_dir == new_dir
        assert downloader.max_retries == 3  # default value
        
    def test_get_filename_from_url(self):
        """Test filename extraction from URLs."""
        # Test with filename in URL
        url1 = "https://example.com/reddit_data.tar.gz"
        filename1 = self.downloader._get_filename_from_url(url1)
        assert filename1 == "reddit_data.tar.gz"
        
        # Test with no filename - should generate one
        url2 = "https://example.com/"
        filename2 = self.downloader._get_filename_from_url(url2)
        assert filename2.startswith("reddit_dataset_")
        assert filename2.endswith(".tar.gz")
        
    def test_calculate_checksum(self):
        """Test checksum calculation for files."""
        # Create test file with known content
        test_file = os.path.join(self.temp_dir, "test.txt")
        test_content = b"Hello, World!"
        
        with open(test_file, 'wb') as f:
            f.write(test_content)
            
        # Calculate expected checksum
        expected_sha256 = hashlib.sha256(test_content).hexdigest()
        expected_md5 = hashlib.md5(test_content).hexdigest()
        
        # Test SHA256
        calculated_sha256 = self.downloader._calculate_checksum(test_file, "sha256")
        assert calculated_sha256 == expected_sha256
        
        # Test MD5
        calculated_md5 = self.downloader._calculate_checksum(test_file, "md5")
        assert calculated_md5 == expected_md5
        
    def test_verify_integrity_success(self):
        """Test successful integrity verification."""
        # Create test file
        test_file = os.path.join(self.temp_dir, "test.txt")
        test_content = b"Test content for integrity check"
        
        with open(test_file, 'wb') as f:
            f.write(test_content)
            
        # Calculate correct checksum
        expected_checksum = hashlib.sha256(test_content).hexdigest()
        
        # Test verification
        result = self.downloader.verify_integrity(test_file, expected_checksum, "sha256")
        assert result is True
        
    def test_verify_integrity_failure(self):
        """Test integrity verification failure."""
        # Create test file
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'wb') as f:
            f.write(b"Test content")
            
        # Use wrong checksum
        wrong_checksum = "wrong_checksum_value"
        
        result = self.downloader.verify_integrity(test_file, wrong_checksum, "sha256")
        assert result is False
        
    def test_verify_integrity_missing_file(self):
        """Test integrity verification with missing file."""
        missing_file = os.path.join(self.temp_dir, "missing.txt")
        
        result = self.downloader.verify_integrity(missing_file, "any_checksum", "sha256")
        assert result is False
        
    def test_calculate_backoff_time(self):
        """Test exponential backoff calculation."""
        # Test backoff times increase exponentially
        backoff_0 = self.downloader._calculate_backoff_time(0)
        backoff_1 = self.downloader._calculate_backoff_time(1)
        backoff_2 = self.downloader._calculate_backoff_time(2)
        
        assert 1 <= backoff_0 <= 3  # 2^0 + jitter
        assert 2 <= backoff_1 <= 5  # 2^1 + jitter
        assert 4 <= backoff_2 <= 9  # 2^2 + jitter
        
    @patch('src.data_downloader.requests.Session.get')
    def test_download_with_progress_success(self, mock_get):
        """Test successful download with progress tracking."""
        # Mock response
        mock_response = Mock()
        mock_response.headers = {'content-length': '100'}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2', b'chunk3']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test download
        test_file = os.path.join(self.temp_dir, "downloaded.txt")
        self.downloader._download_with_progress("http://example.com/file.txt", test_file)
        
        # Verify file was created and contains expected content
        assert os.path.exists(test_file)
        with open(test_file, 'rb') as f:
            content = f.read()
            assert content == b'chunk1chunk2chunk3'
            
    @patch('src.data_downloader.requests.Session.get')
    def test_download_with_progress_network_error(self, mock_get):
        """Test download with network error."""
        # Mock network error
        mock_get.side_effect = requests.RequestException("Network error")
        
        test_file = os.path.join(self.temp_dir, "failed_download.txt")
        
        with pytest.raises(requests.RequestException):
            self.downloader._download_with_progress("http://example.com/file.txt", test_file)
            
    @patch('src.data_downloader.DataDownloader._download_with_progress')
    def test_download_dataset_success(self, mock_download):
        """Test successful dataset download."""
        # Mock successful download
        mock_download.return_value = None
        
        url = "http://example.com/reddit_data.tar.gz"
        result = self.downloader.download_dataset(url)
        
        expected_path = os.path.join(self.temp_dir, "reddit_data.tar.gz")
        assert result == expected_path
        mock_download.assert_called_once()
        
    @patch('src.data_downloader.DataDownloader._download_with_progress')
    def test_download_dataset_with_integrity_check(self, mock_download):
        """Test dataset download with integrity verification."""
        # Create test file with known content
        test_content = b"Test Reddit data"
        expected_checksum = hashlib.sha256(test_content).hexdigest()
        
        def mock_download_func(url, file_path):
            with open(file_path, 'wb') as f:
                f.write(test_content)
                
        mock_download.side_effect = mock_download_func
        
        url = "http://example.com/reddit_data.tar.gz"
        result = self.downloader.download_dataset(url, expected_checksum, "sha256")
        
        expected_path = os.path.join(self.temp_dir, "reddit_data.tar.gz")
        assert result == expected_path
        assert os.path.exists(expected_path)
        
    @patch('src.data_downloader.DataDownloader._download_with_progress')
    def test_download_dataset_retry_on_failure(self, mock_download):
        """Test download retry mechanism on failure."""
        # Mock first two attempts fail, third succeeds
        mock_download.side_effect = [
            Exception("Network error"),
            Exception("Timeout error"),
            None  # Success on third attempt
        ]
        
        url = "http://example.com/reddit_data.tar.gz"
        result = self.downloader.download_dataset(url)
        
        expected_path = os.path.join(self.temp_dir, "reddit_data.tar.gz")
        assert result == expected_path
        assert mock_download.call_count == 3
        
    @patch('src.data_downloader.DataDownloader._download_with_progress')
    def test_download_dataset_max_retries_exceeded(self, mock_download):
        """Test download failure after max retries exceeded."""
        # Mock all attempts fail
        mock_download.side_effect = Exception("Persistent network error")
        
        url = "http://example.com/reddit_data.tar.gz"
        
        with pytest.raises(Exception, match="Failed to download.*after.*attempts"):
            self.downloader.download_dataset(url)
            
        # Should try max_retries + 1 times (2 + 1 = 3)
        assert mock_download.call_count == 3
        
    def test_extract_zip_file(self):
        """Test ZIP file extraction."""
        # Create test ZIP file
        zip_path = os.path.join(self.temp_dir, "test.zip")
        test_files = {
            "file1.txt": b"Content of file 1",
            "subdir/file2.txt": b"Content of file 2"
        }
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for filename, content in test_files.items():
                zf.writestr(filename, content)
                
        # Test extraction
        extracted_files = self.downloader.extract_files(zip_path)
        
        # Verify extraction
        assert len(extracted_files) > 0
        
        # Check that organized directory structure was created
        organized_dir = os.path.join(self.temp_dir, "extracted", "test", "organized")
        assert os.path.exists(organized_dir)
        
    def test_extract_tar_gz_file(self):
        """Test TAR.GZ file extraction."""
        # Create test TAR.GZ file
        tar_path = os.path.join(self.temp_dir, "test.tar.gz")
        test_files = {
            "comments_file.json": b'{"id": "test1", "body": "test comment"}',
            "submissions_file.json": b'{"id": "test2", "title": "test submission"}'
        }
        
        with tarfile.open(tar_path, 'w:gz') as tf:
            for filename, content in test_files.items():
                import io
                tarinfo = tarfile.TarInfo(name=filename)
                tarinfo.size = len(content)
                tf.addfile(tarinfo, io.BytesIO(content))
                
        # Test extraction
        extracted_files = self.downloader.extract_files(tar_path)
        
        # Verify extraction
        assert len(extracted_files) > 0
        
        # Check that files were organized by type
        organized_dir = os.path.join(self.temp_dir, "extracted", "test", "organized")
        comments_dir = os.path.join(organized_dir, "comments")
        submissions_dir = os.path.join(organized_dir, "submissions")
        
        assert os.path.exists(comments_dir)
        assert os.path.exists(submissions_dir)
        
    def test_extract_files_missing_archive(self):
        """Test extraction with missing archive file."""
        missing_archive = os.path.join(self.temp_dir, "missing.zip")
        
        with pytest.raises(FileNotFoundError):
            self.downloader.extract_files(missing_archive)
            
    def test_extract_files_unsupported_format(self):
        """Test extraction with unsupported file format."""
        # Create a file with unsupported extension
        unsupported_file = os.path.join(self.temp_dir, "test.txt")
        with open(unsupported_file, 'w') as f:
            f.write("Not an archive")
            
        with pytest.raises(ValueError, match="Unsupported archive format"):
            self.downloader.extract_files(unsupported_file)
            
    def test_organize_extracted_files(self):
        """Test file organization by type."""
        # Create test directory structure
        extract_dir = os.path.join(self.temp_dir, "extracted")
        os.makedirs(extract_dir)
        
        # Create test files
        test_files = [
            "RC_2023-01_comments.json",
            "RS_2023-01_submissions.json", 
            "readme.txt"
        ]
        
        for filename in test_files:
            file_path = os.path.join(extract_dir, filename)
            with open(file_path, 'w') as f:
                f.write(f"Content of {filename}")
                
        # Test organization
        organized_files = self.downloader._organize_extracted_files(extract_dir)
        
        # Verify organization
        assert len(organized_files) == len(test_files)
        
        organized_dir = os.path.join(extract_dir, "organized")
        assert os.path.exists(os.path.join(organized_dir, "comments"))
        assert os.path.exists(os.path.join(organized_dir, "submissions"))
        assert os.path.exists(os.path.join(organized_dir, "other"))
        
    def test_get_extraction_info(self):
        """Test extraction information gathering."""
        # Create test directory with files
        test_dir = os.path.join(self.temp_dir, "test_extraction")
        os.makedirs(test_dir)
        
        test_files = ["file1.json", "file2.txt"]
        for filename in test_files:
            file_path = os.path.join(test_dir, filename)
            with open(file_path, 'w') as f:
                f.write("Test content")
                
        # Get extraction info
        info = self.downloader.get_extraction_info(test_dir)
        
        # Verify info
        assert info["total_files"] == len(test_files)
        assert info["extract_dir"] == test_dir
        assert ".json" in info["file_types"]
        assert ".txt" in info["file_types"]
        assert info["total_size_mb"] > 0
        
    def test_get_extraction_info_missing_directory(self):
        """Test extraction info with missing directory."""
        missing_dir = os.path.join(self.temp_dir, "missing")
        
        info = self.downloader.get_extraction_info(missing_dir)
        
        assert "error" in info
        assert info["error"] == "Extraction directory not found"
        
    def test_get_download_info(self):
        """Test download directory information."""
        info = self.downloader.get_download_info()
        
        assert info["download_dir"] == self.temp_dir
        assert info["exists"] is True
        assert "free_space_gb" in info
        assert isinstance(info["free_space_gb"], (int, float))
        
    def test_is_zip_file(self):
        """Test ZIP file detection."""
        # Create actual ZIP file
        zip_path = os.path.join(self.temp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test.txt", "test content")
            
        assert self.downloader._is_zip_file(zip_path) is True
        
        # Test with non-ZIP file
        text_path = os.path.join(self.temp_dir, "test.txt")
        with open(text_path, 'w') as f:
            f.write("Not a ZIP file")
            
        assert self.downloader._is_zip_file(text_path) is False
        
    def test_is_tar_file(self):
        """Test TAR file detection."""
        # Create actual TAR file
        tar_path = os.path.join(self.temp_dir, "test.tar")
        with tarfile.open(tar_path, 'w') as tf:
            import io
            tarinfo = tarfile.TarInfo(name="test.txt")
            content = b"test content"
            tarinfo.size = len(content)
            tf.addfile(tarinfo, io.BytesIO(content))
            
        assert self.downloader._is_tar_file(tar_path) is True
        
        # Test with non-TAR file
        text_path = os.path.join(self.temp_dir, "test.txt")
        with open(text_path, 'w') as f:
            f.write("Not a TAR file")
            
        assert self.downloader._is_tar_file(text_path) is False


if __name__ == "__main__":
    pytest.main([__file__])