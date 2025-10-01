"""
Unit tests for drive_uploader module.
Tests Google Drive API integration with mocked responses.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from googleapiclient.errors import HttpError

from src.drive_uploader import DriveUploader


class TestDriveUploader:
    """Test suite for DriveUploader class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.credentials_path = os.path.join(self.temp_dir, "credentials.json")
        self.token_path = os.path.join(self.temp_dir, "token.json")
        
        # Create mock credentials file
        with open(self.credentials_path, 'w') as f:
            f.write('{"client_id": "test", "client_secret": "test"}')
            
        self.uploader = DriveUploader(
            credentials_path=self.credentials_path,
            token_path=self.token_path
        )
        
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_init(self):
        """Test uploader initialization."""
        uploader = DriveUploader("creds.json", "token.json")
        
        assert uploader.credentials_path == "creds.json"
        assert uploader.token_path == "token.json"
        assert uploader.service is None
        assert DriveUploader.SCOPES == ['https://www.googleapis.com/auth/drive.file']
        
    @patch('src.drive_uploader.build')
    @patch('src.drive_uploader.Credentials.from_authorized_user_file')
    def test_authenticate_with_existing_valid_token(self, mock_from_file, mock_build):
        """Test authentication with existing valid token."""
        # Mock existing valid credentials
        mock_creds = Mock()
        mock_creds.valid = True
        mock_from_file.return_value = mock_creds
        
        # Mock service build
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Create token file
        with open(self.token_path, 'w') as f:
            f.write('{"token": "test"}')
            
        result = self.uploader.authenticate()
        
        assert result is True
        assert self.uploader.service == mock_service
        mock_from_file.assert_called_once()
        mock_build.assert_called_once()
        
    @patch('src.drive_uploader.build')
    @patch('src.drive_uploader.Credentials.from_authorized_user_file')
    def test_authenticate_with_expired_token_refresh_success(self, mock_from_file, mock_build):
        """Test authentication with expired token that refreshes successfully."""
        # Mock expired credentials that can be refreshed
        mock_creds = Mock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh_token"
        mock_creds.refresh = Mock()  # Successful refresh
        mock_from_file.return_value = mock_creds
        
        # After refresh, make it valid
        def make_valid(request):
            mock_creds.valid = True
            
        mock_creds.refresh.side_effect = make_valid
        
        # Mock service build
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Create token file
        with open(self.token_path, 'w') as f:
            f.write('{"token": "test"}')
            
        result = self.uploader.authenticate()
        
        assert result is True
        assert self.uploader.service == mock_service
        mock_creds.refresh.assert_called_once()
        
    @patch('src.drive_uploader.build')
    @patch('src.drive_uploader.InstalledAppFlow.from_client_secrets_file')
    def test_authenticate_new_oauth_flow(self, mock_flow_class, mock_build):
        """Test authentication with new OAuth flow."""
        # Mock OAuth flow
        mock_flow = Mock()
        mock_creds = Mock()
        mock_creds.valid = True
        mock_flow.run_local_server.return_value = mock_creds
        mock_flow_class.return_value = mock_flow
        
        # Mock service build
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Mock credentials.to_json for saving
        mock_creds.to_json.return_value = '{"token": "new_token"}'
        
        result = self.uploader.authenticate()
        
        assert result is True
        assert self.uploader.service == mock_service
        mock_flow.run_local_server.assert_called_once()
        
        # Verify token was saved
        assert os.path.exists(self.token_path)
        
    def test_authenticate_missing_credentials_file(self):
        """Test authentication with missing credentials file."""
        # Remove credentials file
        os.remove(self.credentials_path)
        
        result = self.uploader.authenticate()
        
        assert result is False
        assert self.uploader.service is None
        
    @patch('src.drive_uploader.build')
    @patch('src.drive_uploader.InstalledAppFlow.from_client_secrets_file')
    def test_authenticate_oauth_flow_failure(self, mock_flow_class, mock_build):
        """Test authentication with OAuth flow failure."""
        # Mock OAuth flow failure
        mock_flow_class.side_effect = Exception("OAuth flow failed")
        
        result = self.uploader.authenticate()
        
        assert result is False
        assert self.uploader.service is None
        
    def test_test_connection_not_authenticated(self):
        """Test connection test without authentication."""
        result = self.uploader.test_connection()
        
        assert result is False
        
    def test_test_connection_success(self):
        """Test successful connection test."""
        # Mock authenticated service
        mock_service = Mock()
        mock_about = Mock()
        mock_about.get.return_value.execute.return_value = {
            'user': {'emailAddress': 'test@example.com'}
        }
        mock_service.about.return_value = mock_about
        
        self.uploader.service = mock_service
        
        result = self.uploader.test_connection()
        
        assert result is True
        
    def test_test_connection_http_error(self):
        """Test connection test with HTTP error."""
        # Mock authenticated service with error
        mock_service = Mock()
        mock_about = Mock()
        mock_error = HttpError(Mock(status=401), b'Unauthorized')
        mock_about.get.return_value.execute.side_effect = mock_error
        mock_service.about.return_value = mock_about
        
        self.uploader.service = mock_service
        
        result = self.uploader.test_connection()
        
        assert result is False
        
    def test_get_credentials_info_not_authenticated(self):
        """Test getting credentials info when not authenticated."""
        info = self.uploader.get_credentials_info()
        
        assert info['authenticated'] is False
        assert info['credentials_file_exists'] is True  # We created it in setup
        assert info['token_file_exists'] is False
        assert info['user_email'] is None
        
    def test_get_credentials_info_authenticated(self):
        """Test getting credentials info when authenticated."""
        # Mock authenticated service
        mock_service = Mock()
        mock_about = Mock()
        mock_about.get.return_value.execute.return_value = {
            'user': {'emailAddress': 'test@example.com'}
        }
        mock_service.about.return_value = mock_about
        
        self.uploader.service = mock_service
        
        info = self.uploader.get_credentials_info()
        
        assert info['authenticated'] is True
        assert info['user_email'] == 'test@example.com'
        
    def test_create_folder_not_authenticated(self):
        """Test folder creation without authentication."""
        result = self.uploader.create_folder("test_folder")
        
        assert result is None
        
    def test_create_folder_success(self):
        """Test successful folder creation."""
        # Mock authenticated service
        mock_service = Mock()
        mock_files = Mock()
        mock_files.create.return_value.execute.return_value = {'id': 'folder123'}
        mock_service.files.return_value = mock_files
        
        self.uploader.service = mock_service
        
        result = self.uploader.create_folder("test_folder")
        
        assert result == 'folder123'
        
        # Verify API call
        mock_files.create.assert_called_once()
        call_args = mock_files.create.call_args
        assert call_args[1]['body']['name'] == 'test_folder'
        assert call_args[1]['body']['mimeType'] == 'application/vnd.google-apps.folder'
        
    def test_create_folder_with_parent(self):
        """Test folder creation with parent folder."""
        # Mock authenticated service
        mock_service = Mock()
        mock_files = Mock()
        mock_files.create.return_value.execute.return_value = {'id': 'folder123'}
        mock_service.files.return_value = mock_files
        
        self.uploader.service = mock_service
        
        result = self.uploader.create_folder("test_folder", "parent123")
        
        assert result == 'folder123'
        
        # Verify parent was set
        call_args = mock_files.create.call_args
        assert call_args[1]['body']['parents'] == ['parent123']
        
    def test_create_folder_http_error(self):
        """Test folder creation with HTTP error."""
        # Mock authenticated service with error
        mock_service = Mock()
        mock_files = Mock()
        mock_error = HttpError(Mock(status=403), b'Forbidden')
        mock_files.create.return_value.execute.side_effect = mock_error
        mock_service.files.return_value = mock_files
        
        self.uploader.service = mock_service
        
        result = self.uploader.create_folder("test_folder")
        
        assert result is None
        
    def test_find_folder_success(self):
        """Test successful folder finding."""
        # Mock authenticated service
        mock_service = Mock()
        mock_files = Mock()
        mock_files.list.return_value.execute.return_value = {
            'files': [{'id': 'folder123', 'name': 'test_folder'}]
        }
        mock_service.files.return_value = mock_files
        
        self.uploader.service = mock_service
        
        result = self.uploader.find_folder("test_folder")
        
        assert result == 'folder123'
        
        # Verify search query
        call_args = mock_files.list.call_args
        expected_query = "name='test_folder' and mimeType='application/vnd.google-apps.folder'"
        assert call_args[1]['q'] == expected_query
        
    def test_find_folder_not_found(self):
        """Test folder finding when folder doesn't exist."""
        # Mock authenticated service
        mock_service = Mock()
        mock_files = Mock()
        mock_files.list.return_value.execute.return_value = {'files': []}
        mock_service.files.return_value = mock_files
        
        self.uploader.service = mock_service
        
        result = self.uploader.find_folder("nonexistent_folder")
        
        assert result is None
        
    def test_get_or_create_folder_existing(self):
        """Test get_or_create_folder with existing folder."""
        # Mock find_folder to return existing folder
        with patch.object(self.uploader, 'find_folder', return_value='existing123'):
            result = self.uploader.get_or_create_folder("existing_folder")
            
        assert result == 'existing123'
        
    def test_get_or_create_folder_create_new(self):
        """Test get_or_create_folder creating new folder."""
        # Mock find_folder to return None, create_folder to return new ID
        with patch.object(self.uploader, 'find_folder', return_value=None), \
             patch.object(self.uploader, 'create_folder', return_value='new123'):
            result = self.uploader.get_or_create_folder("new_folder")
            
        assert result == 'new123'
        
    def test_upload_file_not_authenticated(self):
        """Test file upload without authentication."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
            
        result = self.uploader.upload_file(test_file)
        
        assert result is None
        
    def test_upload_file_missing_file(self):
        """Test file upload with missing file."""
        # Mock authenticated service
        self.uploader.service = Mock()
        
        missing_file = os.path.join(self.temp_dir, "missing.txt")
        
        result = self.uploader.upload_file(missing_file)
        
        assert result is None
        
    @patch('src.drive_uploader.MediaFileUpload')
    def test_upload_file_success(self, mock_media_upload):
        """Test successful file upload."""
        # Create test file
        test_file = os.path.join(self.temp_dir, "test.txt")
        test_content = "test content for upload"
        with open(test_file, 'w') as f:
            f.write(test_content)
            
        # Mock authenticated service
        mock_service = Mock()
        mock_files = Mock()
        
        # Mock upload request and response
        mock_request = Mock()
        mock_request.next_chunk.return_value = (None, {'id': 'file123'})  # Upload complete
        mock_files.create.return_value = mock_request
        mock_service.files.return_value = mock_files
        
        self.uploader.service = mock_service
        
        # Mock MediaFileUpload
        mock_media = Mock()
        mock_media_upload.return_value = mock_media
        
        result = self.uploader.upload_file(test_file)
        
        assert result == 'file123'
        
        # Verify API calls
        mock_files.create.assert_called_once()
        mock_media_upload.assert_called_once_with(test_file, resumable=True, chunksize=1024*1024)
        
    @patch('src.drive_uploader.MediaFileUpload')
    def test_upload_file_with_progress_callback(self, mock_media_upload):
        """Test file upload with progress callback."""
        # Create test file
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
            
        # Mock authenticated service
        mock_service = Mock()
        mock_files = Mock()
        
        # Mock upload with progress
        mock_status = Mock()
        mock_status.resumable_progress = 50
        mock_request = Mock()
        mock_request.next_chunk.side_effect = [
            (mock_status, None),  # Progress update
            (None, {'id': 'file123'})  # Upload complete
        ]
        mock_files.create.return_value = mock_request
        mock_service.files.return_value = mock_files
        
        self.uploader.service = mock_service
        
        # Mock MediaFileUpload
        mock_media_upload.return_value = Mock()
        
        # Progress callback
        progress_calls = []
        def progress_callback(current, total):
            progress_calls.append((current, total))
            
        result = self.uploader.upload_file(test_file, progress_callback=progress_callback)
        
        assert result == 'file123'
        assert len(progress_calls) == 1
        assert progress_calls[0][0] == 50  # Current bytes
        
    def test_verify_upload_not_authenticated(self):
        """Test upload verification without authentication."""
        result = self.uploader.verify_upload("file123", 1000)
        
        assert result is False
        
    def test_verify_upload_success(self):
        """Test successful upload verification."""
        # Mock authenticated service
        mock_service = Mock()
        mock_files = Mock()
        mock_files.get.return_value.execute.return_value = {
            'name': 'test.txt',
            'size': '1000'
        }
        mock_service.files.return_value = mock_files
        
        self.uploader.service = mock_service
        
        result = self.uploader.verify_upload("file123", 1000)
        
        assert result is True
        
    def test_verify_upload_size_mismatch(self):
        """Test upload verification with size mismatch."""
        # Mock authenticated service
        mock_service = Mock()
        mock_files = Mock()
        mock_files.get.return_value.execute.return_value = {
            'name': 'test.txt',
            'size': '500'  # Different from expected 1000
        }
        mock_service.files.return_value = mock_files
        
        self.uploader.service = mock_service
        
        result = self.uploader.verify_upload("file123", 1000)
        
        assert result is False
        
    def test_get_file_link_success(self):
        """Test successful file link generation."""
        # Mock authenticated service
        mock_service = Mock()
        
        # Mock permissions creation
        mock_permissions = Mock()
        mock_permissions.create.return_value.execute.return_value = {}
        mock_service.permissions.return_value = mock_permissions
        
        # Mock file info retrieval
        mock_files = Mock()
        mock_files.get.return_value.execute.return_value = {
            'webViewLink': 'https://drive.google.com/file/d/file123/view',
            'name': 'test.txt'
        }
        mock_service.files.return_value = mock_files
        
        self.uploader.service = mock_service
        
        result = self.uploader.get_file_link("file123")
        
        assert result == 'https://drive.google.com/file/d/file123/view'
        
        # Verify permission was created
        mock_permissions.create.assert_called_once()
        
    def test_get_storage_quota_success(self):
        """Test successful storage quota retrieval."""
        # Mock authenticated service
        mock_service = Mock()
        mock_about = Mock()
        mock_about.get.return_value.execute.return_value = {
            'storageQuota': {
                'limit': '15000000000',  # 15GB
                'usage': '5000000000',   # 5GB
                'usageInDrive': '4000000000',
                'usageInDriveTrash': '1000000000'
            }
        }
        mock_service.about.return_value = mock_about
        
        self.uploader.service = mock_service
        
        quota_info = self.uploader.get_storage_quota()
        
        assert quota_info['limit'] == 15000000000
        assert quota_info['usage'] == 5000000000
        assert quota_info['available'] == 10000000000  # 15GB - 5GB
        assert abs(quota_info['usage_percent'] - 33.33) < 0.1
        
    def test_check_available_space_sufficient(self):
        """Test space check with sufficient space."""
        # Mock get_storage_quota
        with patch.object(self.uploader, 'get_storage_quota', 
                         return_value={'available': 10000000000}):  # 10GB available
            result = self.uploader.check_available_space(5000000000)  # Need 5GB
            
        assert result is True
        
    def test_check_available_space_insufficient(self):
        """Test space check with insufficient space."""
        # Mock get_storage_quota
        with patch.object(self.uploader, 'get_storage_quota', 
                         return_value={'available': 1000000000}):  # 1GB available
            result = self.uploader.check_available_space(5000000000)  # Need 5GB
            
        assert result is False
        
    def test_check_available_space_unlimited(self):
        """Test space check with unlimited storage."""
        # Mock get_storage_quota
        with patch.object(self.uploader, 'get_storage_quota', 
                         return_value={'available': float('inf')}):  # Unlimited
            result = self.uploader.check_available_space(5000000000)  # Need 5GB
            
        assert result is True
        
    def test_upload_multiple_files_success(self):
        """Test successful multiple file upload."""
        # Create test files
        test_files = []
        for i in range(3):
            test_file = os.path.join(self.temp_dir, f"test{i}.txt")
            with open(test_file, 'w') as f:
                f.write(f"content {i}")
            test_files.append(test_file)
            
        # Mock upload_file method
        with patch.object(self.uploader, 'upload_file', 
                         side_effect=[f'file{i}' for i in range(3)]):
            results = self.uploader.upload_multiple_files(test_files)
            
        assert len(results) == 3
        for i, test_file in enumerate(test_files):
            assert results[test_file] == f'file{i}'
            
    def test_upload_multiple_files_partial_failure(self):
        """Test multiple file upload with some failures."""
        # Create test files
        test_files = []
        for i in range(3):
            test_file = os.path.join(self.temp_dir, f"test{i}.txt")
            with open(test_file, 'w') as f:
                f.write(f"content {i}")
            test_files.append(test_file)
            
        # Mock upload_file method with one failure
        with patch.object(self.uploader, 'upload_file', 
                         side_effect=['file0', None, 'file2']):  # Middle upload fails
            results = self.uploader.upload_multiple_files(test_files)
            
        assert len(results) == 3
        assert results[test_files[0]] == 'file0'
        assert results[test_files[1]] is None  # Failed upload
        assert results[test_files[2]] == 'file2'
        
    def test_handle_auth_error_invalid_grant(self):
        """Test authentication error handling for invalid grant."""
        # Create token file to be removed
        with open(self.token_path, 'w') as f:
            f.write('{"token": "invalid"}')
            
        error = Exception("invalid_grant: Token expired")
        
        self.uploader._handle_auth_error(error)
        
        # Token file should be removed
        assert not os.path.exists(self.token_path)
        
    def test_handle_auth_error_access_denied(self):
        """Test authentication error handling for access denied."""
        error = Exception("access_denied: Permission denied")
        
        # Should not raise exception
        self.uploader._handle_auth_error(error)
        
    def test_handle_auth_error_quota_exceeded(self):
        """Test authentication error handling for quota exceeded."""
        error = Exception("quota_exceeded: API quota exceeded")
        
        # Should not raise exception
        self.uploader._handle_auth_error(error)


if __name__ == "__main__":
    pytest.main([__file__])