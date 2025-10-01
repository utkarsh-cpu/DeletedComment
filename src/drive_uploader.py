"""
Google Drive uploader with OAuth2 authentication and file management.
Handles authentication, file uploads, and folder organization for the deleted comment dataset.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload


class DriveUploader:
    """Handles Google Drive API operations with OAuth2 authentication."""
    
    # Google Drive API scope for file management
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    
    def __init__(self, credentials_path: str = "credentials.json", token_path: str = "token.json"):
        """
        Initialize the Drive uploader.
        
        Args:
            credentials_path: Path to Google API credentials JSON file
            token_path: Path to store/load OAuth2 tokens
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self.logger = logging.getLogger(__name__)
        
    def authenticate(self) -> bool:
        """
        Authenticate with Google Drive API using OAuth2.
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            creds = None
            
            # Load existing token if available
            if os.path.exists(self.token_path):
                try:
                    creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)
                    self.logger.info("Loaded existing credentials from token file")
                except Exception as e:
                    self.logger.warning(f"Failed to load existing token: {e}")
            
            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        self.logger.info("Refreshed expired credentials")
                    except Exception as e:
                        self.logger.error(f"Failed to refresh credentials: {e}")
                        creds = None
                
                if not creds:
                    if not os.path.exists(self.credentials_path):
                        self.logger.error(f"Credentials file not found: {self.credentials_path}")
                        return False
                    
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            self.credentials_path, self.SCOPES
                        )
                        creds = flow.run_local_server(port=0)
                        self.logger.info("Completed OAuth2 flow for new credentials")
                    except Exception as e:
                        self.logger.error(f"OAuth2 flow failed: {e}")
                        return False
                
                # Save the credentials for the next run
                try:
                    with open(self.token_path, 'w') as token:
                        token.write(creds.to_json())
                    self.logger.info(f"Saved credentials to {self.token_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save token: {e}")
            
            # Build the service
            try:
                self.service = build('drive', 'v3', credentials=creds)
                self.logger.info("Successfully authenticated with Google Drive API")
                return True
            except Exception as e:
                self.logger.error(f"Failed to build Drive service: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False
    
    def _handle_auth_error(self, error: Exception) -> None:
        """
        Handle authentication-related errors with helpful messages.
        
        Args:
            error: The exception that occurred
        """
        if "invalid_grant" in str(error):
            self.logger.error("Authentication token expired or invalid. Please re-authenticate.")
            # Remove invalid token file
            if os.path.exists(self.token_path):
                os.remove(self.token_path)
                self.logger.info("Removed invalid token file")
        elif "access_denied" in str(error):
            self.logger.error("Access denied. Please check your Google Drive permissions.")
        elif "quota_exceeded" in str(error):
            self.logger.error("API quota exceeded. Please try again later.")
        else:
            self.logger.error(f"Authentication error: {error}")
    
    def test_connection(self) -> bool:
        """
        Test the connection to Google Drive API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.service:
            self.logger.error("Not authenticated. Call authenticate() first.")
            return False
        
        try:
            # Try to get user info to test connection
            about = self.service.about().get(fields="user").execute()
            user_email = about.get('user', {}).get('emailAddress', 'Unknown')
            self.logger.info(f"Successfully connected to Google Drive for user: {user_email}")
            return True
        except HttpError as e:
            self.logger.error(f"Connection test failed: {e}")
            self._handle_auth_error(e)
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during connection test: {e}")
            return False
    
    def get_credentials_info(self) -> Dict[str, Any]:
        """
        Get information about current credentials.
        
        Returns:
            Dict containing credential status and user info
        """
        info = {
            'authenticated': False,
            'credentials_file_exists': os.path.exists(self.credentials_path),
            'token_file_exists': os.path.exists(self.token_path),
            'user_email': None
        }
        
        if self.service:
            try:
                about = self.service.about().get(fields="user").execute()
                info['authenticated'] = True
                info['user_email'] = about.get('user', {}).get('emailAddress')
            except Exception as e:
                self.logger.warning(f"Could not get user info: {e}")
        
        return info
    
    def create_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> Optional[str]:
        """
        Create a folder in Google Drive.
        
        Args:
            folder_name: Name of the folder to create
            parent_folder_id: ID of parent folder (None for root)
            
        Returns:
            str: Folder ID if successful, None otherwise
        """
        if not self.service:
            self.logger.error("Not authenticated. Call authenticate() first.")
            return None
        
        try:
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_folder_id:
                folder_metadata['parents'] = [parent_folder_id]
            
            folder = self.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            
            folder_id = folder.get('id')
            self.logger.info(f"Created folder '{folder_name}' with ID: {folder_id}")
            return folder_id
            
        except HttpError as e:
            self.logger.error(f"Failed to create folder '{folder_name}': {e}")
            self._handle_auth_error(e)
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error creating folder: {e}")
            return None
    
    def find_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> Optional[str]:
        """
        Find a folder by name in Google Drive.
        
        Args:
            folder_name: Name of the folder to find
            parent_folder_id: ID of parent folder to search in (None for root)
            
        Returns:
            str: Folder ID if found, None otherwise
        """
        if not self.service:
            self.logger.error("Not authenticated. Call authenticate() first.")
            return None
        
        try:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            if parent_folder_id:
                query += f" and '{parent_folder_id}' in parents"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name)"
            ).execute()
            
            folders = results.get('files', [])
            if folders:
                folder_id = folders[0]['id']
                self.logger.info(f"Found folder '{folder_name}' with ID: {folder_id}")
                return folder_id
            else:
                self.logger.info(f"Folder '{folder_name}' not found")
                return None
                
        except HttpError as e:
            self.logger.error(f"Failed to search for folder '{folder_name}': {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error searching for folder: {e}")
            return None
    
    def get_or_create_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> Optional[str]:
        """
        Get existing folder or create it if it doesn't exist.
        
        Args:
            folder_name: Name of the folder
            parent_folder_id: ID of parent folder (None for root)
            
        Returns:
            str: Folder ID if successful, None otherwise
        """
        # Try to find existing folder first
        folder_id = self.find_folder(folder_name, parent_folder_id)
        if folder_id:
            return folder_id
        
        # Create folder if not found
        return self.create_folder(folder_name, parent_folder_id)
    
    def upload_file(self, file_path: str, drive_folder_id: Optional[str] = None, 
                   file_name: Optional[str] = None, progress_callback=None) -> Optional[str]:
        """
        Upload a file to Google Drive with progress tracking.
        
        Args:
            file_path: Local path to the file to upload
            drive_folder_id: ID of the Drive folder to upload to (None for root)
            file_name: Name for the file in Drive (None to use original name)
            progress_callback: Function to call with progress updates (current, total)
            
        Returns:
            str: File ID if successful, None otherwise
        """
        if not self.service:
            self.logger.error("Not authenticated. Call authenticate() first.")
            return None
        
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return None
        
        try:
            file_path_obj = Path(file_path)
            upload_name = file_name or file_path_obj.name
            file_size = file_path_obj.stat().st_size
            
            self.logger.info(f"Starting upload of '{upload_name}' ({file_size:,} bytes)")
            
            # Prepare file metadata
            file_metadata = {'name': upload_name}
            if drive_folder_id:
                file_metadata['parents'] = [drive_folder_id]
            
            # Create media upload object
            media = MediaFileUpload(
                file_path,
                resumable=True,
                chunksize=1024*1024  # 1MB chunks for progress tracking
            )
            
            # Start the upload
            request = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            )
            
            response = None
            uploaded_bytes = 0
            
            while response is None:
                try:
                    status, response = request.next_chunk()
                    
                    if status:
                        uploaded_bytes = status.resumable_progress
                        progress_percent = (uploaded_bytes / file_size) * 100
                        
                        # Call progress callback if provided
                        if progress_callback:
                            progress_callback(uploaded_bytes, file_size)
                        
                        self.logger.debug(f"Upload progress: {progress_percent:.1f}% ({uploaded_bytes:,}/{file_size:,} bytes)")
                
                except HttpError as e:
                    if e.resp.status in [500, 502, 503, 504]:
                        # Recoverable error, continue
                        self.logger.warning(f"Recoverable upload error: {e}. Retrying...")
                        continue
                    else:
                        # Non-recoverable error
                        self.logger.error(f"Upload failed: {e}")
                        self._handle_auth_error(e)
                        return None
            
            file_id = response.get('id')
            self.logger.info(f"Successfully uploaded '{upload_name}' with ID: {file_id}")
            return file_id
            
        except Exception as e:
            self.logger.error(f"Unexpected error during upload: {e}")
            return None
    
    def get_upload_progress(self) -> Dict[str, Any]:
        """
        Get current upload progress information.
        Note: This is a placeholder for tracking multiple concurrent uploads.
        
        Returns:
            Dict containing upload progress information
        """
        # This would be expanded for tracking multiple uploads
        return {
            'active_uploads': 0,
            'completed_uploads': 0,
            'failed_uploads': 0,
            'total_bytes_uploaded': 0
        }
    
    def verify_upload(self, file_id: str, expected_size: int) -> bool:
        """
        Verify that an uploaded file exists and has the correct size.
        
        Args:
            file_id: Google Drive file ID
            expected_size: Expected file size in bytes
            
        Returns:
            bool: True if verification successful, False otherwise
        """
        if not self.service:
            self.logger.error("Not authenticated. Call authenticate() first.")
            return False
        
        try:
            file_info = self.service.files().get(
                fileId=file_id,
                fields='name,size,md5Checksum'
            ).execute()
            
            actual_size = int(file_info.get('size', 0))
            file_name = file_info.get('name', 'Unknown')
            
            if actual_size == expected_size:
                self.logger.info(f"Upload verification successful for '{file_name}' ({actual_size:,} bytes)")
                return True
            else:
                self.logger.error(f"Size mismatch for '{file_name}': expected {expected_size:,}, got {actual_size:,}")
                return False
                
        except HttpError as e:
            self.logger.error(f"Failed to verify upload: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during verification: {e}")
            return False
    
    def get_file_link(self, file_id: str) -> Optional[str]:
        """
        Get a shareable link for a file in Google Drive.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            str: Shareable link if successful, None otherwise
        """
        if not self.service:
            self.logger.error("Not authenticated. Call authenticate() first.")
            return None
        
        try:
            # Make file publicly readable
            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            
            self.service.permissions().create(
                fileId=file_id,
                body=permission
            ).execute()
            
            # Get file info with web view link
            file_info = self.service.files().get(
                fileId=file_id,
                fields='webViewLink,name'
            ).execute()
            
            link = file_info.get('webViewLink')
            file_name = file_info.get('name', 'Unknown')
            
            self.logger.info(f"Generated shareable link for '{file_name}': {link}")
            return link
            
        except HttpError as e:
            self.logger.error(f"Failed to create shareable link: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error creating link: {e}")
            return None
    
    def upload_large_file(self, file_path: str, drive_folder_id: Optional[str] = None,
                         file_name: Optional[str] = None, chunk_size: int = 10*1024*1024,
                         progress_callback=None, retry_attempts: int = 3) -> Optional[str]:
        """
        Upload a large file using chunked upload with enhanced error recovery.
        
        Args:
            file_path: Local path to the file to upload
            drive_folder_id: ID of the Drive folder to upload to
            file_name: Name for the file in Drive
            chunk_size: Size of each upload chunk in bytes (default 10MB)
            progress_callback: Function to call with progress updates
            retry_attempts: Number of retry attempts for failed chunks
            
        Returns:
            str: File ID if successful, None otherwise
        """
        if not self.service:
            self.logger.error("Not authenticated. Call authenticate() first.")
            return None
        
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return None
        
        try:
            file_path_obj = Path(file_path)
            upload_name = file_name or file_path_obj.name
            file_size = file_path_obj.stat().st_size
            
            # Use larger chunk size for large files
            if file_size > 100 * 1024 * 1024:  # > 100MB
                chunk_size = max(chunk_size, 25 * 1024 * 1024)  # Use at least 25MB chunks
            
            self.logger.info(f"Starting chunked upload of '{upload_name}' ({file_size:,} bytes, {chunk_size:,} byte chunks)")
            
            # Prepare file metadata
            file_metadata = {'name': upload_name}
            if drive_folder_id:
                file_metadata['parents'] = [drive_folder_id]
            
            # Create resumable media upload
            media = MediaFileUpload(
                file_path,
                resumable=True,
                chunksize=chunk_size
            )
            
            # Start the upload request
            request = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            )
            
            response = None
            uploaded_bytes = 0
            retry_count = 0
            last_progress = 0
            
            while response is None:
                try:
                    status, response = request.next_chunk()
                    
                    if status:
                        uploaded_bytes = status.resumable_progress
                        progress_percent = (uploaded_bytes / file_size) * 100
                        
                        # Call progress callback if provided
                        if progress_callback:
                            progress_callback(uploaded_bytes, file_size)
                        
                        # Log progress every 10% or significant chunk
                        if progress_percent - last_progress >= 10:
                            self.logger.info(f"Upload progress: {progress_percent:.1f}% ({uploaded_bytes:,}/{file_size:,} bytes)")
                            last_progress = progress_percent
                        
                        # Reset retry count on successful chunk
                        retry_count = 0
                
                except HttpError as e:
                    retry_count += 1
                    
                    if e.resp.status in [500, 502, 503, 504] and retry_count <= retry_attempts:
                        # Recoverable server error, retry with exponential backoff
                        wait_time = min(2 ** retry_count, 60)  # Max 60 seconds
                        self.logger.warning(f"Recoverable upload error (attempt {retry_count}/{retry_attempts}): {e}. Retrying in {wait_time}s...")
                        
                        import time
                        time.sleep(wait_time)
                        continue
                    else:
                        # Non-recoverable error or max retries exceeded
                        self.logger.error(f"Upload failed after {retry_count} attempts: {e}")
                        self._handle_auth_error(e)
                        return None
                
                except Exception as e:
                    retry_count += 1
                    if retry_count <= retry_attempts:
                        wait_time = min(2 ** retry_count, 30)
                        self.logger.warning(f"Unexpected error (attempt {retry_count}/{retry_attempts}): {e}. Retrying in {wait_time}s...")
                        
                        import time
                        time.sleep(wait_time)
                        continue
                    else:
                        self.logger.error(f"Upload failed after {retry_count} attempts: {e}")
                        return None
            
            file_id = response.get('id')
            self.logger.info(f"Successfully uploaded '{upload_name}' with ID: {file_id}")
            
            # Verify the upload
            if self.verify_upload(file_id, file_size):
                return file_id
            else:
                self.logger.error("Upload verification failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Unexpected error during chunked upload: {e}")
            return None
    
    def upload_multiple_files(self, file_paths: list, drive_folder_id: Optional[str] = None,
                             max_concurrent: int = 2, progress_callback=None) -> Dict[str, Optional[str]]:
        """
        Upload multiple files with memory management and concurrency control.
        
        Args:
            file_paths: List of local file paths to upload
            drive_folder_id: ID of the Drive folder to upload to
            max_concurrent: Maximum number of concurrent uploads
            progress_callback: Function to call with overall progress updates
            
        Returns:
            Dict mapping file paths to file IDs (None if upload failed)
        """
        if not self.service:
            self.logger.error("Not authenticated. Call authenticate() first.")
            return {}
        
        results = {}
        total_files = len(file_paths)
        completed_files = 0
        
        self.logger.info(f"Starting batch upload of {total_files} files")
        
        # For now, upload sequentially to manage memory better
        # Could be enhanced with threading for true concurrent uploads
        for i, file_path in enumerate(file_paths):
            try:
                self.logger.info(f"Uploading file {i+1}/{total_files}: {Path(file_path).name}")
                
                # Individual progress callback for this file
                def file_progress(current, total):
                    overall_progress = (completed_files + (current / total)) / total_files
                    if progress_callback:
                        progress_callback(overall_progress, file_path)
                
                # Choose upload method based on file size
                file_size = Path(file_path).stat().st_size
                if file_size > 50 * 1024 * 1024:  # > 50MB, use chunked upload
                    file_id = self.upload_large_file(
                        file_path, 
                        drive_folder_id, 
                        progress_callback=file_progress
                    )
                else:
                    file_id = self.upload_file(
                        file_path, 
                        drive_folder_id, 
                        progress_callback=file_progress
                    )
                
                results[file_path] = file_id
                completed_files += 1
                
                if file_id:
                    self.logger.info(f"Successfully uploaded {Path(file_path).name}")
                else:
                    self.logger.error(f"Failed to upload {Path(file_path).name}")
                
            except Exception as e:
                self.logger.error(f"Error uploading {file_path}: {e}")
                results[file_path] = None
                completed_files += 1
        
        successful_uploads = sum(1 for file_id in results.values() if file_id is not None)
        self.logger.info(f"Batch upload complete: {successful_uploads}/{total_files} files uploaded successfully")
        
        return results
    
    def get_storage_quota(self) -> Dict[str, int]:
        """
        Get Google Drive storage quota information.
        
        Returns:
            Dict containing storage quota details in bytes
        """
        if not self.service:
            self.logger.error("Not authenticated. Call authenticate() first.")
            return {}
        
        try:
            about = self.service.about().get(fields="storageQuota").execute()
            quota = about.get('storageQuota', {})
            
            storage_info = {
                'limit': int(quota.get('limit', 0)),
                'usage': int(quota.get('usage', 0)),
                'usage_in_drive': int(quota.get('usageInDrive', 0)),
                'usage_in_drive_trash': int(quota.get('usageInDriveTrash', 0))
            }
            
            if storage_info['limit'] > 0:
                storage_info['available'] = storage_info['limit'] - storage_info['usage']
                storage_info['usage_percent'] = (storage_info['usage'] / storage_info['limit']) * 100
            else:
                storage_info['available'] = float('inf')  # Unlimited storage
                storage_info['usage_percent'] = 0
            
            self.logger.info(f"Drive storage: {storage_info['usage']:,} / {storage_info['limit']:,} bytes used ({storage_info['usage_percent']:.1f}%)")
            return storage_info
            
        except HttpError as e:
            self.logger.error(f"Failed to get storage quota: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error getting storage quota: {e}")
            return {}
    
    def check_available_space(self, required_bytes: int) -> bool:
        """
        Check if there's enough available space for an upload.
        
        Args:
            required_bytes: Number of bytes needed
            
        Returns:
            bool: True if enough space available, False otherwise
        """
        quota_info = self.get_storage_quota()
        if not quota_info:
            self.logger.warning("Could not check storage quota, proceeding with upload")
            return True
        
        available = quota_info.get('available', 0)
        if available == float('inf'):  # Unlimited storage
            return True
        
        if available >= required_bytes:
            self.logger.info(f"Sufficient storage available: {available:,} bytes available, {required_bytes:,} bytes required")
            return True
        else:
            self.logger.error(f"Insufficient storage: {available:,} bytes available, {required_bytes:,} bytes required")
            return False