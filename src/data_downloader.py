"""
Data downloader module for fetching Reddit datasets from Academic Torrents.

This module provides functionality to download Reddit data from Academic Torrents,
verify file integrity using checksums, and handle network failures with retry logic.
"""

import os
import time
import hashlib
import requests
import zipfile
import tarfile
import shutil
import libtorrent as lt
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class DataDownloader:
    """Handles downloading Reddit datasets from Academic Torrents with integrity verification."""
    
    def __init__(self, download_dir: str = "data", max_retries: int = 3):
        """
        Initialize the data downloader.
        
        Args:
            download_dir: Directory to store downloaded files
            max_retries: Maximum number of retry attempts for failed downloads
        """
        self.download_dir = download_dir
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)
        
    def download_dataset(self, url: str, expected_checksum: Optional[str] = None, 
                        checksum_algorithm: str = "sha256") -> str:
        """
        Download Reddit dataset from Academic Torrents with retry logic.
        
        Args:
            url: URL to download the dataset from
            expected_checksum: Expected checksum for integrity verification
            checksum_algorithm: Algorithm to use for checksum (sha256, md5, etc.)
            
        Returns:
            str: Path to the downloaded file
            
        Raises:
            Exception: If download fails after all retry attempts
        """
        filename = self._get_filename_from_url(url)
        file_path = os.path.join(self.download_dir, filename)
        
        # Skip download if file already exists and passes integrity check
        if os.path.exists(file_path):
            if expected_checksum and self.verify_integrity(file_path, expected_checksum, checksum_algorithm):
                logger.info(f"File {filename} already exists and passes integrity check")
                return file_path
            elif not expected_checksum:
                logger.info(f"File {filename} already exists, skipping download")
                return file_path
        
        logger.info(f"Starting download of {filename} from {url}")
        
        for attempt in range(self.max_retries + 1):
            try:
                self._download_with_progress(url, file_path)
                
                # Verify integrity if checksum provided
                if expected_checksum:
                    if self.verify_integrity(file_path, expected_checksum, checksum_algorithm):
                        logger.info(f"Download completed and verified: {filename}")
                        return file_path
                    else:
                        logger.warning(f"Integrity check failed for {filename}, retrying...")
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        continue
                
                logger.info(f"Download completed: {filename}")
                return file_path
                
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries:
                    wait_time = self._calculate_backoff_time(attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All download attempts failed for {filename}")
                    raise Exception(f"Failed to download {filename} after {self.max_retries + 1} attempts")
    
    def verify_integrity(self, file_path: str, expected_checksum: str, 
                        algorithm: str = "sha256") -> bool:
        """
        Verify file integrity using checksums.
        
        Args:
            file_path: Path to the file to verify
            expected_checksum: Expected checksum value
            algorithm: Checksum algorithm (sha256, md5, sha1)
            
        Returns:
            bool: True if checksum matches, False otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found for integrity check: {file_path}")
            return False
            
        try:
            calculated_checksum = self._calculate_checksum(file_path, algorithm)
            matches = calculated_checksum.lower() == expected_checksum.lower()
            
            if matches:
                logger.info(f"Integrity check passed for {os.path.basename(file_path)}")
            else:
                logger.error(f"Integrity check failed for {os.path.basename(file_path)}")
                logger.error(f"Expected: {expected_checksum}")
                logger.error(f"Calculated: {calculated_checksum}")
                
            return matches
            
        except Exception as e:
            logger.error(f"Error during integrity check: {str(e)}")
            return False
    
    def _download_with_progress(self, url: str, file_path: str) -> None:
        """
        Download file with progress tracking.
        
        Args:
            url: URL to download from
            file_path: Local path to save the file
        """
        response = self.session.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Log progress every 10MB
                    if downloaded_size % (10 * 1024 * 1024) == 0 or downloaded_size == total_size:
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}% ({downloaded_size / (1024*1024):.1f}MB)")
    
    def _calculate_checksum(self, file_path: str, algorithm: str) -> str:
        """
        Calculate checksum for a file.
        
        Args:
            file_path: Path to the file
            algorithm: Checksum algorithm
            
        Returns:
            str: Calculated checksum
        """
        hash_func = getattr(hashlib, algorithm.lower())()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
                
        return hash_func.hexdigest()
    
    def _get_filename_from_url(self, url: str) -> str:
        """
        Extract filename from URL.
        
        Args:
            url: URL to extract filename from
            
        Returns:
            str: Extracted filename
        """
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # If no filename in URL, generate one
        if not filename or '.' not in filename:
            filename = f"reddit_dataset_{int(time.time())}.tar.gz"
            
        return filename
    
    def _calculate_backoff_time(self, attempt: int) -> float:
        """
        Calculate exponential backoff time.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            float: Wait time in seconds
        """
        # Exponential backoff: 2^attempt seconds, with some jitter
        base_wait = 2 ** attempt
        jitter = 0.1 * base_wait  # 10% jitter
        return base_wait + (jitter * (0.5 - hash(str(time.time())) % 100 / 100))
    
    def extract_files(self, archive_path: str, extract_to: Optional[str] = None) -> List[str]:
        """
        Extract files from compressed archives (zip/tar).
        
        Args:
            archive_path: Path to the archive file
            extract_to: Directory to extract to (defaults to organized subdirectory)
            
        Returns:
            List[str]: Paths to extracted files
            
        Raises:
            Exception: If extraction fails
        """
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Archive file not found: {archive_path}")
        
        # Determine extraction directory
        if extract_to is None:
            archive_name = os.path.splitext(os.path.basename(archive_path))[0]
            # Handle .tar.gz, .tar.bz2 etc.
            if archive_name.endswith('.tar'):
                archive_name = os.path.splitext(archive_name)[0]
            extract_to = os.path.join(self.download_dir, "extracted", archive_name)
        
        # Create extraction directory
        os.makedirs(extract_to, exist_ok=True)
        
        logger.info(f"Extracting {os.path.basename(archive_path)} to {extract_to}")
        
        extracted_files = []
        
        try:
            if self._is_zip_file(archive_path):
                extracted_files = self._extract_zip(archive_path, extract_to)
            elif self._is_tar_file(archive_path):
                extracted_files = self._extract_tar(archive_path, extract_to)
            else:
                raise ValueError(f"Unsupported archive format: {archive_path}")
            
            # Organize extracted files
            organized_files = self._organize_extracted_files(extract_to)
            
            logger.info(f"Successfully extracted {len(extracted_files)} files")
            return organized_files
            
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {str(e)}")
            raise
    
    def _extract_zip(self, archive_path: str, extract_to: str) -> List[str]:
        """Extract ZIP archive with progress tracking."""
        extracted_files = []
        
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            for i, file_info in enumerate(zip_ref.infolist()):
                zip_ref.extract(file_info, extract_to)
                extracted_path = os.path.join(extract_to, file_info.filename)
                extracted_files.append(extracted_path)
                
                # Log progress every 100 files or at completion
                if (i + 1) % 100 == 0 or (i + 1) == total_files:
                    progress = ((i + 1) / total_files) * 100
                    logger.info(f"Extraction progress: {progress:.1f}% ({i + 1}/{total_files} files)")
        
        return extracted_files
    
    def _extract_tar(self, archive_path: str, extract_to: str) -> List[str]:
        """Extract TAR archive with progress tracking."""
        extracted_files = []
        
        # Determine compression mode
        if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            mode = 'r:gz'
        elif archive_path.endswith('.tar.bz2') or archive_path.endswith('.tbz2'):
            mode = 'r:bz2'
        elif archive_path.endswith('.tar.xz'):
            mode = 'r:xz'
        else:
            mode = 'r'
        
        with tarfile.open(archive_path, mode) as tar_ref:
            members = tar_ref.getmembers()
            total_files = len(members)
            
            for i, member in enumerate(members):
                tar_ref.extract(member, extract_to)
                extracted_path = os.path.join(extract_to, member.name)
                extracted_files.append(extracted_path)
                
                # Log progress every 100 files or at completion
                if (i + 1) % 100 == 0 or (i + 1) == total_files:
                    progress = ((i + 1) / total_files) * 100
                    logger.info(f"Extraction progress: {progress:.1f}% ({i + 1}/{total_files} files)")
        
        return extracted_files
    
    def _organize_extracted_files(self, extract_dir: str) -> List[str]:
        """
        Organize extracted files into a structured directory layout.
        
        Args:
            extract_dir: Directory containing extracted files
            
        Returns:
            List[str]: Paths to organized files
        """
        organized_files = []
        
        # Create organized directory structure
        organized_dir = os.path.join(extract_dir, "organized")
        os.makedirs(organized_dir, exist_ok=True)
        
        # Subdirectories for different file types
        subdirs = {
            "comments": os.path.join(organized_dir, "comments"),
            "submissions": os.path.join(organized_dir, "submissions"),
            "other": os.path.join(organized_dir, "other")
        }
        
        for subdir in subdirs.values():
            os.makedirs(subdir, exist_ok=True)
        
        # Walk through extracted files and organize them
        for root, dirs, files in os.walk(extract_dir):
            # Skip the organized directory itself
            if "organized" in root:
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                
                # Determine destination based on filename
                if "comment" in file.lower():
                    dest_dir = subdirs["comments"]
                elif "submission" in file.lower():
                    dest_dir = subdirs["submissions"]
                else:
                    dest_dir = subdirs["other"]
                
                # Move file to organized location
                dest_path = os.path.join(dest_dir, file)
                
                # Handle duplicate filenames
                counter = 1
                original_dest = dest_path
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(original_dest)
                    dest_path = f"{name}_{counter}{ext}"
                    counter += 1
                
                shutil.move(file_path, dest_path)
                organized_files.append(dest_path)
                logger.debug(f"Organized: {file} -> {os.path.relpath(dest_path, organized_dir)}")
        
        logger.info(f"Organized {len(organized_files)} files into structured directories")
        return organized_files
    
    def _is_zip_file(self, file_path: str) -> bool:
        """Check if file is a ZIP archive."""
        return zipfile.is_zipfile(file_path)
    
    def _is_tar_file(self, file_path: str) -> bool:
        """Check if file is a TAR archive."""
        return tarfile.is_tarfile(file_path)
    
    def get_extraction_info(self, extract_dir: str) -> Dict[str, Any]:
        """
        Get information about extracted files.
        
        Args:
            extract_dir: Directory containing extracted files
            
        Returns:
            Dict containing extraction info
        """
        if not os.path.exists(extract_dir):
            return {"error": "Extraction directory not found"}
        
        info = {
            "extract_dir": extract_dir,
            "total_files": 0,
            "file_types": {},
            "total_size_mb": 0
        }
        
        try:
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    info["total_files"] += 1
                    info["total_size_mb"] += file_size / (1024 * 1024)
                    
                    if file_ext in info["file_types"]:
                        info["file_types"][file_ext] += 1
                    else:
                        info["file_types"][file_ext] = 1
            
            info["total_size_mb"] = round(info["total_size_mb"], 2)
            
        except Exception as e:
            info["error"] = str(e)
        
        return info
    
    def download_from_torrent(self, magnet_link: str, output_path: Optional[str] = None) -> str:
        """
        Download a file from a torrent magnet link.

        Args:
            magnet_link: The magnet link for the torrent.
            output_path: The directory to save the downloaded file in.

        Returns:
            The path to the downloaded file.
        """
        if output_path is None:
            output_path = self.download_dir

        ses = lt.session()
        ses.listen_on(6881, 6891)

        params = {
            'save_path': output_path,
            'storage_mode': lt.storage_mode_t(2),
        }
        handle = lt.add_magnet_uri(ses, magnet_link, params)
        ses.start_dht()

        logger.info(f"Downloading torrent from {magnet_link} to {output_path}")

        while not handle.has_metadata():
            time.sleep(1)

        logger.info("Metadata downloaded, starting download...")

        while handle.status().state != lt.torrent_status.seeding:
            s = handle.status()
            state_str = [
                'queued', 'checking', 'downloading metadata', 'downloading',
                'finished', 'seeding', 'allocating', 'checking fastresume'
            ]
            logger.info(
                f"%.2f%% complete (down: %.1f kB/s up: %.1f kB/s peers: %d) %s" %
                (s.progress * 100, s.download_rate / 1000, s.upload_rate / 1000,
                 s.num_peers, state_str[s.state]))
            time.sleep(5)

        logger.info("Download finished.")

        # Get the name of the downloaded file
        torrent_info = handle.get_torrent_info()
        files = torrent_info.files()
        if files:
            downloaded_file = os.path.join(output_path, files[0].path)
            return downloaded_file
        else:
            # If no files are in the torrent info, we can't return a specific file path
            return output_path


    def get_download_info(self) -> Dict[str, Any]:
        """
        Get information about the download directory and available space.
        
        Returns:
            Dict containing download directory info
        """
        try:
            stat = os.statvfs(self.download_dir)
            free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            
            return {
                "download_dir": self.download_dir,
                "free_space_gb": round(free_space_gb, 2),
                "exists": os.path.exists(self.download_dir)
            }
        except Exception as e:
            logger.error(f"Error getting download info: {str(e)}")
            return {
                "download_dir": self.download_dir,
                "free_space_gb": 0,
                "exists": False,
                "error": str(e)
            }