# Requirements Document

## Introduction

This feature involves creating a deleted comment dataset from Reddit data available through Academic Torrents. The goal is to process Reddit submissions and comments data to identify, extract, and compile deleted comments into a structured dataset that can be used for research purposes such as content moderation, community behavior analysis, or machine learning applications.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to download and process Reddit data from Academic Torrents, so that I can access the raw submissions and comments data needed for analysis.

#### Acceptance Criteria

1. WHEN the system is initiated THEN it SHALL download the Reddit dataset from the specified Academic Torrents link
2. WHEN the download is complete THEN the system SHALL verify the integrity of the downloaded files
3. WHEN files are verified THEN the system SHALL extract and organize the data files for processing
4. IF the download fails THEN the system SHALL provide clear error messages and retry options

### Requirement 2

**User Story:** As a data analyst, I want to identify and separate deleted comments from moderator-removed comments in the Reddit dataset, so that I can distinguish between different types of content removal.

#### Acceptance Criteria

1. WHEN processing comment data THEN the system SHALL identify comments marked as deleted or removed
2. WHEN a comment has "[deleted]" content THEN the system SHALL classify it as user-deleted
3. WHEN a comment has "[removed]" content THEN the system SHALL classify it as moderator-removed
4. WHEN a comment author is "[deleted]" THEN the system SHALL include it in the user-deleted comments dataset
5. WHEN processing THEN the system SHALL create separate datasets for user-deleted and moderator-removed comments

### Requirement 3

**User Story:** As a researcher, I want to extract relevant metadata for deleted comments in train.csv format, so that I can analyze patterns and context around comment deletion for machine learning applications.

#### Acceptance Criteria

1. WHEN a moderator-removed comment is identified THEN the system SHALL extract comment text (if available before removal), subreddit, timestamp, and removal reason
2. WHEN creating train.csv format THEN the system SHALL include columns: id, comment_text, subreddit, timestamp, removal_type, target_label
3. WHEN processing moderator-removed comments THEN the system SHALL assign appropriate labels (e.g., "toxic", "spam", "rule_violation") based on removal context
4. WHEN extracting metadata THEN the system SHALL handle missing or null values by using appropriate placeholders
5. WHEN available THEN the system SHALL include parent comment context for better training data quality

### Requirement 4

**User Story:** As a data scientist, I want the deleted comment datasets exported in compressed Parquet format, so that I can efficiently store and process large datasets for machine learning workflows.

#### Acceptance Criteria

1. WHEN processing is complete THEN the system SHALL export moderator-removed comments as "removed_by_moderators_train.parquet"
2. WHEN exporting datasets THEN the system SHALL use Parquet format with built-in compression (snappy or gzip)
3. WHEN processing is complete THEN the system SHALL export user-deleted comments as "user_deleted_train.parquet"
4. WHEN creating Parquet files THEN the system SHALL include columns: id, comment_text, subreddit, timestamp, removal_type, target_label
5. WHEN exporting THEN the system SHALL ensure Parquet format is optimized for fast querying and compatible with pandas, dask, and other data processing frameworks
6. WHEN creating output files THEN the system SHALL include data schema documentation explaining column meanings and data types

### Requirement 5

**User Story:** As a user, I want to monitor the processing progress and handle large datasets efficiently, so that I can track completion and manage system resources.

#### Acceptance Criteria

1. WHEN processing large files THEN the system SHALL display progress indicators
2. WHEN memory usage is high THEN the system SHALL process data in chunks to prevent system overload
3. WHEN errors occur during processing THEN the system SHALL log detailed error information and continue processing other files
4. WHEN processing is complete THEN the system SHALL provide a summary report of processed records and any issues encountered

### Requirement 6

**User Story:** As a user with limited local storage, I want the system to use compressed Parquet files and upload them to Google Drive, so that I can efficiently store large datasets without filling up my local disk space.

#### Acceptance Criteria

1. WHEN Parquet datasets are created THEN the system SHALL use built-in Parquet compression (snappy, gzip, or lz4) for maximum space efficiency
2. WHEN local storage is limited THEN the system SHALL automatically upload compressed Parquet files to Google Drive
3. WHEN uploading to Drive THEN the system SHALL authenticate using Google Drive API credentials
4. WHEN upload is complete THEN the system SHALL optionally delete local Parquet files to free up space
5. WHEN uploading THEN the system SHALL provide upload progress indicators and handle network interruptions gracefully
6. WHEN files are uploaded THEN the system SHALL provide Google Drive links for easy access to the compressed Parquet datasets
7. WHEN processing large datasets THEN the system SHALL create Parquet files in chunks to manage memory usage efficiently