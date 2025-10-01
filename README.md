# Reddit Deleted Comment Dataset

This project processes Reddit data from Academic Torrents to create datasets of deleted and moderator-removed comments for research purposes.

## Project Structure

```
├── src/                          # Source code modules
│   ├── data_downloader.py        # Downloads Reddit data from Academic Torrents
│   ├── reddit_parser.py          # Parses Reddit JSON data files
│   ├── comment_classifier.py     # Classifies deleted/removed comments
│   ├── metadata_extractor.py     # Extracts metadata for training datasets
│   ├── parquet_writer.py         # Creates compressed Parquet files
│   ├── drive_uploader.py         # Google Drive integration
│   ├── progress_monitor.py       # Progress tracking and monitoring
│   ├── cleanup_manager.py        # Storage management
│   └── config_loader.py          # Configuration management
├── data/                         # Data directories
│   ├── raw/                      # Raw downloaded files
│   ├── extracted/                # Extracted data files
│   └── processed/                # Final Parquet datasets
├── logs/                         # Application logs
├── config.yaml                   # Main configuration file
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── main.py                       # Main application entry point
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy environment template and configure:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Configure Google Drive API:
   - Create credentials.json from Google Cloud Console
   - Place in project root directory

4. Update config.yaml with your specific settings

## Usage

Run the main processing pipeline:
```bash
python main.py
```

## Configuration

The system uses a combination of:
- `config.yaml` for main configuration
- Environment variables for sensitive data (see `.env.example`)
- Command-line arguments (implemented in main.py)

## Output

The system creates two main datasets:
- `removed_by_moderators_train.parquet` - Comments removed by moderators
- `user_deleted_train.parquet` - Comments deleted by users

Both files are compressed and optimized for machine learning workflows.