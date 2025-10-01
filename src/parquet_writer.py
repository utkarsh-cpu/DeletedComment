"""
Parquet writer component for creating compressed Parquet datasets.
Handles memory-efficient processing and schema optimization for fast querying.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Iterator, Optional, Union
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass


@dataclass
class CompressionStats:
    """Statistics for compression operations."""
    original_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    records_written: int = 0
    processing_time: float = 0.0
    

class ParquetWriter:
    """Writer for creating compressed Parquet datasets with optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Parquet writer.
        
        Args:
            config: Configuration dictionary with writer settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Get storage configuration
        storage_config = self.config.get('storage', {})
        self.default_compression = storage_config.get('compression', 'snappy')
        self.parquet_version = storage_config.get('parquet_version', '2.6')
        self.chunk_size = self.config.get('processing', {}).get('chunk_size', 100000)
        
        # Supported compression algorithms
        self.supported_compressions = ['snappy', 'gzip', 'lz4', 'brotli', 'zstd']
        
        # Statistics tracking
        self.stats = {
            'files_created': 0,
            'total_records': 0,
            'total_compressed_size': 0,
            'total_original_size': 0,
            'compression_ratios': []
        }
        
    def write_dataset(self, data: Iterator[Dict[str, Any]], output_path: Union[str, Path], 
                     compression: Optional[str] = None) -> str:
        """
        Write dataset to Parquet format with configurable compression.
        
        Args:
            data: Iterator of record dictionaries
            output_path: Path for output Parquet file
            compression: Compression algorithm ('snappy', 'gzip', 'lz4', etc.)
            
        Returns:
            str: Path to created Parquet file
            
        Requirements: 4.1, 4.2, 4.5
        """
        output_path = Path(output_path)
        compression = compression or self.default_compression
        
        if compression not in self.supported_compressions:
            self.logger.warning(f"Unsupported compression '{compression}', using 'snappy'")
            compression = 'snappy'
            
        self.logger.info(f"Writing dataset to {output_path} with {compression} compression")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect data into DataFrame for writing
        records = list(data)
        if not records:
            raise ValueError("No data provided for writing")
            
        # Create DataFrame and optimize schema
        df = pd.DataFrame(records)
        schema = self.optimize_schema(records[:1000])  # Sample for schema optimization
        
        # Write to Parquet with compression
        start_time = datetime.now()
        
        try:
            # Get original size estimate
            original_size = self._estimate_original_size(df)
            
            # Write Parquet file
            table = pa.Table.from_pandas(df, schema=schema)
            pq.write_table(
                table, 
                output_path,
                compression=compression,
                version=self.parquet_version,
                use_dictionary=True,  # Enable dictionary encoding
                row_group_size=50000,  # Optimize row group size
                data_page_size=1024*1024,  # 1MB data pages
                write_statistics=True
            )
            
            # Calculate compression statistics
            compressed_size = output_path.stat().st_size
            processing_time = (datetime.now() - start_time).total_seconds()
            
            compression_stats = CompressionStats(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=original_size / compressed_size if compressed_size > 0 else 0,
                records_written=len(records),
                processing_time=processing_time
            )
            
            self._update_stats(compression_stats)
            
            self.logger.info(
                f"Successfully wrote {len(records)} records to {output_path} "
                f"(compression ratio: {compression_stats.compression_ratio:.2f}x)"
            )
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to write Parquet file {output_path}: {e}")
            raise
            
    def create_chunked_parquet(self, data: Iterator[Dict[str, Any]], 
                              output_dir: Union[str, Path],
                              base_filename: str,
                              chunk_size: Optional[int] = None,
                              compression: Optional[str] = None) -> List[str]:
        """
        Create Parquet files in memory-efficient chunks.
        
        Args:
            data: Iterator of record dictionaries
            output_dir: Directory for output files
            base_filename: Base name for chunk files
            chunk_size: Records per chunk (uses instance default if None)
            compression: Compression algorithm
            
        Returns:
            List[str]: Paths to created chunk files
            
        Requirements: 4.1, 4.2, 4.5
        """
        output_dir = Path(output_dir)
        chunk_size = chunk_size or self.chunk_size
        compression = compression or self.default_compression
        
        self.logger.info(f"Creating chunked Parquet files in {output_dir} with chunk size {chunk_size}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_files = []
        chunk_num = 0
        current_chunk = []
        
        try:
            for record in data:
                current_chunk.append(record)
                
                if len(current_chunk) >= chunk_size:
                    # Write current chunk
                    chunk_filename = f"{base_filename}_chunk_{chunk_num:04d}.parquet"
                    chunk_path = output_dir / chunk_filename
                    
                    self.write_dataset(iter(current_chunk), chunk_path, compression)
                    chunk_files.append(str(chunk_path))
                    
                    # Reset for next chunk
                    current_chunk = []
                    chunk_num += 1
                    
                    self.logger.debug(f"Completed chunk {chunk_num} with {chunk_size} records")
                    
            # Write remaining records
            if current_chunk:
                chunk_filename = f"{base_filename}_chunk_{chunk_num:04d}.parquet"
                chunk_path = output_dir / chunk_filename
                
                self.write_dataset(iter(current_chunk), chunk_path, compression)
                chunk_files.append(str(chunk_path))
                
            self.logger.info(f"Created {len(chunk_files)} chunk files")
            return chunk_files
            
        except Exception as e:
            self.logger.error(f"Failed to create chunked Parquet files: {e}")
            raise
            
    def optimize_schema(self, data_sample: List[Dict[str, Any]]) -> pa.Schema:
        """
        Optimize Parquet schema for fast querying and storage efficiency.
        
        Args:
            data_sample: Sample of data records for schema inference
            
        Returns:
            pyarrow.Schema: Optimized schema
            
        Requirements: 4.5
        """
        if not data_sample:
            raise ValueError("Cannot optimize schema with empty data sample")
            
        # Create DataFrame from sample to infer types
        df_sample = pd.DataFrame(data_sample)
        
        # Define optimized field mappings
        field_mappings = {}
        
        for column in df_sample.columns:
            dtype = df_sample[column].dtype
            
            # Optimize based on column name and data type
            if column == 'id':
                field_mappings[column] = pa.string()
            elif column == 'timestamp':
                field_mappings[column] = pa.timestamp('us', tz='UTC')
            elif column in ['score', 'controversiality', 'gilded', 'comment_length', 'thread_score', 'thread_num_comments']:
                field_mappings[column] = pa.int32()
            elif column in ['has_parent', 'is_top_level']:
                field_mappings[column] = pa.bool_()
            elif column in ['subreddit', 'removal_type', 'target_label', 'author']:
                # Use dictionary encoding for categorical data
                field_mappings[column] = pa.dictionary(pa.int16(), pa.string())
            elif column in ['comment_text', 'parent_context', 'thread_title', 'thread_selftext']:
                field_mappings[column] = pa.string()
            elif 'float' in str(dtype):
                field_mappings[column] = pa.float32()
            elif 'int' in str(dtype):
                field_mappings[column] = pa.int32()
            elif 'bool' in str(dtype):
                field_mappings[column] = pa.bool_()
            else:
                field_mappings[column] = pa.string()
                
        # Create schema from field mappings
        fields = [pa.field(name, dtype) for name, dtype in field_mappings.items()]
        schema = pa.schema(fields)
        
        self.logger.debug(f"Optimized schema with {len(fields)} fields")
        return schema
        
    def _estimate_original_size(self, df: pd.DataFrame) -> int:
        """Estimate original data size before compression."""
        # Rough estimate based on DataFrame memory usage
        memory_usage = df.memory_usage(deep=True).sum()
        return int(memory_usage * 1.2)  # Add overhead estimate
        
    def _update_stats(self, compression_stats: CompressionStats) -> None:
        """Update writer statistics."""
        self.stats['files_created'] += 1
        self.stats['total_records'] += compression_stats.records_written
        self.stats['total_compressed_size'] += compression_stats.compressed_size
        self.stats['total_original_size'] += compression_stats.original_size
        self.stats['compression_ratios'].append(compression_stats.compression_ratio)
        
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get current compression statistics."""
        avg_compression_ratio = (
            sum(self.stats['compression_ratios']) / len(self.stats['compression_ratios'])
            if self.stats['compression_ratios'] else 0
        )
        
        return {
            'files_created': self.stats['files_created'],
            'total_records': self.stats['total_records'],
            'total_compressed_size_mb': self.stats['total_compressed_size'] / (1024 * 1024),
            'total_original_size_mb': self.stats['total_original_size'] / (1024 * 1024),
            'average_compression_ratio': avg_compression_ratio,
            'total_space_saved_mb': (self.stats['total_original_size'] - self.stats['total_compressed_size']) / (1024 * 1024)
        }
        
    def reset_stats(self) -> None:
        """Reset writer statistics."""
        self.stats = {
            'files_created': 0,
            'total_records': 0,
            'total_compressed_size': 0,
            'total_original_size': 0,
            'compression_ratios': []
        } 
       
    def export_training_datasets(self, user_deleted_data: Iterator[Dict[str, Any]], 
                                moderator_removed_data: Iterator[Dict[str, Any]],
                                output_dir: Union[str, Path],
                                compression: Optional[str] = None) -> Dict[str, str]:
        """
        Export separate training datasets for user-deleted and moderator-removed comments.
        
        Args:
            user_deleted_data: Iterator of user-deleted comment records
            moderator_removed_data: Iterator of moderator-removed comment records
            output_dir: Directory for output files
            compression: Compression algorithm
            
        Returns:
            Dict[str, str]: Paths to created dataset files
            
        Requirements: 4.3, 4.4
        """
        output_dir = Path(output_dir)
        compression = compression or self.default_compression
        
        self.logger.info(f"Exporting training datasets to {output_dir}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_paths = {}
        
        try:
            # Export user-deleted dataset
            user_deleted_path = output_dir / "user_deleted_train.parquet"
            user_deleted_records = list(user_deleted_data)
            
            if user_deleted_records:
                self.write_dataset(iter(user_deleted_records), user_deleted_path, compression)
                dataset_paths['user_deleted'] = str(user_deleted_path)
                self.logger.info(f"Exported {len(user_deleted_records)} user-deleted records")
            else:
                self.logger.warning("No user-deleted records to export")
                
            # Export moderator-removed dataset
            moderator_removed_path = output_dir / "removed_by_moderators_train.parquet"
            moderator_removed_records = list(moderator_removed_data)
            
            if moderator_removed_records:
                self.write_dataset(iter(moderator_removed_records), moderator_removed_path, compression)
                dataset_paths['moderator_removed'] = str(moderator_removed_path)
                self.logger.info(f"Exported {len(moderator_removed_records)} moderator-removed records")
            else:
                self.logger.warning("No moderator-removed records to export")
                
            return dataset_paths
            
        except Exception as e:
            self.logger.error(f"Failed to export training datasets: {e}")
            raise
            
    def export_combined_dataset(self, all_data: Iterator[Dict[str, Any]],
                              output_path: Union[str, Path],
                              compression: Optional[str] = None) -> str:
        """
        Export combined dataset with all removal types.
        
        Args:
            all_data: Iterator of all comment records
            output_path: Path for combined dataset file
            compression: Compression algorithm
            
        Returns:
            str: Path to created combined dataset file
            
        Requirements: 4.3, 4.4
        """
        output_path = Path(output_path)
        compression = compression or self.default_compression
        
        self.logger.info(f"Exporting combined dataset to {output_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            all_records = list(all_data)
            
            if not all_records:
                raise ValueError("No records to export in combined dataset")
                
            # Add dataset metadata
            for record in all_records:
                record['export_timestamp'] = datetime.now().isoformat()
                record['dataset_version'] = '1.0'
                
            result_path = self.write_dataset(iter(all_records), output_path, compression)
            
            self.logger.info(f"Exported {len(all_records)} total records to combined dataset")
            return result_path
            
        except Exception as e:
            self.logger.error(f"Failed to export combined dataset: {e}")
            raise
            
    def create_dataset_with_schema_validation(self, data: Iterator[Dict[str, Any]],
                                            output_path: Union[str, Path],
                                            expected_columns: List[str],
                                            compression: Optional[str] = None) -> str:
        """
        Create dataset with schema validation to ensure required columns.
        
        Args:
            data: Iterator of record dictionaries
            output_path: Path for output Parquet file
            expected_columns: List of required column names
            compression: Compression algorithm
            
        Returns:
            str: Path to created Parquet file
            
        Requirements: 4.4
        """
        output_path = Path(output_path)
        compression = compression or self.default_compression
        
        self.logger.info(f"Creating dataset with schema validation: {output_path}")
        
        # Collect and validate data
        records = list(data)
        if not records:
            raise ValueError("No data provided for dataset creation")
            
        # Validate schema
        sample_record = records[0]
        missing_columns = set(expected_columns) - set(sample_record.keys())
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Ensure all records have required columns
        validated_records = []
        for i, record in enumerate(records):
            validated_record = {}
            
            for column in expected_columns:
                if column in record:
                    validated_record[column] = record[column]
                else:
                    # Add default value for missing column
                    validated_record[column] = self._get_default_value_for_column(column)
                    self.logger.warning(f"Record {i} missing column '{column}', using default value")
                    
            # Add any extra columns that exist
            for key, value in record.items():
                if key not in validated_record:
                    validated_record[key] = value
                    
            validated_records.append(validated_record)
            
        return self.write_dataset(iter(validated_records), output_path, compression)
        
    def _get_default_value_for_column(self, column: str) -> Any:
        """Get appropriate default value for a missing column."""
        # Define default values based on column name patterns
        if column in ['id', 'comment_text', 'subreddit', 'removal_type', 'target_label', 'author']:
            return ''
        elif column in ['score', 'controversiality', 'gilded', 'comment_length']:
            return 0
        elif column in ['has_parent', 'is_top_level']:
            return False
        elif column == 'timestamp':
            return datetime.fromtimestamp(0)
        else:
            return None
            
    def validate_parquet_compatibility(self, data_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate data compatibility with Parquet format and pandas/dask.
        
        Args:
            data_sample: Sample of data records for validation
            
        Returns:
            Dict: Validation results and recommendations
            
        Requirements: 4.5
        """
        validation_results = {
            'is_compatible': True,
            'issues': [],
            'recommendations': [],
            'schema_info': {}
        }
        
        if not data_sample:
            validation_results['is_compatible'] = False
            validation_results['issues'].append("No data sample provided")
            return validation_results
            
        try:
            # Create test DataFrame
            df_test = pd.DataFrame(data_sample)
            
            # Check for problematic data types
            for column in df_test.columns:
                dtype = df_test[column].dtype
                
                # Check for object columns that might cause issues
                if dtype == 'object':
                    # Check if it's actually mixed types
                    sample_values = df_test[column].dropna().head(100)
                    value_types = set(type(v).__name__ for v in sample_values)
                    
                    if len(value_types) > 1:
                        validation_results['issues'].append(
                            f"Column '{column}' has mixed types: {value_types}"
                        )
                        validation_results['recommendations'].append(
                            f"Convert column '{column}' to consistent type"
                        )
                        
                # Check for very large strings that might cause memory issues
                if dtype == 'object':
                    max_length = df_test[column].astype(str).str.len().max()
                    if max_length > 10000:
                        validation_results['recommendations'].append(
                            f"Column '{column}' has very long strings (max: {max_length}), consider truncation"
                        )
                        
                validation_results['schema_info'][column] = str(dtype)
                
            # Test Parquet schema creation
            try:
                schema = self.optimize_schema(data_sample)
                validation_results['schema_info']['parquet_schema'] = str(schema)
            except Exception as e:
                validation_results['is_compatible'] = False
                validation_results['issues'].append(f"Schema optimization failed: {e}")
                
            # Test actual Parquet conversion
            try:
                table = pa.Table.from_pandas(df_test)
                validation_results['schema_info']['arrow_schema'] = str(table.schema)
            except Exception as e:
                validation_results['is_compatible'] = False
                validation_results['issues'].append(f"Arrow conversion failed: {e}")
                
        except Exception as e:
            validation_results['is_compatible'] = False
            validation_results['issues'].append(f"Validation failed: {e}")
            
        return validation_results        

    def generate_schema_documentation(self, output_path: Union[str, Path], 
                                    data_sample: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate comprehensive schema documentation for the dataset.
        
        Args:
            output_path: Path for schema documentation file
            data_sample: Optional sample data for schema inference
            
        Returns:
            str: Path to created documentation file
            
        Requirements: 4.6
        """
        output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate documentation content
        doc_content = self._generate_schema_doc_content(data_sample)
        
        # Write documentation
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
            
        self.logger.info(f"Generated schema documentation: {output_path}")
        return str(output_path)
        
    def _generate_schema_doc_content(self, data_sample: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate the content for schema documentation."""
        
        doc_lines = [
            "# Reddit Deleted Comments Dataset Schema Documentation",
            "",
            "## Overview",
            "",
            "This document describes the schema and data types for the Reddit deleted comments dataset.",
            "The dataset is stored in Parquet format with compression for efficient storage and querying.",
            "",
            "## Dataset Files",
            "",
            "- `user_deleted_train.parquet`: Comments deleted by users",
            "- `removed_by_moderators_train.parquet`: Comments removed by moderators",
            "",
            "## Schema Definition",
            "",
            "### Core Fields",
            "",
            "| Column | Type | Description | Example |",
            "|--------|------|-------------|---------|",
            "| id | string | Unique Reddit comment ID | 'abc123' |",
            "| comment_text | string | Original comment text or placeholder | 'This is a comment' or '[CONTENT_REMOVED]' |",
            "| subreddit | string | Subreddit name (lowercase) | 'askreddit' |",
            "| timestamp | timestamp(us, UTC) | Comment creation time | 2023-01-01T12:00:00Z |",
            "| removal_type | string (categorical) | Type of removal | 'user_deleted', 'moderator_removed' |",
            "| target_label | string (categorical) | ML training label | 'toxic_content', 'voluntary_deletion' |",
            "",
            "### Relationship Fields",
            "",
            "| Column | Type | Description | Example |",
            "|--------|------|-------------|---------|",
            "| parent_id | string | Parent comment ID (if reply) | 'def456' |",
            "| thread_id | string | Thread/submission ID | 'ghi789' |",
            "| has_parent | boolean | Whether comment is a reply | true |",
            "| is_top_level | boolean | Whether comment is top-level | false |",
            "",
            "### Engagement Fields",
            "",
            "| Column | Type | Description | Example |",
            "|--------|------|-------------|---------|",
            "| score | int32 | Comment score (upvotes - downvotes) | 42 |",
            "| controversiality | int32 | Controversy indicator (0 or 1) | 0 |",
            "| gilded | int32 | Number of Reddit awards | 1 |",
            "| comment_length | int32 | Length of comment text | 150 |",
            "",
            "### Context Fields",
            "",
            "| Column | Type | Description | Example |",
            "|--------|------|-------------|---------|",
            "| author | string (categorical) | Comment author or placeholder | 'username' or '[UNKNOWN_USER]' |",
            "| parent_context | string | Parent comment text (truncated) | 'Previous comment text...' |",
            "| thread_title | string | Thread title | 'What is your favorite...?' |",
            "| thread_selftext | string | Thread description (truncated) | 'Please share your thoughts...' |",
            "| thread_score | int32 | Thread score | 1500 |",
            "| thread_num_comments | int32 | Number of comments in thread | 250 |",
            "",
            "### Quality Fields",
            "",
            "| Column | Type | Description | Example |",
            "|--------|------|-------------|---------|",
            "| data_quality | struct | Data quality assessment | {'completeness_score': 0.85, 'training_ready': true} |",
            "",
            "## Data Types and Encoding",
            "",
            "### String Fields",
            "- **Encoding**: UTF-8",
            "- **Dictionary Encoding**: Applied to categorical fields (subreddit, removal_type, target_label, author)",
            "- **Null Handling**: Empty strings for missing text, placeholders for unavailable content",
            "",
            "### Timestamp Fields",
            "- **Format**: UTC timestamps with microsecond precision",
            "- **Timezone**: All timestamps are in UTC",
            "- **Null Handling**: Unix epoch (1970-01-01T00:00:00Z) for missing timestamps",
            "",
            "### Numeric Fields",
            "- **Integer Fields**: 32-bit signed integers",
            "- **Null Handling**: 0 for missing numeric values",
            "",
            "### Boolean Fields",
            "- **Type**: Boolean true/false",
            "- **Null Handling**: false for missing boolean values",
            "",
            "## Compression and Storage",
            "",
            "### Compression Algorithm",
            "- **Default**: Snappy compression for fast read/write",
            "- **Alternative**: Gzip for better compression ratio",
            "- **Row Groups**: 50,000 records per row group for optimal querying",
            "",
            "### File Organization",
            "- **Format**: Apache Parquet 2.6",
            "- **Dictionary Encoding**: Enabled for categorical columns",
            "- **Statistics**: Column statistics included for query optimization",
            "",
            "## Usage Examples",
            "",
            "### Reading with Pandas",
            "```python",
            "import pandas as pd",
            "",
            "# Read user-deleted comments",
            "user_deleted = pd.read_parquet('user_deleted_train.parquet')",
            "",
            "# Read moderator-removed comments",
            "mod_removed = pd.read_parquet('removed_by_moderators_train.parquet')",
            "",
            "# Filter by subreddit",
            "askreddit_comments = user_deleted[user_deleted['subreddit'] == 'askreddit']",
            "```",
            "",
            "### Reading with Dask",
            "```python",
            "import dask.dataframe as dd",
            "",
            "# Read large dataset with Dask",
            "df = dd.read_parquet('*.parquet')",
            "",
            "# Compute statistics",
            "stats = df.groupby('removal_type')['score'].mean().compute()",
            "```",
            "",
            "### Querying with PyArrow",
            "```python",
            "import pyarrow.parquet as pq",
            "",
            "# Read with filters",
            "table = pq.read_table(",
            "    'user_deleted_train.parquet',",
            "    filters=[('score', '>', 10), ('subreddit', '==', 'science')]",
            ")",
            "```",
            "",
            "## Data Quality Notes",
            "",
            "### Content Availability",
            "- **[CONTENT_REMOVED]**: Original content was deletion marker",
            "- **[CONTENT_UNAVAILABLE]**: Content could not be extracted",
            "- **[UNKNOWN_USER]**: Author information unavailable",
            "",
            "### Training Labels",
            "",
            "#### User Deletion Labels",
            "- `voluntary_deletion`: Standard user deletion",
            "- `regret_deletion`: Deletion of downvoted content",
            "- `controversial_deletion`: Deletion of controversial content",
            "- `high_visibility_deletion`: Deletion of highly upvoted content",
            "",
            "#### Moderator Removal Labels",
            "- `toxic_content`: Removed for toxicity/harassment",
            "- `rule_violation`: Removed for subreddit rule violation",
            "- `misinformation`: Removed for false information",
            "- `policy_violation`: Removed for site-wide policy violation",
            "- `spam`: Removed as spam content",
            "",
            "### Completeness Scoring",
            "The `data_quality.completeness_score` field ranges from 0.0 to 1.0:",
            "- **1.0**: All fields available with high confidence",
            "- **0.8-0.9**: Most fields available, suitable for training",
            "- **0.6-0.7**: Adequate for training with some missing context",
            "- **<0.6**: May not be suitable for high-quality training",
            "",
            "## Compatibility",
            "",
            "### Supported Libraries",
            "- **Pandas**: Full compatibility for data analysis",
            "- **Dask**: Distributed processing of large datasets",
            "- **PyArrow**: Direct Parquet reading with filters",
            "- **Spark**: Compatible with Spark DataFrames",
            "- **DuckDB**: Fast analytical queries",
            "",
            "### Performance Recommendations",
            "- Use column filters when reading subsets of data",
            "- Leverage dictionary encoding for categorical analysis",
            "- Use row group statistics for efficient querying",
            "- Consider partitioning by subreddit for large-scale analysis",
            "",
            f"## Generation Info",
            f"",
            f"- **Generated**: {datetime.now().isoformat()}",
            f"- **Parquet Version**: {self.parquet_version}",
            f"- **Default Compression**: {self.default_compression}",
            f"- **Chunk Size**: {self.chunk_size:,} records",
        ]
        
        # Add sample schema if data is provided
        if data_sample:
            doc_lines.extend([
                "",
                "## Sample Schema (PyArrow)",
                "",
                "```",
                str(self.optimize_schema(data_sample)),
                "```"
            ])
            
        return "\n".join(doc_lines)
        
    def validate_dataset_integrity(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate the integrity and compatibility of a Parquet dataset.
        
        Args:
            file_path: Path to Parquet file to validate
            
        Returns:
            Dict: Validation results
            
        Requirements: 4.6
        """
        file_path = Path(file_path)
        
        validation_results = {
            'file_exists': False,
            'readable': False,
            'schema_valid': False,
            'pandas_compatible': False,
            'dask_compatible': False,
            'file_size_mb': 0,
            'record_count': 0,
            'compression_info': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check file existence
            if not file_path.exists():
                validation_results['issues'].append(f"File does not exist: {file_path}")
                return validation_results
                
            validation_results['file_exists'] = True
            validation_results['file_size_mb'] = file_path.stat().st_size / (1024 * 1024)
            
            # Test PyArrow reading
            try:
                parquet_file = pq.ParquetFile(file_path)
                validation_results['readable'] = True
                validation_results['record_count'] = parquet_file.metadata.num_rows
                validation_results['compression_info'] = {
                    'algorithm': str(parquet_file.metadata.row_group(0).column(0).compression),
                    'num_row_groups': parquet_file.metadata.num_row_groups
                }
                validation_results['schema_valid'] = True
                
            except Exception as e:
                validation_results['issues'].append(f"PyArrow reading failed: {e}")
                
            # Test Pandas compatibility
            try:
                df = pd.read_parquet(file_path, nrows=100)  # Test with small sample
                validation_results['pandas_compatible'] = True
            except Exception as e:
                validation_results['issues'].append(f"Pandas compatibility failed: {e}")
                
            # Test Dask compatibility
            try:
                import dask.dataframe as dd
                ddf = dd.read_parquet(file_path)
                ddf.head()  # Trigger computation
                validation_results['dask_compatible'] = True
            except ImportError:
                validation_results['recommendations'].append("Install Dask for distributed processing")
            except Exception as e:
                validation_results['issues'].append(f"Dask compatibility failed: {e}")
                
            # Performance recommendations
            if validation_results['file_size_mb'] > 1000:
                validation_results['recommendations'].append(
                    "Large file detected - consider using Dask for processing"
                )
                
            if validation_results['compression_info'].get('num_row_groups', 0) > 100:
                validation_results['recommendations'].append(
                    "Many row groups detected - may impact query performance"
                )
                
        except Exception as e:
            validation_results['issues'].append(f"Validation failed: {e}")
            
        return validation_results