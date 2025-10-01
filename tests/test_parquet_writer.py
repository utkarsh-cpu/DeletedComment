"""
Unit tests for parquet_writer module.
Tests Parquet file creation, compression, and schema optimization.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import pandas as pd
import pyarrow as pa

from src.parquet_writer import ParquetWriter, CompressionStats


class TestParquetWriter:
    """Test suite for ParquetWriter class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.writer = ParquetWriter()
        
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def create_sample_data(self, num_records=10):
        """Helper to create sample training data."""
        data = []
        for i in range(num_records):
            record = {
                'id': f'comment_{i}',
                'comment_text': f'This is test comment number {i}',
                'subreddit': 'test',
                'timestamp': datetime.now(timezone.utc),
                'removal_type': 'user_deleted' if i % 2 == 0 else 'moderator_removed',
                'target_label': 'voluntary_deletion' if i % 2 == 0 else 'rule_violation',
                'parent_id': f'parent_{i}' if i > 0 else '',
                'thread_id': f'thread_{i // 3}',
                'score': i * 2,
                'author': f'user_{i}',
                'controversiality': 1 if i % 5 == 0 else 0,
                'gilded': 1 if i % 10 == 0 else 0,
                'comment_length': len(f'This is test comment number {i}'),
                'has_parent': i > 0,
                'is_top_level': i == 0
            }
            data.append(record)
        return data
        
    def test_init_default_config(self):
        """Test writer initialization with default configuration."""
        writer = ParquetWriter()
        
        assert writer.default_compression == 'snappy'
        assert writer.parquet_version == '2.6'
        assert writer.chunk_size == 100000
        assert 'snappy' in writer.supported_compressions
        assert writer.stats['files_created'] == 0
        
    def test_init_custom_config(self):
        """Test writer initialization with custom configuration."""
        config = {
            'storage': {
                'compression': 'gzip',
                'parquet_version': '2.4'
            },
            'processing': {
                'chunk_size': 50000
            }
        }
        
        writer = ParquetWriter(config)
        
        assert writer.default_compression == 'gzip'
        assert writer.parquet_version == '2.4'
        assert writer.chunk_size == 50000
        
    def test_write_dataset_success(self):
        """Test successful dataset writing."""
        data = self.create_sample_data(5)
        output_path = Path(self.temp_dir) / "test_dataset.parquet"
        
        result_path = self.writer.write_dataset(iter(data), output_path, 'snappy')
        
        assert result_path == str(output_path)
        assert output_path.exists()
        
        # Verify file can be read back
        df = pd.read_parquet(output_path)
        assert len(df) == 5
        assert 'id' in df.columns
        assert 'comment_text' in df.columns
        
        # Check stats
        stats = self.writer.get_compression_stats()
        assert stats['files_created'] == 1
        assert stats['total_records'] == 5
        
    def test_write_dataset_empty_data(self):
        """Test writing dataset with empty data."""
        output_path = Path(self.temp_dir) / "empty_dataset.parquet"
        
        with pytest.raises(ValueError, match="No data provided for writing"):
            self.writer.write_dataset(iter([]), output_path)
            
    def test_write_dataset_unsupported_compression(self):
        """Test writing with unsupported compression algorithm."""
        data = self.create_sample_data(3)
        output_path = Path(self.temp_dir) / "test_dataset.parquet"
        
        # Should fall back to snappy
        result_path = self.writer.write_dataset(iter(data), output_path, 'unsupported_compression')
        
        assert result_path == str(output_path)
        assert output_path.exists()
        
    def test_write_dataset_creates_directory(self):
        """Test that write_dataset creates output directory if needed."""
        data = self.create_sample_data(3)
        nested_path = Path(self.temp_dir) / "nested" / "directory" / "test.parquet"
        
        result_path = self.writer.write_dataset(iter(data), nested_path)
        
        assert result_path == str(nested_path)
        assert nested_path.exists()
        assert nested_path.parent.exists()
        
    def test_create_chunked_parquet(self):
        """Test creating chunked Parquet files."""
        data = self.create_sample_data(12)  # More than chunk size
        output_dir = Path(self.temp_dir) / "chunks"
        
        chunk_files = self.writer.create_chunked_parquet(
            iter(data), 
            output_dir, 
            "test_chunks", 
            chunk_size=5
        )
        
        # Should create 3 chunks: 5, 5, 2
        assert len(chunk_files) == 3
        
        for chunk_file in chunk_files:
            assert Path(chunk_file).exists()
            assert "test_chunks_chunk_" in chunk_file
            
        # Verify chunk contents
        df1 = pd.read_parquet(chunk_files[0])
        df2 = pd.read_parquet(chunk_files[1])
        df3 = pd.read_parquet(chunk_files[2])
        
        assert len(df1) == 5
        assert len(df2) == 5
        assert len(df3) == 2
        
    def test_optimize_schema(self):
        """Test schema optimization."""
        data_sample = self.create_sample_data(3)
        
        schema = self.writer.optimize_schema(data_sample)
        
        assert isinstance(schema, pa.Schema)
        
        # Check specific field optimizations
        field_names = [field.name for field in schema]
        assert 'id' in field_names
        assert 'timestamp' in field_names
        assert 'score' in field_names
        
        # Check field types
        id_field = schema.field('id')
        assert id_field.type == pa.string()
        
        timestamp_field = schema.field('timestamp')
        assert str(timestamp_field.type).startswith('timestamp')
        
        score_field = schema.field('score')
        assert score_field.type == pa.int32()
        
    def test_optimize_schema_empty_sample(self):
        """Test schema optimization with empty sample."""
        with pytest.raises(ValueError, match="Cannot optimize schema with empty data sample"):
            self.writer.optimize_schema([])
            
    def test_export_training_datasets(self):
        """Test exporting separate training datasets."""
        user_deleted_data = [
            record for record in self.create_sample_data(10) 
            if record['removal_type'] == 'user_deleted'
        ]
        
        moderator_removed_data = [
            record for record in self.create_sample_data(10) 
            if record['removal_type'] == 'moderator_removed'
        ]
        
        output_dir = Path(self.temp_dir) / "training_datasets"
        
        dataset_paths = self.writer.export_training_datasets(
            iter(user_deleted_data),
            iter(moderator_removed_data),
            output_dir
        )
        
        assert 'user_deleted' in dataset_paths
        assert 'moderator_removed' in dataset_paths
        
        user_deleted_path = Path(dataset_paths['user_deleted'])
        moderator_removed_path = Path(dataset_paths['moderator_removed'])
        
        assert user_deleted_path.exists()
        assert moderator_removed_path.exists()
        assert user_deleted_path.name == "user_deleted_train.parquet"
        assert moderator_removed_path.name == "removed_by_moderators_train.parquet"
        
        # Verify contents
        user_df = pd.read_parquet(user_deleted_path)
        mod_df = pd.read_parquet(moderator_removed_path)
        
        assert len(user_df) == len(user_deleted_data)
        assert len(mod_df) == len(moderator_removed_data)
        
        # Verify all records have correct removal type
        assert all(user_df['removal_type'] == 'user_deleted')
        assert all(mod_df['removal_type'] == 'moderator_removed')
        
    def test_export_training_datasets_empty_data(self):
        """Test exporting training datasets with empty data."""
        output_dir = Path(self.temp_dir) / "empty_datasets"
        
        dataset_paths = self.writer.export_training_datasets(
            iter([]),  # Empty user deleted
            iter([]),  # Empty moderator removed
            output_dir
        )
        
        # Should return empty dict when no data
        assert dataset_paths == {}
        
    def test_export_combined_dataset(self):
        """Test exporting combined dataset."""
        all_data = self.create_sample_data(8)
        output_path = Path(self.temp_dir) / "combined_dataset.parquet"
        
        result_path = self.writer.export_combined_dataset(iter(all_data), output_path)
        
        assert result_path == str(output_path)
        assert output_path.exists()
        
        # Verify contents
        df = pd.read_parquet(output_path)
        assert len(df) == 8
        
        # Check that metadata was added
        assert 'export_timestamp' in df.columns
        assert 'dataset_version' in df.columns
        assert all(df['dataset_version'] == '1.0')
        
    def test_export_combined_dataset_empty(self):
        """Test exporting combined dataset with empty data."""
        output_path = Path(self.temp_dir) / "empty_combined.parquet"
        
        with pytest.raises(ValueError, match="No records to export in combined dataset"):
            self.writer.export_combined_dataset(iter([]), output_path)
            
    def test_create_dataset_with_schema_validation(self):
        """Test dataset creation with schema validation."""
        data = self.create_sample_data(5)
        output_path = Path(self.temp_dir) / "validated_dataset.parquet"
        expected_columns = ['id', 'comment_text', 'subreddit', 'removal_type', 'score']
        
        result_path = self.writer.create_dataset_with_schema_validation(
            iter(data), 
            output_path, 
            expected_columns
        )
        
        assert result_path == str(output_path)
        assert output_path.exists()
        
        # Verify all expected columns are present
        df = pd.read_parquet(output_path)
        for column in expected_columns:
            assert column in df.columns
            
    def test_create_dataset_with_schema_validation_missing_columns(self):
        """Test schema validation with missing required columns."""
        # Create data missing some required columns
        incomplete_data = [
            {
                'id': 'test1',
                'comment_text': 'Test comment'
                # Missing other required columns
            }
        ]
        
        output_path = Path(self.temp_dir) / "incomplete_dataset.parquet"
        expected_columns = ['id', 'comment_text', 'subreddit', 'removal_type']
        
        # Should still work, filling in missing columns with defaults
        result_path = self.writer.create_dataset_with_schema_validation(
            iter(incomplete_data), 
            output_path, 
            expected_columns
        )
        
        assert result_path == str(output_path)
        
        # Verify missing columns were filled with defaults
        df = pd.read_parquet(output_path)
        assert 'subreddit' in df.columns
        assert 'removal_type' in df.columns
        assert df.iloc[0]['subreddit'] == ''  # Default value
        assert df.iloc[0]['removal_type'] == ''  # Default value
        
    def test_get_default_value_for_column(self):
        """Test default value generation for missing columns."""
        assert self.writer._get_default_value_for_column('id') == ''
        assert self.writer._get_default_value_for_column('comment_text') == ''
        assert self.writer._get_default_value_for_column('score') == 0
        assert self.writer._get_default_value_for_column('has_parent') is False
        assert isinstance(self.writer._get_default_value_for_column('timestamp'), datetime)
        assert self.writer._get_default_value_for_column('unknown_column') is None
        
    def test_validate_parquet_compatibility(self):
        """Test Parquet compatibility validation."""
        # Good data
        good_data = self.create_sample_data(3)
        validation = self.writer.validate_parquet_compatibility(good_data)
        
        assert validation['is_compatible'] is True
        assert len(validation['issues']) == 0
        assert 'schema_info' in validation
        
        # Data with mixed types (problematic)
        mixed_data = [
            {'id': 'test1', 'mixed_field': 'string_value'},
            {'id': 'test2', 'mixed_field': 123},  # Different type
            {'id': 'test3', 'mixed_field': True}   # Another different type
        ]
        
        validation = self.writer.validate_parquet_compatibility(mixed_data)
        
        # Should detect mixed types issue
        mixed_type_issues = [issue for issue in validation['issues'] if 'mixed types' in issue]
        assert len(mixed_type_issues) > 0
        
    def test_validate_parquet_compatibility_empty_data(self):
        """Test Parquet compatibility validation with empty data."""
        validation = self.writer.validate_parquet_compatibility([])
        
        assert validation['is_compatible'] is False
        assert 'No data sample provided' in validation['issues']
        
    def test_generate_schema_documentation(self):
        """Test schema documentation generation."""
        output_path = Path(self.temp_dir) / "schema_docs.md"
        data_sample = self.create_sample_data(2)
        
        result_path = self.writer.generate_schema_documentation(output_path, data_sample)
        
        assert result_path == str(output_path)
        assert output_path.exists()
        
        # Verify documentation content
        with open(output_path, 'r') as f:
            content = f.read()
            
        assert "Reddit Deleted Comments Dataset Schema Documentation" in content
        assert "Core Fields" in content
        assert "Compression and Storage" in content
        assert "Usage Examples" in content
        assert "pandas" in content
        assert "dask" in content
        
    def test_estimate_original_size(self):
        """Test original size estimation."""
        data = self.create_sample_data(5)
        df = pd.DataFrame(data)
        
        estimated_size = self.writer._estimate_original_size(df)
        
        assert isinstance(estimated_size, int)
        assert estimated_size > 0
        
    def test_update_stats(self):
        """Test statistics updating."""
        compression_stats = CompressionStats(
            original_size=1000,
            compressed_size=500,
            compression_ratio=2.0,
            records_written=10,
            processing_time=1.5
        )
        
        initial_files = self.writer.stats['files_created']
        
        self.writer._update_stats(compression_stats)
        
        assert self.writer.stats['files_created'] == initial_files + 1
        assert self.writer.stats['total_records'] == 10
        assert self.writer.stats['total_compressed_size'] == 500
        assert self.writer.stats['total_original_size'] == 1000
        assert 2.0 in self.writer.stats['compression_ratios']
        
    def test_get_compression_stats(self):
        """Test compression statistics retrieval."""
        # Add some stats
        compression_stats1 = CompressionStats(
            original_size=1000,
            compressed_size=400,
            compression_ratio=2.5,
            records_written=5,
            processing_time=1.0
        )
        
        compression_stats2 = CompressionStats(
            original_size=2000,
            compressed_size=600,
            compression_ratio=3.33,
            records_written=10,
            processing_time=2.0
        )
        
        self.writer._update_stats(compression_stats1)
        self.writer._update_stats(compression_stats2)
        
        stats = self.writer.get_compression_stats()
        
        assert stats['files_created'] == 2
        assert stats['total_records'] == 15
        assert stats['total_compressed_size_mb'] == (400 + 600) / (1024 * 1024)
        assert stats['total_original_size_mb'] == (1000 + 2000) / (1024 * 1024)
        assert abs(stats['average_compression_ratio'] - ((2.5 + 3.33) / 2)) < 0.01
        assert stats['total_space_saved_mb'] == (3000 - 1000) / (1024 * 1024)
        
    def test_reset_stats(self):
        """Test statistics reset."""
        # Generate some stats
        data = self.create_sample_data(3)
        output_path = Path(self.temp_dir) / "stats_test.parquet"
        self.writer.write_dataset(iter(data), output_path)
        
        # Verify stats exist
        stats = self.writer.get_compression_stats()
        assert stats['files_created'] > 0
        
        # Reset and verify
        self.writer.reset_stats()
        new_stats = self.writer.get_compression_stats()
        assert new_stats['files_created'] == 0
        assert new_stats['total_records'] == 0
        assert new_stats['average_compression_ratio'] == 0
        
    def test_different_compression_algorithms(self):
        """Test writing with different compression algorithms."""
        data = self.create_sample_data(5)
        
        # Test different compression types
        compressions = ['snappy', 'gzip', 'lz4']
        
        for compression in compressions:
            if compression in self.writer.supported_compressions:
                output_path = Path(self.temp_dir) / f"test_{compression}.parquet"
                
                result_path = self.writer.write_dataset(iter(data), output_path, compression)
                
                assert result_path == str(output_path)
                assert output_path.exists()
                
                # Verify file can be read
                df = pd.read_parquet(output_path)
                assert len(df) == 5
                
    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        # Create larger dataset
        large_data = self.create_sample_data(1000)
        output_path = Path(self.temp_dir) / "large_dataset.parquet"
        
        result_path = self.writer.write_dataset(iter(large_data), output_path)
        
        assert result_path == str(output_path)
        assert output_path.exists()
        
        # Verify contents
        df = pd.read_parquet(output_path)
        assert len(df) == 1000
        
        # Check compression was effective
        stats = self.writer.get_compression_stats()
        assert stats['average_compression_ratio'] > 1.0  # Should have some compression


if __name__ == "__main__":
    pytest.main([__file__])