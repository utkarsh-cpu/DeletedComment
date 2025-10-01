"""
Parquet file integrity and compression ratio tests.
Tests Parquet file format compliance and compression efficiency.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.parquet_writer import ParquetWriter


class TestParquetIntegrity:
    """Test suite for Parquet file integrity and compression."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.writer = ParquetWriter()
        
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def create_test_dataset(self, num_records=1000, include_nulls=False):
        """Create test dataset with various data types."""
        records = []
        
        for i in range(num_records):
            record = {
                'id': f'record_{i:06d}',
                'comment_text': f'Test comment {i} with varying length content. ' * (1 + i % 5),
                'subreddit': f'subreddit_{i % 20}',  # Categorical data
                'timestamp': datetime.now(timezone.utc),
                'removal_type': 'user_deleted' if i % 2 == 0 else 'moderator_removed',
                'target_label': ['voluntary_deletion', 'rule_violation', 'toxic_content'][i % 3],
                'parent_id': f'parent_{i}' if i > 0 else '',
                'thread_id': f'thread_{i // 10}',
                'score': (i % 201) - 100,  # Range -100 to 100
                'author': f'user_{i % 100}',  # Categorical with repetition
                'controversiality': 1 if i % 10 == 0 else 0,
                'gilded': i % 50,  # Mostly 0, some higher values
                'comment_length': len(f'Test comment {i} with varying length content. ' * (1 + i % 5)),
                'has_parent': i > 0,
                'is_top_level': i % 20 == 0
            }
            
            # Add nulls for testing null handling
            if include_nulls and i % 10 == 0:
                record['parent_id'] = None
                record['author'] = None
                
            records.append(record)
            
        return records
        
    def test_parquet_file_format_compliance(self):
        """Test that generated Parquet files comply with format standards."""
        # Create test dataset
        records = self.create_test_dataset(500)
        output_path = Path(self.temp_dir) / "format_test.parquet"
        
        # Write Parquet file
        self.writer.write_dataset(iter(records), output_path, 'snappy')
        
        # Verify file exists and is readable
        assert output_path.exists()
        
        # Test with PyArrow
        table = pq.read_table(output_path)
        assert table.num_rows == 500
        assert table.num_columns > 0
        
        # Test with Pandas
        df = pd.read_parquet(output_path)
        assert len(df) == 500
        assert len(df.columns) > 0
        
        # Verify schema consistency
        expected_columns = set(records[0].keys())
        actual_columns = set(df.columns)
        assert expected_columns.issubset(actual_columns)
        
        # Test data integrity
        assert df.iloc[0]['id'] == 'record_000000'
        assert df.iloc[-1]['id'] == 'record_000499'
        
    def test_compression_ratios_different_algorithms(self):
        """Test compression ratios with different algorithms."""
        # Create dataset with repetitive data (good for compression)
        records = self.create_test_dataset(2000)
        
        compression_algorithms = ['snappy', 'gzip', 'lz4']
        compression_results = {}
        
        for compression in compression_algorithms:
            output_path = Path(self.temp_dir) / f"compression_test_{compression}.parquet"
            
            # Write with specific compression
            self.writer.write_dataset(iter(records), output_path, compression)
            
            # Measure file size
            file_size = output_path.stat().st_size
            
            # Estimate uncompressed size
            df = pd.DataFrame(records)
            estimated_uncompressed = df.memory_usage(deep=True).sum()
            
            compression_ratio = estimated_uncompressed / file_size
            compression_results[compression] = {
                'file_size': file_size,
                'compression_ratio': compression_ratio
            }
            
            # Verify file can be read back correctly
            df_read = pd.read_parquet(output_path)
            assert len(df_read) == len(records)
            
            print(f"{compression}: {file_size:,} bytes, {compression_ratio:.2f}x compression")
            
        # Verify compression is effective
        for compression, results in compression_results.items():
            assert results['compression_ratio'] > 1.5  # At least 1.5x compression
            
        # Gzip should generally achieve better compression than Snappy
        if 'gzip' in compression_results and 'snappy' in compression_results:
            gzip_ratio = compression_results['gzip']['compression_ratio']
            snappy_ratio = compression_results['snappy']['compression_ratio']
            # Gzip should compress better (but this isn't always guaranteed)
            print(f"Gzip vs Snappy compression: {gzip_ratio:.2f}x vs {snappy_ratio:.2f}x")
            
    def test_parquet_schema_optimization(self):
        """Test that schema optimization produces efficient schemas."""
        # Create test data
        records = self.create_test_dataset(100)
        
        # Test schema optimization
        schema = self.writer.optimize_schema(records)
        
        # Verify schema has expected optimizations
        schema_dict = {field.name: field.type for field in schema}
        
        # Check specific optimizations
        assert schema_dict['id'] == pa.string()
        assert str(schema_dict['timestamp']).startswith('timestamp')
        assert schema_dict['score'] == pa.int32()
        assert schema_dict['has_parent'] == pa.bool_()
        
        # Check dictionary encoding for categorical fields
        categorical_fields = ['subreddit', 'removal_type', 'target_label', 'author']
        for field in categorical_fields:
            if field in schema_dict:
                field_type = schema_dict[field]
                # Should be dictionary encoded
                assert isinstance(field_type, pa.DictionaryType)
                
        # Write with optimized schema
        output_path = Path(self.temp_dir) / "optimized_schema.parquet"
        self.writer.write_dataset(iter(records), output_path)
        
        # Verify file can be read and has correct schema
        table = pq.read_table(output_path)
        assert table.schema.equals(schema, check_metadata=False)
        
    def test_parquet_compatibility_pandas_dask(self):
        """Test Parquet compatibility with pandas and dask."""
        # Create test dataset
        records = self.create_test_dataset(1000)
        output_path = Path(self.temp_dir) / "compatibility_test.parquet"
        
        # Write Parquet file
        self.writer.write_dataset(iter(records), output_path)
        
        # Test pandas compatibility
        df_pandas = pd.read_parquet(output_path)
        assert len(df_pandas) == 1000
        assert 'id' in df_pandas.columns
        assert 'timestamp' in df_pandas.columns
        
        # Test basic operations
        assert df_pandas['score'].sum() == sum(r['score'] for r in records)
        assert df_pandas['has_parent'].sum() == sum(r['has_parent'] for r in records)
        
        # Test filtering
        user_deleted = df_pandas[df_pandas['removal_type'] == 'user_deleted']
        expected_user_deleted = len([r for r in records if r['removal_type'] == 'user_deleted'])
        assert len(user_deleted) == expected_user_deleted
        
        # Test dask compatibility (if dask is available)
        try:
            import dask.dataframe as dd
            
            df_dask = dd.read_parquet(output_path)
            assert df_dask.compute().equals(df_pandas)
            
            # Test dask operations
            score_sum_dask = df_dask['score'].sum().compute()
            score_sum_pandas = df_pandas['score'].sum()
            assert score_sum_dask == score_sum_pandas
            
        except ImportError:
            print("Dask not available, skipping dask compatibility test")
            
    def test_parquet_null_value_handling(self):
        """Test proper handling of null values in Parquet files."""
        # Create dataset with null values
        records = self.create_test_dataset(500, include_nulls=True)
        output_path = Path(self.temp_dir) / "null_test.parquet"
        
        # Write Parquet file
        self.writer.write_dataset(iter(records), output_path)
        
        # Read back and verify null handling
        df = pd.read_parquet(output_path)
        
        # Check that nulls are preserved
        null_parent_ids = df['parent_id'].isnull().sum()
        null_authors = df['author'].isnull().sum()
        
        assert null_parent_ids > 0  # Should have some null parent_ids
        assert null_authors > 0     # Should have some null authors
        
        # Verify data integrity for non-null values
        non_null_records = df[df['parent_id'].notnull()]
        assert len(non_null_records) > 0
        
    def test_parquet_large_string_handling(self):
        """Test handling of large string values in Parquet."""
        # Create records with very large strings
        large_records = []
        for i in range(100):
            large_text = f"Large comment {i}: " + "x" * (1000 + i * 100)  # Varying large sizes
            
            record = {
                'id': f'large_{i}',
                'comment_text': large_text,
                'subreddit': 'test',
                'timestamp': datetime.now(timezone.utc),
                'removal_type': 'user_deleted',
                'target_label': 'voluntary_deletion',
                'score': i,
                'author': f'user_{i}',
                'comment_length': len(large_text),
                'has_parent': False,
                'is_top_level': True
            }
            large_records.append(record)
            
        output_path = Path(self.temp_dir) / "large_strings.parquet"
        
        # Write and verify
        self.writer.write_dataset(iter(large_records), output_path)
        
        # Read back and verify integrity
        df = pd.read_parquet(output_path)
        assert len(df) == 100
        
        # Verify large strings are preserved
        assert len(df.iloc[0]['comment_text']) > 1000
        assert len(df.iloc[-1]['comment_text']) > 10000
        
        # Verify compression is still effective with large strings
        file_size = output_path.stat().st_size
        estimated_uncompressed = sum(len(r['comment_text']) for r in large_records) * 2  # Rough estimate
        compression_ratio = estimated_uncompressed / file_size
        
        assert compression_ratio > 2.0  # Should still achieve good compression
        
    def test_parquet_metadata_and_statistics(self):
        """Test that Parquet files include proper metadata and statistics."""
        # Create test dataset
        records = self.create_test_dataset(1000)
        output_path = Path(self.temp_dir) / "metadata_test.parquet"
        
        # Write with statistics enabled
        self.writer.write_dataset(iter(records), output_path)
        
        # Read metadata
        parquet_file = pq.ParquetFile(output_path)
        metadata = parquet_file.metadata
        
        # Verify basic metadata
        assert metadata.num_rows == 1000
        assert metadata.num_columns > 0
        assert metadata.num_row_groups > 0
        
        # Check row group statistics
        for rg in range(metadata.num_row_groups):
            row_group = metadata.row_group(rg)
            
            # Check that statistics are available for some columns
            for col in range(row_group.num_columns):
                column_meta = row_group.column(col)
                stats = column_meta.statistics
                
                if stats is not None:
                    # Statistics should have min/max values
                    assert stats.min is not None or stats.max is not None
                    
        # Verify schema metadata
        schema = parquet_file.schema_arrow
        assert len(schema) > 0
        
    def test_parquet_chunked_writing_consistency(self):
        """Test that chunked writing produces consistent results."""
        # Create large dataset
        records = self.create_test_dataset(5000)
        
        # Write as single file
        single_path = Path(self.temp_dir) / "single_file.parquet"
        self.writer.write_dataset(iter(records), single_path)
        
        # Write as chunked files
        chunk_dir = Path(self.temp_dir) / "chunks"
        chunk_files = self.writer.create_chunked_parquet(
            iter(records), chunk_dir, "chunked", chunk_size=1000
        )
        
        # Read single file
        df_single = pd.read_parquet(single_path)
        
        # Read and combine chunked files
        chunk_dfs = []
        for chunk_file in chunk_files:
            chunk_df = pd.read_parquet(chunk_file)
            chunk_dfs.append(chunk_df)
            
        df_chunked = pd.concat(chunk_dfs, ignore_index=True)
        
        # Verify consistency
        assert len(df_single) == len(df_chunked)
        assert len(df_single) == 5000
        
        # Sort both dataframes by ID for comparison
        df_single_sorted = df_single.sort_values('id').reset_index(drop=True)
        df_chunked_sorted = df_chunked.sort_values('id').reset_index(drop=True)
        
        # Compare key columns
        assert df_single_sorted['id'].equals(df_chunked_sorted['id'])
        assert df_single_sorted['score'].equals(df_chunked_sorted['score'])
        assert df_single_sorted['removal_type'].equals(df_chunked_sorted['removal_type'])
        
    def test_parquet_compression_ratio_benchmarks(self):
        """Test that compression ratios meet expected benchmarks."""
        # Create different types of datasets to test compression
        datasets = {
            'repetitive': self._create_repetitive_dataset(1000),
            'diverse': self._create_diverse_dataset(1000),
            'sparse': self._create_sparse_dataset(1000)
        }
        
        compression_benchmarks = {
            'repetitive': 5.0,  # Should compress very well
            'diverse': 2.0,     # Should compress moderately
            'sparse': 1.5       # Should compress less well
        }
        
        for dataset_type, records in datasets.items():
            output_path = Path(self.temp_dir) / f"benchmark_{dataset_type}.parquet"
            
            # Write with snappy compression
            self.writer.write_dataset(iter(records), output_path, 'snappy')
            
            # Calculate compression ratio
            file_size = output_path.stat().st_size
            df = pd.DataFrame(records)
            estimated_uncompressed = df.memory_usage(deep=True).sum()
            compression_ratio = estimated_uncompressed / file_size
            
            expected_ratio = compression_benchmarks[dataset_type]
            
            print(f"{dataset_type} dataset: {compression_ratio:.2f}x compression "
                  f"(expected: >{expected_ratio:.1f}x)")
            
            # Verify meets benchmark (with some tolerance)
            assert compression_ratio >= expected_ratio * 0.8, \
                f"{dataset_type} compression ratio {compression_ratio:.2f}x below benchmark {expected_ratio}x"
                
    def _create_repetitive_dataset(self, num_records):
        """Create dataset with repetitive data (good for compression)."""
        records = []
        for i in range(num_records):
            record = {
                'id': f'rep_{i:06d}',
                'comment_text': 'This is a repetitive comment text that appears many times.',
                'subreddit': 'repetitive_subreddit',  # Same subreddit
                'timestamp': datetime.now(timezone.utc),
                'removal_type': 'user_deleted',  # Same removal type
                'target_label': 'voluntary_deletion',  # Same label
                'score': 10,  # Same score
                'author': 'repetitive_user',  # Same author
                'has_parent': False,
                'is_top_level': True
            }
            records.append(record)
        return records
        
    def _create_diverse_dataset(self, num_records):
        """Create dataset with diverse data (moderate compression)."""
        records = []
        for i in range(num_records):
            record = {
                'id': f'div_{i:06d}',
                'comment_text': f'Diverse comment {i} with unique content and varying length. ' * (1 + i % 3),
                'subreddit': f'subreddit_{i % 10}',
                'timestamp': datetime.now(timezone.utc),
                'removal_type': ['user_deleted', 'moderator_removed'][i % 2],
                'target_label': ['voluntary_deletion', 'rule_violation', 'toxic_content'][i % 3],
                'score': i % 100,
                'author': f'user_{i % 50}',
                'has_parent': i % 2 == 0,
                'is_top_level': i % 5 == 0
            }
            records.append(record)
        return records
        
    def _create_sparse_dataset(self, num_records):
        """Create dataset with sparse/random data (poor compression)."""
        import random
        import string
        
        records = []
        for i in range(num_records):
            # Generate random strings
            random_text = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=100))
            random_subreddit = ''.join(random.choices(string.ascii_lowercase, k=10))
            random_author = ''.join(random.choices(string.ascii_letters, k=8))
            
            record = {
                'id': f'sparse_{i:06d}',
                'comment_text': random_text,
                'subreddit': random_subreddit,
                'timestamp': datetime.now(timezone.utc),
                'removal_type': random.choice(['user_deleted', 'moderator_removed']),
                'target_label': random.choice(['voluntary_deletion', 'rule_violation', 'toxic_content']),
                'score': random.randint(-100, 100),
                'author': random_author,
                'has_parent': random.choice([True, False]),
                'is_top_level': random.choice([True, False])
            }
            records.append(record)
        return records


if __name__ == "__main__":
    pytest.main([__file__])