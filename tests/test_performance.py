"""
Performance tests for the deleted comment dataset pipeline.
Tests memory usage, processing speed, and compression efficiency.
"""

import os
import tempfile
import shutil
import json
import time
import psutil
import pytest
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from src.reddit_parser import RedditParser
from src.comment_classifier import CommentClassifier
from src.metadata_extractor import MetadataExtractor
from src.parquet_writer import ParquetWriter


class TestPerformance:
    """Performance test suite for the pipeline components."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize components with performance-oriented settings
        self.parser = RedditParser(chunk_size=1000, strict_validation=False)
        self.classifier = CommentClassifier()
        self.extractor = MetadataExtractor()
        self.writer = ParquetWriter()
        
        # Performance tracking
        self.process = psutil.Process()
        
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
        
    def create_large_dataset(self, num_comments=10000):
        """Create a large dataset for performance testing."""
        comments = []
        
        for i in range(num_comments):
            # Create realistic comment data with varying content lengths
            content_length = 50 + (i % 500)  # Varying content length
            
            if i % 5 == 0:
                # Deleted comment
                comment = {
                    'id': f'del_{i:06d}',
                    'body': '[deleted]',
                    'author': f'user_{i}',
                    'subreddit': f'subreddit_{i % 100}',  # 100 different subreddits
                    'created_utc': 1640995200 + i * 60,
                    'score': (i % 201) - 100,  # Score range -100 to 100
                    'parent_id': f't1_parent_{i-1}' if i > 0 else f't3_thread_{i//50}',
                    'link_id': f't3_thread_{i//50}',
                    'controversiality': 1 if i % 20 == 0 else 0,
                    'gilded': 1 if i % 100 == 0 else 0
                }
            elif i % 5 == 1:
                # Removed comment
                comment = {
                    'id': f'rem_{i:06d}',
                    'body': '[removed]',
                    'author': f'user_{i}',
                    'subreddit': f'subreddit_{i % 100}',
                    'created_utc': 1640995200 + i * 60,
                    'score': (i % 201) - 100,
                    'parent_id': f't1_parent_{i-1}' if i > 0 else f't3_thread_{i//50}',
                    'link_id': f't3_thread_{i//50}',
                    'controversiality': 1 if i % 15 == 0 else 0,
                    'gilded': 0
                }
            else:
                # Active comment with realistic content
                comment = {
                    'id': f'act_{i:06d}',
                    'body': f'This is comment number {i} with some realistic content. ' * (content_length // 50),
                    'author': f'active_user_{i}',
                    'subreddit': f'subreddit_{i % 100}',
                    'created_utc': 1640995200 + i * 60,
                    'score': (i % 201) - 50,  # Generally more positive
                    'parent_id': f't1_parent_{i-1}' if i > 0 else f't3_thread_{i//50}',
                    'link_id': f't3_thread_{i//50}',
                    'controversiality': 1 if i % 30 == 0 else 0,
                    'gilded': 1 if i % 200 == 0 else 0
                }
                
            comments.append(comment)
            
        return comments
        
    def create_large_data_file(self, comments, filename="large_dataset.json"):
        """Create a large data file for testing."""
        file_path = Path(self.temp_dir) / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for comment in comments:
                f.write(json.dumps(comment) + '\n')
                
        return file_path
        
    @pytest.mark.performance
    def test_parsing_performance_large_dataset(self):
        """Test parsing performance with large dataset."""
        # Create large dataset
        num_comments = 50000
        comments = self.create_large_dataset(num_comments)
        data_file = self.create_large_data_file(comments)
        
        # Measure file size
        file_size_mb = data_file.stat().st_size / 1024 / 1024
        print(f"Test file size: {file_size_mb:.2f} MB")
        
        # Measure parsing performance
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        parsed_count = 0
        for comment in self.parser.parse_comments_file(data_file):
            parsed_count += 1
            
            # Check memory usage periodically
            if parsed_count % 10000 == 0:
                current_memory = self.get_memory_usage()
                memory_increase = current_memory - start_memory
                print(f"Parsed {parsed_count} comments, memory increase: {memory_increase:.2f} MB")
                
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        # Calculate performance metrics
        processing_time = end_time - start_time
        comments_per_second = parsed_count / processing_time
        memory_increase = end_memory - start_memory
        
        print(f"Parsing Performance:")
        print(f"  Comments processed: {parsed_count}")
        print(f"  Processing time: {processing_time:.2f} seconds")
        print(f"  Comments per second: {comments_per_second:.2f}")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        print(f"  Memory per comment: {memory_increase / parsed_count * 1024:.2f} KB")
        
        # Performance assertions
        assert parsed_count == num_comments
        assert comments_per_second > 1000  # Should process at least 1000 comments/sec
        assert memory_increase < 500  # Should use less than 500MB additional memory
        
        # Check parsing stats
        stats = self.parser.get_parsing_stats()
        assert stats.valid_records == num_comments
        assert stats.json_errors == 0
        
    @pytest.mark.performance
    def test_classification_performance(self):
        """Test classification performance with large dataset."""
        # Create dataset focused on classification
        num_comments = 20000
        comments = self.create_large_dataset(num_comments)
        
        # Measure classification performance
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        classified_count = 0
        classification_results = []
        
        for comment in comments:
            classification = self.classifier.classify_comment(comment)
            classification_results.append(classification)
            classified_count += 1
            
            # Memory check
            if classified_count % 5000 == 0:
                current_memory = self.get_memory_usage()
                memory_increase = current_memory - start_memory
                print(f"Classified {classified_count} comments, memory: {memory_increase:.2f} MB")
                
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        # Calculate performance metrics
        processing_time = end_time - start_time
        classifications_per_second = classified_count / processing_time
        memory_increase = end_memory - start_memory
        
        print(f"Classification Performance:")
        print(f"  Comments classified: {classified_count}")
        print(f"  Processing time: {processing_time:.2f} seconds")
        print(f"  Classifications per second: {classifications_per_second:.2f}")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        
        # Performance assertions
        assert classified_count == num_comments
        assert classifications_per_second > 5000  # Should be very fast
        assert memory_increase < 100  # Should use minimal additional memory
        
        # Check classification distribution
        stats = self.classifier.get_classification_stats()
        assert stats['total_classified'] == num_comments
        assert stats['user_deleted'] > 0
        assert stats['moderator_removed'] > 0
        assert stats['active'] > 0
        
    @pytest.mark.performance
    def test_metadata_extraction_performance(self):
        """Test metadata extraction performance."""
        # Create dataset for metadata extraction
        num_comments = 15000
        comments = self.create_large_dataset(num_comments)
        
        # Measure extraction performance
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        extracted_count = 0
        training_records = []
        
        for comment in comments:
            # Only extract for deleted/removed comments (more realistic)
            classification = self.classifier.classify_comment(comment)
            
            if classification in ['user_deleted', 'moderator_removed']:
                removal_context = self.classifier.extract_removal_context(comment)
                record = self.extractor.build_training_record(comment, removal_context)
                training_records.append(record)
                
            extracted_count += 1
            
            if extracted_count % 3000 == 0:
                current_memory = self.get_memory_usage()
                memory_increase = current_memory - start_memory
                print(f"Processed {extracted_count} comments, {len(training_records)} records, memory: {memory_increase:.2f} MB")
                
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        # Calculate performance metrics
        processing_time = end_time - start_time
        extractions_per_second = extracted_count / processing_time
        memory_increase = end_memory - start_memory
        
        print(f"Metadata Extraction Performance:")
        print(f"  Comments processed: {extracted_count}")
        print(f"  Training records created: {len(training_records)}")
        print(f"  Processing time: {processing_time:.2f} seconds")
        print(f"  Extractions per second: {extractions_per_second:.2f}")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        
        # Performance assertions
        assert extracted_count == num_comments
        assert len(training_records) > 0
        assert extractions_per_second > 2000  # Should be reasonably fast
        assert memory_increase < 200  # Should use reasonable memory
        
        # Check extraction stats
        stats = self.extractor.get_extraction_stats()
        assert stats['total_extracted'] >= len(training_records)
        assert stats['success_rate'] > 95  # Should have high success rate
        
    @pytest.mark.performance
    def test_parquet_writing_performance(self):
        """Test Parquet writing performance and compression efficiency."""
        # Create training dataset
        num_records = 10000
        training_records = []
        
        for i in range(num_records):
            record = {
                'id': f'record_{i:06d}',
                'comment_text': f'Training comment {i} with some content for compression testing. ' * 3,
                'subreddit': f'subreddit_{i % 50}',
                'timestamp': datetime.now(timezone.utc),
                'removal_type': 'user_deleted' if i % 2 == 0 else 'moderator_removed',
                'target_label': 'voluntary_deletion' if i % 2 == 0 else 'rule_violation',
                'parent_id': f'parent_{i}',
                'thread_id': f'thread_{i // 100}',
                'score': i % 201 - 100,
                'author': f'user_{i}',
                'controversiality': 1 if i % 10 == 0 else 0,
                'gilded': 1 if i % 50 == 0 else 0,
                'comment_length': len(f'Training comment {i} with some content for compression testing. ' * 3),
                'has_parent': True,
                'is_top_level': i % 20 == 0
            }
            training_records.append(record)
            
        # Test different compression algorithms
        compression_algorithms = ['snappy', 'gzip', 'lz4']
        compression_results = {}
        
        for compression in compression_algorithms:
            print(f"Testing {compression} compression...")
            
            output_path = Path(self.temp_dir) / f"performance_test_{compression}.parquet"
            
            # Measure writing performance
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            self.writer.write_dataset(iter(training_records), output_path, compression)
            
            end_time = time.time()
            end_memory = self.get_memory_usage()
            
            # Calculate metrics
            processing_time = end_time - start_time
            records_per_second = num_records / processing_time
            memory_increase = end_memory - start_memory
            
            # File size and compression ratio
            file_size = output_path.stat().st_size
            file_size_mb = file_size / 1024 / 1024
            
            # Estimate original size (rough)
            df = pd.DataFrame(training_records)
            estimated_original_size = df.memory_usage(deep=True).sum()
            compression_ratio = estimated_original_size / file_size
            
            compression_results[compression] = {
                'processing_time': processing_time,
                'records_per_second': records_per_second,
                'memory_increase': memory_increase,
                'file_size_mb': file_size_mb,
                'compression_ratio': compression_ratio
            }
            
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Records per second: {records_per_second:.2f}")
            print(f"  File size: {file_size_mb:.2f} MB")
            print(f"  Compression ratio: {compression_ratio:.2f}x")
            print(f"  Memory increase: {memory_increase:.2f} MB")
            
            # Performance assertions
            assert records_per_second > 1000  # Should write at least 1000 records/sec
            assert compression_ratio > 1.5  # Should achieve reasonable compression
            assert memory_increase < 300  # Should use reasonable memory
            
        # Compare compression algorithms
        print("\nCompression Algorithm Comparison:")
        for alg, results in compression_results.items():
            print(f"  {alg}: {results['compression_ratio']:.2f}x compression, "
                  f"{results['records_per_second']:.0f} rec/sec, "
                  f"{results['file_size_mb']:.2f} MB")
            
        # Verify files can be read back
        for compression in compression_algorithms:
            file_path = Path(self.temp_dir) / f"performance_test_{compression}.parquet"
            df = pd.read_parquet(file_path)
            assert len(df) == num_records
            
    @pytest.mark.performance
    def test_chunked_processing_memory_efficiency(self):
        """Test memory efficiency of chunked processing."""
        # Create very large dataset
        num_comments = 100000
        print(f"Creating dataset with {num_comments} comments...")
        
        comments = self.create_large_dataset(num_comments)
        data_file = self.create_large_data_file(comments)
        
        file_size_mb = data_file.stat().st_size / 1024 / 1024
        print(f"Dataset file size: {file_size_mb:.2f} MB")
        
        # Test different chunk sizes
        chunk_sizes = [1000, 5000, 10000]
        memory_results = {}
        
        for chunk_size in chunk_sizes:
            print(f"Testing chunk size: {chunk_size}")
            
            start_time = time.time()
            start_memory = self.get_memory_usage()
            max_memory = start_memory
            
            processed_count = 0
            training_records = []
            
            # Process in chunks
            parser = RedditParser(chunk_size=chunk_size, strict_validation=False)
            
            for chunk in parser.chunk_processor(data_file):
                chunk_start_memory = self.get_memory_usage()
                
                # Process chunk
                for comment in chunk:
                    classification = self.classifier.classify_comment(comment)
                    
                    if classification in ['user_deleted', 'moderator_removed']:
                        removal_context = self.classifier.extract_removal_context(comment)
                        record = self.extractor.build_training_record(comment, removal_context)
                        training_records.append(record)
                        
                    processed_count += 1
                    
                chunk_end_memory = self.get_memory_usage()
                max_memory = max(max_memory, chunk_end_memory)
                
                if processed_count % 20000 == 0:
                    print(f"  Processed {processed_count} comments, "
                          f"current memory: {chunk_end_memory:.2f} MB, "
                          f"max memory: {max_memory:.2f} MB")
                    
            end_time = time.time()
            end_memory = self.get_memory_usage()
            
            # Calculate metrics
            processing_time = end_time - start_time
            comments_per_second = processed_count / processing_time
            peak_memory_increase = max_memory - start_memory
            final_memory_increase = end_memory - start_memory
            
            memory_results[chunk_size] = {
                'processing_time': processing_time,
                'comments_per_second': comments_per_second,
                'peak_memory_increase': peak_memory_increase,
                'final_memory_increase': final_memory_increase,
                'training_records': len(training_records)
            }
            
            print(f"  Results: {comments_per_second:.0f} comments/sec, "
                  f"peak memory: +{peak_memory_increase:.2f} MB, "
                  f"final memory: +{final_memory_increase:.2f} MB")
            
            # Performance assertions
            assert processed_count == num_comments
            assert comments_per_second > 500  # Should maintain reasonable speed
            assert peak_memory_increase < 1000  # Should not use excessive memory
            
        # Compare chunk sizes
        print("\nChunk Size Comparison:")
        for chunk_size, results in memory_results.items():
            print(f"  {chunk_size}: {results['comments_per_second']:.0f} c/s, "
                  f"peak: +{results['peak_memory_increase']:.0f} MB, "
                  f"records: {results['training_records']}")
            
        # Smaller chunks should use less peak memory
        assert memory_results[1000]['peak_memory_increase'] <= memory_results[10000]['peak_memory_increase']
        
    @pytest.mark.performance
    def test_end_to_end_pipeline_performance(self):
        """Test performance of complete end-to-end pipeline."""
        # Create realistic dataset
        num_comments = 25000
        print(f"Testing end-to-end pipeline with {num_comments} comments...")
        
        comments = self.create_large_dataset(num_comments)
        data_file = self.create_large_data_file(comments)
        
        # Measure complete pipeline performance
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # Step 1: Parse comments
        print("Step 1: Parsing comments...")
        parse_start = time.time()
        parsed_comments = list(self.parser.parse_comments_file(data_file))
        parse_time = time.time() - parse_start
        parse_memory = self.get_memory_usage()
        
        # Step 2: Classify comments
        print("Step 2: Classifying comments...")
        classify_start = time.time()
        classified_comments = []
        for comment in parsed_comments:
            classification = self.classifier.classify_comment(comment)
            removal_context = self.classifier.extract_removal_context(comment)
            
            classified_comment = comment.copy()
            classified_comment['classification'] = classification
            classified_comment['removal_context'] = removal_context
            classified_comments.append(classified_comment)
            
        classify_time = time.time() - classify_start
        classify_memory = self.get_memory_usage()
        
        # Step 3: Extract metadata
        print("Step 3: Extracting metadata...")
        extract_start = time.time()
        training_records = []
        for comment in classified_comments:
            if comment['classification'] in ['user_deleted', 'moderator_removed']:
                record = self.extractor.build_training_record(
                    comment, comment['removal_context']
                )
                training_records.append(record)
                
        extract_time = time.time() - extract_start
        extract_memory = self.get_memory_usage()
        
        # Step 4: Write Parquet files
        print("Step 4: Writing Parquet files...")
        write_start = time.time()
        
        if training_records:
            # Separate by removal type
            user_deleted = [r for r in training_records if r['removal_type'] == 'user_deleted']
            moderator_removed = [r for r in training_records if r['removal_type'] == 'moderator_removed']
            
            output_dir = Path(self.temp_dir) / "performance_output"
            dataset_paths = self.writer.export_training_datasets(
                iter(user_deleted),
                iter(moderator_removed),
                output_dir
            )
            
        write_time = time.time() - write_start
        write_memory = self.get_memory_usage()
        
        # Calculate overall metrics
        total_time = time.time() - start_time
        total_memory_increase = write_memory - start_memory
        overall_throughput = num_comments / total_time
        
        # Print performance breakdown
        print(f"\nEnd-to-End Pipeline Performance:")
        print(f"  Total comments: {num_comments}")
        print(f"  Training records: {len(training_records)}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Overall throughput: {overall_throughput:.2f} comments/second")
        print(f"  Total memory increase: {total_memory_increase:.2f} MB")
        print(f"\nStep Breakdown:")
        print(f"  Parse: {parse_time:.2f}s ({parse_memory - start_memory:.1f} MB)")
        print(f"  Classify: {classify_time:.2f}s ({classify_memory - parse_memory:.1f} MB)")
        print(f"  Extract: {extract_time:.2f}s ({extract_memory - classify_memory:.1f} MB)")
        print(f"  Write: {write_time:.2f}s ({write_memory - extract_memory:.1f} MB)")
        
        # Performance assertions
        assert len(parsed_comments) == num_comments
        assert len(training_records) > 0
        assert overall_throughput > 200  # Should process at least 200 comments/sec overall
        assert total_memory_increase < 800  # Should use less than 800MB total
        
        # Verify output files
        if 'user_deleted' in dataset_paths:
            user_df = pd.read_parquet(dataset_paths['user_deleted'])
            assert len(user_df) == len(user_deleted)
            
        if 'moderator_removed' in dataset_paths:
            mod_df = pd.read_parquet(dataset_paths['moderator_removed'])
            assert len(mod_df) == len(moderator_removed)
            
        # Check compression efficiency
        writer_stats = self.writer.get_compression_stats()
        if writer_stats['files_created'] > 0:
            avg_compression = writer_stats['average_compression_ratio']
            print(f"  Average compression ratio: {avg_compression:.2f}x")
            assert avg_compression > 2.0  # Should achieve good compression
            
    @pytest.mark.performance
    def test_memory_stress_large_dataset(self):
        """Stress test with very large dataset to check memory limits."""
        # Create stress test dataset
        num_comments = 200000  # Large dataset for stress testing
        print(f"Memory stress test with {num_comments} comments...")
        
        # Monitor memory during dataset creation
        creation_start_memory = self.get_memory_usage()
        comments = self.create_large_dataset(num_comments)
        creation_end_memory = self.get_memory_usage()
        
        print(f"Dataset creation memory: {creation_end_memory - creation_start_memory:.2f} MB")
        
        data_file = self.create_large_data_file(comments)
        file_size_mb = data_file.stat().st_size / 1024 / 1024
        print(f"Dataset file size: {file_size_mb:.2f} MB")
        
        # Clear comments from memory
        del comments
        
        # Process with small chunks to manage memory
        chunk_size = 2000
        start_memory = self.get_memory_usage()
        max_memory = start_memory
        
        processed_count = 0
        total_training_records = 0
        
        # Process in chunks with memory monitoring
        parser = RedditParser(chunk_size=chunk_size, strict_validation=False)
        
        for chunk_num, chunk in enumerate(parser.chunk_processor(data_file)):
            chunk_start_memory = self.get_memory_usage()
            
            # Process chunk
            chunk_training_records = 0
            for comment in chunk:
                classification = self.classifier.classify_comment(comment)
                
                if classification in ['user_deleted', 'moderator_removed']:
                    removal_context = self.classifier.extract_removal_context(comment)
                    # Don't store records to save memory, just count
                    chunk_training_records += 1
                    
                processed_count += 1
                
            total_training_records += chunk_training_records
            chunk_end_memory = self.get_memory_usage()
            max_memory = max(max_memory, chunk_end_memory)
            
            # Log progress every 50 chunks
            if chunk_num % 50 == 0:
                memory_increase = chunk_end_memory - start_memory
                print(f"  Chunk {chunk_num}: processed {processed_count} comments, "
                      f"memory: {chunk_end_memory:.1f} MB (+{memory_increase:.1f})")
                
        end_memory = self.get_memory_usage()
        peak_memory_increase = max_memory - start_memory
        final_memory_increase = end_memory - start_memory
        
        print(f"\nMemory Stress Test Results:")
        print(f"  Comments processed: {processed_count}")
        print(f"  Training records identified: {total_training_records}")
        print(f"  Peak memory increase: {peak_memory_increase:.2f} MB")
        print(f"  Final memory increase: {final_memory_increase:.2f} MB")
        print(f"  Memory efficiency: {processed_count / peak_memory_increase:.0f} comments/MB")
        
        # Stress test assertions
        assert processed_count == num_comments
        assert total_training_records > 0
        assert peak_memory_increase < 1500  # Should handle large datasets efficiently
        assert final_memory_increase < 500  # Should not leak significant memory


if __name__ == "__main__":
    # Run performance tests with verbose output
    pytest.main([__file__, "-v", "-s", "-m", "performance"])