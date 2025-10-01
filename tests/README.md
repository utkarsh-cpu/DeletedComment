# Testing Suite for Reddit Deleted Comments Dataset

This directory contains a comprehensive testing suite for the Reddit deleted comments dataset processing pipeline.

## Test Structure

### Unit Tests
- `test_data_downloader.py` - Tests for data downloading and extraction functionality
- `test_reddit_parser.py` - Tests for Reddit JSON data parsing with error handling
- `test_comment_classifier.py` - Tests for comment classification (deleted/removed/active)
- `test_metadata_extractor.py` - Tests for metadata extraction and training record creation
- `test_parquet_writer.py` - Tests for Parquet file creation and compression
- `test_drive_uploader.py` - Tests for Google Drive API integration (mocked)

### Integration Tests
- `test_integration.py` - End-to-end pipeline tests with sample Reddit data

### Performance Tests
- `test_performance.py` - Memory usage, processing speed, and throughput tests
- `test_parquet_integrity.py` - Parquet file format compliance and compression efficiency tests

### Test Configuration
- `conftest.py` - Pytest configuration, fixtures, and test markers
- `run_tests.py` - Test runner script with various options

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements-test.txt
```

### Quick Start

Run all tests:
```bash
python tests/run_tests.py --all
```

Run only unit tests:
```bash
python tests/run_tests.py --unit
```

Run with coverage:
```bash
python tests/run_tests.py --unit --coverage
```

### Test Categories

#### Unit Tests (Fast)
```bash
python tests/run_tests.py --unit
# or
pytest tests/test_*.py -m "not performance and not integration"
```

#### Integration Tests
```bash
python tests/run_tests.py --integration
# or
pytest tests/test_integration.py
```

#### Performance Tests
```bash
python tests/run_tests.py --performance
# or
pytest tests/test_performance.py -m performance
```

#### Parquet Integrity Tests
```bash
python tests/run_tests.py --integrity
# or
pytest tests/test_parquet_integrity.py
```

### Advanced Options

Run tests in parallel:
```bash
python tests/run_tests.py --unit --parallel 4
```

Run fast tests only (exclude slow tests):
```bash
python tests/run_tests.py --all --fast
```

Verbose output:
```bash
python tests/run_tests.py --unit --verbose
```

## Test Markers

The test suite uses pytest markers to categorize tests:

- `@pytest.mark.performance` - Performance and benchmarking tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests that take longer to run

## Test Data

Tests use various types of sample data:

### Sample Reddit Comments
- Active comments with normal content
- User-deleted comments (`[deleted]` body or `[deleted]` author)
- Moderator-removed comments (`[removed]` body)
- Comments with missing or malformed data

### Sample Reddit Submissions
- Thread context for comment hierarchy testing
- Various subreddits and metadata

### Performance Test Data
- Large datasets (10K-200K comments) for stress testing
- Repetitive data for compression testing
- Diverse data for realistic performance measurement

## Test Coverage

The test suite covers:

### Functionality Testing
- ✅ Data downloading and extraction
- ✅ JSON parsing with error handling
- ✅ Comment classification accuracy
- ✅ Metadata extraction completeness
- ✅ Parquet file creation and compression
- ✅ Google Drive API integration (mocked)

### Integration Testing
- ✅ End-to-end pipeline processing
- ✅ Component interaction and data flow
- ✅ Error handling across components
- ✅ Context resolution (parent comments, threads)
- ✅ Data quality assessment

### Performance Testing
- ✅ Memory usage efficiency
- ✅ Processing throughput (comments/second)
- ✅ Compression ratios
- ✅ Large dataset handling
- ✅ Chunked processing efficiency

### Data Integrity Testing
- ✅ Parquet format compliance
- ✅ Schema consistency
- ✅ Compression algorithm comparison
- ✅ Pandas/Dask compatibility
- ✅ Null value handling

## Performance Benchmarks

The test suite includes performance benchmarks:

### Processing Speed
- **Parsing**: >1,000 comments/second
- **Classification**: >5,000 comments/second  
- **Metadata Extraction**: >2,000 comments/second
- **Parquet Writing**: >1,000 records/second

### Memory Efficiency
- **Large Dataset Processing**: <500MB for 50K comments
- **Chunked Processing**: <200MB peak memory increase
- **End-to-End Pipeline**: <800MB total memory usage

### Compression Ratios
- **Repetitive Data**: >5.0x compression
- **Diverse Data**: >2.0x compression
- **Sparse Data**: >1.5x compression

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Fast tests for quick feedback
python tests/run_tests.py --fast --coverage

# Full test suite for comprehensive validation
python tests/run_tests.py --all --coverage
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the `src` directory is in your Python path
2. **Missing Dependencies**: Install test requirements with `pip install -r requirements-test.txt`
3. **Google API Tests**: These use mocked responses and don't require actual credentials
4. **Performance Test Failures**: May indicate system resource constraints

### Test Data Cleanup

Tests automatically clean up temporary files, but you can manually clean:

```bash
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
rm -rf .pytest_cache/
rm -rf htmlcov/
```

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Use appropriate pytest markers (`@pytest.mark.performance`, etc.)
3. Include docstrings explaining what the test validates
4. Add sample data fixtures in `conftest.py` if needed
5. Update this README if adding new test categories

## Test Requirements Validation

The test suite validates all requirements from the specification:

- **Requirements 1.1-1.4**: Data downloading and integrity verification
- **Requirements 2.1-2.5**: Comment classification and separation
- **Requirements 3.1-3.5**: Metadata extraction and training data formatting
- **Requirements 4.1-4.6**: Parquet file creation and compression
- **Requirements 5.1-5.4**: Progress monitoring and error handling
- **Requirements 6.1-6.7**: Google Drive integration and storage management