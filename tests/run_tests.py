#!/usr/bin/env python3
"""
Test runner script for the deleted comment dataset project.
Provides convenient commands to run different test suites.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for the deleted comment dataset project")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--integrity", action="store_true", help="Run Parquet integrity tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only (exclude slow tests)")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, help="Run tests in parallel (requires pytest-xdist)")
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        base_cmd.append("-v")
    else:
        base_cmd.append("-q")
    
    # Add parallel execution
    if args.parallel:
        base_cmd.extend(["-n", str(args.parallel)])
    
    # Add coverage
    if args.coverage:
        base_cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Determine which tests to run
    test_suites = []
    
    if args.all:
        test_suites = [
            ("Unit Tests", ["tests/test_*.py", "-m", "not performance and not integration"]),
            ("Integration Tests", ["tests/test_integration.py"]),
            ("Parquet Integrity Tests", ["tests/test_parquet_integrity.py"]),
            ("Performance Tests", ["tests/test_performance.py", "-m", "performance"])
        ]
    elif args.unit:
        test_suites = [
            ("Unit Tests", ["tests/test_*.py", "-m", "not performance and not integration"])
        ]
    elif args.integration:
        test_suites = [
            ("Integration Tests", ["tests/test_integration.py"])
        ]
    elif args.performance:
        test_suites = [
            ("Performance Tests", ["tests/test_performance.py", "-m", "performance"])
        ]
    elif args.integrity:
        test_suites = [
            ("Parquet Integrity Tests", ["tests/test_parquet_integrity.py"])
        ]
    else:
        # Default: run unit tests and integration tests
        test_suites = [
            ("Unit Tests", ["tests/test_*.py", "-m", "not performance and not integration"]),
            ("Integration Tests", ["tests/test_integration.py"])
        ]
    
    # Add fast filter if requested
    if args.fast:
        for i, (name, cmd_args) in enumerate(test_suites):
            if "-m" in cmd_args:
                # Modify existing marker expression
                marker_idx = cmd_args.index("-m") + 1
                cmd_args[marker_idx] += " and not slow"
            else:
                # Add new marker expression
                cmd_args.extend(["-m", "not slow"])
    
    # Run test suites
    all_passed = True
    results = []
    
    for suite_name, test_args in test_suites:
        cmd = base_cmd + test_args
        success = run_command(cmd, suite_name)
        results.append((suite_name, success))
        if not success:
            all_passed = False
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for suite_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{suite_name}: {status}")
    
    if all_passed:
        print(f"\n🎉 All test suites passed!")
        return 0
    else:
        print(f"\n💥 Some test suites failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())