#!/usr/bin/env python3
"""
Test runner script for AI Lego Bricks test suite.

This script provides convenient commands for running different types of tests.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def run_unit_tests(verbose=False):
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    if verbose:
        cmd.append("-v")
    return run_command(cmd, "Running unit tests")


def run_integration_tests(verbose=False):
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]
    if verbose:
        cmd.append("-v")
    return run_command(cmd, "Running integration tests")


def run_performance_tests(verbose=False):
    """Run performance tests."""
    cmd = ["python", "-m", "pytest", "tests/performance/", "-m", "performance"]
    if verbose:
        cmd.append("-v")
    return run_command(cmd, "Running performance tests")


def run_all_tests(verbose=False, skip_integration=False, skip_performance=False):
    """Run all tests."""
    success = True
    
    # Always run unit tests
    success &= run_unit_tests(verbose)
    
    # Optionally run integration tests
    if not skip_integration:
        success &= run_integration_tests(verbose)
    
    # Optionally run performance tests
    if not skip_performance:
        success &= run_performance_tests(verbose)
    
    return success


def run_coverage_report():
    """Generate coverage report."""
    cmd = ["python", "-m", "pytest", "--cov-report=html", "--cov-report=term"]
    return run_command(cmd, "Generating coverage report")


def run_type_checking():
    """Run mypy type checking."""
    cmd = ["python", "-m", "mypy", "ailego/", "agent_orchestration/", "llm/", "memory/", "chat/", "prompt/", "tts/"]
    return run_command(cmd, "Running type checking")


def run_linting():
    """Run code linting."""
    success = True
    
    # Black formatting check
    cmd = ["python", "-m", "black", "--check", "."]
    success &= run_command(cmd, "Checking code formatting with Black")
    
    # Flake8 linting
    cmd = ["python", "-m", "flake8", "ailego/", "agent_orchestration/", "llm/", "memory/", "chat/", "prompt/", "tts/"]
    success &= run_command(cmd, "Running flake8 linting")
    
    return success


def run_security_checks():
    """Run security checks."""
    # Bandit security linting
    cmd = ["python", "-m", "bandit", "-r", "ailego/", "agent_orchestration/", "llm/", "memory/", "chat/", "prompt/", "tts/"]
    return run_command(cmd, "Running Bandit security checks")


def run_dependency_checks():
    """Run dependency vulnerability checks."""
    cmd = ["python", "-m", "pip", "check"]
    return run_command(cmd, "Checking dependencies")


def run_quick_tests():
    """Run a quick subset of tests for development."""
    cmd = ["python", "-m", "pytest", "tests/unit/", "-x", "--ff"]
    return run_command(cmd, "Running quick development tests")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Lego Bricks Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Test commands")
    
    # Unit tests
    unit_parser = subparsers.add_parser("unit", help="Run unit tests")
    
    # Integration tests
    integration_parser = subparsers.add_parser("integration", help="Run integration tests")
    
    # Performance tests
    performance_parser = subparsers.add_parser("performance", help="Run performance tests")
    
    # All tests
    all_parser = subparsers.add_parser("all", help="Run all tests")
    all_parser.add_argument("--skip-integration", action="store_true", help="Skip integration tests")
    all_parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests")
    
    # Coverage
    coverage_parser = subparsers.add_parser("coverage", help="Generate coverage report")
    
    # Quality checks
    quality_parser = subparsers.add_parser("quality", help="Run all quality checks")
    
    # Quick tests
    quick_parser = subparsers.add_parser("quick", help="Run quick development tests")
    
    # CI mode
    ci_parser = subparsers.add_parser("ci", help="Run CI pipeline")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    success = True
    
    if args.command == "unit":
        success = run_unit_tests(args.verbose)
    
    elif args.command == "integration":
        success = run_integration_tests(args.verbose)
    
    elif args.command == "performance":
        success = run_performance_tests(args.verbose)
    
    elif args.command == "all":
        success = run_all_tests(
            args.verbose, 
            args.skip_integration, 
            args.skip_performance
        )
    
    elif args.command == "coverage":
        success = run_coverage_report()
    
    elif args.command == "quality":
        print("🔍 Running quality checks...")
        success &= run_linting()
        success &= run_type_checking()
        success &= run_security_checks()
        success &= run_dependency_checks()
    
    elif args.command == "quick":
        success = run_quick_tests()
    
    elif args.command == "ci":
        print("🚀 Running CI pipeline...")
        success &= run_linting()
        success &= run_type_checking()
        success &= run_unit_tests(verbose=True)
        success &= run_integration_tests(verbose=True)
        success &= run_coverage_report()
        success &= run_security_checks()
    
    if success:
        print("\n✅ All tests/checks passed!")
        return 0
    else:
        print("\n❌ Some tests/checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())