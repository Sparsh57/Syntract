#!/usr/bin/env python3
"""
Pytest integration for the comprehensive test runner.

This allows running the comprehensive tests via pytest:
    pytest tests/test_comprehensive_integration.py -v
    pytest tests/test_comprehensive_integration.py::test_comprehensive_runner -v
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path so we can import the comprehensive test runner
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_comprehensive_runner():
    """
    Run the comprehensive test suite via pytest.
    
    This test will pass if all comprehensive tests pass, and fail if any fail.
    It provides integration between pytest and our custom comprehensive test runner.
    """
    try:
        # Import the comprehensive test runner
        import run_comprehensive_tests
        
        # Run the comprehensive tests
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE TESTS VIA PYTEST")
        print("="*60)
        
        # Call the main function and capture the exit code
        exit_code = run_comprehensive_tests.main()
        
        # Assert that all tests passed (exit code 0)
        assert exit_code == 0, "Comprehensive tests failed - see output above for details"
        
        print("\n✅ All comprehensive tests passed!")
        
    except ImportError as e:
        pytest.fail(f"Failed to import comprehensive test runner: {e}")
    except Exception as e:
        pytest.fail(f"Comprehensive test runner failed with error: {e}")


def test_comprehensive_runner_import():
    """
    Test that the comprehensive test runner can be imported successfully.
    """
    try:
        import run_comprehensive_tests
        assert hasattr(run_comprehensive_tests, 'main'), "main function not found in comprehensive test runner"
        assert hasattr(run_comprehensive_tests, 'TestResults'), "TestResults class not found in comprehensive test runner"
        
    except ImportError as e:
        pytest.fail(f"Failed to import comprehensive test runner: {e}")


@pytest.mark.slow
def test_comprehensive_runner_detailed():
    """
    Run comprehensive tests with detailed reporting.
    
    This is marked as 'slow' so it can be skipped with: pytest -m "not slow"
    """
    try:
        import run_comprehensive_tests
        
        print("\n" + "="*60)
        print("RUNNING DETAILED COMPREHENSIVE TESTS VIA PYTEST")
        print("="*60)
        
        # Run with detailed output
        exit_code = run_comprehensive_tests.main()
        
        if exit_code != 0:
            pytest.fail("Comprehensive tests failed - check the detailed output above")
            
        print("\n✅ All detailed comprehensive tests passed!")
        
    except Exception as e:
        pytest.fail(f"Detailed comprehensive test runner failed: {e}")


if __name__ == "__main__":
    # Allow running this file directly
    pytest.main([__file__, "-v"]) 