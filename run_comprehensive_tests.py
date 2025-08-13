#!/usr/bin/env python3
"""
Comprehensive Test Runner for SynTract Project

This script systematically tests all modules in the synthesis/ and syntract_viewer/ folders.
It runs tests step by step and provides detailed feedback on the testing progress.

Usage:
    python run_comprehensive_tests.py
"""

import os
import sys
import traceback
import importlib
import numpy as np
import nibabel as nib
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TestResults:
    """Class to track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
        
    def add_pass(self):
        self.passed += 1
        
    def add_fail(self, error_msg):
        self.failed += 1
        self.errors.append(error_msg)
        
    def add_skip(self, reason):
        self.skipped += 1
        self.errors.append(f"SKIPPED: {reason}")
        
    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*50}")
        print(f"TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Skipped: {self.skipped}")
        
        if self.errors:
            print(f"\nErrors/Issues:")
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")
        
        return self.failed == 0

def setup_test_environment():
    """Set up the test environment"""
    print("Setting up test environment...")
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp(prefix="SynTract_test_")
    print(f"Created temporary directory: {temp_dir}")
    
    return temp_dir

def cleanup_test_environment(temp_dir):
    """Clean up test environment"""
    print(f"Cleaning up temporary directory: {temp_dir}")
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print("Cleanup completed successfully")
    except Exception as e:
        print(f"Warning: Could not clean up {temp_dir}: {e}")

def create_test_nifti_file(temp_dir: str, shape=(50, 50, 50)) -> str:
    """Create a test NIfTI file"""
    data = np.random.random(shape).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    
    filepath = os.path.join(temp_dir, "test_brain.nii.gz")
    nib.save(img, filepath)
    return filepath

def create_test_trk_file(temp_dir: str, n_streamlines=10) -> str:
    """Create a test TRK file"""
    try:
        from nibabel.streamlines import Tractogram
        from nibabel.streamlines.trk import TrkFile
        
        # Create simple test streamlines
        streamlines = []
        for i in range(n_streamlines):
            # Create a simple curved streamline
            t = np.linspace(0, 2*np.pi, 20)
            x = np.cos(t) + i
            y = np.sin(t) + i
            z = t * 0.1 + i
            streamline = np.column_stack([x, y, z]).astype(np.float32)
            streamlines.append(streamline)
        
        tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        filepath = os.path.join(temp_dir, "test_streamlines.trk")
        
        TrkFile(tractogram).save(filepath)
        return filepath
    except ImportError as e:
        print(f"Could not create TRK file: {e}")
        return None

def test_module_structure():
    """Test 1: Module Structure Tests"""
    print("\n" + "="*60)
    print("STEP 1: MODULE STRUCTURE TESTS")
    print("="*60)
    
    results = TestResults()
    
    # Test synthesis module structure
    synthesis_modules = [
        'synthesis.__init__',
        'synthesis.ants_transform', 
        'synthesis.compare_interpolation',
        'synthesis.densify',
        'synthesis.main',
        'synthesis.nifti_preprocessing',
        'synthesis.streamline_processing',
        'synthesis.transform',
        'synthesis.visualize'
    ]
    
    print("\nTesting synthesis module imports...")
    for module_name in synthesis_modules:
        try:
            module = importlib.import_module(module_name)
            print(f"✓ {module_name}")
            results.add_pass()
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
            results.add_fail(f"Import error in {module_name}: {e}")
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            results.add_fail(f"Error in {module_name}: {e}")
    
    # Test syntract_viewer module structure  
    syntract_modules = [
        'syntract_viewer.__init__',
        'syntract_viewer.background_enhancement',
        'syntract_viewer.contrast',
        'syntract_viewer.core',
        'syntract_viewer.cornucopia_augmentation',
        'syntract_viewer.effects',
        'syntract_viewer.generate_fiber_examples',
        'syntract_viewer.generation', 
        'syntract_viewer.improved_cornucopia',
        'syntract_viewer.masking',
        'syntract_viewer.utils'
    ]
    
    print("\nTesting syntract_viewer module imports...")
    for module_name in syntract_modules:
        try:
            module = importlib.import_module(module_name)
            print(f"✓ {module_name}")
            results.add_pass()
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
            results.add_fail(f"Import error in {module_name}: {e}")
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            results.add_fail(f"Error in {module_name}: {e}")
    
    return results

def test_basic_functionality(temp_dir: str):
    """Test 2: Basic Functionality Tests"""
    print("\n" + "="*60)
    print("STEP 2: BASIC FUNCTIONALITY TESTS")
    print("="*60)
    
    results = TestResults()
    
    # Test synthesis.transform
    print("\nTesting synthesis.transform...")
    try:
        from synthesis.transform import build_new_affine
        
        old_affine = np.eye(4)
        old_affine[:3, :3] = np.diag([2.0, 2.0, 2.0])
        old_shape = (100, 100, 100)
        new_voxel_size = 1.0
        new_shape = (200, 200, 200)
        
        new_affine = build_new_affine(old_affine, old_shape, new_voxel_size, new_shape, use_gpu=False)
        assert new_affine.shape == (4, 4)
        print("✓ build_new_affine basic test")
        results.add_pass()
    except Exception as e:
        print(f"✗ build_new_affine: {e}")
        results.add_fail(f"build_new_affine error: {e}")
    
    # Test synthesis.densify
    print("\nTesting synthesis.densify...")
    try:
        from synthesis.densify import linear_interpolate, densify_streamline_subvoxel
        
        # Test linear interpolation
        p0 = np.array([0, 0, 0])
        p1 = np.array([2, 2, 2])
        result = linear_interpolate(p0, p1, 0.5)
        expected = np.array([1, 1, 1])
        np.testing.assert_allclose(result, expected)
        print("✓ linear_interpolate test")
        results.add_pass()
        
        # Test streamline densification
        streamline = np.array([
            [0, 0, 0],
            [1, 1, 0],
            [2, 0, 0]
        ], dtype=np.float32)
        
        densified = densify_streamline_subvoxel(streamline, step_size=0.5, use_gpu=False, interp_method='linear')
        assert len(densified) >= len(streamline)
        print("✓ densify_streamline_subvoxel test")
        results.add_pass()
        
    except Exception as e:
        print(f"✗ densify functions: {e}")
        results.add_fail(f"densify functions error: {e}")
    
    # Test syntract_viewer.utils
    print("\nTesting syntract_viewer.utils...")
    try:
        from syntract_viewer.utils import select_random_streamlines, densify_streamline
        
        # Create test streamlines
        streamlines = [
            np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32),
            np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]], dtype=np.float32),
            np.array([[0, 1, 0], [1, 1, 1], [2, 1, 2]], dtype=np.float32)
        ]
        
        # Test random selection
        selected = select_random_streamlines(streamlines, percentage=50.0, random_state=42)
        assert isinstance(selected, list)
        assert len(selected) <= len(streamlines)
        print("✓ select_random_streamlines test")
        results.add_pass()
        
        # Test densification
        simple_streamline = np.array([
            [0, 0, 0],
            [5, 0, 0],
            [10, 0, 0]
        ], dtype=np.float32)
        
        densified = densify_streamline(simple_streamline, step=1.0)
        assert len(densified) >= len(simple_streamline)
        print("✓ densify_streamline test")
        results.add_pass()
        
    except Exception as e:
        print(f"✗ syntract_viewer.utils: {e}")
        results.add_fail(f"syntract_viewer.utils error: {e}")
    
    return results

def test_file_operations(temp_dir: str):
    """Test 3: File Operations Tests"""
    print("\n" + "="*60)
    print("STEP 3: FILE OPERATIONS TESTS")
    print("="*60)
    
    results = TestResults()
    
    # Create test files
    nifti_file = create_test_nifti_file(temp_dir)
    trk_file = create_test_trk_file(temp_dir)
    
    if not nifti_file:
        results.add_skip("Could not create test NIfTI file")
        return results
        
    print(f"Created test NIfTI file: {nifti_file}")
    
    # Test NIfTI loading
    try:
        from synthesis.nifti_preprocessing import estimate_memory_usage
        
        memory_gb = estimate_memory_usage((50, 50, 50), np.float32)
        assert memory_gb > 0
        print("✓ estimate_memory_usage test")
        results.add_pass()
        
    except Exception as e:
        print(f"✗ NIfTI preprocessing: {e}")
        results.add_fail(f"NIfTI preprocessing error: {e}")
    
    # Test streamline processing if TRK file was created
    if trk_file:
        print(f"Created test TRK file: {trk_file}")
        try:
            from synthesis.streamline_processing import clip_streamline_to_fov
            
            # Test streamline clipping
            streamline = np.array([
                [-1, -1, -1],  # Outside FOV
                [1, 1, 1],     # Inside FOV
                [2, 2, 2],     # Inside FOV
                [100, 100, 100]  # Outside FOV
            ], dtype=np.float32)
            
            new_shape = (50, 50, 50)
            clipped = clip_streamline_to_fov(streamline, new_shape, use_gpu=False)
            print("✓ clip_streamline_to_fov test")
            results.add_pass()
            
        except Exception as e:
            print(f"✗ streamline processing: {e}")
            results.add_fail(f"streamline processing error: {e}")
    else:
        results.add_skip("Could not create test TRK file")
    
    return results

def test_visualization_functions(temp_dir: str):
    """Test 4: Visualization Functions Tests"""
    print("\n" + "="*60)
    print("STEP 4: VISUALIZATION FUNCTIONS TESTS")
    print("="*60)
    
    results = TestResults()
    
    # Test matplotlib backend (headless for CI)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Test synthesis.visualize functions
        try:
            from synthesis.visualize import overlay_streamlines_on_blockface_coronal, visualize_trk_with_nifti
            print("✓ synthesis.visualize imports successful")
            results.add_pass()
        except Exception as e:
            print(f"✗ synthesis.visualize imports: {e}")
            results.add_fail(f"synthesis.visualize import error: {e}")
        
        # Test syntract_viewer visualization
        try:
            from syntract_viewer.core import visualize_nifti_with_trk
            print("✓ syntract_viewer.core imports successful")
            results.add_pass()
        except Exception as e:
            print(f"✗ syntract_viewer.core imports: {e}")
            results.add_fail(f"syntract_viewer.core import error: {e}")
            
        # Test utils functions
        try:
            from syntract_viewer.utils import get_colormap, generate_tract_color_variation
            
            # Test colormap generation
            cmap = get_colormap(color_scheme='bw')
            assert cmap is not None
            print("✓ get_colormap test")
            results.add_pass()
            
            # Test color variation
            base_color = (1.0, 0.8, 0.1)
            varied_color = generate_tract_color_variation(base_color, variation=0.2, random_state=42)
            assert isinstance(varied_color, tuple)
            assert len(varied_color) == 3
            print("✓ generate_tract_color_variation test")
            results.add_pass()
            
        except Exception as e:
            print(f"✗ visualization utils: {e}")
            results.add_fail(f"visualization utils error: {e}")
        
    except Exception as e:
        print(f"✗ matplotlib setup: {e}")
        results.add_fail(f"matplotlib setup error: {e}")
    
    return results

def test_edge_cases():
    """Test 5: Edge Cases and Error Handling"""
    print("\n" + "="*60)
    print("STEP 5: EDGE CASES AND ERROR HANDLING TESTS")
    print("="*60)
    
    results = TestResults()
    
    # Test empty inputs
    try:
        from syntract_viewer.utils import select_random_streamlines
        
        # Test with empty streamline list
        empty_result = select_random_streamlines([], percentage=50.0, random_state=42)
        assert len(empty_result) == 0
        print("✓ Empty streamlines handling")
        results.add_pass()
        
    except Exception as e:
        print(f"✗ Empty input handling: {e}")
        results.add_fail(f"Empty input handling error: {e}")
    
    # Test invalid parameters
    try:
        from synthesis.densify import linear_interpolate
        
        # Test with extreme values
        p0 = np.array([0, 0, 0])
        p1 = np.array([1e6, 1e6, 1e6])
        result = linear_interpolate(p0, p1, 0.5)
        assert not np.any(np.isnan(result))
        print("✓ Extreme value handling")
        results.add_pass()
        
    except Exception as e:
        print(f"✗ Extreme value handling: {e}")
        results.add_fail(f"Extreme value handling error: {e}")
    
    # Test GPU/CPU fallback
    try:
        from synthesis.densify import densify_streamline_subvoxel
        
        streamline = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2]
        ], dtype=np.float32)
        
        # Test with GPU disabled
        result_cpu = densify_streamline_subvoxel(streamline, step_size=0.5, use_gpu=False)
        
        # Test with GPU enabled (should fallback to CPU if no GPU)
        result_gpu_fallback = densify_streamline_subvoxel(streamline, step_size=0.5, use_gpu=True)
        
        print("✓ GPU/CPU fallback handling")
        results.add_pass()
        
    except Exception as e:
        print(f"✗ GPU/CPU fallback: {e}")
        results.add_fail(f"GPU/CPU fallback error: {e}")
    
    return results

def test_integration():
    """Test 6: Integration Tests"""
    print("\n" + "="*60)
    print("STEP 6: INTEGRATION TESTS")
    print("="*60)
    
    results = TestResults()
    
    # Test module interaction
    try:
        from synthesis.transform import build_new_affine
        from synthesis.densify import densify_streamline_subvoxel
        from syntract_viewer.utils import select_random_streamlines
        
        # Create test data
        streamlines = [
            np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32),
            np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]], dtype=np.float32)
        ]
        
        # Test pipeline: select -> densify -> transform
        selected = select_random_streamlines(streamlines, percentage=100.0)
        densified = [densify_streamline_subvoxel(s, step_size=0.5, use_gpu=False) for s in selected]
        
        old_affine = np.eye(4)
        new_affine = build_new_affine(old_affine, (100, 100, 100), 1.0, (200, 200, 200), use_gpu=False)
        
        print("✓ Module integration test")
        results.add_pass()
        
    except Exception as e:
        print(f"✗ Module integration: {e}")
        results.add_fail(f"Module integration error: {e}")
    
    # Test cross-module compatibility
    try:
        from synthesis.densify import calculate_streamline_metrics
        
        streamlines = [
            np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32),
            np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]], dtype=np.float32)
        ]
        
        metrics = calculate_streamline_metrics(streamlines)
        assert isinstance(metrics, dict)
        assert 'mean_length' in metrics
        print("✓ Cross-module compatibility test")
        results.add_pass()
        
    except Exception as e:
        print(f"✗ Cross-module compatibility: {e}")
        results.add_fail(f"Cross-module compatibility error: {e}")
    
    return results

def main():
    """Main test runner"""
    print("="*60)
    print("COMPREHENSIVE TEST RUNNER FOR SynTract")
    print("="*60)
    print("Testing all modules in synthesis/ and syntract_viewer/ folders")
    print("This will run step-by-step comprehensive tests...")
    
    # Setup
    temp_dir = setup_test_environment()
    
    try:
        # Run all test steps
        all_results = TestResults()
        
        # Step 1: Module Structure
        step1_results = test_module_structure()
        all_results.passed += step1_results.passed
        all_results.failed += step1_results.failed
        all_results.skipped += step1_results.skipped
        all_results.errors.extend(step1_results.errors)
        
        # Step 2: Basic Functionality
        step2_results = test_basic_functionality(temp_dir)
        all_results.passed += step2_results.passed
        all_results.failed += step2_results.failed
        all_results.skipped += step2_results.skipped
        all_results.errors.extend(step2_results.errors)
        
        # Step 3: File Operations
        step3_results = test_file_operations(temp_dir)
        all_results.passed += step3_results.passed
        all_results.failed += step3_results.failed
        all_results.skipped += step3_results.skipped
        all_results.errors.extend(step3_results.errors)
        
        # Step 4: Visualization
        step4_results = test_visualization_functions(temp_dir)
        all_results.passed += step4_results.passed
        all_results.failed += step4_results.failed
        all_results.skipped += step4_results.skipped
        all_results.errors.extend(step4_results.errors)
        
        # Step 5: Edge Cases
        step5_results = test_edge_cases()
        all_results.passed += step5_results.passed
        all_results.failed += step5_results.failed
        all_results.skipped += step5_results.skipped
        all_results.errors.extend(step5_results.errors)
        
        # Step 6: Integration
        step6_results = test_integration()
        all_results.passed += step6_results.passed
        all_results.failed += step6_results.failed
        all_results.skipped += step6_results.skipped
        all_results.errors.extend(step6_results.errors)
        
        # Final summary
        success = all_results.summary()
        
        if success:
            print(f"\n🎉 ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
            return 0
        else:
            print(f"\n❌ SOME TESTS FAILED. Please review the errors above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        cleanup_test_environment(temp_dir)

if __name__ == "__main__":
    sys.exit(main()) 