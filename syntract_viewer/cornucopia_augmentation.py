#!/usr/bin/env python
"""
Cornucopia Integration for Medical Imaging Augmentations

This module integrates Yael's Cornucopia package (https://cornucopia.readthedocs.io/)
for advanced medical imaging augmentations with our fiber tract visualization system.

Cornucopia provides GPU-accelerated, medical imaging-specific transforms including:
- Rician noise (MRI-characteristic)
- Bias field artifacts  
- Gaussian mixture noise
- Anatomy-preserving spatial transforms
- Medical imaging-specific intensity corrections

Key Features:
- GPU acceleration with automatic device detection
- Medical imaging-specific augmentations
- Graceful fallback when Cornucopia unavailable
- Multiple preset configurations
- Compatible with existing pipeline

Installation:
    pip install cornucopia

Authors: LINC Team, with Cornucopia integration
License: MIT
"""

import warnings
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import random

# Import torch first
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import Cornucopia with the correct package name
try:
    import cornucopia
    from cornucopia import (
        RandomGaussianMixtureTransform,
        RandomGaussianNoiseTransform,
        RandomGammaTransform,
        RandomAffineTransform,
        IntensityTransform,
        SequentialTransform
    )
    CORNUCOPIA_AVAILABLE = True
    print("✅ Cornucopia successfully imported!")
except ImportError as e:
    print(f"⚠️  Cornucopia not available: {e}")
    print("   Install with: pip install cornucopia")
    print("   Falling back to basic augmentations only.")
    warnings.warn(
        "Cornucopia not available. Install with: pip install cornucopia\n"
        "Falling back to basic augmentations only.",
        UserWarning
    )
    CORNUCOPIA_AVAILABLE = False

# Graceful import for fallback functions
try:
    from skimage import filters, exposure, transform
    import scipy.ndimage as ndimage
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class CornucopiaAugmenter:
    """
    Advanced medical imaging augmentation using Cornucopia transforms.
    
    This class provides sophisticated augmentation capabilities specifically
    designed for medical imaging, with particular attention to preserving
    anatomical structures while introducing realistic variations.
    """
    
    def __init__(self, device='auto', random_state=None):
        """
        Initialize the Cornucopia augmenter.
        
        Parameters
        ----------
        device : str or torch.device
            Device to use for computations ('auto', 'cpu', 'cuda', or torch.device)
        random_state : int, optional
            Random seed for reproducible augmentations
        """
        self.device = self._get_device(device)
        self.random_state = random_state
        
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        # Initialize transform collections
        self.noise_transforms = self._init_noise_transforms()
        self.intensity_transforms = self._init_intensity_transforms()
        self.spatial_transforms = self._init_spatial_transforms()
        self.contrast_transforms = self._init_contrast_transforms()
    
    def _get_device(self, device):
        """Determine the appropriate device for computations."""
        if not TORCH_AVAILABLE:
            return 'cpu'
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            return torch.device(device)
        else:
            return device
    
    def _init_noise_transforms(self):
        """Initialize medical imaging-specific noise transforms."""
        if not CORNUCOPIA_AVAILABLE:
            return {}
        
        return {
            'gaussian_mixture': RandomGaussianMixtureTransform(
                mu=1,
                sigma=0.05,
                fwhm=2
            ),
            'gaussian_noise': RandomGaussianNoiseTransform(
                sigma=0.02
            ),
            'rician_noise': self._create_rician_noise_transform(),
            'structured_noise': self._create_structured_noise_transform()
        }
    
    def _init_intensity_transforms(self):
        """Initialize intensity transforms for medical imaging."""
        if not CORNUCOPIA_AVAILABLE:
            return {}
        
        return {
            'adaptive_intensity': self._create_adaptive_intensity_transform(),
            'gamma_medical': RandomGammaTransform(
                gamma=(0.7, 1.5)
            ),
            'bias_field': self._create_bias_field_transform()
        }
    
    def _init_spatial_transforms(self):
        """Initialize spatial transforms preserving anatomical integrity."""
        if not CORNUCOPIA_AVAILABLE:
            return {}
        
        return {
            'affine_medical': RandomAffineTransform(
                rotations=15,  # Moderate rotation preserving anatomy
                translations=0.1,  # Small translations
                zooms=0.1  # Conservative scaling
            ),
            'elastic_deformation': self._create_elastic_transform()
        }
    
    def _init_contrast_transforms(self):
        """Initialize contrast transforms for fiber visualization."""
        if not CORNUCOPIA_AVAILABLE:
            return {}
        
        return {
            'adaptive_contrast': self._create_adaptive_contrast_transform(),
            'local_contrast': self._create_local_contrast_transform()
        }
    
    def _create_rician_noise_transform(self):
        """Create Rician noise transform (common in MRI)."""
        class RicianNoiseTransform:
            def __init__(self, sigma_range=(0.01, 0.03), prob=0.5):
                self.sigma_range = sigma_range
                self.prob = prob
            
            def __call__(self, x):
                try:
                    if torch.rand(1) > self.prob:
                        return x
                    
                    # Check for edge cases
                    if torch.all(x == 0):
                        return x
                    
                    # Check for NaN or Inf values
                    if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                        return x
                    
                    sigma = torch.rand(1) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
                    
                    # Ensure sigma is not zero
                    sigma = torch.clamp(sigma, min=1e-6)
                    
                    # Add Rician noise (magnitude of complex Gaussian)
                    real_noise = torch.randn_like(x) * sigma
                    imag_noise = torch.randn_like(x) * sigma
                    
                    noisy_real = x + real_noise
                    noisy_imag = imag_noise
                    
                    # Compute magnitude safely
                    magnitude_squared = noisy_real**2 + noisy_imag**2
                    
                    # Avoid taking sqrt of negative numbers
                    magnitude_squared = torch.clamp(magnitude_squared, min=0)
                    result = torch.sqrt(magnitude_squared)
                    
                    # Check result for issues
                    if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
                        return x
                    
                    return torch.clamp(result, 0, 1)
                    
                except Exception as e:
                    return x
        
        return RicianNoiseTransform()
    
    def _create_structured_noise_transform(self):
        """Create structured noise that mimics acquisition artifacts."""
        class StructuredNoiseTransform:
            def __init__(self, intensity_range=(0.01, 0.05), prob=0.3):
                self.intensity_range = intensity_range
                self.prob = prob
            
            def __call__(self, x):
                if torch.rand(1) > self.prob:
                    return x
                
                intensity = torch.rand(1) * (self.intensity_range[1] - self.intensity_range[0]) + self.intensity_range[0]
                
                # Create structured noise pattern
                h, w = x.shape[-2:]
                pattern = torch.sin(torch.linspace(0, 20*np.pi, w)).unsqueeze(0).expand(h, w)
                pattern = pattern.to(x.device) * intensity
                
                return x + pattern
        
        return StructuredNoiseTransform()
    
    def _create_bias_field_transform(self):
        """Create bias field transform (common MRI artifact)."""
        class BiasFieldTransform:
            def __init__(self, intensity_range=(0.08, 0.25), prob=0.5):
                self.intensity_range = intensity_range
                self.prob = prob
            
            def __call__(self, x):
                try:
                    if torch.rand(1) > self.prob:
                        return x
                    
                    # Check for edge cases
                    if torch.all(x == 0):
                        return x
                    
                    h, w = x.shape[-2:]
                    
                    # Create smooth bias field
                    y_coords, x_coords = torch.meshgrid(
                        torch.linspace(-1, 1, h),
                        torch.linspace(-1, 1, w),
                        indexing='ij'
                    )
                    
                    # Polynomial bias field - avoid zero values (make less aggressive)
                    intensity_factor = torch.rand(1) * (self.intensity_range[1] - self.intensity_range[0]) + self.intensity_range[0]
                    bias_field = 1.0 + (x_coords**2 + y_coords**2) * intensity_factor
                    
                    # Ensure bias field doesn't have zeros or extreme values (less aggressive range)
                    bias_field = torch.clamp(bias_field, min=0.5, max=1.5)
                    
                    bias_field = bias_field.to(x.device)
                    result = x * bias_field
                    
                    # Check result for issues
                    if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
                        return x
                    
                    return torch.clamp(result, 0, 1)
                    
                except Exception as e:
                    return x
        
        return BiasFieldTransform()
    
    def _create_adaptive_intensity_transform(self):
        """Create adaptive intensity transform."""
        class AdaptiveIntensityTransform:
            def __init__(self, scale_range=(0.8, 1.2), shift_range=(-0.1, 0.1)):
                self.scale_range = scale_range
                self.shift_range = shift_range
            
            def __call__(self, x):
                try:
                    # Check for edge cases
                    if torch.all(x == 0):
                        return x
                    
                    # Check for NaN or Inf values
                    if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                        return x
                    
                    scale = torch.rand(1) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
                    shift = torch.rand(1) * (self.shift_range[1] - self.shift_range[0]) + self.shift_range[0]
                    
                    # Ensure scale is not zero
                    scale = torch.clamp(scale, min=0.1, max=2.0)
                    
                    result = x * scale + shift
                    
                    # Check result for issues
                    if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
                        return x
                    
                    return torch.clamp(result, 0, 1)
                    
                except Exception as e:
                    return x
        
        return AdaptiveIntensityTransform()
    
    def _create_elastic_transform(self):
        """Create elastic deformation transform."""
        try:
            from cornucopia import RandomElasticTransform
            return RandomElasticTransform(
                dmax=0.1,  # Maximum displacement
                shape=5   # Number of control points
            )
        except ImportError:
            # Fallback to custom implementation
            class ElasticTransform:
                def __call__(self, x):
                    return x  # Simple fallback
            return ElasticTransform()
    
    def _create_adaptive_contrast_transform(self):
        """Create adaptive contrast transform."""
        class AdaptiveContrastTransform:
            def __init__(self, factor_range=(0.8, 1.3)):
                self.factor_range = factor_range
            
            def __call__(self, x):
                try:
                    # Check for edge cases
                    if torch.all(x == 0):
                        return x
                    
                    # Check standard deviation to avoid division by zero
                    x_std = torch.std(x)
                    if x_std < 1e-6:
                        return x
                    
                    # Check mean to avoid issues
                    x_mean = torch.mean(x)
                    if torch.isnan(x_mean) or torch.isinf(x_mean):
                        return x
                    
                    factor = torch.rand(1) * (self.factor_range[1] - self.factor_range[0]) + self.factor_range[0]
                    result = (x - x_mean) * factor + x_mean
                    
                    # Additional safety checks
                    if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
                        return x
                    
                    return torch.clamp(result, 0, 1)
                    
                except Exception as e:
                    # If anything goes wrong, return original tensor
                    return x
        
        return AdaptiveContrastTransform()
    
    def _create_local_contrast_transform(self):
        """Create local contrast enhancement transform."""
        class LocalContrastTransform:
            def __init__(self, kernel_size=31, prob=0.5):
                self.kernel_size = kernel_size
                self.prob = prob
            
            def __call__(self, x):
                if torch.rand(1) > self.prob:
                    return x
                
                # Simple local contrast enhancement
                from scipy import ndimage
                x_np = x.cpu().numpy()
                
                # Local mean subtraction
                local_mean = ndimage.uniform_filter(x_np, size=self.kernel_size)
                enhanced = x_np - 0.5 * local_mean
                
                return torch.from_numpy(enhanced).to(x.device)
        
        return LocalContrastTransform()
    
    def apply_noise_augmentation(self, image: np.ndarray, 
                                noise_type: str = 'gaussian_mixture',
                                intensity: float = 1.0) -> np.ndarray:
        """
        Apply noise augmentation to an image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (2D slice)
        noise_type : str
            Type of noise ('gaussian_mixture', 'rician_noise', 'structured_noise')
        intensity : float
            Intensity multiplier for noise (0.0-2.0)
        
        Returns
        -------
        np.ndarray
            Augmented image
        """
        if not CORNUCOPIA_AVAILABLE:
            return self._fallback_noise_augmentation(image, noise_type, intensity)
        
        try:
            # Convert to torch tensor with error checking
            tensor_img = torch.from_numpy(image).float()
            
            # Check for issues in tensor
            if torch.any(torch.isnan(tensor_img)) or torch.any(torch.isinf(tensor_img)):
                return self._fallback_noise_augmentation(image, noise_type, intensity)
            
            tensor_img = tensor_img.unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Apply transform
            if noise_type in self.noise_transforms:
                transform = self.noise_transforms[noise_type]
                # Adjust intensity if needed
                if hasattr(transform, 'std_range') and intensity != 1.0:
                    original_range = transform.std_range
                    transform.std_range = (original_range[0] * intensity, original_range[1] * intensity)
                    augmented = transform(tensor_img)
                    transform.std_range = original_range  # Reset
                else:
                    augmented = transform(tensor_img)
            else:
                augmented = tensor_img
            
            # Convert back to numpy with error checking
            result = augmented.squeeze().cpu().numpy()
            
            # Check result for issues
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return self._fallback_noise_augmentation(image, noise_type, intensity)
            
            return result
            
        except Exception as e:
            # Fall back to simple augmentation if tensor operations fail
            return self._fallback_noise_augmentation(image, noise_type, intensity)
    
    def apply_intensity_augmentation(self, image: np.ndarray,
                                   transform_type: str = 'adaptive_intensity',
                                   **kwargs) -> np.ndarray:
        """
        Apply intensity augmentation to an image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (2D slice)
        transform_type : str
            Type of intensity transform
        **kwargs : dict
            Additional parameters for the transform
        
        Returns
        -------
        np.ndarray
            Augmented image
        """
        if not CORNUCOPIA_AVAILABLE:
            return self._fallback_intensity_augmentation(image, transform_type, **kwargs)
        
        try:
            # Convert to torch tensor with error checking
            tensor_img = torch.from_numpy(image).float()
            
            # Check for issues in tensor
            if torch.any(torch.isnan(tensor_img)) or torch.any(torch.isinf(tensor_img)):
                return self._fallback_intensity_augmentation(image, transform_type, **kwargs)
            
            tensor_img = tensor_img.unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Apply transform
            if transform_type in self.intensity_transforms:
                transform = self.intensity_transforms[transform_type]
                augmented = transform(tensor_img)
            else:
                augmented = tensor_img
            
            # Convert back to numpy with error checking
            result = augmented.squeeze().cpu().numpy()
            
            # Check result for issues
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return self._fallback_intensity_augmentation(image, transform_type, **kwargs)
            
            return result
            
        except Exception as e:
            # Fall back to simple augmentation if tensor operations fail
            return self._fallback_intensity_augmentation(image, transform_type, **kwargs)
    
    def apply_spatial_augmentation(self, image: np.ndarray,
                                 transform_type: str = 'affine_medical',
                                 **kwargs) -> np.ndarray:
        """
        Apply spatial augmentation to an image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (2D slice)
        transform_type : str
            Type of spatial transform
        **kwargs : dict
            Additional parameters for the transform
        
        Returns
        -------
        np.ndarray
            Augmented image
        """
        if not CORNUCOPIA_AVAILABLE:
            return self._fallback_spatial_augmentation(image, transform_type, **kwargs)
        
        try:
            # Convert to torch tensor with error checking
            tensor_img = torch.from_numpy(image).float()
            
            # Check for issues in tensor
            if torch.any(torch.isnan(tensor_img)) or torch.any(torch.isinf(tensor_img)):
                return self._fallback_spatial_augmentation(image, transform_type, **kwargs)
            
            tensor_img = tensor_img.unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Apply transform
            if transform_type in self.spatial_transforms:
                transform = self.spatial_transforms[transform_type]
                augmented = transform(tensor_img)
            else:
                augmented = tensor_img
            
            # Convert back to numpy with error checking
            result = augmented.squeeze().cpu().numpy()
            
            # Check result for issues
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return self._fallback_spatial_augmentation(image, transform_type, **kwargs)
            
            return result
            
        except Exception as e:
            # Fall back to simple augmentation if tensor operations fail
            return self._fallback_spatial_augmentation(image, transform_type, **kwargs)
    
    def apply_contrast_augmentation(self, image: np.ndarray,
                                  transform_type: str = 'adaptive_contrast',
                                  **kwargs) -> np.ndarray:
        """
        Apply contrast augmentation to an image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (2D slice)
        transform_type : str
            Type of contrast transform
        **kwargs : dict
            Additional parameters for the transform
        
        Returns
        -------
        np.ndarray
            Augmented image
        """
        if not CORNUCOPIA_AVAILABLE:
            return self._fallback_contrast_augmentation(image, transform_type, **kwargs)
        
        try:
            # Convert to torch tensor with error checking
            tensor_img = torch.from_numpy(image).float()
            
            # Check for issues in tensor
            if torch.any(torch.isnan(tensor_img)) or torch.any(torch.isinf(tensor_img)):
                return self._fallback_contrast_augmentation(image, transform_type, **kwargs)
            
            tensor_img = tensor_img.unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Apply transform
            if transform_type in self.contrast_transforms:
                transform = self.contrast_transforms[transform_type]
                augmented = transform(tensor_img)
            else:
                augmented = tensor_img
            
            # Convert back to numpy with error checking
            result = augmented.squeeze().cpu().numpy()
            
            # Check result for issues
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return self._fallback_contrast_augmentation(image, transform_type, **kwargs)
            
            return result
            
        except Exception as e:
            # Fall back to simple augmentation if tensor operations fail
            return self._fallback_contrast_augmentation(image, transform_type, **kwargs)
    
    def apply_comprehensive_augmentation(self, image: np.ndarray,
                                       augmentation_config: Dict[str, Any]) -> np.ndarray:
        """
        Apply a comprehensive set of augmentations based on configuration.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (2D slice)
        augmentation_config : dict
            Configuration dictionary specifying which augmentations to apply
            and their parameters
        
        Returns
        -------
        np.ndarray
            Fully augmented image
        """
        # Check for edge cases in input image
        if np.all(image == 0):
            return image
        
        if np.std(image) < 1e-6:
            return image
        
        # Check for NaN or Inf values in input
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            return image
        
        result = image.copy()
        
        # Ensure input is in [0,1] range
        result = np.clip(result, 0, 1)
        
        try:
            # Apply augmentations in order: spatial -> intensity -> noise -> contrast
            if 'spatial' in augmentation_config:
                try:
                    spatial_config = augmentation_config['spatial']
                    result = self.apply_spatial_augmentation(
                        result, 
                        spatial_config.get('type', 'affine_medical'),
                        **spatial_config.get('params', {})
                    )
                    # Check result after spatial transform
                    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                        result = image.copy()
                except Exception as e:
                    print(f"   ⚠️  Spatial augmentation failed: {e}")
                    result = image.copy()
            
            if 'intensity' in augmentation_config:
                try:
                    intensity_config = augmentation_config['intensity']
                    result = self.apply_intensity_augmentation(
                        result,
                        intensity_config.get('type', 'adaptive_intensity'),
                        **intensity_config.get('params', {})
                    )
                    # Check result after intensity transform
                    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                        result = image.copy()
                except Exception as e:
                    print(f"   ⚠️  Intensity augmentation failed: {e}")
                    # Don't reset to original, keep previous result
            
            if 'noise' in augmentation_config:
                try:
                    noise_config = augmentation_config['noise']
                    result = self.apply_noise_augmentation(
                        result,
                        noise_config.get('type', 'gaussian_mixture'),
                        noise_config.get('intensity', 1.0)
                    )
                    # Check result after noise transform
                    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                        # Keep previous result without noise
                        pass
                except Exception as e:
                    print(f"   ⚠️  Noise augmentation failed: {e}")
                    # Don't reset, keep previous result
            
            if 'contrast' in augmentation_config:
                try:
                    contrast_config = augmentation_config['contrast']
                    result = self.apply_contrast_augmentation(
                        result,
                        contrast_config.get('type', 'adaptive_contrast'),
                        **contrast_config.get('params', {})
                    )
                    # Check result after contrast transform
                    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                        # Keep previous result without contrast enhancement
                        pass
                except Exception as e:
                    print(f"   ⚠️  Contrast augmentation failed: {e}")
                    # Don't reset, keep previous result
        
        except Exception as e:
            # If any augmentation fails catastrophically, return the original image
            print(f"   ⚠️  Comprehensive augmentation failed: {e}, returning original image")
            return image
        
        # Ensure output is properly clipped and valid
        result = np.clip(result, 0, 1)
        
        # Final safety check
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            return image
        
        return result
    
    # Fallback methods for when Cornucopia is not available
    def _fallback_noise_augmentation(self, image, noise_type, intensity):
        """Fallback noise augmentation using basic numpy operations."""
        if noise_type == 'gaussian_mixture':
            noise = np.random.normal(0, 0.02 * intensity, image.shape)
            return np.clip(image + noise, 0, 1)
        elif noise_type == 'rician_noise':
            noise = np.random.normal(0, 0.01 * intensity, image.shape)
            return np.clip(image + noise, 0, 1)
        else:
            return image
    
    def _fallback_intensity_augmentation(self, image, transform_type, **kwargs):
        """Fallback intensity augmentation using basic operations."""
        if transform_type == 'adaptive_intensity':
            scale = np.random.uniform(0.8, 1.2)
            shift = np.random.uniform(-0.1, 0.1)
            return np.clip(image * scale + shift, 0, 1)
        elif transform_type == 'gamma_medical':
            gamma = np.random.uniform(0.7, 1.5)
            return np.power(image, gamma)
        else:
            return image
    
    def _fallback_spatial_augmentation(self, image, transform_type, **kwargs):
        """Fallback spatial augmentation (minimal for now)."""
        # For now, just return the image unchanged
        # Could implement basic scipy-based transforms here
        return image
    
    def _fallback_contrast_augmentation(self, image, transform_type, **kwargs):
        """Fallback contrast augmentation using basic operations."""
        if transform_type == 'adaptive_contrast':
            factor = np.random.uniform(0.8, 1.3)
            mean = np.mean(image)
            return np.clip((image - mean) * factor + mean, 0, 1)
        else:
            return image


def create_augmentation_presets():
    """
    Create predefined augmentation presets for fiber tract visualization.
    
    Returns
    -------
    dict
        Dictionary of augmentation preset configurations
    """
    return {
        'aggressive': {
            'spatial': {'type': 'elastic_deformation'},
            'intensity': {'type': 'bias_field'},
            'noise': {'type': 'rician_noise', 'intensity': 0.6},  # Further reduced from 1.0 to 0.6 for smoother results
            'contrast': {'type': 'local_contrast'}
        },
        
        'clinical_simulation': {
            'spatial': {'type': 'affine_medical'},
            'intensity': {'type': 'bias_field'},
            'noise': {'type': 'rician_noise', 'intensity': 0.2},  # Further reduced from 0.4 to 0.2 for very smooth results
            'contrast': {'type': 'adaptive_contrast'}
        }
    }


# Convenience function for easy integration
def augment_fiber_slice(slice_data: np.ndarray, 
                       preset: str = 'clinical_simulation',
                       custom_config: Optional[Dict[str, Any]] = None,
                       random_state: Optional[int] = None) -> np.ndarray:
    """
    Convenient function to augment a single fiber slice.
    
    Parameters
    ----------
    slice_data : np.ndarray
        Input slice data (2D array)
    preset : str
        Augmentation preset name
    custom_config : dict, optional
        Custom augmentation configuration (overrides preset)
    random_state : int, optional
        Random seed for reproducible results
    
    Returns
    -------
    np.ndarray
        Augmented slice data
    """
    # Normalize input data to [0,1] more carefully to preserve brain-background contrast
    original_min = np.min(slice_data)
    original_max = np.max(slice_data)
    
    # Special handling for brain data: use a more conservative normalization
    # that doesn't push brain tissue too close to 1.0
    if original_max > 1.0:  # Assume this is non-normalized medical data
        # Find background and brain tissue intensity ranges
        background_thresh = np.percentile(slice_data[slice_data > 0], 5) if np.any(slice_data > 0) else 0
        brain_values = slice_data[slice_data > background_thresh]
        
        if len(brain_values) > 0:
            # Use a normalization that maps brain tissue to ~0.6 range instead of ~0.8+
            # This gives more headroom for intensity transforms
            brain_95th = np.percentile(brain_values, 95)
            normalization_factor = brain_95th / 0.6  # Map 95th percentile to 0.6
            
            normalized_data = slice_data / normalization_factor
            normalized_data = np.clip(normalized_data, 0, 1)
            
            # Smart normalization applied successfully
        else:
            # Fallback to simple normalization
            normalized_data = (slice_data - original_min) / (original_max - original_min + 1e-8)
    else:
        # Data already in [0,1] range
        normalized_data = slice_data.copy()
    
    augmenter = CornucopiaAugmenter(random_state=random_state)
    
    if custom_config is not None:
        config = custom_config
    else:
        presets = create_augmentation_presets()
        config = presets.get(preset, presets['clinical_simulation'])
    
    result = augmenter.apply_comprehensive_augmentation(normalized_data, config)
    
    # The result is already in [0,1] range and suitable for further processing
    return result