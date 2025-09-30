#!/usr/bin/env python
"""
Cornucopia Integration for Medical Imaging Augmentations
"""

import warnings
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import random

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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
except ImportError as e:
    CORNUCOPIA_AVAILABLE = False

try:
    from skimage import filters, exposure, transform
    import scipy.ndimage as ndimage
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class CornucopiaAugmenter:
    """Advanced medical imaging augmentation using Cornucopia transforms."""
    
    def __init__(self, device='auto', random_state=None, truly_random=False):
        self.device = self._get_device(device)
        self.random_state = random_state
        self.truly_random = truly_random
        
        if truly_random:
            # Use current time for truly random augmentations
            import time
            true_random_seed = int(time.time() * 1000000) % (2**32)
            if TORCH_AVAILABLE:
                torch.manual_seed(true_random_seed)
            np.random.seed(true_random_seed)
            random.seed(true_random_seed)
        elif random_state is not None:
            if TORCH_AVAILABLE:
                torch.manual_seed(random_state)
            np.random.seed(random_state)
            random.seed(random_state)
        
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
            'gaussian_mixture': RandomGaussianMixtureTransform(mu=1, sigma=0.05, fwhm=2),
            'gaussian_noise': RandomGaussianNoiseTransform(sigma=0.02),
            'rician_noise': self._create_rician_noise_transform(),
            'structured_noise': self._create_structured_noise_transform()
        }
    
    def _init_intensity_transforms(self):
        """Initialize intensity transforms for medical imaging."""
        if not CORNUCOPIA_AVAILABLE:
            return {}
        
        return {
            'adaptive_intensity': self._create_adaptive_intensity_transform(),
            'gamma_medical': RandomGammaTransform(gamma=(0.7, 1.5)),
            'bias_field': self._create_bias_field_transform()
        }
    
    def _init_spatial_transforms(self):
        """Initialize spatial transforms preserving anatomical integrity."""
        if not CORNUCOPIA_AVAILABLE:
            return {}
        
        return {
            'affine_medical': RandomAffineTransform(rotations=15, translations=0.1, zooms=0.1),
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
                    
                    if torch.all(x == 0) or torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                        return x
                    
                    sigma = torch.rand(1) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
                    sigma = torch.clamp(sigma, min=1e-6)
                    
                    real_noise = torch.randn_like(x) * sigma
                    imag_noise = torch.randn_like(x) * sigma
                    
                    noisy_real = x + real_noise
                    noisy_imag = imag_noise
                    
                    magnitude_squared = noisy_real**2 + noisy_imag**2
                    magnitude_squared = torch.clamp(magnitude_squared, min=0)
                    result = torch.sqrt(magnitude_squared)
                    
                    if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
                        return x
                    
                    return torch.clamp(result, 0, 1)
                    
                except Exception as e:
                    return x
        
        return RicianNoiseTransform()
    
    def _create_structured_noise_transform(self):
        """Create structured noise that mimics acquisition artifacts."""
        class StructuredNoiseTransform:
            def __call__(self, x):
                intensity = torch.rand(1) * 0.04 + 0.01
                h, w = x.shape[-2:]
                pattern = torch.sin(torch.linspace(0, 20*np.pi, w)).unsqueeze(0).expand(h, w)
                pattern = pattern.to(x.device) * intensity
                return x + pattern
        return StructuredNoiseTransform()
    
    def _create_bias_field_transform(self):
        """Create bias field transform (common MRI artifact)."""
        class BiasFieldTransform:
            def __call__(self, x):
                if torch.all(x == 0):
                    return x
                h, w = x.shape[-2:]
                y_coords, x_coords = torch.meshgrid(
                    torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
                intensity_factor = torch.rand(1) * 0.17 + 0.08
                bias_field = 1.0 + (x_coords**2 + y_coords**2) * intensity_factor
                bias_field = torch.clamp(bias_field, min=0.5, max=1.5).to(x.device)
                return torch.clamp(x * bias_field, 0, 1)
        return BiasFieldTransform()
    
    def _create_adaptive_intensity_transform(self):
        """Create adaptive intensity transform."""
        class AdaptiveIntensityTransform:
            def __call__(self, x):
                if torch.all(x == 0):
                    return x
                scale = torch.clamp(torch.rand(1) * 0.4 + 0.8, min=0.1, max=2.0)
                shift = torch.rand(1) * 0.2 - 0.1
                return torch.clamp(x * scale + shift, 0, 1)
        return AdaptiveIntensityTransform()
    
    def _create_elastic_transform(self):
        """Create elastic deformation transform."""
        try:
            from cornucopia import RandomElasticTransform
            return RandomElasticTransform(dmax=0.1, shape=5)
        except ImportError:
            class ElasticTransform:
                def __call__(self, x):
                    return x
            return ElasticTransform()
    
    def _create_adaptive_contrast_transform(self):
        """Create adaptive contrast transform."""
        class AdaptiveContrastTransform:
            def __call__(self, x):
                if torch.all(x == 0) or torch.std(x) < 1e-6:
                    return x
                factor = torch.rand(1) * 0.5 + 0.8
                x_mean = torch.mean(x)
                return torch.clamp((x - x_mean) * factor + x_mean, 0, 1)
        return AdaptiveContrastTransform()
    
    def _create_local_contrast_transform(self):
        """Create local contrast enhancement transform."""
        class LocalContrastTransform:
            def __call__(self, x):
                x_np = x.cpu().numpy()
                local_mean = ndimage.uniform_filter(x_np, size=31)
                enhanced = x_np - 0.5 * local_mean
                return torch.from_numpy(enhanced).to(x.device)
        return LocalContrastTransform()
    
    def apply_noise_augmentation(self, image: np.ndarray, 
                                noise_type: str = 'gaussian_mixture',
                                intensity: float = 1.0) -> np.ndarray:
        """Apply noise augmentation to an image."""
        if not CORNUCOPIA_AVAILABLE:
            return self._fallback_noise_augmentation(image, noise_type, intensity)
        
        try:
            tensor_img = torch.from_numpy(image).float()
            if torch.any(torch.isnan(tensor_img)) or torch.any(torch.isinf(tensor_img)):
                return self._fallback_noise_augmentation(image, noise_type, intensity)
            
            tensor_img = tensor_img.unsqueeze(0).unsqueeze(0).to(self.device)
            
            if noise_type in self.noise_transforms:
                transform = self.noise_transforms[noise_type]
                augmented = transform(tensor_img)
            else:
                augmented = tensor_img
            
            result = augmented.squeeze().cpu().numpy()
            
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return self._fallback_noise_augmentation(image, noise_type, intensity)
            
            return result
        except Exception:
            return self._fallback_noise_augmentation(image, noise_type, intensity)
    
    def apply_intensity_augmentation(self, image: np.ndarray,
                                   transform_type: str = 'adaptive_intensity',
                                   **kwargs) -> np.ndarray:
        """Apply intensity augmentation to an image."""
        if not CORNUCOPIA_AVAILABLE:
            return self._fallback_intensity_augmentation(image, transform_type, **kwargs)
        
        try:
            tensor_img = torch.from_numpy(image).float()
            if torch.any(torch.isnan(tensor_img)) or torch.any(torch.isinf(tensor_img)):
                return self._fallback_intensity_augmentation(image, transform_type, **kwargs)
            
            tensor_img = tensor_img.unsqueeze(0).unsqueeze(0).to(self.device)
            
            if transform_type in self.intensity_transforms:
                transform = self.intensity_transforms[transform_type]
                augmented = transform(tensor_img)
            else:
                augmented = tensor_img
            
            result = augmented.squeeze().cpu().numpy()
            
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return self._fallback_intensity_augmentation(image, transform_type, **kwargs)
            
            return result
        except Exception:
            return self._fallback_intensity_augmentation(image, transform_type, **kwargs)
    
    def apply_spatial_augmentation(self, image: np.ndarray,
                                 transform_type: str = 'affine_medical',
                                 **kwargs) -> np.ndarray:
        """Apply spatial augmentation to an image."""
        if not CORNUCOPIA_AVAILABLE:
            return self._fallback_spatial_augmentation(image, transform_type, **kwargs)
        
        try:
            tensor_img = torch.from_numpy(image).float()
            if torch.any(torch.isnan(tensor_img)) or torch.any(torch.isinf(tensor_img)):
                return self._fallback_spatial_augmentation(image, transform_type, **kwargs)
            
            tensor_img = tensor_img.unsqueeze(0).unsqueeze(0).to(self.device)
            
            if transform_type in self.spatial_transforms:
                transform = self.spatial_transforms[transform_type]
                augmented = transform(tensor_img)
            else:
                augmented = tensor_img
            
            result = augmented.squeeze().cpu().numpy()
            
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return self._fallback_spatial_augmentation(image, transform_type, **kwargs)
            
            return result
        except Exception:
            return self._fallback_spatial_augmentation(image, transform_type, **kwargs)
    
    def apply_contrast_augmentation(self, image: np.ndarray,
                                  transform_type: str = 'adaptive_contrast',
                                  **kwargs) -> np.ndarray:
        """Apply contrast augmentation to an image."""
        if not CORNUCOPIA_AVAILABLE:
            return self._fallback_contrast_augmentation(image, transform_type, **kwargs)
        
        try:
            tensor_img = torch.from_numpy(image).float()
            if torch.any(torch.isnan(tensor_img)) or torch.any(torch.isinf(tensor_img)):
                return self._fallback_contrast_augmentation(image, transform_type, **kwargs)
            
            tensor_img = tensor_img.unsqueeze(0).unsqueeze(0).to(self.device)
            
            if transform_type in self.contrast_transforms:
                transform = self.contrast_transforms[transform_type]
                augmented = transform(tensor_img)
            else:
                augmented = tensor_img
            
            result = augmented.squeeze().cpu().numpy()
            
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return self._fallback_contrast_augmentation(image, transform_type, **kwargs)
            
            return result
        except Exception:
            return self._fallback_contrast_augmentation(image, transform_type, **kwargs)
    
    def apply_comprehensive_augmentation(self, image: np.ndarray,
                                       augmentation_config: Dict[str, Any]) -> np.ndarray:
        """Apply a comprehensive set of augmentations based on configuration."""
        if np.all(image == 0) or np.std(image) < 1e-6 or np.any(np.isnan(image)) or np.any(np.isinf(image)):
            return image
        
        result = np.clip(image.copy(), 0, 1)
        
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
                    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                        result = image.copy()
                except Exception:
                    result = image.copy()
            
            if 'intensity' in augmentation_config:
                try:
                    intensity_config = augmentation_config['intensity']
                    result = self.apply_intensity_augmentation(
                        result,
                        intensity_config.get('type', 'adaptive_intensity'),
                        **intensity_config.get('params', {})
                    )
                    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                        result = image.copy()
                except Exception:
                    pass
            
            if 'noise' in augmentation_config:
                try:
                    noise_config = augmentation_config['noise']
                    result = self.apply_noise_augmentation(
                        result,
                        noise_config.get('type', 'gaussian_mixture'),
                        noise_config.get('intensity', 1.0)
                    )
                    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                        pass
                except Exception:
                    pass
            
            if 'contrast' in augmentation_config:
                try:
                    contrast_config = augmentation_config['contrast']
                    result = self.apply_contrast_augmentation(
                        result,
                        contrast_config.get('type', 'adaptive_contrast'),
                        **contrast_config.get('params', {})
                    )
                    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                        pass
                except Exception:
                    pass
        
        except Exception:
            return image
        
        result = np.clip(result, 0, 1)
        
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
    """Create predefined augmentation presets for fiber tract visualization."""
    return {
        'aggressive': {
            'spatial': {'type': 'elastic_deformation'},
            'intensity': {'type': 'bias_field'},
            'noise': {'type': 'rician_noise', 'intensity': 0.1},
            'contrast': {'type': 'local_contrast'}
        },
        
        'clinical_simulation': {
            'spatial': {'type': 'affine_medical'},
            'intensity': {'type': 'bias_field'},
            'noise': {'type': 'rician_noise', 'intensity': 0.2},
            'contrast': {'type': 'adaptive_contrast'}
        }
    }


    
    
def create_augmentation_presets(truly_random=False):
    """Create different presets for medical imaging augmentations."""
    presets = {}
    
    # Standard presets (existing)
    if CORNUCOPIA_AVAILABLE:
        presets['clinical_simulation'] = CornucopiaAugmenter(
            random_state=None if truly_random else 42, 
            truly_random=truly_random
        )
        presets['aggressive'] = CornucopiaAugmenter(
            random_state=None if truly_random else 123, 
            truly_random=truly_random
        )
    
    return presets


def augment_fiber_slice(slice_data: np.ndarray, 
                       preset: str = 'clinical_simulation',
                       custom_config: Optional[Dict[str, Any]] = None,
                       random_state: Optional[int] = None,
                       truly_random: bool = False) -> np.ndarray:
    """Convenient function to augment a single fiber slice with true randomization support."""
    # Normalize input data to [0,1] more carefully to preserve brain-background contrast
    original_min = np.min(slice_data)
    original_max = np.max(slice_data)
    
    if original_max > 1.0:
        background_thresh = np.percentile(slice_data[slice_data > 0], 5) if np.any(slice_data > 0) else 0
        brain_values = slice_data[slice_data > background_thresh]
        
        if len(brain_values) > 0:
            brain_95th = np.percentile(brain_values, 95)
            normalization_factor = brain_95th / 0.6
            
            normalized_data = slice_data / normalization_factor
            normalized_data = np.clip(normalized_data, 0, 1)
        else:
            normalized_data = (slice_data - original_min) / (original_max - original_min + 1e-8)
    else:
        normalized_data = slice_data.copy()
    
    augmenter = CornucopiaAugmenter(random_state=random_state)
    
    if custom_config is not None:
        config = custom_config
    else:
        presets = create_augmentation_presets()
        config = presets.get(preset, presets['clinical_simulation'])
    
    result = augmenter.apply_comprehensive_augmentation(normalized_data, config)
    
    return result