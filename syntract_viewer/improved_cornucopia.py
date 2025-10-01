"""
Improved Cornucopia augmentation for optical imaging simulation.
"""

import numpy as np
import random
from typing import Dict, Any, Optional, Tuple
import warnings

try:
    import torch
    from cornucopia import GaussianNoiseTransform, ChiNoiseTransform
    from cornucopia import RandomMulFieldTransform, RandomGammaTransform
    from cornucopia import ElasticTransform, RandomElasticTransform
    CORNUCOPIA_AVAILABLE = True
except ImportError:
    CORNUCOPIA_AVAILABLE = False

try:
    from scipy import ndimage, interpolate
    from skimage import morphology, measure, filters
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ImprovedCornucopiaAugmenter:
    """Enhanced Cornucopia augmenter for optical imaging simulation."""
    
    def __init__(self, device='auto', random_state=None):
        """Initialize the improved augmenter."""
        self.device = self._get_device(device)
        self.random_state = random_state
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            if CORNUCOPIA_AVAILABLE:
                torch.manual_seed(random_state)
        
        # Initialize transforms
        self._init_noise_transforms()
        self._init_intensity_transforms()
        self._init_spatial_transforms()
        self._init_debris_transforms()
    
    def _get_device(self, device):
        """Get the appropriate device for tensor operations."""
        if not CORNUCOPIA_AVAILABLE:
            return 'cpu'
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _init_noise_transforms(self):
        """Initialize noise transforms optimized for optical imaging."""
        self.noise_transforms = {}
        
        if CORNUCOPIA_AVAILABLE:
            # Gamma noise (speckle) - ideal for optical imaging
            self.noise_transforms['gamma_speckle'] = self._create_gamma_speckle_transform()
            self.noise_transforms['gamma_multiplicative'] = self._create_gamma_multiplicative_transform()
            self.noise_transforms['optical_speckle'] = self._create_optical_speckle_transform()
        
        # Fallback implementations
        self.noise_transforms['fallback_speckle'] = self._create_fallback_speckle()
    
    def _init_intensity_transforms(self):
        """Initialize intensity transforms with multiplicative fields."""
        self.intensity_transforms = {}
        
        if CORNUCOPIA_AVAILABLE:
            # Smooth multiplicative fields
            self.intensity_transforms['smooth_multiplicative'] = self._create_smooth_multiplicative_field()
            self.intensity_transforms['multiplicative_field'] = self._create_multiplicative_field()
            self.intensity_transforms['optical_intensity'] = self._create_optical_intensity_transform()
        
        # Fallback implementations
        self.intensity_transforms['fallback_multiplicative'] = self._create_fallback_multiplicative()
    
    def _init_spatial_transforms(self):
        """Initialize spatial transforms."""
        self.spatial_transforms = {}
        
        if CORNUCOPIA_AVAILABLE:
            self.spatial_transforms['elastic_medical'] = self._create_elastic_transform()
        
        # Fallback implementations
        self.spatial_transforms['fallback_elastic'] = self._create_fallback_elastic()
    
    def _init_debris_transforms(self):
        """Initialize debris simulation transforms."""
        self.debris_transforms = {}
        
        # These work with or without cornucopia
        self.debris_transforms['random_spheres'] = self._create_random_spheres_debris()
        self.debris_transforms['random_shapes'] = self._create_random_shapes_debris()
        self.debris_transforms['morphological_debris'] = self._create_morphological_debris()
    
    def _create_gamma_speckle_transform(self):
        """Create gamma speckle transform for optical imaging."""
        if not CORNUCOPIA_AVAILABLE:
            return self._create_fallback_speckle()
        
        class GammaSpeckleTransform:
            def __init__(self, intensity_range=(0.85, 1.3), prob=0.8):  # Higher prob, more visible range
                self.intensity_range = intensity_range
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                # Create very subtle gamma noise
                intensity = random.uniform(*self.intensity_range)
                gamma_transform = RandomGammaTransform(gamma=intensity)
                return gamma_transform(x)
        
        return GammaSpeckleTransform()
    
    def _create_gamma_multiplicative_transform(self):
        """Create gamma multiplicative transform."""
        if not CORNUCOPIA_AVAILABLE:
            return self._create_fallback_multiplicative()
        
        class GammaMultiplicativeTransform:
            def __init__(self, scale_range=(0.02, 0.06), prob=0.7):  # Balanced range
                self.scale_range = scale_range
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                # Use GaussianNoiseTransform for controlled multiplicative effect
                scale = random.uniform(*self.scale_range)
                gamma_noise = GaussianNoiseTransform(sigma=scale)
                return gamma_noise(x)
        
        return GammaMultiplicativeTransform()
    
    def _create_custom_gamma_multiplicative(self, scale_range=(0.02, 0.06), prob=0.7):
        """Create gamma multiplicative transform with custom parameters."""
        if not CORNUCOPIA_AVAILABLE:
            return self._create_fallback_multiplicative()
        
        class CustomGammaMultiplicativeTransform:
            def __init__(self, scale_range, prob):
                self.scale_range = scale_range
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                scale = random.uniform(*self.scale_range)
                gamma_noise = GaussianNoiseTransform(sigma=scale)
                return gamma_noise(x)
        
        return CustomGammaMultiplicativeTransform(scale_range, prob)
    
    def _create_custom_gamma_speckle(self, intensity_range=(0.9, 1.05), prob=0.8):
        """Create gamma speckle transform with custom parameters."""
        if not CORNUCOPIA_AVAILABLE:
            return self._create_fallback_speckle()
        
        class CustomGammaSpeckleTransform:
            def __init__(self, intensity_range, prob):
                self.intensity_range = intensity_range
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                intensity = random.uniform(*self.intensity_range)
                gamma_transform = RandomGammaTransform(gamma=intensity)
                return gamma_transform(x)
        
        return CustomGammaSpeckleTransform(intensity_range, prob)
    
    def _create_optical_speckle_transform(self):
        """Create optical speckle transform."""
        if not CORNUCOPIA_AVAILABLE:
            return self._create_fallback_speckle()
        
        class OpticalSpeckleTransform:
            def __init__(self, speckle_strength=(0.01, 0.05), prob=0.7):  # Reduced for less grain
                self.speckle_strength = speckle_strength
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                # Very subtle Chi noise with multiplicative effect
                strength = random.uniform(*self.speckle_strength)
                chi_noise = ChiNoiseTransform(sigma=strength)
                result = chi_noise(x)
                
                # Very slight multiplicative field - handle parameter name variations
                try:
                    mul_field = RandomMulFieldTransform(vmax=0.05, order=2, shape=8)
                    result = mul_field(result)
                except Exception as e:
                    # Fallback - just use the noise without multiplicative field
                    print(f"    Multiplicative field failed: {e}")
                
                return result
        
        return OpticalSpeckleTransform()
    
    def _create_smooth_multiplicative_field(self):
        """Create smooth multiplicative field using RandomMulFieldTransform."""
        if not CORNUCOPIA_AVAILABLE:
            return self._create_fallback_multiplicative()
        
        class SmoothMultiplicativeField:
            def __init__(self, field_strength=(0.02, 0.08), smoothness=(2, 3), prob=0.6):  # Much more conservative
                self.field_strength = field_strength
                self.smoothness = smoothness
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                # Use RandomMulFieldTransform for very subtle smooth multiplicative effect
                strength = random.uniform(*self.field_strength)
                degree = random.randint(*self.smoothness)
                try:
                    mul_field = RandomMulFieldTransform(vmax=strength, order=degree, shape=8)
                    return mul_field(x)
                except Exception as e:
                    print(f"    Smooth multiplicative field failed: {e}")
                    # Fallback - return original
                    return x
        
        return SmoothMultiplicativeField()
    
    def _create_multiplicative_field(self):
        """Create multiplicative field transform."""
        if not CORNUCOPIA_AVAILABLE:
            return self._create_fallback_multiplicative()
        
        class MultiplicativeField:
            def __init__(self, strength_range=(0.06, 0.12), prob=0.75):  # Balanced range
                self.strength_range = strength_range
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                # Use RandomMulFieldTransform for multiplicative effect
                strength = random.uniform(*self.strength_range)
                try:
                    mul_field = RandomMulFieldTransform(vmax=strength, order=3, shape=6)
                    return mul_field(x)
                except Exception as e:
                    print(f"    Multiplicative field failed: {e}")
                    # Fallback - return original
                    return x
        
        return MultiplicativeField()
    
    def _create_custom_multiplicative_field(self, strength_range=(0.06, 0.12), prob=0.75):
        """Create multiplicative field transform with custom parameters."""
        if not CORNUCOPIA_AVAILABLE:
            return self._create_fallback_multiplicative()
        
        class CustomMultiplicativeField:
            def __init__(self, strength_range, prob):
                self.strength_range = strength_range
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                strength = random.uniform(*self.strength_range)
                try:
                    mul_field = RandomMulFieldTransform(vmax=strength, order=3, shape=6)
                    return mul_field(x)
                except Exception as e:
                    print(f"    Custom multiplicative field failed: {e}")
                    return x
        
        return CustomMultiplicativeField(strength_range, prob)
    
    def _create_optical_intensity_transform(self):
        """Create optical intensity transform."""
        if not CORNUCOPIA_AVAILABLE:
            return self._create_fallback_multiplicative()
        
        class OpticalIntensityTransform:
            def __init__(self, gamma_range=(0.9, 1.1), bias_strength=(0.01, 0.05), prob=0.7):  # Increased from 0.4
                self.gamma_range = gamma_range
                self.bias_strength = bias_strength
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                # Apply very subtle gamma correction
                gamma_val = random.uniform(*self.gamma_range)
                gamma_transform = RandomGammaTransform(gamma=gamma_val)
                result = gamma_transform(x)
                
                # Add subtle multiplicative field
                bias_strength = random.uniform(*self.bias_strength)
                try:
                    mul_field = RandomMulFieldTransform(vmax=bias_strength, order=2, shape=6)
                    result = mul_field(result)
                except Exception as e:
                    print(f"    Optical intensity multiplicative field failed: {e}")
                    # Fallback - return just the gamma corrected result
                
                return result
        
        return OpticalIntensityTransform()
    
    def _create_elastic_transform(self):
        """Create elastic deformation transform."""
        if not CORNUCOPIA_AVAILABLE:
            return self._create_fallback_elastic()
        
        class ElasticTransform:
            def __init__(self, max_displacement=5, prob=0.5):
                self.max_displacement = max_displacement
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                # Use RandomElasticTransform with reduced displacement
                elastic = RandomElasticTransform(dmax=self.max_displacement, shape=5)
                return elastic(x)
        
        return ElasticTransform()
    
    def _create_random_spheres_debris(self):
        """Create random spheres debris simulation."""
        class RandomSpheresDebris:
            def __init__(self, n_spheres=(1, 5), radius_range=(2, 8), intensity_range=(0.1, 0.8), prob=0.3):
                self.n_spheres = n_spheres
                self.radius_range = radius_range
                self.intensity_range = intensity_range
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                # Convert to numpy for processing
                if hasattr(x, 'cpu'):
                    x_np = x.squeeze().cpu().numpy()
                    was_tensor = True
                else:
                    x_np = x
                    was_tensor = False
                
                result = x_np.copy()
                h, w = result.shape
                n_spheres = random.randint(*self.n_spheres)
                
                for _ in range(n_spheres):
                    # Random sphere parameters
                    radius = random.randint(*self.radius_range)
                    center_x = random.randint(radius, w - radius)
                    center_y = random.randint(radius, h - radius)
                    intensity = random.uniform(*self.intensity_range)
                    
                    # Create sphere mask
                    y_coords, x_coords = np.ogrid[:h, :w]
                    mask = ((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) <= radius ** 2
                    
                    # Apply debris with alpha blending
                    alpha = random.uniform(0.3, 0.8)
                    result[mask] = result[mask] * (1 - alpha) + intensity * alpha
                
                # Convert back to tensor if needed
                if was_tensor:
                    return torch.from_numpy(result).unsqueeze(0).unsqueeze(0).to(x.device)
                else:
                    return result
        
        return RandomSpheresDebris()
    
    def _create_random_shapes_debris(self):
        """Create random shapes debris simulation."""
        class RandomShapesDebris:
            def __init__(self, n_shapes=(1, 3), size_range=(4, 10), intensity_range=(0.15, 0.35), prob=0.5):
                self.n_shapes = n_shapes
                self.size_range = size_range
                self.intensity_range = intensity_range
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                # Convert to numpy for processing
                if hasattr(x, 'cpu'):
                    x_np = x.squeeze().cpu().numpy()
                    was_tensor = True
                else:
                    x_np = x
                    was_tensor = False
                
                result = x_np.copy()
                h, w = result.shape
                n_shapes = random.randint(*self.n_shapes)
                
                for _ in range(n_shapes):
                    size = random.randint(*self.size_range)
                    center_x = random.randint(size, w - size)
                    center_y = random.randint(size, h - size)
                    intensity = random.uniform(*self.intensity_range)
                    
                    # Create random shape (circle, rectangle, or ellipse)
                    shape_type = random.choice(['circle', 'rectangle', 'ellipse'])
                    
                    if shape_type == 'circle':
                        y_coords, x_coords = np.ogrid[:h, :w]
                        mask = ((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) <= size ** 2
                    elif shape_type == 'rectangle':
                        mask = np.zeros((h, w), dtype=bool)
                        y_start = max(0, center_y - size//2)
                        y_end = min(h, center_y + size//2)
                        x_start = max(0, center_x - size//2)
                        x_end = min(w, center_x + size//2)
                        mask[y_start:y_end, x_start:x_end] = True
                    else:  # ellipse
                        y_coords, x_coords = np.ogrid[:h, :w]
                        a = size
                        b = int(size * random.uniform(0.8, 1.2))  # More consistent aspect ratio
                        mask = ((x_coords - center_x) ** 2 / a ** 2 + (y_coords - center_y) ** 2 / b ** 2) <= 1
                    
                    # Apply morphological operations if available (controlled)
                    if SCIPY_AVAILABLE and random.random() < 0.4:
                        operation = random.choice(['erosion', 'dilation', 'opening'])
                        if operation == 'erosion':
                            mask = morphology.binary_erosion(mask, morphology.disk(1))
                        elif operation == 'dilation':
                            mask = morphology.binary_dilation(mask, morphology.disk(1))
                        else:  # opening
                            mask = morphology.binary_opening(mask, morphology.disk(1))
                    
                    # Apply debris with controlled alpha blending
                    alpha = random.uniform(0.25, 0.45)  # More controlled blending
                    result[mask] = result[mask] * (1 - alpha) + intensity * alpha
                
                # Convert back to tensor if needed
                if was_tensor:
                    return torch.from_numpy(result).unsqueeze(0).unsqueeze(0).to(x.device)
                else:
                    return result
        
        return RandomShapesDebris()
    
    def _create_custom_random_shapes_debris(self, n_shapes=(1, 3), size_range=(4, 10), intensity_range=(0.15, 0.35), prob=0.5):
        """Create random shapes debris simulation with custom parameters."""
        class CustomRandomShapesDebris:
            def __init__(self, n_shapes, size_range, intensity_range, prob):
                self.n_shapes = n_shapes
                self.size_range = size_range
                self.intensity_range = intensity_range
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                # Convert to numpy for processing
                if hasattr(x, 'cpu'):
                    x_np = x.squeeze().cpu().numpy()
                    was_tensor = True
                else:
                    x_np = x
                    was_tensor = False
                
                result = x_np.copy()
                h, w = result.shape
                n_shapes = random.randint(*self.n_shapes)
                
                for _ in range(n_shapes):
                    size = random.randint(*self.size_range)
                    center_x = random.randint(size, w - size)
                    center_y = random.randint(size, h - size)
                    intensity = random.uniform(*self.intensity_range)
                    
                    # Create random shape (circle, rectangle, or ellipse)
                    shape_type = random.choice(['circle', 'rectangle', 'ellipse'])
                    
                    if shape_type == 'circle':
                        y_coords, x_coords = np.ogrid[:h, :w]
                        mask = ((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) <= size ** 2
                    elif shape_type == 'rectangle':
                        mask = np.zeros((h, w), dtype=bool)
                        y_start = max(0, center_y - size//2)
                        y_end = min(h, center_y + size//2)
                        x_start = max(0, center_x - size//2)
                        x_end = min(w, center_x + size//2)
                        mask[y_start:y_end, x_start:x_end] = True
                    else:  # ellipse
                        y_coords, x_coords = np.ogrid[:h, :w]
                        a = size
                        b = int(size * random.uniform(0.8, 1.2))
                        mask = ((x_coords - center_x) ** 2 / a ** 2 + (y_coords - center_y) ** 2 / b ** 2) <= 1
                    
                    # Apply morphological operations if available (controlled)
                    if SCIPY_AVAILABLE and random.random() < 0.3:
                        operation = random.choice(['erosion', 'dilation'])
                        if operation == 'erosion':
                            mask = morphology.binary_erosion(mask, morphology.disk(1))
                        else:  # dilation
                            mask = morphology.binary_dilation(mask, morphology.disk(1))
                    
                    # Apply debris with controlled alpha blending
                    alpha = random.uniform(0.2, 0.4)
                    result[mask] = result[mask] * (1 - alpha) + intensity * alpha
                
                # Convert back to tensor if needed
                if was_tensor:
                    return torch.from_numpy(result).unsqueeze(0).unsqueeze(0).to(x.device)
                else:
                    return result
        
        return CustomRandomShapesDebris(n_shapes, size_range, intensity_range, prob)
    
    def _create_morphological_debris(self):
        """Create morphological debris simulation."""
        class MorphologicalDebris:
            def __init__(self, n_regions=(1, 3), base_size=(4, 10), intensity_range=(0.15, 0.6), prob=0.3):
                self.n_regions = n_regions
                self.base_size = base_size
                self.intensity_range = intensity_range
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                # Convert to numpy for processing
                if hasattr(x, 'cpu'):
                    x_np = x.squeeze().cpu().numpy()
                    was_tensor = True
                else:
                    x_np = x
                    was_tensor = False
                
                result = x_np.copy()
                h, w = result.shape
                n_regions = random.randint(*self.n_regions)
                
                for _ in range(n_regions):
                    size = random.randint(*self.base_size)
                    center_x = random.randint(size, w - size)
                    center_y = random.randint(size, h - size)
                    intensity = random.uniform(*self.intensity_range)
                    
                    # Start with basic circular region
                    y_coords, x_coords = np.ogrid[:h, :w]
                    mask = ((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) <= size ** 2
                    
                    # Apply series of morphological operations if available
                    if SCIPY_AVAILABLE:
                        # Random sequence of operations
                        n_ops = random.randint(2, 5)
                        for _ in range(n_ops):
                            op = random.choice(['erosion', 'dilation', 'opening', 'closing'])
                            disk_size = random.randint(1, 3)
                            
                            if op == 'erosion':
                                mask = morphology.binary_erosion(mask, morphology.disk(disk_size))
                            elif op == 'dilation':
                                mask = morphology.binary_dilation(mask, morphology.disk(disk_size))
                            elif op == 'opening':
                                mask = morphology.binary_opening(mask, morphology.disk(disk_size))
                            else:  # closing
                                mask = morphology.binary_closing(mask, morphology.disk(disk_size))
                    
                    # Apply debris with alpha blending
                    alpha = random.uniform(0.2, 0.6)
                    result[mask] = result[mask] * (1 - alpha) + intensity * alpha
                
                # Convert back to tensor if needed
                if was_tensor:
                    return torch.from_numpy(result).unsqueeze(0).unsqueeze(0).to(x.device)
                else:
                    return result
        
        return MorphologicalDebris()
    
    def _create_fallback_speckle(self):
        """Create fallback speckle implementation."""
        class FallbackSpeckle:
            def __init__(self, strength=(0.01, 0.05), prob=0.3):
                self.strength = strength
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                # Convert to numpy if needed
                if hasattr(x, 'cpu'):
                    x_np = x.squeeze().cpu().numpy()
                    was_tensor = True
                else:
                    x_np = x
                    was_tensor = False
                
                strength = random.uniform(*self.strength)
                # Very subtle gamma multiplicative noise simulation
                gamma_noise = np.random.gamma(1.0, strength, x_np.shape)
                result = x_np * gamma_noise
                
                # Convert back to tensor if needed
                if was_tensor:
                    return torch.from_numpy(result).unsqueeze(0).unsqueeze(0).to(x.device)
                else:
                    return result
        
        return FallbackSpeckle()
    
    def _create_fallback_multiplicative(self):
        """Create fallback multiplicative field implementation."""
        class FallbackMultiplicative:
            def __init__(self, strength=(0.02, 0.08), prob=0.4):
                self.strength = strength
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob:
                    return x
                
                # Convert to numpy if needed
                if hasattr(x, 'cpu'):
                    x_np = x.squeeze().cpu().numpy()
                    was_tensor = True
                else:
                    x_np = x
                    was_tensor = False
                
                h, w = x_np.shape
                strength = random.uniform(*self.strength)
                
                # Create smooth polynomial multiplicative field
                y_coords, x_coords = np.mgrid[0:h, 0:w]
                # Normalize coordinates to [-1, 1]
                y_norm = (y_coords - h/2) / (h/2)
                x_norm = (x_coords - w/2) / (w/2)
                
                # Create smooth polynomial field
                field = 1.0 + strength * (x_norm + y_norm + x_norm**2 - y_norm**2)
                
                # Apply Gaussian smoothing if SCIPY_AVAILABLE
                if SCIPY_AVAILABLE:
                    field = ndimage.gaussian_filter(field, sigma=min(h, w) / 20)
                
                result = x_np * field
                
                # Convert back to tensor if needed
                if was_tensor:
                    return torch.from_numpy(result).unsqueeze(0).unsqueeze(0).to(x.device)
                else:
                    return result
        
        return FallbackMultiplicative()
    
    def _create_fallback_elastic(self):
        """Create fallback elastic deformation implementation."""
        class FallbackElastic:
            def __init__(self, prob=0.4):
                self.prob = prob
            
            def __call__(self, x):
                if random.random() > self.prob or not SCIPY_AVAILABLE:
                    return x
                
                # Convert to numpy if needed
                if hasattr(x, 'cpu'):
                    x_np = x.squeeze().cpu().numpy()
                    was_tensor = True
                else:
                    x_np = x
                    was_tensor = False
                
                h, w = x_np.shape
                
                # Create simple displacement field
                max_displacement = 2
                
                # Create smooth displacement fields
                dy = np.random.normal(0, max_displacement/3, (h//8, w//8))
                dx = np.random.normal(0, max_displacement/3, (h//8, w//8))
                
                # Upscale displacement fields
                dy = ndimage.zoom(dy, (8, 8), order=1)
                dx = ndimage.zoom(dx, (8, 8), order=1)
                
                # Ensure same size
                dy = dy[:h, :w]
                dx = dx[:h, :w]
                
                # Apply displacement
                y_coords, x_coords = np.mgrid[0:h, 0:w]
                y_displaced = np.clip(y_coords + dy, 0, h-1)
                x_displaced = np.clip(x_coords + dx, 0, w-1)
                
                # Interpolate
                result = ndimage.map_coordinates(x_np, [y_displaced, x_displaced], order=1)
                
                # Convert back to tensor if needed
                if was_tensor:
                    return torch.from_numpy(result).unsqueeze(0).unsqueeze(0).to(x.device)
                else:
                    return result
        
        return FallbackElastic()
    
    def apply_optical_augmentation(self, image: np.ndarray, augmentation_config: Dict[str, Any]) -> np.ndarray:
        """
        Apply optical imaging augmentation based on configuration.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (2D slice)
        augmentation_config : dict
            Configuration dictionary specifying augmentations to apply
        
        Returns
        -------
        np.ndarray
            Augmented image
        """
        if np.all(image == 0) or np.std(image) < 1e-6:
            return image
        
        # Normalize input to [0,1] range
        original_min = np.min(image)
        original_max = np.max(image)
        if original_max > original_min:
            normalized_image = (image - original_min) / (original_max - original_min)
        else:
            return image
        
        result = normalized_image.copy()
        
        try:
            # Convert to tensor if cornucopia is available
            if CORNUCOPIA_AVAILABLE:
                tensor_img = torch.from_numpy(result).float().unsqueeze(0).unsqueeze(0).to(self.device)
            else:
                tensor_img = result
            
            # Apply augmentations in order
            if 'spatial' in augmentation_config:
                spatial_config = augmentation_config['spatial']
                transform_type = spatial_config.get('type', 'elastic_medical')
                if transform_type in self.spatial_transforms:
                    tensor_img = self.spatial_transforms[transform_type](tensor_img)
            
            if 'intensity' in augmentation_config:
                intensity_config = augmentation_config['intensity']
                transform_type = intensity_config.get('type', 'multiplicative_field')
                if transform_type in self.intensity_transforms:
                    # Create transform with custom parameters if provided
                    if transform_type == 'multiplicative_field' and 'strength_range' in intensity_config:
                        custom_transform = self._create_custom_multiplicative_field(
                            strength_range=intensity_config.get('strength_range', (0.06, 0.12)),
                            prob=intensity_config.get('prob', 0.75)
                        )
                        tensor_img = custom_transform(tensor_img)
                    else:
                        tensor_img = self.intensity_transforms[transform_type](tensor_img)
            
            if 'noise' in augmentation_config:
                noise_config = augmentation_config['noise']
                noise_type = noise_config.get('type', 'gamma_speckle')
                if noise_type in self.noise_transforms:
                    # Create transform with custom parameters if provided
                    if noise_type == 'gamma_multiplicative' and 'scale_range' in noise_config:
                        custom_transform = self._create_custom_gamma_multiplicative(
                            scale_range=noise_config.get('scale_range', (0.02, 0.06)),
                            prob=noise_config.get('prob', 0.7)
                        )
                        tensor_img = custom_transform(tensor_img)
                    elif noise_type == 'gamma_speckle' and 'intensity_range' in noise_config:
                        custom_transform = self._create_custom_gamma_speckle(
                            intensity_range=noise_config.get('intensity_range', (0.9, 1.05)),
                            prob=noise_config.get('prob', 0.8)
                        )
                        tensor_img = custom_transform(tensor_img)
                    else:
                        tensor_img = self.noise_transforms[noise_type](tensor_img)
            
            if 'debris' in augmentation_config:
                debris_config = augmentation_config['debris']
                debris_type = debris_config.get('type', 'random_shapes')
                if debris_type in self.debris_transforms:
                    # Create transform with custom parameters if provided
                    if debris_type == 'random_shapes' and 'n_shapes' in debris_config:
                        custom_transform = self._create_custom_random_shapes_debris(
                            n_shapes=debris_config.get('n_shapes', (1, 3)),
                            size_range=debris_config.get('size_range', (4, 10)),
                            intensity_range=debris_config.get('intensity_range', (0.15, 0.35)),
                            prob=debris_config.get('prob', 0.5)
                        )
                        tensor_img = custom_transform(tensor_img)
                    else:
                        tensor_img = self.debris_transforms[debris_type](tensor_img)
                    tensor_img = self.debris_transforms[debris_type](tensor_img)
            
            # Convert back to numpy
            if CORNUCOPIA_AVAILABLE and hasattr(tensor_img, 'cpu'):
                result = tensor_img.squeeze().cpu().numpy()
            else:
                result = tensor_img
            
            # Denormalize
            result = result * (original_max - original_min) + original_min
            return np.clip(result, original_min, original_max).astype(image.dtype)
            
        except Exception as e:
            print(f"   ️  Optical augmentation failed: {e}, returning original")
            return image


def create_optical_presets():
    """
    Create optical imaging augmentation presets.
    
    Returns
    -------
    dict
        Dictionary of preset configurations
    """
    return {
        'gamma_speckle': {
            'noise': {'type': 'gamma_speckle', 'intensity_range': (0.9995, 1.0005), 'prob': 0.3},  # Like subtle_debris level
            'intensity': {'type': 'smooth_multiplicative'}
        },
        
        'optical_with_debris': {
            'noise': {'type': 'gamma_multiplicative', 'scale_range': (0.0005, 0.0005), 'prob': 0.2},  # Same as subtle_debris
            'intensity': {'type': 'smooth_multiplicative'}
        },
        
        'heavy_speckle': {
            'noise': {'type': 'gamma_multiplicative', 'scale_range': (0.0008, 0.0008), 'prob': 0.25},  # Just slightly more than subtle_debris
            'intensity': {'type': 'smooth_multiplicative'}  # Change to smooth like others
        },
        
        'clean_optical': {
            'intensity': {'type': 'smooth_multiplicative'}  # Keep this one completely clean
        },
        
        'subtle_debris': {
            'noise': {'type': 'gamma_multiplicative', 'scale_range': (0.0005, 0.0005), 'prob': 0.2},  # Keep this as is - it works!
            'intensity': {'type': 'smooth_multiplicative'}
        },
        
        # Compatibility presets with old cornucopia_augmentation module
        'clinical_simulation': {
            'intensity': {'type': 'smooth_multiplicative'}  # Keep this completely clean
        }
    }


def augment_fiber_slice(slice_data: np.ndarray, 
                       preset: str = 'clean_optical',
                       custom_config: Optional[Dict[str, Any]] = None,
                       random_state: Optional[int] = None) -> np.ndarray:
    """
    Convenient function to augment a single slice with optical imaging effects.
    
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
    # Print debug information about Cornucopia settings
    print(f" Applying Cornucopia augmentation:")
    print(f"    Preset: '{preset}'")
    print(f"    Cornucopia available: {CORNUCOPIA_AVAILABLE}")
    if custom_config:
        print(f"   ️  Custom config: {custom_config}")
    
    augmenter = ImprovedCornucopiaAugmenter(random_state=random_state)
    
    if custom_config is not None:
        config = custom_config
        print(f"    Using custom configuration")
    else:
        presets = create_optical_presets()
        if preset in presets:
            config = presets[preset]
            print(f"    Using preset '{preset}': {config}")
        else:
            available_presets = list(presets.keys())
            print(f"   ️  Unknown preset '{preset}', available: {available_presets}")
            config = presets['clean_optical']
            print(f"    Falling back to 'clean_optical': {config}")
    
    result = augmenter.apply_optical_augmentation(slice_data, config)
    
    if np.array_equal(result, slice_data):
        print(f"   ️  Cornucopia had no effect (result identical to input)")
    else:
        print(f"    Cornucopia augmentation applied successfully")
    
    return result 