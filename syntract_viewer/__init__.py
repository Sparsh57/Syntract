"""
NIfTI Tractography Visualization Package

A comprehensive package for visualizing NIfTI images with overlaid tractography data,
featuring dark field microscopy-style visualization and advanced augmentation capabilities.
"""

from .core import (
    visualize_nifti_with_trk,
    visualize_nifti_with_trk_coronal,
    visualize_multiple_views
)

from .generation import (
    generate_varied_examples,
    generate_enhanced_varied_examples
)

from .masking import (
    create_fiber_mask,
    create_smart_brain_mask
)

from .contrast import (
    apply_contrast_enhancement,
    apply_enhanced_contrast_and_augmentation,
    apply_comprehensive_slice_processing
)

from .effects import (
    apply_dark_field_effect,
    apply_smart_dark_field_effect,
    apply_conservative_dark_field_effect
)

from .utils import (
    select_random_streamlines,
    densify_streamline,
    generate_tract_color_variation,
    get_colormap
)

# Try to import background enhancement functionality
try:
    from .background_enhancement import (
        enhance_slice_background,
        enhance_background_smoothness,
        apply_smart_sharpening,
        create_enhancement_presets
    )
    BACKGROUND_ENHANCEMENT_AVAILABLE = True
except ImportError:
    BACKGROUND_ENHANCEMENT_AVAILABLE = False

# Try to import Cornucopia functionality
try:
    from .cornucopia_augmentation import (
        CornucopiaAugmenter,
        create_augmentation_presets,
        augment_fiber_slice
    )
    CORNUCOPIA_INTEGRATION_AVAILABLE = True
except ImportError:
    CORNUCOPIA_INTEGRATION_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "Sparsh Makharia, LINC Team"
__license__ = "MIT"

__all__ = [
    # Core visualization
    'visualize_nifti_with_trk',
    'visualize_nifti_with_trk_coronal',
    'visualize_multiple_views',
    
    # Generation
    'generate_varied_examples',
    'generate_enhanced_varied_examples',
    
    # Masking
    'create_fiber_mask',
    'create_smart_brain_mask',
    
    # Contrast
    'apply_contrast_enhancement',
    'apply_enhanced_contrast_and_augmentation',
    'apply_comprehensive_slice_processing',
    
    # Effects
    'apply_dark_field_effect',
    'apply_smart_dark_field_effect',
    'apply_conservative_dark_field_effect',
    
    # Utils
    'select_random_streamlines',
    'densify_streamline',
    'generate_tract_color_variation',
    'get_colormap',
    
    # Constants
    'BACKGROUND_ENHANCEMENT_AVAILABLE',
    'CORNUCOPIA_INTEGRATION_AVAILABLE'
]

# Add background enhancement exports if available
if BACKGROUND_ENHANCEMENT_AVAILABLE:
    __all__.extend([
        'enhance_slice_background',
        'enhance_background_smoothness',
        'apply_smart_sharpening',
        'create_enhancement_presets'
    ])

# Add Cornucopia exports if available
if CORNUCOPIA_INTEGRATION_AVAILABLE:
    __all__.extend([
        'CornucopiaAugmenter',
        'create_augmentation_presets',
        'augment_fiber_slice'
    ]) 