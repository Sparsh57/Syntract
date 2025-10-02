"""
Example generation functions for creating synthetic fiber tract datasets.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.streamlines import load
from dipy.tracking.streamline import transform_streamlines
import random
from pathlib import Path

try:
    from .core import visualize_nifti_with_trk, visualize_nifti_with_trk_coronal, visualize_multiple_views
    from .contrast import apply_enhanced_contrast_and_augmentation, CORNUCOPIA_INTEGRATION_AVAILABLE
    from .masking import create_aggressive_brain_mask, create_fiber_mask
    from .effects import apply_balanced_dark_field_effect, apply_blockface_preserving_dark_field_effect
    from .utils import select_random_streamlines, densify_streamline, generate_tract_color_variation, get_colormap
    from .orange_blob_generator import apply_orange_artifacts
except ImportError:
    from core import visualize_nifti_with_trk, visualize_nifti_with_trk_coronal, visualize_multiple_views
    from contrast import apply_enhanced_contrast_and_augmentation, CORNUCOPIA_INTEGRATION_AVAILABLE
    from masking import create_aggressive_brain_mask, create_fiber_mask
    from effects import apply_balanced_dark_field_effect, apply_blockface_preserving_dark_field_effect
    from utils import select_random_streamlines, densify_streamline, generate_tract_color_variation, get_colormap
    from orange_blob_generator import apply_orange_artifacts


def generate_varied_examples(nifti_file, trk_file, output_dir, n_examples=5, prefix="synthetic_", 
                             slice_mode="coronal", intensity_variation=True, tract_color_variation=True,
                             specific_slice=None, streamline_percentage=100.0, roi_sphere=None,
                             tract_linewidth=1.0, save_masks=False, mask_thickness=1, 
                             min_fiber_percentage=10.0, max_fiber_percentage=100.0,
                             density_threshold=0.15, gaussian_sigma=2.0, random_state=None,
                             close_gaps=False, closing_footprint_size=5, label_bundles=False,
                             min_bundle_size=20, use_high_density_masks=False,
                             contrast_method='clahe', contrast_params=None, 
                             enable_orange_blobs=False, orange_blob_probability=0.3):
    """
    Generate multiple varied examples with different contrast settings.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    if contrast_params is None:
        contrast_params = {
            'clip_limit': 0.01,
            'tile_grid_size': (8, 8)
        }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine slice index if specific_slice is provided
    if specific_slice is not None:
        nii_img = nib.load(nifti_file)
        dims = nii_img.shape
        
        if slice_mode == "axial" and (specific_slice < 0 or specific_slice >= dims[2]):
            print(f"Warning: Slice index {specific_slice} out of bounds for axial dimension {dims[2]}")
            specific_slice = dims[2] // 2
        elif slice_mode == "coronal" and (specific_slice < 0 or specific_slice >= dims[1]):
            print(f"Warning: Slice index {specific_slice} out of bounds for coronal dimension {dims[1]}")
            specific_slice = dims[1] // 2
    
    # Use fiber density range if provided
    use_fiber_range = (min_fiber_percentage > 0 and max_fiber_percentage > min_fiber_percentage)
    
    # Generate high-density masks if requested
    high_density_masks = {}
    high_density_labeled_masks = {}
    
    if use_high_density_masks and save_masks:
        print(f"Generating high-density masks using {max_fiber_percentage}% of fibers...")
        _generate_high_density_masks(
            nifti_file, trk_file, output_dir, prefix, slice_mode, specific_slice,
            max_fiber_percentage, roi_sphere, tract_linewidth, mask_thickness,
            density_threshold, gaussian_sigma, close_gaps, closing_footprint_size,
            label_bundles, min_bundle_size, contrast_method, contrast_params,
            high_density_masks, high_density_labeled_masks, random_state
        )
    
    # Generate examples
    for i in range(n_examples):
        example_random_state = None
        if random_state is not None:
            example_random_state = random_state + i
        
        # Generate random parameters
        if intensity_variation:
            if i % 3 == 0:
                color_scheme = 'bw'
                blue_tint = 0
            elif i % 3 == 1:
                color_scheme = 'blue'
                blue_tint = random.uniform(0.2, 0.4)
            else:
                color_scheme = random.choices(['bw', 'blue'], weights=[0.3, 0.7])[0]
                blue_tint = random.uniform(0.1, 0.4) if color_scheme == 'blue' else 0
            
            intensity_params = {
                'gamma': random.uniform(0.8, 1.2),
                'threshold': random.uniform(0.02, 0.08),
                'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
                'background_boost': random.uniform(0.9, 1.1),
                'color_scheme': color_scheme,
                'blue_tint': blue_tint
            }
        else:
            intensity_params = None
            
        # Generate random tract color (saturated orange-yellow for realism)
        if tract_color_variation:
            base_r = random.uniform(0.95, 1.0)  # High red for orange-yellow
            base_g = random.uniform(0.7, 0.85)  # Medium-high green for orange-yellow
            base_b = random.uniform(0.0, 0.15)  # Low blue for warmth
            tract_color_base = (base_r, base_g, base_b)
            color_var = random.uniform(0.05, 0.2)
        else:
            tract_color_base = (1.0, 0.8, 0.1)  # Saturated orange-yellow instead of pure yellow
            color_var = 0.0
            
        output_file = os.path.join(output_dir, f"{prefix}{i+1:03d}.png")
        
        # Vary fiber density if range is specified
        if use_fiber_range:
            if n_examples > 1:
                t = i / (n_examples - 1)
                fiber_pct = min_fiber_percentage + t * (max_fiber_percentage - min_fiber_percentage)
            else:
                fiber_pct = (min_fiber_percentage + max_fiber_percentage) / 2
            print(f"Example {i+1}: Using {fiber_pct:.1f}% of fibers")
        else:
            fiber_pct = streamline_percentage
        
        # Determine whether to save masks
        should_save_masks = save_masks and (not use_high_density_masks or fiber_pct >= max_fiber_percentage)
        
        # Generate visualization
        if slice_mode == "axial":
            visualize_nifti_with_trk(
                nifti_file, trk_file, output_file, n_slices=1,
                intensity_params=intensity_params,
                tract_color_base=tract_color_base, 
                tract_color_variation=color_var,
                slice_idx=specific_slice,
                streamline_percentage=fiber_pct,
                roi_sphere=roi_sphere,
                tract_linewidth=tract_linewidth,
                save_masks=should_save_masks,
                mask_thickness=mask_thickness,
                density_threshold=density_threshold,
                gaussian_sigma=gaussian_sigma,
                random_state=example_random_state,
                close_gaps=close_gaps,
                closing_footprint_size=closing_footprint_size,
                label_bundles=label_bundles,
                min_bundle_size=min_bundle_size,
                contrast_method=contrast_method,
                contrast_params=contrast_params
            )
            
            # Apply high-density masks if requested
            if use_high_density_masks and save_masks and fiber_pct < max_fiber_percentage:
                _apply_high_density_masks_axial(
                    output_file, high_density_masks, high_density_labeled_masks,
                    specific_slice, label_bundles, nifti_file
                )
            
        elif slice_mode == "coronal":
            visualize_nifti_with_trk_coronal(
                nifti_file, trk_file, output_file, n_slices=1,
                intensity_params=intensity_params,
                tract_color_base=tract_color_base, 
                tract_color_variation=color_var,
                slice_idx=specific_slice,
                streamline_percentage=fiber_pct,
                roi_sphere=roi_sphere,
                tract_linewidth=tract_linewidth,
                save_masks=should_save_masks,
                mask_thickness=mask_thickness,
                density_threshold=density_threshold,
                gaussian_sigma=gaussian_sigma,
                random_state=example_random_state,
                close_gaps=close_gaps,
                closing_footprint_size=closing_footprint_size,
                label_bundles=label_bundles,
                min_bundle_size=min_bundle_size,
                contrast_method=contrast_method,
                contrast_params=contrast_params
            )
            
            # Apply high-density masks if requested
            if use_high_density_masks and save_masks and fiber_pct < max_fiber_percentage:
                _apply_high_density_masks_coronal(
                    output_file, high_density_masks, high_density_labeled_masks,
                    specific_slice, label_bundles, nifti_file
                )
            
        else:  # "all"
            visualize_multiple_views(
                nifti_file, trk_file, output_file,
                intensity_params=intensity_params,
                tract_color_base=tract_color_base, 
                tract_color_variation=color_var,
                streamline_percentage=fiber_pct,
                roi_sphere=roi_sphere,
                tract_linewidth=tract_linewidth,
                save_masks=should_save_masks,
                mask_thickness=mask_thickness,
                density_threshold=density_threshold,
                gaussian_sigma=gaussian_sigma,
                random_state=example_random_state,
                close_gaps=close_gaps,
                closing_footprint_size=closing_footprint_size,
                label_bundles=label_bundles,
                min_bundle_size=min_bundle_size,
                contrast_method=contrast_method,
                contrast_params=contrast_params
            )
            
            # Apply high-density masks if requested
            if use_high_density_masks and save_masks and fiber_pct < max_fiber_percentage:
                _apply_high_density_masks_multiview(
                    output_file, high_density_masks, high_density_labeled_masks, label_bundles
                )
        
        print(f"Generated example {i+1}/{n_examples}: {output_file}")
        
        # Apply orange blobs if enabled
        if enable_orange_blobs and random.random() < orange_blob_probability:
            apply_orange_blobs_to_saved_image(output_file, random_state=example_random_state)
            print(f"Applied orange blobs to example {i+1}")


def apply_orange_blobs_to_saved_image(image_path, random_state=None):
    """
    Apply orange blobs to an already saved image file.
    
    Parameters:
    -----------
    image_path : str
        Path to the saved image file
    random_state : int, optional
        Random seed for reproducible blob generation
    """
    try:
        from PIL import Image
        import numpy as np
        
        # Load the saved image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Apply orange artifacts
        orange_image, artifact_mask = apply_orange_artifacts(
            image_array, 
            enable=True,
            num_sites=random.randint(1, 2) if random_state is None else None,
            random_state=random_state
        )
        
        # Save the modified image
        if orange_image.dtype != np.uint8:
            orange_image = (orange_image * 255).astype(np.uint8)
        
        orange_pil = Image.fromarray(orange_image)
        orange_pil.save(image_path)
        
    except Exception as e:
        print(f"Warning: Failed to apply orange blobs to {image_path}: {e}")


def generate_enhanced_varied_examples(nifti_file, trk_file, output_dir, 
                                    n_examples=5, prefix="enhanced_",
                                    slice_mode="coronal", specific_slice=None,
                                    streamline_percentage=100.0, roi_sphere=None,
                                    tract_linewidth=1.0, save_masks=False,
                                    min_fiber_percentage=10.0, max_fiber_percentage=100.0,
                                    contrast_method='clahe', contrast_params=None,
                                    cornucopia_preset=None,
                                    background_preset='preserve_edges',
                                    enable_sharpening=True,
                                    sharpening_strength=0.5,
                                    use_cornucopia_per_example=False,
                                    use_background_enhancement=True,
                                    close_gaps=False, closing_footprint_size=5,
                                    randomize=False,
                                    random_state=None, 
                                    enable_orange_blobs=False, orange_blob_probability=0.3,
                                    **kwargs):
    """
    Generate varied examples with enhanced augmentations using Cornucopia and background enhancement.
    
    Parameters
    ----------
    randomize : bool
        If True, randomize min/max streamline percentages, streamline appearance,
        cornucopia preset (present or not, which one), and background effect type
        for each example
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set up base augmentation configuration
    augmentation_config = None
    if use_cornucopia_per_example and cornucopia_preset and CORNUCOPIA_INTEGRATION_AVAILABLE and not randomize:
        try:
            try:
                from .cornucopia_augmentation import create_augmentation_presets
            except ImportError:
                from cornucopia_augmentation import create_augmentation_presets

            presets = create_augmentation_presets()
            if cornucopia_preset in presets:
                augmentation_config = presets[cornucopia_preset]
            else:
                print(f"Warning: Unknown Cornucopia preset '{cornucopia_preset}', using 'clinical_simulation'")
                augmentation_config = presets.get('clinical_simulation', None)
        except Exception as e:
            print(f"Warning: Failed to create augmentation config: {e}")
            augmentation_config = None

    # Set up base background enhancement configuration
    background_config = None
    if use_background_enhancement and not randomize:
        try:
            try:
                from .background_enhancement import create_enhancement_presets
            except ImportError:
                from background_enhancement import create_enhancement_presets

            presets = create_enhancement_presets()
            if background_preset in presets:
                background_config = background_preset
            else:
                print(f"Warning: Unknown background preset '{background_preset}', using 'smooth_realistic'")
                background_config = 'smooth_realistic'
        except Exception as e:
            print(f"Warning: Failed to create background config: {e}")
            background_config = None

    if random_state is not None:
        np.random.seed(random_state)

    if contrast_params is None:
        contrast_params = {
            'clip_limit': 0.01,
            'tile_grid_size': (8, 8)
        }

    print(f" Generating {n_examples} enhanced examples with advanced processing")
    print(f"   Randomize mode: {randomize}")
    if not randomize:
        print(f"   Contrast method: {contrast_method}")
        print(f"   Background enhancement: {background_config if background_config else 'disabled'}")
        print(f"   Cornucopia preset: {cornucopia_preset if augmentation_config else 'disabled'}")
    else:
        print(f"   Parameters will be randomized for each example")
    print(f"   Cornucopia available: {CORNUCOPIA_INTEGRATION_AVAILABLE}")

    # Generate enhanced examples
    if randomize or \
       (use_cornucopia_per_example and CORNUCOPIA_INTEGRATION_AVAILABLE and augmentation_config is not None) or \
       (use_background_enhancement and background_config is not None):
        print(" Generating examples with comprehensive slice processing...")

        base_results = _generate_examples_with_comprehensive_processing(
            nifti_file=nifti_file,
            trk_file=trk_file,
            output_dir=output_dir,
            n_examples=n_examples,
            prefix=prefix,
            slice_mode=slice_mode,
            specific_slice=specific_slice,
            streamline_percentage=streamline_percentage,
            roi_sphere=roi_sphere,
            tract_linewidth=tract_linewidth,
            save_masks=save_masks,
            min_fiber_percentage=min_fiber_percentage,
            max_fiber_percentage=max_fiber_percentage,
            contrast_method=contrast_method,
            contrast_params=contrast_params,
            enable_sharpening=enable_sharpening,
            sharpening_strength=sharpening_strength,
            cornucopia_config=augmentation_config,
            background_config=background_config,
            close_gaps=close_gaps,
            closing_footprint_size=closing_footprint_size,
            randomize=randomize,
            random_state=random_state,
            **kwargs
        )
    else:
        # Generate standard examples
        base_results = generate_varied_examples(
            nifti_file=nifti_file,
            trk_file=trk_file,
            output_dir=output_dir,
            n_examples=n_examples,
            prefix=prefix,
            slice_mode=slice_mode,
            specific_slice=specific_slice,
            streamline_percentage=streamline_percentage,
            roi_sphere=roi_sphere,
            tract_linewidth=tract_linewidth,
            save_masks=save_masks,
            min_fiber_percentage=min_fiber_percentage,
            max_fiber_percentage=max_fiber_percentage,
            contrast_method=contrast_method,
            contrast_params=contrast_params,
            enable_orange_blobs=enable_orange_blobs,
            orange_blob_probability=orange_blob_probability,
            random_state=random_state,
            **kwargs
        )

    # Create summary report
    summary = {
        'n_examples_generated': n_examples,
        'randomize_mode': randomize,
        'background_enhancement_available': background_config is not None,
        'background_preset_used': background_preset if background_config else None,
        'cornucopia_available': CORNUCOPIA_INTEGRATION_AVAILABLE,
        'cornucopia_preset_used': cornucopia_preset if use_cornucopia_per_example else None,
        'contrast_method': contrast_method,
        'output_directory': str(output_dir),
        'base_results': base_results
    }

    return summary


def _generate_examples_with_comprehensive_processing(nifti_file, trk_file, output_dir,
                                             n_examples=5, prefix="enhanced_",
                                             slice_mode="coronal", specific_slice=None,
                                             streamline_percentage=100.0, roi_sphere=None,
                                             tract_linewidth=1.0, save_masks=False,
                                             min_fiber_percentage=10.0, max_fiber_percentage=100.0,
                                             enable_sharpening=True, sharpening_strength=0.5,
                                             contrast_method='clahe', contrast_params=None,
                                             cornucopia_config=None, background_config=None,
                                             close_gaps=False, closing_footprint_size=5,
                                             randomize=False, random_state=None, 
                                             enable_orange_blobs=False, orange_blob_probability=0.3, **kwargs):
    """Generate examples with comprehensive processing including background enhancement and Cornucopia."""
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    nii_img = nib.load(nifti_file)
    nii_data = nii_img.get_fdata()
    dims = nii_data.shape

    tractogram = load(trk_file)
    streamlines = tractogram.streamlines
    print(f"Loaded {len(streamlines)} streamlines from {trk_file}")

    affine_inv = np.linalg.inv(nii_img.affine)
    streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))

    # Determine slice index
    if specific_slice is not None:
        slice_idx = specific_slice
    else:
        if slice_mode == "coronal":
            slice_idx = dims[1] // 2
        elif slice_mode == "axial":
            slice_idx = dims[2] // 2
        else:  # sagittal
            slice_idx = dims[0] // 2

    if contrast_params is None:
        contrast_params = {'clip_limit': 0.01, 'tile_grid_size': (8, 8)}

    # Prepare available options for randomization
    if randomize:
        # Available cornucopia presets (including None for no cornucopia)
        cornucopia_options = [None, 'clinical_simulation']
        
        # Available background effects 
        background_effects = ['balanced', 'blockface_preserving']
        
        # Prepare cornucopia configs
        cornucopia_configs = {None: None}
        if CORNUCOPIA_INTEGRATION_AVAILABLE:
            try:
                try:
                    from .cornucopia_augmentation import create_augmentation_presets
                except ImportError:
                    from cornucopia_augmentation import create_augmentation_presets
                
                # Use truly random presets when randomizing
                presets = create_augmentation_presets(truly_random=True)
                cornucopia_configs.update(presets)
            except Exception as e:
                print(f"Warning: Failed to load cornucopia presets: {e}")

    # Generate examples
    for i in range(n_examples):
        example_random_state = None
        if random_state is not None and not randomize:
            # Only use fixed random state if randomization is disabled
            example_random_state = random_state + i
            random.seed(example_random_state)
            np.random.seed(example_random_state)
        elif randomize:
            # For true randomization, use current time + example index as seed
            import time
            true_random_seed = int(time.time() * 1000000) + i * 1000 + random.randint(1, 999)
            random.seed(true_random_seed)
            np.random.seed(true_random_seed)
            print(f"Example {i+1} using truly random seed: {true_random_seed}")

        # Randomize parameters if requested
        if randomize:
            # 4. Randomize background effect first (so we can adjust fiber percentage based on it)
            current_background_effect = random.choice(background_effects)
            
            # 1. Randomize min/max fiber percentages based on background effect
            if current_background_effect == 'blockface_preserving':
                # Higher fiber percentages for blockface preserving (better contrast against bright areas)
                random_min_fiber = random.uniform(15.0, 40.0)  # Higher min: 15-40% instead of 5-30%
                random_max_fiber = random.uniform(80.0, 100.0)  # Higher max: 80-100% instead of 70-100%
                print(f"   Using higher fiber percentages for blockface_preserving effect")
            else:  # 'balanced'
                # Standard fiber percentages for balanced effect
                random_min_fiber = random.uniform(5.0, 30.0)
                random_max_fiber = random.uniform(70.0, 100.0)
            
            # Calculate actual fiber percentage for this example
            if n_examples > 1:
                t = i / (n_examples - 1)
                fiber_pct = random_min_fiber + t * (random_max_fiber - random_min_fiber)
            else:
                fiber_pct = (random_min_fiber + random_max_fiber) / 2
            
            # 2. Randomize streamline appearance
            random_tract_linewidth = random.uniform(0.5, 1.5)
            
            # 3. Randomize cornucopia preset
            current_cornucopia_preset = random.choice(cornucopia_options)
            current_cornucopia_config = cornucopia_configs.get(current_cornucopia_preset, None)

            print(f"Example {i+1} randomized parameters:")
            print(f"   Background effect: {current_background_effect}")
            print(f"   Fiber range: {random_min_fiber:.1f}% - {random_max_fiber:.1f}% (using {fiber_pct:.1f}%)")
            print(f"   Tract linewidth: {random_tract_linewidth:.2f}")
            print(f"   Cornucopia: {current_cornucopia_preset}")
            
        else:
            # Use base parameters
            if n_examples > 1:
                t = i / (n_examples - 1)
                fiber_pct = min_fiber_percentage + t * (max_fiber_percentage - min_fiber_percentage)
            else:
                fiber_pct = (min_fiber_percentage + max_fiber_percentage) / 2
            
            random_tract_linewidth = tract_linewidth
            current_cornucopia_config = cornucopia_config
            current_background_effect = 'balanced'  # default
            
            print(f"Example {i+1}: Using {fiber_pct:.1f}% of fibers")

        # Select streamlines
        selected_streamlines = select_random_streamlines(
            streamlines_voxel, fiber_pct, random_state=example_random_state
        )

        print(f"   Selected {len(selected_streamlines)} streamlines")

        # Extract slice data
        if slice_mode == "coronal":
            slice_data = nii_data[:, slice_idx, :]
        elif slice_mode == "axial":
            slice_data = nii_data[:, :, slice_idx]
        else:  # sagittal
            slice_data = nii_data[slice_idx, :, :]

        # Apply comprehensive slice processing
        try:
            from .contrast import apply_comprehensive_slice_processing
        except ImportError:
            from contrast import apply_comprehensive_slice_processing

        # Use randomized or base background config
        if randomize:
            # For randomized mode, always use background enhancement with preserve_edges preset
            current_background_config = 'preserve_edges'
        else:
            current_background_config = background_config

        enhanced_slice = apply_comprehensive_slice_processing(
            slice_data,
            background_preset=current_background_config,
            cornucopia_preset=current_cornucopia_preset if randomize else None,
            contrast_method=contrast_method,
            background_params=None,
            cornucopia_params=current_cornucopia_config if isinstance(current_cornucopia_config, dict) else None,
            contrast_params=contrast_params,
            enable_sharpening=kwargs.get('enable_sharpening', False),
            sharpening_strength=kwargs.get('sharpening_strength', 0.5),
            random_state=example_random_state
        )
        
        # Safety check
        if np.all(enhanced_slice == 0) or np.std(enhanced_slice) < 1e-6:
            print(f"     Enhanced slice is empty, using original slice data")
            enhanced_slice = slice_data.copy()
            try:
                from .contrast import apply_contrast_enhancement
            except ImportError:
                from contrast import apply_contrast_enhancement
            enhanced_slice = apply_contrast_enhancement(
                enhanced_slice, 
                clip_limit=contrast_params.get('clip_limit', 0.01),
                tile_grid_size=contrast_params.get('tile_grid_size', (8, 8))
            )
        
        # Additional background cleanup for Cornucopia artifacts
        if (randomize and current_cornucopia_preset is not None) or \
           (not randomize and cornucopia_config is not None):
            # Make background cleanup optional and less aggressive
            apply_background_cleanup = kwargs.get('apply_background_cleanup', False)
            if apply_background_cleanup:
                brain_mask = create_aggressive_brain_mask(slice_data, enhanced_slice)
                background_areas = ~brain_mask.astype(bool)
                
                # Instead of forcing to pure black, just dim the background
                background_dimming_factor = kwargs.get('background_dimming_factor', 0.4)
                enhanced_slice[background_areas] *= background_dimming_factor
        
        # Generate visualization with randomized background effect
        # Remove dims from kwargs to avoid conflict with positional parameter
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'dims'}
        _create_enhanced_visualization(
            enhanced_slice, selected_streamlines, slice_mode, slice_idx, dims,
            output_dir, prefix, i, save_masks, 
            current_cornucopia_config if randomize else cornucopia_config, 
            slice_data, random_tract_linewidth, example_random_state, 
            background_effect=current_background_effect if randomize else 'balanced',
            enable_orange_blobs=enable_orange_blobs, orange_blob_probability=orange_blob_probability,
            close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
            **filtered_kwargs
        )
    
    return {'examples_generated': n_examples}


def _create_enhanced_visualization(enhanced_slice, selected_streamlines, slice_mode, slice_idx, dims,
                                 output_dir, prefix, example_idx, save_masks, cornucopia_config, slice_data,
                                 tract_linewidth, example_random_state, 
                                 background_effect='balanced', enable_orange_blobs=False, orange_blob_probability=0.3, **kwargs):
    """Create enhanced visualization with streamlines."""
    # Generate intensity parameters
    intensity_params = {
        'gamma': random.uniform(0.8, 1.2),
        'threshold': random.uniform(0.02, 0.08),
        'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
        'background_boost': random.uniform(0.9, 1.1),
        'color_scheme': random.choice(['bw', 'blue']),
        'blue_tint': random.uniform(0.1, 0.4)
    }

    # Apply the specified background effect
    if background_effect == 'blockface_preserving':
        dark_field_slice = apply_blockface_preserving_dark_field_effect(
            enhanced_slice,
            intensity_params,
            random_state=example_random_state,
            force_background_black=True
        )
    else:  # 'balanced' is default
        dark_field_slice = apply_balanced_dark_field_effect(
            enhanced_slice,
            intensity_params,
            random_state=example_random_state,
            force_background_black=True
        )
    
    # Additional background cleanup for Cornucopia artifacts
    if cornucopia_config is not None:
        # Make background cleanup optional and less aggressive
        apply_background_cleanup = kwargs.get('apply_background_cleanup', False)
        if apply_background_cleanup:
            brain_mask = create_aggressive_brain_mask(slice_data, enhanced_slice)
            background_areas = ~brain_mask.astype(bool)
            
            # Instead of forcing to pure black, just dim the background
            background_dimming_factor = kwargs.get('background_dimming_factor', 0.4)
            dark_field_slice[background_areas] *= background_dimming_factor
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))  # Increased for better 1024x1024 output
    fig.patch.set_facecolor('black')
    
    # Get colormap
    color_scheme = intensity_params.get('color_scheme', 'bw')
    blue_tint = intensity_params.get('blue_tint', 0.3)
    dark_field_cmap = get_colormap(color_scheme, blue_tint)
    
    # Display brain slice with pure black background
    brain_min = 0.0  # Force minimum to pure black
    brain_max = np.max(dark_field_slice) if np.any(dark_field_slice > 0) else 1.0

    ax.imshow(np.rot90(dark_field_slice), cmap=dark_field_cmap, aspect='equal',
             interpolation='bicubic', vmin=brain_min, vmax=brain_max)
    ax.set_facecolor('black')
    
    # Create mask if requested
    if save_masks:
        label_bundles = kwargs.get('label_bundles', False)
        if label_bundles:
            mask, labeled_mask = create_fiber_mask(
                selected_streamlines, slice_idx, orientation=slice_mode,
                dims=dims, thickness=kwargs.get('mask_thickness', 1),
                density_threshold=kwargs.get('density_threshold', 0.15),
                gaussian_sigma=kwargs.get('gaussian_sigma', 2.0),
                close_gaps=kwargs.get('close_gaps', False),
                closing_footprint_size=kwargs.get('closing_footprint_size', 5),
                label_bundles=True,
                min_bundle_size=kwargs.get('min_bundle_size', 20)
            )
            mask = np.rot90(mask)
            labeled_mask = np.rot90(labeled_mask)
        else:
            mask = create_fiber_mask(
                selected_streamlines, slice_idx, orientation=slice_mode,
                dims=dims, thickness=kwargs.get('mask_thickness', 1),
                density_threshold=kwargs.get('density_threshold', 0.15),
                gaussian_sigma=kwargs.get('gaussian_sigma', 2.0),
                close_gaps=kwargs.get('close_gaps', False),
                closing_footprint_size=kwargs.get('closing_footprint_size', 5),
                min_bundle_size=kwargs.get('min_bundle_size', 20)
            )
            mask = np.rot90(mask)
    
    # Overlay streamlines
    _add_streamlines_to_plot(ax, selected_streamlines, slice_mode, slice_idx, dims, tract_linewidth, example_random_state, background_effect)
    
    # Add orange injection site streamlines if enabled - EXTREME VISIBILITY VERSION
    if enable_orange_blobs:  # Always add when enabled, ignore probability for now
        print(f"GENERATING UNMISSABLE INJECTION SITE")
        import random as rnd
        
        # Get image dimensions
        height, width = dims[1], dims[0]  # Note: dims might be (width, height)
        if slice_mode == 'coronal':
            height, width = dims[2], dims[0]
        elif slice_mode == 'sagittal':
            height, width = dims[2], dims[1]
        
        print(f"Image dimensions: {width} x {height}")
        
        # Create injection site at random location
        margin = int(min(width, height) * 0.1)  # Keep some margin from edges
        center_x = rnd.randint(margin, width - margin)
        center_y = rnd.randint(margin, height - margin)
        injection_radius = min(width, height) * 0.05  # Slightly larger injection area
        
        print(f"Injection center: ({center_x:.0f}, {center_y:.0f}), radius: {injection_radius:.0f}")
        
        # Number of orange streamlines to generate
        num_orange_streamlines = rnd.randint(300, 500)
        
        # Simple orange circle
        circle = plt.Circle((center_x, center_y), injection_radius, color='orange', alpha=0.6, zorder=25)
        ax.add_patch(circle)
        for i in range(num_orange_streamlines):
            # Random start point within injection area
            angle = rnd.uniform(0, 2*np.pi)
            radius = rnd.uniform(0, injection_radius)
            start_x = center_x + radius * np.cos(angle)
            start_y = center_y + radius * np.sin(angle)
            
            # Generate curved orange streamline - much shorter for small area
            streamline_length = rnd.randint(20, 30)  # Slightly longer streamlines
            x_coords = [start_x]
            y_coords = [start_y]
            
            # Direction radiating outward with some randomness
            direction_x = np.cos(angle) + rnd.gauss(0, 0.3)
            direction_y = np.sin(angle) + rnd.gauss(0, 0.3)
            
            current_x, current_y = start_x, start_y
            
            # Add curvature parameters for natural fiber appearance
            curve_amount = rnd.uniform(0.1, 0.5)  # How much to curve
            curve_frequency = rnd.uniform(0.05, 0.2)  # How often to change direction
            
            for step in range(streamline_length):
                # Add progressive curvature and noise for realistic brain fiber appearance
                step_size = rnd.uniform(0.5, 1.0)  # Much smaller steps
                
                # Add smooth curvature
                curve_offset_x = curve_amount * np.sin(step * curve_frequency) * rnd.uniform(0.5, 1.5)
                curve_offset_y = curve_amount * np.cos(step * curve_frequency) * rnd.uniform(0.5, 1.5)
                
                # Add random noise for natural variation
                noise_x = rnd.gauss(0, 0.3)
                noise_y = rnd.gauss(0, 0.3)
                
                # Update direction with curvature and noise
                direction_x += (curve_offset_x + noise_x) * 0.1
                direction_y += (curve_offset_y + noise_y) * 0.1
                
                # Normalize to prevent runaway
                direction_length = (direction_x**2 + direction_y**2)**0.5
                if direction_length > 0:
                    direction_x /= direction_length
                    direction_y /= direction_length
                
                current_x += direction_x * step_size
                current_y += direction_y * step_size
                
                if (current_x < 0 or current_x >= width or 
                    current_y < 0 or current_y >= height):
                    break
                    
                x_coords.append(current_x)
                y_coords.append(current_y)
            
            # Plot orange streamlines with natural appearance
            if len(x_coords) > 5:
                # Use more natural orange colors with lower brightness
                orange_colors = ['#CC5500', '#BB4400', '#DD6600', '#AA3300', '#EE7700']
                color = rnd.choice(orange_colors)
                ax.plot(x_coords, y_coords, color=color, linewidth=1.5, 
                       alpha=0.6, solid_capstyle='round', zorder=24)
        
        # Add text label for absolute certainty
        ax.text(center_x, center_y - injection_radius - 50, 'INJECTION SITE', 
                fontsize=20, color='#FF00FF', weight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                zorder=35)
        
        print(f"ADDED UNMISSABLE INJECTION SITE with {num_orange_streamlines} multicolor streamlines")
        print(f"Giant markers: MAGENTA, CYAN, YELLOW, RED at ({center_x:.0f}, {center_y:.0f})")
        print(f"12 radiating lines + circle outline + text label")
        print(f"If you don't see THIS, please check if the image file is corrupted!")
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save result
    output_file = os.path.join(output_dir, f"{prefix}{example_idx+1:03d}.png")
    # Import resize utility
    try:
        from .utils import save_image_1024
    except ImportError:
        from utils import save_image_1024
    save_image_1024(output_file, fig, is_mask=False)
    print(f"Generated example {example_idx+1}: {output_file} (1024x1024)")
    
    # Apply orange blobs if enabled
    enable_orange_blobs = kwargs.get('enable_orange_blobs', False)
    orange_blob_probability = kwargs.get('orange_blob_probability', 0.3)
    if enable_orange_blobs and random.random() < orange_blob_probability:
        apply_orange_blobs_to_saved_image(output_file, random_state=example_random_state)
        print(f"Applied orange blobs to example {example_idx+1}")
    
    # Save mask if requested
    if save_masks:
        mask_dir = os.path.dirname(output_file)
        mask_basename = os.path.splitext(os.path.basename(output_file))[0]
        mask_filename = f"{mask_dir}/{mask_basename}_mask_slice{slice_idx}.png"
        save_image_1024(mask_filename, mask, is_mask=True)
        print(f"Saved mask for slice {slice_idx} to {mask_filename} (1024x1024)")
        
        # Save labeled bundles if requested
        label_bundles = kwargs.get('label_bundles', False)
        if label_bundles and 'labeled_mask' in locals():
            try:
                from .utils import visualize_labeled_bundles
            except ImportError:
                from utils import visualize_labeled_bundles
            labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_slice{slice_idx}.png"
            visualize_labeled_bundles(labeled_mask, labeled_filename)
            print(f"Saved labeled bundles for slice {slice_idx} to {labeled_filename}")
    
    plt.close()


def _add_streamlines_to_plot(ax, streamlines, slice_mode, slice_idx, dims, tract_linewidth, random_state, background_effect='balanced'):
    """Add streamlines to the plot with opacity adjusted based on background effect."""
    from matplotlib.collections import LineCollection
    
    segments = []
    colors = []

    opacity_multiplier = 0.5
    
    for sl in streamlines:
        sl_dense = densify_streamline(sl)
        
        # Project streamline based on slice mode
        if slice_mode == "coronal":
            x = sl_dense[:, 0]
            y = sl_dense[:, 1]
            z = sl_dense[:, 2]
            distance_to_slice = np.abs(y - slice_idx)
            min_distance = np.min(distance_to_slice)
            if min_distance > 2.0:
                continue
            z_plot = dims[2] - z - 1
            points = np.array([x, z_plot]).T.reshape(-1, 1, 2)
        elif slice_mode == "axial":
            x = sl_dense[:, 0]
            y = sl_dense[:, 1]
            z = sl_dense[:, 2]
            distance_to_slice = np.abs(z - slice_idx)
            min_distance = np.min(distance_to_slice)
            if min_distance > 2.0:
                continue
            y_plot = dims[1] - y - 1
            points = np.array([x, y_plot]).T.reshape(-1, 1, 2)
        else:  # sagittal
            x = sl_dense[:, 0]
            y = sl_dense[:, 1]
            z = sl_dense[:, 2]
            distance_to_slice = np.abs(x - slice_idx)
            min_distance = np.min(distance_to_slice)
            if min_distance > 2.0:
                continue
            z_plot = dims[2] - z - 1
            points = np.array([y, z_plot]).T.reshape(-1, 1, 2)
        
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Generate varied color (saturated orange-yellow for realism with randomization)
        # For randomized mode, generate more varied colors
        base_r = random.uniform(0.8, 1.0)  # Allow more variation in red
        base_g = random.uniform(0.6, 0.9)  # Allow more variation in green  
        base_b = random.uniform(0.0, 0.2)  # Allow slight variation in blue
        tract_color_base = (base_r, base_g, base_b)
        
        tract_color = generate_tract_color_variation(
            tract_color_base, 0.3, random_state=random_state  # Increased variation
        )
        
        # Adjust opacity based on background effect and distance
        base_opacity = max(0.0, (1.0 - min_distance / 2.0) * 0.5)
        adjusted_opacity = min(1.0, base_opacity * opacity_multiplier)  # Cap at 1.0
        
        for seg in segs:
            segments.append(seg)
            colors.append(tract_color + (adjusted_opacity,))
    
    # Add streamlines to plot
    if segments:
        linewidth = tract_linewidth * random.uniform(0.9, 1.1)
        lc = LineCollection(segments, colors=colors, linewidths=linewidth)
        ax.add_collection(lc)


# Helper functions for high-density mask handling
def _generate_high_density_masks(nifti_file, trk_file, output_dir, prefix, slice_mode, specific_slice,
                                max_fiber_percentage, roi_sphere, tract_linewidth, mask_thickness,
                                density_threshold, gaussian_sigma, close_gaps, closing_footprint_size,
                                label_bundles, min_bundle_size, contrast_method, contrast_params,
                                high_density_masks, high_density_labeled_masks, random_state):
    """Generate high-density masks for use across all examples."""
    # Load data to determine slice position
    nii_img = nib.load(nifti_file)
    dims = nii_img.shape
    
    if specific_slice is None:
        if slice_mode == "axial":
            slice_idx = dims[2] // 2
        elif slice_mode == "coronal":
            slice_idx = dims[1] // 2
        else:  # sagittal
            slice_idx = dims[0] // 2
    else:
        slice_idx = specific_slice
    
    # Generate mask using high fiber density
    tractogram = load(trk_file)
    streamlines = tractogram.streamlines
    affine_inv = np.linalg.inv(nii_img.affine)
    streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))
    
    # Select high percentage of streamlines
    selected_streamlines = select_random_streamlines(
        streamlines_voxel, max_fiber_percentage, random_state=random_state
    )
    
    # Create mask
    if label_bundles:
        mask, labeled_mask = create_fiber_mask(
            selected_streamlines, slice_idx, orientation=slice_mode,
            dims=dims, thickness=mask_thickness, dilate=True,
            density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
            close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
            label_bundles=True, min_bundle_size=min_bundle_size
        )
        high_density_masks[slice_idx] = np.rot90(mask)
        high_density_labeled_masks[slice_idx] = np.rot90(labeled_mask)
    else:
        mask = create_fiber_mask(
            selected_streamlines, slice_idx, orientation=slice_mode,
            dims=dims, thickness=mask_thickness, dilate=True,
            density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
            close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
            min_bundle_size=min_bundle_size
        )
        high_density_masks[slice_idx] = np.rot90(mask)


def _apply_high_density_masks_axial(output_file, high_density_masks, high_density_labeled_masks,
                                   specific_slice, label_bundles, nifti_file):
    """Apply high-density masks for axial view."""
    nii_img = nib.load(nifti_file)
    dims = nii_img.shape
    
    slice_idx = specific_slice if specific_slice is not None else dims[2] // 2
    
    if slice_idx in high_density_masks:
        mask_dir = os.path.dirname(output_file)
        mask_basename = os.path.splitext(os.path.basename(output_file))[0]
        mask_filename = f"{mask_dir}/{mask_basename}_mask_slice{slice_idx}.png"
        
        from .utils import save_image_1024
        save_image_1024(mask_filename, high_density_masks[slice_idx], is_mask=True)
        print(f"Applied high-density mask for axial slice {slice_idx}: {mask_filename} (1024x1024)")
        
        if label_bundles and slice_idx in high_density_labeled_masks:
            from .utils import visualize_labeled_bundles
            labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_slice{slice_idx}.png"
            visualize_labeled_bundles(high_density_labeled_masks[slice_idx], labeled_filename)
            print(f"Applied high-density labeled bundles for axial slice {slice_idx}: {labeled_filename}")


def _apply_high_density_masks_coronal(output_file, high_density_masks, high_density_labeled_masks,
                                     specific_slice, label_bundles, nifti_file):
    """Apply high-density masks for coronal view."""
    nii_img = nib.load(nifti_file)
    dims = nii_img.shape
    
    slice_idx = specific_slice if specific_slice is not None else dims[1] // 2
    
    if slice_idx in high_density_masks:
        mask_dir = os.path.dirname(output_file)
        mask_basename = os.path.splitext(os.path.basename(output_file))[0]
        mask_filename = f"{mask_dir}/{mask_basename}_mask_slice{slice_idx}.png"
        
        from .utils import save_image_1024
        save_image_1024(mask_filename, high_density_masks[slice_idx], is_mask=True)
        print(f"Applied high-density mask for coronal slice {slice_idx}: {mask_filename} (1024x1024)")
        
        if label_bundles and slice_idx in high_density_labeled_masks:
            from .utils import visualize_labeled_bundles
            labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_slice{slice_idx}.png"
            visualize_labeled_bundles(high_density_labeled_masks[slice_idx], labeled_filename)
            print(f"Applied high-density labeled bundles for coronal slice {slice_idx}: {labeled_filename}")


def _apply_high_density_masks_multiview(output_file, high_density_masks, high_density_labeled_masks, label_bundles):
    """Apply high-density masks for multiple views."""
    mask_dir = os.path.dirname(output_file)
    if not mask_dir:
        mask_dir = "../synthesis"
    mask_basename = os.path.splitext(os.path.basename(output_file))[0]
    
    for view, mask in high_density_masks.items():
        if mask is not None:
            mask_filename = f"{mask_dir}/{mask_basename}_mask_{view}.png"
            from .utils import save_image_1024
            save_image_1024(mask_filename, mask, is_mask=True)
            print(f"Applied high-density mask for {view} view: {mask_filename} (1024x1024)")
            
            if label_bundles and view in high_density_labeled_masks:
                from .utils import visualize_labeled_bundles
                labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_{view}.png"
                visualize_labeled_bundles(high_density_labeled_masks[view], labeled_filename)
                print(f"Applied high-density labeled bundles for {view} view: {labeled_filename}")


def generate_enhanced_varied_examples_with_preserved_background(nifti_file, trk_file, output_dir, 
                                                              n_examples=5, prefix="blockface_preserved_",
                                                              preserve_bright_background=True,
                                                              apply_background_cleanup=False,
                                                              **kwargs):
    """
    Convenience function to generate enhanced examples with preserved bright blockface areas.
    
    This function automatically sets the parameters to preserve bright background areas
    that might otherwise be forced to black.
    
    Parameters
    ----------
    nifti_file : str
        Path to NIfTI file
    trk_file : str  
        Path to tractography file
    output_dir : str
        Output directory
    n_examples : int
        Number of examples to generate
    prefix : str
        File prefix for examples
    preserve_bright_background : bool
        Whether to preserve bright background areas (default: True)
    apply_background_cleanup : bool
        Whether to apply aggressive background cleanup (default: False)
    **kwargs : dict
        Additional parameters passed to generate_enhanced_varied_examples
        
    Returns
    -------
    dict
        Summary of generation results
    """
    print(f" Generating {n_examples} examples with PRESERVED bright background areas")
    print(f"   Preserve bright background: {preserve_bright_background}")
    print(f"   Apply background cleanup: {apply_background_cleanup}")
    
    # Set parameters to preserve bright areas
    kwargs.update({
        'preserve_bright_background': preserve_bright_background,
        'apply_background_cleanup': apply_background_cleanup
    })
    
    return generate_enhanced_varied_examples(
        nifti_file=nifti_file,
        trk_file=trk_file, 
        output_dir=output_dir,
        n_examples=n_examples,
        prefix=prefix,
        **kwargs
    ) 