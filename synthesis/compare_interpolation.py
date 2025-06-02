#!/usr/bin/env python
"""
compare_interpolation.py

Compare original (no interpolation), linear, Hermite, and RBF interpolation methods for streamlines.

Features:
1. Load streamlines from a TRK file.
2. Process them (densify) with multiple interpolation methods.
3. Compute metrics (length, curvature, torsion).
4. Compare metrics numerically (pairwise comparisons).
5. Visualize:
   - 3D overlay of streamlines
   - Point-by-point differences
   - Curvature/torsion distributions
   - Color-coded 3D differences for the top outlier streamlines
"""

import os
import numpy as np
import nibabel as nib
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed

# Import your local densify functions and metrics.
# Make sure these are in your Python path or same folder.
from densify import (
    densify_streamline_subvoxel,
    calculate_streamline_metrics,
)

###############################################################################
#                           STREAMLINE PROCESSING                             #
###############################################################################

def process_streamlines_with_method(streamlines, step_size, method, voxel_size=1.0, use_gpu=False):
    """
    Densify a set of streamlines using a specified interpolation method.
    If method is "original", return the raw streamlines without any interpolation.
    """
    if method == "original":
        print(f"Using original streamlines (no interpolation).")
        return [np.asarray(stream) for stream in streamlines]

    print(f"Processing {len(streamlines)} streamlines with {method} interpolation (voxel size: {voxel_size}mm)...")
    processed = []
    for idx, stream in enumerate(streamlines):
        if idx % 50 == 0:
            print(f"  -> Streamline {idx}/{len(streamlines)}")
        try:
            # Optionally enable detailed debugging for the first few streamlines
            os.environ["DEBUG_TANGENTS"] = "1" if idx < 3 else "0"

            # Scale step size by voxel size if desired
            scaled_step_size = step_size * (voxel_size / 1.0)

            densified = densify_streamline_subvoxel(
                stream,
                scaled_step_size,
                use_gpu=use_gpu,
                interp_method=method,
                voxel_size=voxel_size
            )
            processed.append(densified)
        except Exception as e:
            print(f"Error processing streamline {idx}: {e}")
    print(f"Processed {len(processed)}/{len(streamlines)} with {method} interpolation.")
    return processed

###############################################################################
#                              METRICS & COMPARISON                           #
###############################################################################

def calculate_metrics_from_processed(processed_streams, methods):
    """
    Calculate streamline metrics for each interpolation method from the already processed streamlines.
    """
    metrics = {}
    for method in methods:
        print(f"Calculating metrics for {method} interpolation...")
        metrics[method] = calculate_streamline_metrics(processed_streams[method])
    return metrics

def compare_metrics_for_methods(metrics, methods):
    """
    Compare metrics across multiple interpolation methods, including all pairwise comparisons.
    """
    print("\n================== Metrics Comparison ==================")
    for metric in ['mean_curvature', 'max_curvature', 'mean_length', 'total_length', 'mean_torsion']:
        print(f"\nMetric: {metric}")
        for method in methods:
            value = metrics[method].get(metric, 0)
            print(f"  {method.capitalize()}: {value:.6f}")
        if len(methods) > 1:
            # Compute pairwise differences for every combination
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    method1 = methods[i]
                    method2 = methods[j]
                    base_value = metrics[method1].get(metric, 0)
                    comp_value = metrics[method2].get(metric, 0)
                    if base_value:
                        diff = comp_value - base_value
                        pct_diff = (diff / base_value) * 100
                        print(f"  Difference ({method2.capitalize()} vs {method1.capitalize()}): {diff:.6f} ({pct_diff:+.2f}%)")
                    else:
                        print(f"  Cannot compute percentage difference for {method2.capitalize()} vs {method1.capitalize()} as base value is zero.")
    print("========================================================\n")

###############################################################################
#                        BASIC VISUAL COMPARISONS (3D)                        #
###############################################################################

def visualize_comparison_for_all_methods(processed_streams, methods, max_streamlines=5):
    """
    Show a 3D overlay of a few streamlines for all specified interpolation methods.
    Also show a point-by-point difference plot for each displayed streamline.
    """
    num_streams = min(len(processed_streams[methods[0]]), max_streamlines)
    if num_streams == 0:
        print("No streamlines to visualize.")
        return

    fig = plt.figure(figsize=(15, 5 * num_streams))

    for i in range(num_streams):
        ax = fig.add_subplot(num_streams, 2, 2 * i + 1, projection='3d')
        for method in methods:
            stream = processed_streams[method][i]
            ax.plot(stream[:, 0], stream[:, 1], stream[:, 2], label=method.capitalize())
        ax.set_title(f"Streamline {i} (3D Overlay)")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        ax2 = fig.add_subplot(num_streams, 2, 2 * i + 2)
        for method1 in methods:
            for method2 in methods:
                if method1 != method2:
                    stream1 = processed_streams[method1][i]
                    stream2 = processed_streams[method2][i]
                    min_len = min(len(stream1), len(stream2))
                    if min_len > 0:
                        diff = np.linalg.norm(stream1[:min_len] - stream2[:min_len], axis=1)
                        ax2.plot(diff, label=f"{method1.capitalize()} vs {method2.capitalize()}")
        ax2.set_title(f"Streamline {i} - Point Differences")
        ax2.set_xlabel("Point Index")
        ax2.set_ylabel("Euclidean Distance")
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.show()

###############################################################################
#                      METRICS DISTRIBUTION COMPARISONS                       #
###############################################################################

def plot_metrics_comparison_for_all_methods(metrics, methods):
    """
    Create histograms/scatter plots comparing curvature, length, torsion for all methods.
    """
    for metric in ['curvature', 'length', 'torsion']:
        if metric in metrics[methods[0]]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for method in methods:
                data = metrics[method].get(metric, [])
                if metric in ['curvature', 'torsion']:
                    flat_data = [item for sublist in data for item in sublist]
                else:
                    flat_data = data
                ax.hist(flat_data, bins=50, alpha=0.5, label=method.capitalize())
            ax.set_title(f'{metric.capitalize()} Distribution')
            ax.set_xlabel(metric.capitalize())
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()

        if metric == 'length' and all('length' in metrics[method] for method in methods):
            fig, ax = plt.subplots(figsize=(6, 6))
            for method1 in methods:
                for method2 in methods:
                    if method1 != method2:
                        ax.scatter(metrics[method1]['length'], metrics[method2]['length'], alpha=0.6,
                                   label=f"{method1.capitalize()} vs {method2.capitalize()}")
            mn = min(min(metrics[method]['length']) for method in methods)
            mx = max(max(metrics[method]['length']) for method in methods)
            ax.plot([mn, mx], [mn, mx], 'k--')
            ax.set_title('Length Scatter')
            ax.set_xlabel('Length')
            ax.set_ylabel('Length')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()

###############################################################################
#                   COLOR-CODED DIFFERENCES IN 3D (PER STREAMLINE)            #
###############################################################################

def color_code_difference_3d(reference_stream, compared_stream, ax=None, title=""):
    """
    Plot a streamline (e.g., Hermite or RBF) color-coded by its distance to a reference streamline.
    The reference streamline is shown in solid blue.
    """
    min_len = min(len(reference_stream), len(compared_stream))
    if min_len < 2:
        return

    ref_trim = reference_stream[:min_len]
    comp_trim = compared_stream[:min_len]

    distances = np.linalg.norm(comp_trim - ref_trim, axis=1)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.set_title(title if title else "Color-coded by distance")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Plot the reference streamline in solid blue
    ax.plot(ref_trim[:, 0], ref_trim[:, 1], ref_trim[:, 2],
            color='blue', label='Reference')

    # Plot the compared streamline as a scatter with color coding
    scatter = ax.scatter(
        comp_trim[:, 0],
        comp_trim[:, 1],
        comp_trim[:, 2],
        c=distances,
        cmap='hot',
        marker='o',
        s=20,
        label='Compared'
    )
    # Add colorbar
    if hasattr(ax, 'get_figure'):
        fig = ax.get_figure()
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("Distance to Reference")

    ax.legend()

def visualize_color_coded_differences_for_all_methods(processed_streams, methods, top_n=5):
    """
    Identify the top N streamline pairs with the largest mean difference for all method pairs,
    then show each pair in 3D with color-coded differences.
    """
    for method1 in methods:
        for method2 in methods:
            if method1 != method2:
                differences = []
                for stream1, stream2 in zip(processed_streams[method1], processed_streams[method2]):
                    min_len = min(len(stream1), len(stream2))
                    if min_len == 0:
                        differences.append(0)
                        continue
                    diff_vals = np.linalg.norm(stream1[:min_len] - stream2[:min_len], axis=1)
                    differences.append(np.mean(diff_vals))
                differences = np.array(differences)

                top_indices = np.argsort(differences)[-top_n:]
                print(f"Top outlier streamline indices for {method1.capitalize()} vs {method2.capitalize()}: {top_indices}")

                fig = plt.figure(figsize=(10, 4 * top_n))
                for i, idx in enumerate(top_indices):
                    ax = fig.add_subplot(top_n, 1, i + 1, projection='3d')
                    title = f"Streamline {idx} (Mean Diff={differences[idx]:.4f})"
                    color_code_difference_3d(processed_streams[method1][idx],
                                             processed_streams[method2][idx],
                                             ax=ax,
                                             title=title)
                plt.tight_layout()
                plt.show()

###############################################################################
#                          MAIN COMPARISON FUNCTION                           #
###############################################################################

def compare_interpolations(trk_file, step_size=0.5, voxel_size=None, num_streamlines=None, use_gpu=False, methods=None):
    """
    High-level function to:
      1) Load TRK streamlines
      2) Process them with specified interpolation methods (including "original" for no interpolation)
      3) Compute metrics, compare, and visualize
    """
    print(f"Comparing interpolation methods on {trk_file}")
    print(f"Step size: {step_size}, Using GPU: {use_gpu}")

    # Load streamlines
    trk_data = nib.streamlines.load(trk_file)
    streamlines = trk_data.streamlines
    header = trk_data.header
    original_voxel_sizes = header.get('voxel_sizes', [1.0, 1.0, 1.0])
    original_voxel_size = float(np.mean(original_voxel_sizes))

    if voxel_size is None:
        voxel_size = original_voxel_size

    print("Original voxel sizes:", original_voxel_sizes)
    print(f"Mean voxel size: {original_voxel_size:.3f}mm")
    print(f"Using voxel size: {voxel_size:.3f}mm")
    print(f"Loaded {len(streamlines)} streamlines.")

    if num_streamlines is not None and num_streamlines < len(streamlines):
        print(f"Limiting to {num_streamlines} streamlines.")
        streamlines = streamlines[:num_streamlines]

    # Process streamlines for all methods (only done once)
    processed_streams = {
        method: process_streamlines_with_method(streamlines, step_size, method, voxel_size=voxel_size, use_gpu=use_gpu)
        for method in methods
    }

    # Calculate metrics using the already processed streamlines
    metrics = calculate_metrics_from_processed(processed_streams, methods)

    # Compare metrics (with all pairwise differences)
    compare_metrics_for_methods(metrics, methods)

    # Visual comparisons
    visualize_comparison_for_all_methods(processed_streams, methods, max_streamlines=5)
    plot_metrics_comparison_for_all_methods(metrics, methods)

    # Color-coded differences for top outliers
    visualize_color_coded_differences_for_all_methods(processed_streams, methods, top_n=5)

    print("Comparison completed. Check the generated plots for details.")

###############################################################################
#                                    MAIN                                      #
###############################################################################

def main():
    parser = argparse.ArgumentParser(description='Compare interpolation methods for streamlines.')
    parser.add_argument('trk_file', type=str, help='Path to input TRK file.')
    parser.add_argument('--step_size', type=float, default=0.5,
                        help='Step size for densification (default: 0.5).')
    parser.add_argument('--voxel_size', type=float, default=None,
                        help='Voxel size for analysis (default: use TRK header).')
    parser.add_argument('--num_streamlines', type=int, default=None,
                        help='Number of streamlines to process (default: all).')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU acceleration if available.')
    parser.add_argument('--methods', type=str, nargs='+',
                        choices=['original', 'linear', 'hermite', 'rbf'],
                        default=['original', 'linear', 'hermite', 'rbf'],
                        help='Interpolation methods to compare (default: original linear hermite rbf)')
    args = parser.parse_args()

    compare_interpolations(
        trk_file=args.trk_file,
        step_size=args.step_size,
        voxel_size=args.voxel_size,
        num_streamlines=args.num_streamlines,
        use_gpu=args.use_gpu,
        methods=args.methods
    )

if __name__ == "__main__":
    main()