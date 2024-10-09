import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import os
import sys
from time import time

def main():
    start_time = time()
    print("=== Eigenvalue-Eigenvector 3D Visualization Script ===\n")

    # Step 1: Check if required files exist
    print("Step 1: Checking for required CSV files...")
    required_files = ['submatrix.eigenvalues.csv', 'submatrix.eigenvectors.csv']
    missing_files = [file for file in required_files if not os.path.isfile(file)]
    if missing_files:
        print(f"Error: The following required file(s) are missing: {', '.join(missing_files)}")
        sys.exit(1)
    print("All required files are present.\n")

    # Step 2: Load the CSV files
    print("Step 2: Loading CSV files...")
    try:
        eigenvalues = pd.read_csv('submatrix.eigenvalues.csv', header=None).values.flatten()
        eigenvectors = pd.read_csv('submatrix.eigenvectors.csv', header=None).values
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        sys.exit(1)
    print(f"Loaded {len(eigenvalues)} eigenvalues and eigenvectors with shape {eigenvectors.shape}.\n")

    def create_3d_plot(eigenvalues, eigenvectors, sorted_data=False, filename="eigenplot.html", downsample=True, max_points=100000):
        plot_start_time = time()
        print(f"=== Creating {'sorted' if sorted_data else 'original'} visualization ===")
        print("Starting plot creation...\n")

        # Step 3: Sort data if required
        if sorted_data:
            print("Step 3: Sorting eigenvalues and eigenvectors...")
            sort_start = time()
            idx = np.argsort(eigenvalues)
            eigenvalues_sorted = eigenvalues[idx]
            eigenvectors_sorted = eigenvectors[:, idx]
            eigenvalues_plot = eigenvalues_sorted
            eigenvectors_plot = eigenvectors_sorted
            sort_end = time()
            print(f"Sorting completed in {sort_end - sort_start:.2f} seconds.\n")
        else:
            eigenvalues_plot = eigenvalues
            eigenvectors_plot = eigenvectors
            print("Step 3: Sorting not required. Using original data.\n")

        # Step 4: Prepare meshgrid for axes
        print("Step 4: Preparing meshgrid for axes...")
        mesh_start = time()
        x_indices = np.arange(eigenvectors_plot.shape[1])  # Eigenvector indices (X-axis)
        y_indices = np.arange(eigenvectors_plot.shape[0])  # Component indices (Y-axis)
        x, y = np.meshgrid(x_indices, y_indices)
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = np.repeat(eigenvalues_plot, eigenvectors_plot.shape[0])  # Eigenvalues as Z-axis
        eigenvectors_flat = eigenvectors_plot.flatten()
        mesh_end = time()
        print(f"Meshgrid prepared in {mesh_end - mesh_start:.2f} seconds.\n")

        # Step 5: Filter out zero-valued eigenvectors
        print("Step 5: Filtering out zero-valued eigenvectors...")
        filter_start = time()
        non_zero_mask = eigenvectors_flat != 0
        x_non_zero = x_flat[non_zero_mask]
        y_non_zero = y_flat[non_zero_mask]
        z_non_zero = z_flat[non_zero_mask]
        eigenvectors_non_zero = eigenvectors_flat[non_zero_mask]
        num_non_zero = np.sum(non_zero_mask)
        filter_end = time()
        print(f"Filtered out zero values. Remaining points to plot: {num_non_zero}.\n")

        if num_non_zero == 0:
            print("No non-zero eigenvector values to plot. Skipping plot creation.\n")
            return

        # Step 6: Downsample the data for performance (optional)
        if downsample:
            print("Step 6: Downsampling data for performance...")
            downsample_start = time()
            if num_non_zero > max_points:
                print(f"Number of points ({num_non_zero}) exceeds the maximum allowed ({max_points}). Downsampling...")
                # Randomly select indices without replacement
                np.random.seed(42)  # For reproducibility
                indices = np.random.choice(num_non_zero, size=max_points, replace=False)
                x_non_zero = x_non_zero[indices]
                y_non_zero = y_non_zero[indices]
                z_non_zero = z_non_zero[indices]
                eigenvectors_non_zero = eigenvectors_non_zero[indices]
                print(f"Downsampling completed. Plotting {max_points} points.\n")
            else:
                print(f"Number of points ({num_non_zero}) is within the limit. No downsampling needed.\n")
            downsample_end = time()
            print(f"Downsampling step completed in {downsample_end - downsample_start:.2f} seconds.\n")
        else:
            print("Step 6: Downsampling not enabled. Proceeding without downsampling.\n")

        # Step 7: Compute Z-normalization excluding zeros
        print("Step 7: Computing Z-normalization for color mapping...")
        znorm_start = time()
        mean = np.mean(eigenvectors_non_zero)
        std = np.std(eigenvectors_non_zero)
        print(f"Mean of non-zero eigenvector values: {mean:.4f}")
        print(f"Standard deviation of non-zero eigenvector values: {std:.4f}")

        # Compute intensity based on z-normalization
        intensity = (eigenvectors_non_zero - (mean - 2 * std)) / (4 * std)
        intensity = np.clip(intensity, 0, 1)  # Values are between 0 and 1
        znorm_end = time()
        print(f"Z-normalization completed in {znorm_end - znorm_start:.2f} seconds.\n")

        # Step 8: Assign colors based on intensity
        print("Step 8: Assigning colors based on intensity...")
        color_start = time()
        # Define a custom colorscale from pure blue to pure red
        colorscale = [
            [0.0, 'rgb(0,0,255)'],   # Pure Blue
            [1.0, 'rgb(255,0,0)']    # Pure Red
        ]
        color_end = time()
        print(f"Color assignment completed in {color_end - color_start:.2f} seconds.\n")

        # Debugging: Verify color mapping
        print("Step 8.1: Verifying color mapping...")
        sample_indices = np.random.choice(len(intensity), size=5, replace=False)
        for idx in sample_indices:
            print(f"Point {idx}: Intensity={intensity[idx]:.2f}")
        print("Color mapping verification completed.\n")

        # Step 9: Create 3D scatter plot
        print("Step 9: Creating 3D scatter plot...")
        plotly_start = time()
        try:
            scatter = go.Scatter3d(
                x=x_non_zero,
                y=y_non_zero,
                z=z_non_zero,
                mode='markers',
                marker=dict(
                    size=1,          # Smaller size for better performance
                    color=intensity, # Numerical intensity for colorscale mapping
                    colorscale=colorscale,
                    cmin=0,
                    cmax=1,
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(
                        title="Eigenvector Value",
                        titleside="right",
                        tickvals=[0, 0.25, 0.5, 0.75, 1],
                        ticktext=["≤ -2 SD", " -1 SD", "Mean", " +1 SD", "≥ +2 SD"],
                        len=0.75,
                        thickness=10,
                        yanchor="middle",
                        y=0.5
                    )
                ),
                name='Eigenvectors',
                showlegend=False,
                hoverinfo='none'  # Disable hover info for faster rendering
            )
        except Exception as e:
            print(f"Error creating Scatter3d: {e}")
            return
        plot_end = time()
        print(f"3D scatter plot created in {plot_end - plotly_start:.2f} seconds.\n")

        # Step 10: Configure plot layout
        print("Step 10: Configuring plot layout...")
        layout_start = time()
        layout = go.Layout(
            title='Eigenvalue-Eigenvector 3D Visualization' + (' (Sorted)' if sorted_data else ''),
            scene=dict(
                xaxis=dict(title='Eigenvector Index', backgroundcolor="black", gridcolor="gray", showbackground=True),
                yaxis=dict(title='Component Index', backgroundcolor="black", gridcolor="gray", showbackground=True),
                zaxis=dict(title='Eigenvalue', backgroundcolor="black", gridcolor="gray", showbackground=True),
                bgcolor="black",  # Set scene background to black
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            paper_bgcolor='black',  # Set paper background to black
            plot_bgcolor='black',
            margin=dict(l=0, r=0, b=0, t=50)
        )
        layout_end = time()
        print(f"Plot layout configured in {layout_end - layout_start:.2f} seconds.\n")

        # Step 11: Generate and save the figure
        print("Step 11: Generating and saving the figure...")
        generate_start = time()
        fig = go.Figure(data=[scatter], layout=layout)
        try:
            pio.write_html(fig, file=filename, auto_open=False)
            generate_end = time()
            print(f"Plot saved successfully as '{filename}' in {generate_end - generate_start:.2f} seconds.\n")
        except Exception as e:
            print(f"Error saving plot: {e}\n")
            return

        plot_total_time = time() - plot_start_time
        print(f"Visualization creation completed in {plot_total_time:.2f} seconds.\n")

    # Step 12: Create both unsorted and sorted visualizations with downsampling
    try:
        create_3d_plot(
            eigenvalues,
            eigenvectors,
            sorted_data=False,
            filename="eigenplot_original.html",
            downsample=True,       # Enable downsampling
            max_points=100000
        )
        create_3d_plot(
            eigenvalues,
            eigenvectors,
            sorted_data=True,
            filename="eigenplot_sorted.html",
            downsample=True,       # Downsampling
            max_points=100000
        )
    except Exception as e:
        print(f"An error occurred during plot creation: {e}\n")
        sys.exit(1)

    total_time = time() - start_time
    print(f"All visualizations have been generated successfully in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
