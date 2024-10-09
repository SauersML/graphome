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
    
    def create_3d_plot(eigenvalues, eigenvectors, sorted_data=False, filename="eigenplot.html"):
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
        x_indices = np.arange(eigenvectors_plot.shape[1])  # Eigenvector indices
        y_indices = np.arange(eigenvectors_plot.shape[0])  # Component indices
        x, y = np.meshgrid(x_indices, y_indices)
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = np.repeat(eigenvalues_plot, eigenvectors_plot.shape[0])
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
        print(f"Filtered out zero values. Remaining points to plot: {num_non_zero}.")
        print(f"Filtering completed in {filter_end - filter_start:.2f} seconds.\n")
        
        if num_non_zero == 0:
            print("No non-zero eigenvector values to plot. Skipping plot creation.\n")
            return
        
        # Step 6: Assign colors based on eigenvector values
        print("Step 6: Assigning colors based on eigenvector values...")
        color_start = time()
        # Normalize eigenvector values for color mapping
        min_val = eigenvectors_non_zero.min()
        max_val = eigenvectors_non_zero.max()
        if max_val != min_val:
            norm = (eigenvectors_non_zero - min_val) / (max_val - min_val)
        else:
            norm = np.zeros_like(eigenvectors_non_zero)
        
        # Use Plotly's 'RdBu' color scale (red for high, blue for low)
        colors = norm  # Plotly will map normalized values to the color scale
        color_end = time()
        print(f"Color assignment completed in {color_end - color_start:.2f} seconds.\n")
        
        # Step 7: Create 3D scatter plot
        print("Step 7: Creating 3D scatter plot...")
        plotly_start = time()
        scatter = go.Scatter3d(
            x=x_non_zero,
            y=y_non_zero,
            z=z_non_zero,
            mode='markers',
            marker=dict(
                size=3,
                color=colors,
                colorscale='RdBu',
                cmin=0,
                cmax=1,
                opacity=0.8
            ),
            name='Eigenvectors',
            showlegend=False
        )
        plot_end = time()
        print(f"3D scatter plot created in {plot_end - plotly_start:.2f} seconds.\n")
        
        # Step 8: Configure plot layout
        print("Step 8: Configuring plot layout...")
        layout_start = time()
        layout = go.Layout(
            title='Eigenvalue-Eigenvector 3D Visualization' + (' (Sorted)' if sorted_data else ''),
            scene=dict(
                xaxis=dict(title='Eigenvector Index', backgroundcolor="black", gridcolor="gray"),
                yaxis=dict(title='Component Index', backgroundcolor="black", gridcolor="gray"),
                zaxis=dict(title='Eigenvalue', backgroundcolor="black", gridcolor="gray"),
                bgcolor="black",  # Set scene background to black
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            paper_bgcolor='black',
            plot_bgcolor='black',
            margin=dict(l=0, r=0, b=0, t=50)
        )
        layout_end = time()
        print(f"Plot layout configured in {layout_end - layout_start:.2f} seconds.\n")
        
        # Step 9: Generate and save the figure
        print("Step 9: Generating and saving the figure...")
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
    
    # Step 10: Create both unsorted and sorted visualizations
    try:
        create_3d_plot(eigenvalues, eigenvectors, sorted_data=False, filename="eigenplot_original.html")
        create_3d_plot(eigenvalues, eigenvectors, sorted_data=True, filename="eigenplot_sorted.html")
    except Exception as e:
        print(f"An error occurred during plot creation: {e}\n")
        sys.exit(1)
    
    total_time = time() - start_time
    print(f"All visualizations have been generated successfully in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
