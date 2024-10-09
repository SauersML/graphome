import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
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

    if eigenvectors.shape[1] != len(eigenvalues):
        print("Error: The number of eigenvalues does not match the number of eigenvectors.")
        sys.exit(1)

    def create_3d_plot(eigenvalues, eigenvectors, sorted_data=False, filename="eigenplot_3d.png"):
        plot_start_time = time()
        print(f"=== Creating {'sorted' if sorted_data else 'unsorted'} 3D visualization ===")
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
            print(f"Sorting completed in {sort_end - sort_start:.4f} seconds.\n")
        else:
            eigenvalues_plot = eigenvalues
            eigenvectors_plot = eigenvectors
            print("Step 3: Sorting not required. Using original data.\n")

        # Step 4: Prepare grid positions
        print("Step 4: Preparing grid positions...")
        grid_start = time()
        num_eigenvectors = eigenvectors_plot.shape[1]
        num_components = eigenvectors_plot.shape[0]

        # Create meshgrid for X (component index) and Y (eigenvector index)
        X, Y = np.meshgrid(np.arange(num_components), np.arange(num_eigenvectors))
        X = X.flatten()
        Y = Y.flatten()
        Z = np.repeat(eigenvalues_plot, num_components)

        # Flatten eigenvector values
        data = eigenvectors_plot.flatten()
        grid_end = time()
        print(f"Grid positions prepared in {grid_end - grid_start:.4f} seconds.\n")

        # Step 5: Compute Z-normalization excluding zeros
        print("Step 5: Computing Z-normalization for color mapping...")
        znorm_start = time()
        # Exclude zeros from normalization
        non_zero_values = data[data != 0]
        if non_zero_values.size == 0:
            print("All eigenvector values are zero. Skipping plot creation.\n")
            return
        mean = np.mean(non_zero_values)
        std = np.std(non_zero_values)
        print(f"Mean of non-zero eigenvector values: {mean:.4f}")
        print(f"Standard deviation of non-zero eigenvector values: {std:.4f}")

        # Compute intensity based on z-normalization
        # Map values <= (mean - 2*std) to 0 and >= (mean + 2*std) to 1
        intensity = (data - (mean - 2 * std)) / (4 * std)
        intensity = np.clip(intensity, 0, 1)
        znorm_end = time()
        print(f"Z-normalization completed in {znorm_end - znorm_start:.4f} seconds.\n")

        print(f"Intensity range: min={intensity.min()}, max={intensity.max()}\n")

        # Step 6: Assign colors based on intensity
        print("Step 6: Assigning colors based on intensity...")
        color_start = time()
        # Define a custom colorscale from pure blue to pure red
        cmap = mcolors.LinearSegmentedColormap.from_list('blue_red', ['blue', 'cyan', 'green', 'yellow', 'red'], N=256)
        cmap.set_under('black')  # Set color for values below the normalization range (i.e., intensity=0)

        # Normalize intensity for colormap
        norm = mcolors.Normalize(vmin=0.01, vmax=1.0)  # Set vmin slightly above 0 to use 'under' color for intensity=0

        color_end = time()
        print(f"Color assignment setup completed in {color_end - color_start:.4f} seconds.\n")

        # Step 7: Create 3D scatter plot
        print("Step 7: Creating 3D scatter plot...")
        plot3d_start = time()
        try:
            fig = plt.figure(figsize=(14, 10), facecolor='black')
            ax = fig.add_subplot(111, projection='3d', facecolor='black')

            # Scatter plot with intensity as color
            scatter = ax.scatter(
                X, Y, Z,
                c=intensity,
                cmap=cmap,
                norm=norm,
                marker='o',
                s=20,
                depthshade=True
            )

            # Set labels
            ax.set_title('Eigenvalue-Eigenvector 3D Visualization' + (' (Sorted)' if sorted_data else ''), color='white', fontsize=16)
            ax.set_xlabel('Component Index', color='white', fontsize=14)
            ax.set_ylabel('Eigenvector Index', color='white', fontsize=14)
            ax.set_zlabel('Eigenvalue', color='white', fontsize=14)

            # Customize the view angle for better aesthetics
            ax.view_init(elev=30, azim=45)

            # Set background to black for panes
            ax.xaxis.pane.set_facecolor((0, 0, 0, 1.0))
            ax.yaxis.pane.set_facecolor((0, 0, 0, 1.0))
            ax.zaxis.pane.set_facecolor((0, 0, 0, 1.0))

            # Set grid lines color
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

            # Remove tick colors
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.tick_params(axis='z', colors='white')

            # Create a colorbar linked to the scatter plot
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.1, extend='min')
            cbar.set_label('Eigenvector Value (Normalized)', color='white', fontsize=14)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            # Set facecolor to black
            fig.patch.set_facecolor('black')
            ax.patch.set_facecolor('black')

            plt.tight_layout()
        except Exception as e:
            print(f"Error creating 3D scatter plot: {e}")
            return
        plot3d_end = time()
        print(f"3D scatter plot created in {plot3d_end - plot3d_start:.4f} seconds.\n")

        # Step 8: Save the plot as PNG
        print("Step 8: Saving the figure as PNG...")
        save_start = time()
        try:
            plt.savefig(filename, dpi=300, facecolor='black')
            plt.close()
            print(f"Plot saved successfully as '{filename}' in {time() - save_start:.4f} seconds.\n")
        except Exception as e:
            print(f"Error saving plot: {e}\n")
            return

        plot_total_time = time() - plot_start_time
        print(f"Visualization creation completed in {plot_total_time:.4f} seconds.\n")

    # Step 4: Create both unsorted and sorted visualizations
    try:
        create_3d_plot(
            eigenvalues,
            eigenvectors,
            sorted_data=False,
            filename="eigenplot_original_3d.png"
        )
        create_3d_plot(
            eigenvalues,
            eigenvectors,
            sorted_data=True,
            filename="eigenplot_sorted_3d.png"
        )
    except Exception as e:
        print(f"An error occurred during plot creation: {e}\n")
        sys.exit(1)

    total_time = time() - start_time
    print(f"All visualizations have been generated successfully in {total_time:.4f} seconds.")

if __name__ == "__main__":
    main()
