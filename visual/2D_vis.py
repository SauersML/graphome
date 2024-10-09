import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
from time import time

def main():
    start_time = time()
    print("=== Eigenvalue-Eigenvector 2D Visualization Script ===\n")

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

    def create_2d_plot(eigenvalues, eigenvectors, sorted_data=False, filename="eigenplot_2d.png"):
        plot_start_time = time()
        print(f"=== Creating {'sorted' if sorted_data else 'original'} 2D visualization ===")
        print("Starting plot creation...\n")

        # Step 3: Sort data
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

        # Step 4: Prepare grid positions
        print("Step 4: Preparing grid positions...")
        grid_start = time()
        num_eigenvectors = eigenvectors_plot.shape[1]
        num_components = eigenvectors_plot.shape[0]
        # For imshow, data is a 2D array where rows are y-axis and columns are x-axis
        data = eigenvectors_plot  # shape (num_components, num_eigenvectors)
        grid_end = time()
        print(f"Grid positions prepared in {grid_end - grid_start:.2f} seconds.\n")

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
        intensity = np.clip(intensity, 0, 1)  # Ensure values are between 0 and 1
        znorm_end = time()
        print(f"Z-normalization completed in {znorm_end - znorm_start:.2f} seconds.\n")

        print(f"Intensity range: min={intensity.min()}, max={intensity.max()}\n")

        # Step 6: Assign colors based on intensity
        print("Step 6: Assigning colors based on intensity...")
        color_start = time()
        # Define a custom colorscale from pure blue to pure red
        cmap = mcolors.LinearSegmentedColormap.from_list('bwr_custom', ['blue', 'white', 'red'], N=256) # Try black
        color_end = time()
        print(f"Color assignment completed in {color_end - color_start:.2f} seconds.\n")

        # Step 7: Create 2D scatter plot with square markers
        print("Step 7: Creating 2D scatter plot with square markers...")
        plotly_start = time()
        try:
            plt.figure(figsize=(12, 10), facecolor='black')
            ax = plt.gca()
            ax.set_facecolor('black')

            # Mask zero values
            masked_intensity = np.ma.masked_where(data == 0, intensity)

            # Plot using imshow
            img = ax.imshow(masked_intensity, cmap=cmap, aspect='auto', interpolation='nearest')

            # Set ticks
            ax.set_xticks(np.linspace(0, num_eigenvectors - 1, min(10, num_eigenvectors)))
            ax.set_yticks(np.linspace(0, num_components - 1, min(10, num_components)))
            ax.set_xticklabels([int(x) for x in np.linspace(0, num_eigenvectors - 1, min(10, num_eigenvectors))])
            ax.set_yticklabels([int(y) for y in np.linspace(0, num_components - 1, min(10, num_components))])

            # Invert y-axis to have the first component at the top
            ax.invert_yaxis()

            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Set titles and labels
            plt.title('Eigenvalue-Eigenvector 2D Visualization' + (' (Sorted)' if sorted_data else ''), color='white', fontsize=16)
            plt.xlabel('Eigenvector Index', color='white', fontsize=14)
            plt.ylabel('Component Index', color='white', fontsize=14)

            # Configure colorbar
            cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Eigenvector Value', color='white', fontsize=14)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            # Adjust tick colors
            ax.tick_params(axis='x', colors='white', labelsize=10)
            ax.tick_params(axis='y', colors='white', labelsize=10)

            plt.tight_layout()
        except Exception as e:
            print(f"Error creating scatter plot: {e}")
            return
        plot_end = time()
        print(f"2D scatter plot created in {plot_end - plotly_start:.2f} seconds.\n")

        # Step 8: Save the plot as PNG
        print("Step 8: Saving the figure as PNG...")
        save_start = time()
        try:
            plt.savefig(filename, dpi=300, facecolor='black')
            plt.close()
            print(f"Plot saved successfully as '{filename}' in {time() - save_start:.2f} seconds.\n")
        except Exception as e:
            print(f"Error saving plot: {e}\n")
            return

        plot_total_time = time() - plot_start_time
        print(f"Visualization creation completed in {plot_total_time:.2f} seconds.\n")

    # Step 4: Create both unsorted and sorted visualizations
    try:
        create_2d_plot(
            eigenvalues,
            eigenvectors,
            sorted_data=False,
            filename="eigenplot_original_2d.png"
        )
        create_2d_plot(
            eigenvalues,
            eigenvectors,
            sorted_data=True,
            filename="eigenplot_sorted_2d.png"
        )
    except Exception as e:
        print(f"An error occurred during plot creation: {e}\n")
        sys.exit(1)

    total_time = time() - start_time
    print(f"All visualizations have been generated successfully in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
