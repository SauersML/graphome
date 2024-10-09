import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import glob
import re
from multiprocessing import Pool, cpu_count
import imageio

def extract_start_number(filename):
    match = re.search(r'multi_submatrix(\d+)_\d+\.eigenvalues\.csv', filename)
    return int(match.group(1)) if match else -1

def create_plot(eigenvalues_file, eigenvectors_file, output_dir_sorted, output_dir_unsorted, index):
    try:
        eigenvalues = pd.read_csv(eigenvalues_file, header=None).values.flatten()
        eigenvectors = pd.read_csv(eigenvectors_file, header=None).values

        # Unsorted plot
        plot_eigenvectors(eigenvalues, eigenvectors, sorted_data=False, output_dir=output_dir_unsorted, index=index)

        # Sorted plot
        plot_eigenvectors(eigenvalues, eigenvectors, sorted_data=True, output_dir=output_dir_sorted, index=index)

    except Exception:
        pass

def plot_eigenvectors(eigenvalues, eigenvectors, sorted_data, output_dir, index):
    if sorted_data:
        idx = np.argsort(eigenvalues)
        eigenvalues_plot = eigenvalues[idx]
        eigenvectors_plot = eigenvectors[:, idx]
    else:
        eigenvalues_plot = eigenvalues
        eigenvectors_plot = eigenvectors

    data = eigenvectors_plot
    non_zero = data[data != 0]
    if non_zero.size == 0:
        return
    mean = np.mean(non_zero)
    std = np.std(non_zero)
    intensity = (data - (mean - 2 * std)) / (4 * std)
    intensity = np.clip(intensity, 0, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list('bwr_custom', ['blue', 'black', 'red'], N=256)

    plt.figure(figsize=(12, 10), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    img = ax.imshow(intensity, cmap=cmap, aspect='auto', interpolation='nearest')

    num_eigenvectors = eigenvectors_plot.shape[1]
    num_components = eigenvectors_plot.shape[0]
    ax.set_xticks(np.linspace(0, num_eigenvectors - 1, min(10, num_eigenvectors)))
    ax.set_yticks(np.linspace(0, num_components - 1, min(10, num_components)))
    ax.set_xticklabels([int(x) for x in np.linspace(0, num_eigenvectors - 1, min(10, num_eigenvectors))])
    ax.set_yticklabels([int(y) for y in np.linspace(0, num_components - 1, min(10, num_components))])

    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)

    title = 'Eigenvalue-Eigenvector Visualization (Sorted)' if sorted_data else 'Eigenvalue-Eigenvector Visualization (Unsorted)'
    plt.title(title, color='white', fontsize=16)
    plt.xlabel('Eigenvector Index', color='white', fontsize=14)
    plt.ylabel('Component Index', color='white', fontsize=14)

    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Eigenvector Value', color='white', fontsize=14)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)

    plt.tight_layout()
    filename = f"frame_{index:04d}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, facecolor='black')
    plt.close()

def main():
    input_dir = "input"
    eigenvalues_pattern = os.path.join(input_dir, "multi_submatrix*_*eigenvalues.csv")
    eigenvalues_files = sorted(glob.glob(eigenvalues_pattern), key=extract_start_number)

    pairs = []
    for ev in eigenvalues_files:
        base = ev.replace('.eigenvalues.csv', '')
        evectors = f"{base}.eigenvectors.csv"
        if os.path.isfile(evectors):
            pairs.append((ev, evectors))

    if not pairs:
        print("No matching eigenvalues and eigenvectors files found.")
        sys.exit(1)

    output_dir_sorted = "frames_sorted"
    output_dir_unsorted = "frames_unsorted"
    os.makedirs(output_dir_sorted, exist_ok=True)
    os.makedirs(output_dir_unsorted, exist_ok=True)

    with Pool(cpu_count()) as pool:
        pool.starmap(create_plot, [(ev, evectors, output_dir_sorted, output_dir_unsorted, idx) 
                                   for idx, (ev, evectors) in enumerate(pairs)])

    # Create sorted animation
    sorted_images = sorted(glob.glob(os.path.join(output_dir_sorted, "frame_*.png")))
    if sorted_images:
        sorted_frames = [imageio.imread(img) for img in sorted_images]
        imageio.mimsave('animation_sorted.mp4', sorted_frames, fps=20)
        print("Sorted animation saved as 'animation_sorted.mp4'.")
    else:
        print("No sorted images to create animation.")

    # Create unsorted animation
    unsorted_images = sorted(glob.glob(os.path.join(output_dir_unsorted, "frame_*.png")))
    if unsorted_images:
        unsorted_frames = [imageio.imread(img) for img in unsorted_images]
        imageio.mimsave('animation_unsorted.mp4', unsorted_frames, fps=20)
        print("Unsorted animation saved as 'animation_unsorted.mp4'.")
    else:
        print("No unsorted images to create animation.")

if __name__ == "__main__":
    main()
