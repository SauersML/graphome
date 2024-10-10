import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import glob
import re
import argparse
from multiprocessing import Pool, cpu_count
import imageio.v2 as imageio
from time import time

def extract_start_number(filename):
    match = re.search(r'multi_submatrix(\d+)_\d+\.eigenvalues\.csv', filename)
    return int(match.group(1)) if match else -1

def plot_eigenvectors(job):
    eigenvalues_file, eigenvectors_file, sorted_flag, output_dir, frame_index = job
    try:
        eigenvalues = pd.read_csv(eigenvalues_file, header=None).values.flatten()
        eigenvectors = pd.read_csv(eigenvectors_file, header=None).values

        if sorted_flag:
            idx = np.argsort(eigenvalues)
            eigenvalues_plot = eigenvalues[idx]
            eigenvectors_plot = eigenvectors[:, idx]
        else:
            eigenvalues_plot = eigenvalues
            eigenvectors_plot = eigenvectors

        data = eigenvectors_plot
        non_zero = data[data != 0]
        if non_zero.size == 0:
            return False, frame_index, sorted_flag

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

        title = 'Sorted by Eigenvalue' if sorted_flag else 'Unsorted'
        plt.title(f'Eigenvector Visualization ({title})', color='white', fontsize=16)
        plt.xlabel('Eigenvector Index', color='white', fontsize=14)
        plt.ylabel('Component Index', color='white', fontsize=14)

        cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Eigenvector Value', color='white', fontsize=14)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        ax.tick_params(axis='x', colors='white', labelsize=10)
        ax.tick_params(axis='y', colors='white', labelsize=10)

        plt.tight_layout()
        filename = f"frame_{frame_index:04d}_{title}.jpg"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=200, facecolor='black', format='jpeg')
        plt.close()

        return True, frame_index, sorted_flag

    except Exception as e:
        return False, frame_index, sorted_flag, str(e)

def create_animation(output_dir, sorted_flag):
    pattern = os.path.join(output_dir, f"frame_*_{'Sorted' if sorted_flag else 'Unsorted'}.jpg")
    images = sorted(glob.glob(pattern))
    if images:
        frames = []
        for img in images:
            frames.append(imageio.imread(img))
        animation_name = f'animation_{"sorted" if sorted_flag else "unsorted"}.mp4'
        imageio.mimsave(animation_name, frames, fps=20, codec='libx264')
        print(f"Animation saved as '{animation_name}'.")
    else:
        print(f"No images found for '{'sorted' if sorted_flag else 'unsorted'}' animation.")

def main():
    parser = argparse.ArgumentParser(description="Eigenvalue-Eigenvector Visualization")
    parser.add_argument('input_dir', type=str, help='Path to the directory containing eigenvalues and eigenvectors CSV files')
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist.")
        sys.exit(1)

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

    jobs = []
    frame_index = 0
    for ev, evectors in pairs:
        jobs.append((ev, evectors, True, output_dir_sorted, frame_index))
        frame_index += 1
        jobs.append((ev, evectors, False, output_dir_unsorted, frame_index))
        frame_index += 1

    total_jobs = len(jobs)
    print(f"Starting plot creation for {len(pairs)} file pairs ({total_jobs} jobs)...")
    start_time = time()

    with Pool(processes=cpu_count()) as pool:
        for i, result in enumerate(pool.imap_unordered(plot_eigenvectors, jobs, chunksize=10), 1):
            if result[0]:
                sorted_flag = result[2]
                percent = (i / total_jobs) * 100
                print(f"Processed {i}/{total_jobs} ({percent:.2f}%) - {'Sorted' if sorted_flag else 'Unsorted'}")
            else:
                frame_index = result[1]
                sorted_flag = result[2]
                error_message = result[3] if len(result) > 3 else 'Unknown error'
                print(f"Error processing frame {frame_index} - {'Sorted' if sorted_flag else 'Unsorted'}: {error_message}")

    elapsed = time() - start_time
    print(f"Plot creation completed in {elapsed:.2f} seconds.")

    print("Creating sorted animation...")
    create_animation(output_dir_sorted, True)

    print("Creating unsorted animation...")
    create_animation(output_dir_unsorted, False)

    print("All animations have been generated successfully.")

if __name__ == "__main__":
    main()
