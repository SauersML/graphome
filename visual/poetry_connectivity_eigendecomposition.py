import requests
import zipfile
import numpy as np
from scipy.linalg import eigh
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

url = "https://www.kaggle.com/api/v1/datasets/download/sauers/seamus"
response = requests.get(url)

with open("seamus.zip", "wb") as f:
    f.write(response.content)

with zipfile.ZipFile('seamus.zip', 'r') as zip_ref:
    zip_ref.extractall()

with open('seamus.txt', 'r') as file:
    text = file.read()

words = text.split()  # Assuming space-delimited
unique_words = list(set(words))
n = len(unique_words)

# Create word-to-index mapping
word_to_index = {word: i for i, word in enumerate(unique_words)}

# Initialize adjacency matrix
adj_matrix = np.zeros((n, n), dtype=int)

# Build adjacency matrix
for i in range(len(words) - 1):
    word1, word2 = words[i], words[i + 1]
    idx1, idx2 = word_to_index[word1], word_to_index[word2]
    adj_matrix[idx1, idx2] = 1
    adj_matrix[idx2, idx1] = 1  # Assuming undirected

degree_matrix = np.diag(adj_matrix.sum(axis=1))
laplacian_matrix = degree_matrix - adj_matrix

eigenvalues, eigenvectors = eigh(laplacian_matrix)

# Save adjacency matrix, eigenvalues, and eigenvectors
np.savetxt("adjacency_matrix.csv", adj_matrix, delimiter=",")
np.savetxt("eigenvalues.csv", eigenvalues, delimiter=",")
np.savetxt("eigenvectors.csv", eigenvectors, delimiter=",")

def create_2d_plot(eigenvalues, eigenvectors, sorted_data=False, filename="eigenplot_2d.png"):
    plot_start_time = time.time()
    print(f"=== Creating {'sorted' if sorted_data else 'original'} 2D visualization ===")
    print("Starting plot creation...\n")

    if sorted_data:
        idx = np.argsort(eigenvalues)
        eigenvalues_plot = eigenvalues[idx]
        eigenvectors_plot = eigenvectors[:, idx]
    else:
        eigenvalues_plot = eigenvalues
        eigenvectors_plot = eigenvectors

    num_eigenvectors = eigenvectors_plot.shape[1]
    num_components = eigenvectors_plot.shape[0]
    data = eigenvectors_plot  # shape (num_components, num_eigenvectors)

    # Compute Z-normalization excluding zeros
    non_zero_values = data[data != 0]
    if non_zero_values.size == 0:
        print("All eigenvector values are zero. Skipping plot creation.\n")
        return
    mean = np.mean(non_zero_values)
    std = np.std(non_zero_values)

    # Compute intensity based on z-normalization
    intensity = (data - (mean - 2 * std)) / (4 * std)
    intensity = np.clip(intensity, 0, 1)  # Ensure values are between 0 and 1

    # Assign colors based on intensity
    cmap = mcolors.LinearSegmentedColormap.from_list('bwr_custom', ['blue', 'black', 'red'], N=256)

    # Create the plot
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

    # Invert y-axis
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

    # Save the plot as PNG
    plt.savefig(filename, dpi=300, facecolor='black')
    plt.close()

    print(f"Plot saved successfully as '{filename}'.")

create_2d_plot(eigenvalues, eigenvectors, sorted_data=False, filename="eigenplot_original_2d.png")
create_2d_plot(eigenvalues, eigenvectors, sorted_data=True, filename="eigenplot_sorted_2d.png")
