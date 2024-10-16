import requests
import zipfile
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from scipy.sparse import csr_matrix, lil_matrix
from scipy.linalg import eig  # Full eigenvalue solver for dense matrices
from scipy.sparse.linalg import eigs  # For computing a few eigenvalues/eigenvectors for sparse matrices
import sys

# Parameters
LIMIT_WORDS = True  # Set to False if you want to process the entire dataset
MAX_WORDS = 2000 if LIMIT_WORDS else None

# Download the dataset
url = "https://www.kaggle.com/api/v1/datasets/download/sauers/seamus"
print("Starting dataset download...")
start_time = time.time()
response = requests.get(url)

with open("seamus.zip", "wb") as f:
    f.write(response.content)
print(f"Dataset downloaded in {time.time() - start_time:.2f} seconds.")

# Extract the dataset
print("Extracting dataset...")
start_time = time.time()
with zipfile.ZipFile('seamus.zip', 'r') as zip_ref:
    zip_ref.extractall()
print(f"Dataset extracted in {time.time() - start_time:.2f} seconds.")

# Read the text file
print("Reading text file...")
start_time = time.time()
with open('seamus.txt', 'r') as file:
    text = file.read()
print(f"Text file read in {time.time() - start_time:.2f} seconds.")

# Clean up the text: Keep only letters and spaces, and make lowercase
print("Cleaning text...")
start_time = time.time()
text = re.sub(r'[^a-z\s]', '', text.lower())  # Remove non-letters, keep spaces, lowercase
print(f"Text cleaned in {time.time() - start_time:.2f} seconds.")

# Split words and find unique words
print("Splitting words and finding unique words...")
start_time = time.time()
words = text.split()  # Assuming space-delimited
if LIMIT_WORDS:
    words = words[:MAX_WORDS]
unique_words = list(set(words))
n = len(unique_words)
print(f"Found {n} unique words in {time.time() - start_time:.2f} seconds.")

# Create word-to-index mapping
print("Creating word-to-index mapping...")
start_time = time.time()
word_to_index = {word: i for i, word in enumerate(unique_words)}
print(f"Word-to-index mapping created in {time.time() - start_time:.2f} seconds.")

# Initialize adjacency matrix using lil_matrix for efficiency
print("Initializing adjacency matrix...")
start_time = time.time()
adj_matrix = lil_matrix((n, n), dtype=int)

# Build adjacency matrix with progress updates
print("Building adjacency matrix...")
start_time = time.time()
for i in range(len(words) - 1):
    word1, word2 = words[i], words[i + 1]
    idx1, idx2 = word_to_index[word1], word_to_index[word2]
    adj_matrix[idx1, idx2] = 1
    adj_matrix[idx2, idx1] = 1  # Assuming undirected
    if i % 10000 == 0:
        sys.stdout.write(f"\rProgress: {(i/len(words))*100:.2f}%")
        sys.stdout.flush()
print(f"\nAdjacency matrix built in {time.time() - start_time:.2f} seconds.")

# Convert adjacency matrix to csr_matrix for faster operations
adj_matrix = adj_matrix.tocsr()

# Compute Laplacian matrix
print("Computing Laplacian matrix...")
start_time = time.time()
degree_matrix = csr_matrix(np.diag(adj_matrix.sum(axis=1).A1))
laplacian_matrix = degree_matrix - adj_matrix
print(f"Laplacian matrix computed in {time.time() - start_time:.2f} seconds.")

# Perform eigendecomposition with progress updates
print("Performing eigendecomposition (this may take some time)...")
start_time = time.time()

def eigendecomposition_progress(laplacian_matrix):
    num_eigenvalues = laplacian_matrix.shape[0]

    if num_eigenvalues <= 1000:
        # Use dense solver (scipy.linalg.eig) for small or full matrices
        print(f"Matrix is small or full, using scipy.linalg.eig for dense matrix.")
        eigenvalues, eigenvectors = eig(laplacian_matrix.toarray())
    else:
        # Use sparse solver (scipy.sparse.linalg.eigs) for large sparse matrices
        print(f"Matrix is large, using scipy.sparse.linalg.eigs for sparse matrix.")
        k = min(1000, num_eigenvalues - 1)  # Compute the first k smallest eigenvalues/eigenvectors
        eigenvalues, eigenvectors = eigs(laplacian_matrix, k=k, which='SM', tol=0.0, maxiter=1000)

    return eigenvalues, eigenvectors

eigenvalues, eigenvectors = eigendecomposition_progress(laplacian_matrix)
print(f"\nEigendecomposition completed in {time.time() - start_time:.2f} seconds.")

# Save adjacency matrix, eigenvalues, and eigenvectors
print("Saving adjacency matrix, eigenvalues, and eigenvectors to CSV...")
start_time = time.time()
np.savetxt("adjacency_matrix.csv", adj_matrix.toarray(), delimiter=",")
np.savetxt("eigenvalues.csv", eigenvalues, delimiter=",")
np.savetxt("eigenvectors.csv", eigenvectors, delimiter=",")
print(f"Data saved in {time.time() - start_time:.2f} seconds.")

# Visualization function
def create_2d_plot(eigenvalues, eigenvectors, sorted_data=False, filename="eigenplot_2d.png"):
    print(f"=== Creating {'sorted' if sorted_data else 'original'} 2D visualization ===")
    plot_start_time = time.time()

    if sorted_data:
        print("Sorting eigenvalues and eigenvectors...")
        sort_start = time.time()
        idx = np.argsort(eigenvalues)
        eigenvalues_plot = eigenvalues[idx]
        eigenvectors_plot = eigenvectors[:, idx]
        print(f"Sorting completed in {time.time() - sort_start:.2f} seconds.")
    else:
        eigenvalues_plot = eigenvalues
        eigenvectors_plot = eigenvectors

    num_eigenvectors = eigenvectors_plot.shape[1]
    num_components = eigenvectors_plot.shape[0]

    # Use the real part of the eigenvectors for visualization
    data = eigenvectors_plot.real  # shape (num_components, num_eigenvectors)

    print("Computing Z-normalization for color mapping...")
    znorm_start = time.time()
    non_zero_values = data[data != 0]
    if non_zero_values.size == 0:
        print("All eigenvector values are zero. Skipping plot creation.\n")
        return
    mean = np.mean(non_zero_values)
    std = np.std(non_zero_values)

    intensity = (data - (mean - 2 * std)) / (4 * std)
    intensity = np.clip(intensity, 0, 1)  # Ensure values are between 0 and 1
    print(f"Z-normalization completed in {time.time() - znorm_start:.2f} seconds.")

    print("Creating plot...")
    plt.figure(figsize=(12, 10), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    masked_intensity = np.ma.masked_where(data == 0, intensity)
    cmap = mcolors.LinearSegmentedColormap.from_list('bwr_custom', ['blue', 'black', 'red'], N=256)

    img = ax.imshow(masked_intensity, cmap=cmap, aspect='auto', interpolation='nearest')

    ax.set_xticks(np.linspace(0, num_eigenvectors - 1, min(10, num_eigenvectors)))
    ax.set_yticks(np.linspace(0, num_components - 1, min(10, num_components)))
    ax.set_xticklabels([int(x) for x in np.linspace(0, num_eigenvectors - 1, min(10, num_eigenvectors))])
    ax.set_yticklabels([int(y) for y in np.linspace(0, num_components - 1, min(10, num_components))])
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.title('Eigenvalue-Eigenvector 2D Visualization' + (' (Sorted)' if sorted_data else ''), color='white', fontsize=16)
    plt.xlabel('Eigenvector Index', color='white', fontsize=14)
    plt.ylabel('Component Index', color='white', fontsize=14)

    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Eigenvector Value', color='white', fontsize=14)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor='black')
    plt.close()
    print(f"Plot saved as '{filename}' in {time.time() - plot_start_time:.2f} seconds.")

create_2d_plot(eigenvalues, eigenvectors, sorted_data=False, filename="eigenplot_original_2d.png")
create_2d_plot(eigenvalues, eigenvectors, sorted_data=True, filename="eigenplot_sorted_2d.png")

print("All tasks completed.")
