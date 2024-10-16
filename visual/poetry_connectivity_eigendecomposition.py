import requests
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh  # For faster sparse matrix eigendecomposition

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

# Split words and find unique words
print("Splitting words and finding unique words...")
start_time = time.time()
words = text.split()  # Assuming space-delimited
unique_words = list(set(words))
n = len(unique_words)
print(f"Found {n} unique words in {time.time() - start_time:.2f} seconds.")

# Create word-to-index mapping
print("Creating word-to-index mapping...")
start_time = time.time()
word_to_index = {word: i for i, word in enumerate(unique_words)}
print(f"Word-to-index mapping created in {time.time() - start_time:.2f} seconds.")

# Initialize adjacency matrix
print("Initializing adjacency matrix...")
start_time = time.time()
adj_matrix = csr_matrix((n, n), dtype=int)  # Use sparse matrix

# Build adjacency matrix
print("Building adjacency matrix...")
for i in range(len(words) - 1):
    word1, word2 = words[i], words[i + 1]
    idx1, idx2 = word_to_index[word1], word_to_index[word2]
    adj_matrix[idx1, idx2] = 1
    adj_matrix[idx2, idx1] = 1  # Assuming undirected
print(f"Adjacency matrix built in {time.time() - start_time:.2f} seconds.")

# Compute Laplacian matrix
print("Computing Laplacian matrix...")
start_time = time.time()
degree_matrix = csr_matrix(np.diag(adj_matrix.sum(axis=1)))
laplacian_matrix = degree_matrix - adj_matrix
print(f"Laplacian matrix computed in {time.time() - start_time:.2f} seconds.")

# Perform eigendecomposition with periodic progress
print("Performing eigendecomposition (with progress updates)...")
start_time = time.time()

# Use sparse solver (eigsh) and only compute the first few eigenvectors for speed
k = 100  # Number of eigenvalues/vectors to compute
total_eigenvalues = laplacian_matrix.shape[0]

def eigendecomposition_progress(laplacian_matrix, k):
    # Progress tracker and callback for iterative eigenvalue solver
    num_iter = 0

    def progress_callback(x):
        nonlocal num_iter
        num_iter += 1
        progress = (num_iter / k) * 100
        print(f"Progress: {progress:.2f}% complete")

    # Perform eigendecomposition on sparse Laplacian matrix with callback
    eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=k, which='SM', tol=0.0, maxiter=1000)
    return eigenvalues, eigenvectors

eigenvalues, eigenvectors = eigendecomposition_progress(laplacian_matrix, k)
print(f"Eigendecomposition completed in {time.time() - start_time:.2f} seconds.")

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
    data = eigenvectors_plot  # shape (num_components, num_eigenvectors)

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
