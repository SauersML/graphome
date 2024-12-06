import numpy as np
from scipy.optimize import minimize

class SparseAutoencoder:
    def __init__(self, input_size, hidden_size, sparsity_param=500.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sparsity_param = sparsity_param
        
        # Initialize parameters
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        total_params = hidden_size * input_size + input_size * hidden_size
        self.params = np.random.randn(total_params) * scale
        
    def _unpack_parameters(self, params):
        """Convert flat parameter array into weight matrices"""
        mark = self.hidden_size * self.input_size
        W1 = params[:mark].reshape(self.hidden_size, self.input_size)
        W2 = params[mark:].reshape(self.input_size, self.hidden_size)
        return W1, W2

    def _pack_parameters(self, W1, W2):
        """Convert weight matrices into flat array"""
        return np.concatenate([W1.ravel(), W2.ravel()])
        
    def sigmoid(self, x):
        """Numerically stable sigmoid"""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))
    
    def encode(self, x, params=None):
        """Get hidden layer activation"""
        if params is None:
            params = self.params
        W1, _ = self._unpack_parameters(params)
        return self.sigmoid(W1 @ x)
        
    def decode(self, h, params=None):
        """Reconstruct from hidden layer"""
        if params is None:
            params = self.params
        _, W2 = self._unpack_parameters(params)
        return W2 @ h
    
    def loss_function(self, params, x):
        """Compute total loss with L1 sparsity penalty"""
        W1, W2 = self._unpack_parameters(params)
        
        # Forward pass
        h = self.sigmoid(W1 @ x)
        x_hat = W2 @ h
        
        # Reconstruction error (MSE)
        reconstruction_error = np.mean((x - x_hat) ** 2)
        
        # L1 sparsity penalty
        sparsity_penalty = self.sparsity_param * np.mean(np.abs(h))
        
        return reconstruction_error + sparsity_penalty
        
    def train(self, data,     maxiter=100):
        """Train using L-BFGS-B optimizer"""
        x = data.reshape(-1, 1)  # Reshape to column vector
        
        def callback(params):
            loss = self.loss_function(params, x)
            h = self.encode(x, params)
            sparsity = np.mean(np.abs(h) > 1e-3)
            print(f"Loss: {loss:.6f}, Sparsity: {sparsity:.4f}")
            
        result = minimize(
            fun=self.loss_function,
            x0=self.params,
            args=(x,),
            method='L-BFGS-B',
            callback=callback,
            options={'maxiter': maxiter, 'disp': True}
        )
        
        self.params = result.x
        return result

def train_autoencoder(matrix, hidden_size=32, maxiter=1000):
    """Train autoencoder on a single matrix"""
    input_size = matrix.size
    autoencoder = SparseAutoencoder(input_size, hidden_size)
    
    print(f"Training on {input_size}-dimensional input -> {hidden_size} hidden units")
    result = autoencoder.train(matrix, maxiter=maxiter)
    
    return autoencoder

if __name__ == "__main__":
    # Generate sample matrix
    size = 10
    matrix = np.random.randn(size, size)
    matrix = (matrix + matrix.T) / 2  # Make symmetric
    
    # Train autoencoder
    print("Training autoencoder...")
    ae = train_autoencoder(matrix, hidden_size=32, maxiter=1000)
    
    # Get encoded representation
    x = matrix.reshape(-1, 1)
    encoded = ae.encode(x)
    print(f"\nFinal encoding sparsity: {np.mean(np.abs(encoded) > 1e-3):.4f}")
    
    # Check reconstruction
    decoded = ae.decode(encoded).reshape(matrix.shape)
    error = np.mean((matrix - decoded) ** 2)
    print(f"Reconstruction MSE: {error:.6f}")
    
    np.savez('autoencoder_state.npz', 
             params=ae.params,
             original=matrix,
             encoded=encoded.reshape(-1),
             decoded=decoded)
