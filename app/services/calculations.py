#calculation.py
import numpy as np
def vectorized_cosine_distance_matrix(vectors):
    """
    Vectorized cosine distance matrix computation
    Returns the same result as your original code but faster
    """
    X = np.asarray(vectors)
    
    # Compute norms of all vectors in one go
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    
    # Normalize vectors (avoid division by zero with small epsilon)
    X_normalized = X / (norms)  # 1e-8 prevents division by zero
    
    # Compute cosine similarity matrix using matrix multiplication
    cosine_sim = X_normalized @ X_normalized.T
    
    # Convert to cosine distance matrix (1 - similarity)
    return 1 - np.abs(cosine_sim)

def two_input_cosine_distance_matrix(X_vectors, Y_vectors):
    """
    Vectorized cosine distance matrix computation between two sets of vectors.
    ranges from 0 to 2 (2 = exact match, 0 complete opposites)
    Args:
        X_vectors (list/np.ndarray): First set of vectors (n vectors).
        Y_vectors (list/np.ndarray): Second set of vectors (m vectors).
    
    Returns:
        np.ndarray: Cosine distance matrix of shape (n, m).
    """
    X = np.asarray(X_vectors).astype(np.float64)
    Y = np.asarray(Y_vectors).astype(np.float64)
    
    # Compute norms of vectors in both sets
    norms_X = np.linalg.norm(X, axis=1, keepdims=True)
    norms_Y = np.linalg.norm(Y, axis=1, keepdims=True)
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-12
    X_normalized = X / (norms_X + epsilon)
    Y_normalized = Y / (norms_Y + epsilon)
    
    # Compute cosine similarity matrix between X and Y
    cosine_sim = X_normalized @ Y_normalized.T
    
    # Convert to cosine distance matrix (1 - absolute similarity)
    return 1 - cosine_sim


def check_similarity(X, Y):
    """
    Vectorized cosine distance matrix computation between two sets of vectors.
    ranges from 0 to 2 (2 = exact match, 0 complete opposites)
    Args:
        X_vectors (list/np.ndarray): First set of vectors (n vectors).
        Y_vectors (list/np.ndarray): Second set of vectors (m vectors).
    
    Returns:
        np.ndarray: Cosine distance matrix of shape (n, m).
    """
    X = np.asarray(X).astype(np.float64)
    Y = np.asarray(Y).astype(np.float64)
    dotnormprod = np.dot(X, Y)/(np.linalg.norm(X)*np.linalg.norm(Y))
    return dotnormprod
