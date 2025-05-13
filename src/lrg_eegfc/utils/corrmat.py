from ..shared import *

def clean_correlation_matrix(X: np.ndarray, rowvar: bool = True):
    """
    Given data matrix X (T×N), returns:
      - C_clean: denoised correlation matrix
      - eigvals: all eigenvalues of the empirical C
      - lambda_min, lambda_max: Marchenko–Pastur bounds
      - signal_mask: boolean mask of eigenvalues > lambda_max
    """
    N, T = X.shape
    C = np.corrcoef(X, rowvar=rowvar)
    eigvals, eigvecs = np.linalg.eigh(C)

    Q = T / N
    lambda_min = (1 - np.sqrt(1/Q))**2
    lambda_max = (1 + np.sqrt(1/Q))**2

    signal_mask = eigvals > lambda_max
    V = eigvecs[:, signal_mask]
    L = np.diag(eigvals[signal_mask])

    C_clean = V @ L @ V.T
    np.fill_diagonal(C_clean, 1.0)

    return C_clean, eigvals, eigvecs, lambda_min, lambda_max, signal_mask