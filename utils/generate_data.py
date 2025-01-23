# Imports
import numpy as np


def generate_data(n, d, rho, eta, seed=0):
    rng = np.random.default_rng(seed)
    
    # Construction of mu_1 and mu_2
    v1 = rng.normal(size=d)
    v1 /= np.linalg.norm(v1) # normalize

    v2 = rng.normal(size=d)
    v2 -= (v2 @ v1) * v1 # remove projection of mu_2 onto mu1 to ensure orthogonality
    v2 /= np.linalg.norm(v2) # normalize

    mu_1 = rho * v1
    mu_2 = rho * v2

    # Covariance matrix
    I_d = np.eye(d)
    Sigma = I_d - (np.outer(mu_1, mu_1) + np.outer(mu_2, mu_2)) / (rho**2)

    # Generate labels
    y = rng.choice([+1, -1], size=n)

    # Generate noise and signal
    noise = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
    
    # mu_1 if y = +1, mu_2 if y = -1
    signals = np.where(y[:, None] == 1, mu_1, mu_2)

    # Assign to tokens
    X = np.empty((n, 2, d), dtype=float)
    token_choices = rng.integers(low=0, high=2, size=n)
    X[np.arange(n), token_choices, :] = signals
    X[np.arange(n), 1 - token_choices, :] = noise

    # Flip labels with probability eta
    rng_flip = np.random.default_rng(seed + 1)
    flip_mask = rng_flip.random(n) < eta
    y_flipped = y.copy()
    y_flipped[flip_mask] = -y_flipped[flip_mask]

    # Indices
    clean_indices = np.where(~flip_mask)[0]
    noisy_indices = np.where(flip_mask)[0]

    return X, y_flipped, clean_indices, noisy_indices
