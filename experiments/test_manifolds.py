import numpy as np
import torch
from experiments.tangential_delaunay import build_tangential_complex, euler_characteristic

def sample_sphere(n_samples: int, dim: int, noise: float = 0.0) -> np.ndarray:
    """Samples points uniformly from a `dim`-sphere embedded in R^{dim+1}."""
    X = np.random.randn(n_samples, dim + 1)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    if noise > 0:
        X += np.random.randn(*X.shape) * noise
    return X

def test_manifold(name: str, X: np.ndarray, k: int, expected_chi: int, n_neighbors: int):
    print(f"\n--- Testing {name} ---")
    print(f"Points: {X.shape[0]}, Ambient dim: {X.shape[1]}, Manifold dim (k): {k}")
    
    result = build_tangential_complex(
        X, 
        k_manifold=k, 
        n_neighbors=n_neighbors, 
        enforce_manifold=True,
        account_for_noise=True,
        max_rounds=100,
        seed=42,
        gpu=-1 # CPU
    )
    
    chi = euler_characteristic(result.simplices)
    print(f"Converged: {result.converged}")
    print(f"Simplices: {len(result.simplices)}")
    print(f"Euler characteristic: {chi} (Expected: {expected_chi})")
    
    if chi == expected_chi:
        print("✅ SUCCESS")
    else:
        print("❌ FAILED")

if __name__ == "__main__":
    np.random.seed(42)
    
    # 2-sphere (k=2), S^2 in R^3, chi = 2
    X_s2 = sample_sphere(800, dim=2, noise=0.05)
    test_manifold("2-Sphere", X_s2, k=2, expected_chi=2, n_neighbors=20)
    
    # 3-sphere (k=3), S^3 in R^4, chi = 0
    X_s3 = sample_sphere(2000, dim=3, noise=0.05)
    test_manifold("3-Sphere", X_s3, k=3, expected_chi=0, n_neighbors=30)
    
    # 4-sphere (k=4), S^4 in R^5, chi = 2
    X_s4 = sample_sphere(4000, dim=4, noise=0.05)
    test_manifold("4-Sphere", X_s4, k=4, expected_chi=2, n_neighbors=40)
