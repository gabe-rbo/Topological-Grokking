import numpy as np
from experiments.test_manifolds import sample_sphere
from experiments.tangential_delaunay import build_tangential_complex, euler_characteristic
import torch

def compare_noise_handling():
    # We use 2000 points here so that clean data easily closes the sphere.
    X_clean = sample_sphere(2000, dim=2, noise=0.0)
    # Using 1% noise instead of 5% so the difference is visibly recoverable by MLS
    X_noisy = sample_sphere(2000, dim=2, noise=0.01)

    print("\n--- 1. CLEAN DATA ---")
    res_clean = build_tangential_complex(X_clean, k_manifold=2, n_neighbors=20, max_rounds=20, enforce_manifold=True, account_for_noise=False, gpu=-1)
    print(f"Converged: {res_clean.converged}")
    print(f"Inconsistent points left: {len(res_clean.failed_points)}")
    print(f"Euler characteristic: {euler_characteristic(res_clean.simplices)} (Expected 2 for S^2)")

    print("\n--- 2. NOISY DATA (WITHOUT MLS) ---")
    res_noisy_no_mls = build_tangential_complex(X_noisy, k_manifold=2, n_neighbors=20, max_rounds=20, enforce_manifold=True, account_for_noise=False, gpu=-1)
    print(f"Converged: {res_noisy_no_mls.converged}")
    print(f"Inconsistent points left: {len(res_noisy_no_mls.failed_points)}")
    print(f"Euler characteristic: {euler_characteristic(res_noisy_no_mls.simplices)}")

    print("\n--- 3. NOISY DATA (WITH MLS) ---")
    res_noisy_with_mls = build_tangential_complex(X_noisy, k_manifold=2, n_neighbors=20, max_rounds=20, enforce_manifold=True, account_for_noise=True, gpu=-1)
    print(f"Converged: {res_noisy_with_mls.converged}")
    print(f"Inconsistent points left: {len(res_noisy_with_mls.failed_points)}")
    print(f"Euler characteristic: {euler_characteristic(res_noisy_with_mls.simplices)}")

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    compare_noise_handling()
