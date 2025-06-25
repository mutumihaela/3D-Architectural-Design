import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def estimate_normals_knn(pc, k=30):
    neigh = NearestNeighbors(n_neighbors=k).fit(pc)
    _, idx = neigh.kneighbors(pc)
    normals = []
    for i in range(pc.shape[0]):
        neighbors = pc[idx[i]]
        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals.append(eigvecs[:, 0])  # smallest eigenvector
    return np.array(normals)

# Load (N, 2048, 3) point clouds
pcs = np.load("/shared_storage/mutumihaela/michelangelo/p-fid/modelnet40_real_pointclouds.npy")
pcs_with_normals = []

for pc in tqdm(pcs, desc="Estimating normals (PCA fallback)"):
    try:
        normals = estimate_normals_knn(pc)
        pcs_with_normals.append(np.concatenate([pc, normals], axis=1))
    except Exception as e:
        print("Error:", e)
        continue

pcs_with_normals = np.stack(pcs_with_normals)
np.save("/shared_storage/mutumihaela/michelangelo/p-fid/modelnet40_real_pointclouds_6D.npy", pcs_with_normals)
print(f"✅ Done! Final shape: {pcs_with_normals.shape}")
