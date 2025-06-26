import os
import sys
import torch
import numpy as np
import trimesh
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy import linalg

#PointNet++ repo path
sys.path.append("/shared_storage/mutumihaela/michelangelo/Pointnet_Pointnet2_pytorch")
from models.pointnet2_cls_msg import get_model

#CONFIG
OBJ_FOLDER = "/shared_storage/mutumihaela/shape-e-bun/output"
REAL_CLOUDS_PATH = "/shared_storage/mutumihaela/michelangelo/p-fid/modelnet40_real_pointclouds_6D.npy"
N_POINTS = 2048
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/shared_storage/mutumihaela/michelangelo/Pointnet_Pointnet2_pytorch/pretrained_pointnet2_msg.pth"
OUTPUT_TXT = "/shared_storage/mutumihaela/shape-e-bun/p-fid/p_fid_results.txt"

#Load Pretrained Classifier
model = get_model(num_class=40, normal_channel=True)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

#Generated Shapes Dataset (.obj to point cloud with normals)
class ObjDataset(Dataset):
    def __init__(self, folder, n_points):
        self.files = [f for f in os.listdir(folder) if f.endswith(".obj")]
        self.folder = folder
        self.n_points = n_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        mesh = trimesh.load(os.path.join(self.folder, fname))
        if not mesh.face_normals.any():
            mesh.compute_vertex_normals()
        pts, face_indices = trimesh.sample.sample_surface(mesh, self.n_points)
        normals = mesh.face_normals[face_indices]
        pts_normals = np.concatenate([pts, normals], axis=1)
        return torch.from_numpy(pts_normals).float(), fname

#Feature extractor
def get_features(model, dataloader):
    feats = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Handle both (points,) and (points, filename)
            if isinstance(batch, (list, tuple)):
                points = batch[0]
            else:
                points = batch
            points = points.to(DEVICE)  # (B, N, 6)
            points = points.permute(0, 2, 1)  # (B, 6, N)
            out, _ = model(points)
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)

#Load Datasets
dataset_gen = ObjDataset(OBJ_FOLDER, N_POINTS)
loader_gen = DataLoader(dataset_gen, batch_size=BATCH_SIZE, shuffle=False)

real_pcs = np.load(REAL_CLOUDS_PATH)  # (N, 2048, 6)
dataset_real = torch.utils.data.TensorDataset(torch.from_numpy(real_pcs).float())
loader_real = DataLoader(dataset_real, batch_size=BATCH_SIZE)

# Extract Features
print("🔍 Extracting features from generated shapes...")
gen_features = get_features(model, loader_gen)

print("🔍 Extracting features from real shapes...")
real_features = get_features(model, loader_real)

#Compute FID
def compute_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

mu_gen = np.mean(gen_features, axis=0)
sigma_gen = np.cov(gen_features, rowvar=False)

mu_real = np.mean(real_features, axis=0)
sigma_real = np.cov(real_features, rowvar=False)

fid_score = compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)

#Save Result
with open(OUTPUT_TXT, "w") as f:
    f.write(f"P-FID: {fid_score:.6f}\n")

print(f" Done! P-FID: {fid_score:.6f}")
print(f" Saved to: {OUTPUT_TXT}")
