
import os
import sys
import torch
import numpy as np
import trimesh
from tqdm import tqdm
from scipy.stats import entropy
from torch.utils.data import Dataset, DataLoader

#Add repo path
sys.path.append("/shared_storage/mutumihaela/michelangelo/Pointnet_Pointnet2_pytorch")
from models.pointnet2_cls_msg import get_model

#CONFIG
OBJ_FOLDER = "/shared_storage/mutumihaela/shape-e-bun/output"
N_POINTS = 2048
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/shared_storage/mutumihaela/michelangelo/p-is/pretrained_pointnet2_msg.pth"
OUTPUT_TXT = "/shared_storage/mutumihaela/shape-e-bun/p-is/p-is_scores.txt"

#Load Pretrained PointNet++ MSG
classifier = get_model(num_class=40, normal_channel=True)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.to(DEVICE)
classifier.eval()

print(f" Using device: {DEVICE}")

#Dataset with normals (Option 1)
class ObjPointCloudDataset(Dataset):
    def __init__(self, obj_folder, n_points=2048):
        self.obj_folder = obj_folder
        self.obj_files = [f for f in os.listdir(obj_folder) if f.endswith(".obj")]
        self.n_points = n_points

    def __len__(self):
        return len(self.obj_files)

    def __getitem__(self, idx):
        obj_file = self.obj_files[idx]
        obj_path = os.path.join(self.obj_folder, obj_file)
        mesh = trimesh.load(obj_path)

        if mesh.is_empty:
            raise ValueError(f"Mesh {obj_file} is empty!")

        # Ensure normals are available
        if not mesh.face_normals.any():
            mesh.compute_vertex_normals()

        points, face_indices = trimesh.sample.sample_surface(mesh, self.n_points)
        normals = mesh.face_normals[face_indices]
        points_with_normals = np.concatenate([points, normals], axis=1)  # (n_points, 6)
        points_tensor = torch.from_numpy(points_with_normals).float()

        return points_tensor, obj_file

dataset = ObjPointCloudDataset(OBJ_FOLDER, N_POINTS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# Compute Predictions 
all_predictions = []
all_obj_names = []
individual_scores = []

with torch.no_grad():
    for points, obj_names in tqdm(dataloader):
        points = points.to(DEVICE)  # (B, N, 6)
        points = points.permute(0, 2, 1)  # (B, 6, N)
        outputs, _ = classifier(points)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        all_predictions.append(probs)
        all_obj_names.extend(obj_names)

all_predictions = np.concatenate(all_predictions, axis=0)

#Compute P-IS
p_y = np.mean(all_predictions, axis=0)
kl_divergences = []
for idx, p_y_given_x in enumerate(all_predictions):
    kl = entropy(p_y_given_x, p_y)
    kl_divergences.append(kl)
    individual_scores.append((all_obj_names[idx], np.exp(kl)))

mean_kl = np.mean(kl_divergences)
p_is_total = np.exp(mean_kl)

#Write Results to TXT
with open(OUTPUT_TXT, "w") as f:
    for obj_name, score in individual_scores:
        f.write(f"{obj_name}\t{score:.6f}\n")
    f.write(f"\nTotal P-IS: {p_is_total:.6f}\n")

print(f" Done! P-IS results saved to {OUTPUT_TXT}")
