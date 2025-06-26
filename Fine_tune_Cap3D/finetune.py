import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import pickle
import pandas as pd
from torch.cuda.amp import autocast, GradScaler

# Path spre Shape-E
sys.path.append("/shared_storage/mutumihaela/shape-e-bun/shap-e/shap_e/models")
from shap_e.models.generation.transformer import Transformer
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_config

# CLIP inclus direct
class FrozenImageCLIP(nn.Module):
    def __init__(self, device):
        super().__init__()
        from transformers import CLIPProcessor, CLIPModel
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.feature_dim = self.model.text_projection.out_features

    def forward(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        return self.model.get_text_features(**inputs)

class ShapEDataset(Dataset):
    def __init__(self, base_path, split):
        print(f"[DATASET] Initializing dataset for split: {split}")
        self.latent_path = os.path.join(base_path, 'Cap3D_8k')
        self.captions = pd.read_csv(os.path.join(base_path, 'Furniture_8k_dataset.csv'), header=None)
        self.valid_uids = pickle.load(open(os.path.join(base_path, f'{split}_set.pkl'), 'rb'))
        self.name_to_index = {self.captions[0][i]: i for i in range(len(self.captions))}
        print(f"[DATASET] Loaded {len(self.valid_uids)} samples")

    def __len__(self):
        return len(self.valid_uids)

    def __getitem__(self, idx):
        uid = self.valid_uids[idx]
        cap_idx = self.name_to_index[uid]
        latent_path = os.path.join(self.latent_path, uid + '.pt')
        latent = torch.load(latent_path)[:, :512]
        caption = self.captions[1][cap_idx]
        return {'caption': caption, 'latent': latent}

class CustomTextConditionedDiffusionModel(nn.Module):
    def __init__(self, device, dtype=torch.float32, input_channels=6, width=512, n_ctx=512, layers=4, heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, width).to(device=device, dtype=dtype)
        self.transformer = Transformer(device=device, dtype=dtype, n_ctx=n_ctx, width=width, layers=layers, heads=heads).to(device=device, dtype=dtype)
        self.ln_post = nn.LayerNorm(width).to(device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, input_channels).to(device=device, dtype=dtype)
        self.clip = FrozenImageCLIP(device=device)
        self.clip_embed = nn.Linear(self.clip.feature_dim, width).to(device=device, dtype=dtype)

    def forward(self, x, t, texts=None):
        h = self.input_proj(x.permute(0, 2, 1))
        if texts is not None:
            clip_features = self.clip(texts=texts)
            cond = self.clip_embed(clip_features)
            h = h + cond.unsqueeze(1)
        h = self.transformer(h)
        h = self.ln_post(h)
        h = self.output_proj(h)
        return h.permute(0, 2, 1)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    diffusion = diffusion_from_config(load_config('diffusion'))
    model = CustomTextConditionedDiffusionModel(device=device, dtype=dtype).to(device)

    # === Load existing fine-tuned checkpoint ===
    ckpt_path = os.path.join(args.latent_code_path, "checkpoints.pth")
    if os.path.exists(ckpt_path):
        print(f"[LOAD] Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(" Checkpoint loaded.")
    else:
        print(" No checkpoint found. Starting from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    train_dataset = ShapEDataset(args.latent_code_path, 'training')
    val_dataset = ShapEDataset(args.latent_code_path, 'validation')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 3

    for epoch in range(args.epoch):
        print(f"\n[TRAIN] Epoch {epoch + 1}/{args.epoch}")
        model.train()
        for i, batch in enumerate(train_loader):
            captions = batch['caption']
            x_start = batch['latent'].to(device=device, dtype=dtype)
            t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device).long()

            optimizer.zero_grad()
            with autocast():
                output = model(x_start, t, texts=captions)
                loss = ((output - x_start) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"[LOSS] Step {i+1}, Loss: {loss.item():.6f}")

        print(f"[VAL] Evaluating on validation set...")
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                captions = batch['caption']
                x_start = batch['latent'].to(device=device, dtype=dtype)
                t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device).long()
                with autocast():
                    output = model(x_start, t, texts=captions)
                    val_loss = ((output - x_start) ** 2).mean()
                    val_losses.append(val_loss.item())
        mean_val_loss = sum(val_losses) / len(val_losses)
        print(f"[VAL] Mean loss: {mean_val_loss:.6f}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.latent_code_path, "finetuned_3.pth"))
            print(f" Model salvat (val_loss: {best_val_loss:.2f})")
        else:
            epochs_no_improve += 1
            print(f" Fără îmbunătățire. Count: {epochs_no_improve}/{early_stop_patience}")
            if epochs_no_improve >= early_stop_patience:
                print(" Early stopping activat.")
                break

    print(f" Antrenarea s-a încheiat. Cel mai bun loss: {best_val_loss:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_code_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    train(args)
