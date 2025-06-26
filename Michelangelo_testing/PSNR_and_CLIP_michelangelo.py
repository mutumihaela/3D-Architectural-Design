import os
import re
import torch
import pandas as pd
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import open3d as o3d
from transformers import CLIPProcessor, CLIPModel

DATA_DIR = "/shared_storage/mutumihaela/michelangelo/output"
OUT_CSV = "/shared_storage/mutumihaela/michelangelo/output/metrix/clip_psnr_results.csv"
SUMMARY_CSV = "/shared_storage/mutumihaela/michelangelo/output/metrix/clip_psnr_summary.csv"
TMP_RENDER_DIR = os.path.join(DATA_DIR, "tmp_renders")
os.makedirs(TMP_RENDER_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_prompt(filename):
    name = os.path.splitext(filename)[0]
    words = name.split("_")[:-1]
    text = " ".join(words)
    return re.sub(r'\d+', '', text).strip()

def render_mesh(mesh_path, image_path, width=256, height=256):
    import open3d.visualization.rendering as rendering

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    renderer = rendering.OffscreenRenderer(width, height)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"

    renderer.scene.set_background([1, 1, 1, 1])
    renderer.scene.add_geometry("mesh", mesh, mat)

    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent().max()

    eye = center + np.array([0.0, 0.0, 2.5 * extent])
    up = np.array([0.0, 1.0, 0.0])

    renderer.setup_camera(60.0, center, eye, up)

    img = renderer.render_to_image()
    o3d.io.write_image(image_path, img)

    renderer.scene.clear_geometry()
    try:
        renderer.release()
    except AttributeError:
        del renderer

def load_image(path):
    return Image.open(path).convert("RGB")

def compute_psnr(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    return peak_signal_noise_ratio(arr1, arr2, data_range=255)

#Collect samples
samples = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".png"):
        base = os.path.splitext(file)[0]
        obj_path = os.path.join(DATA_DIR, base + ".obj")
        ply_path = os.path.join(DATA_DIR, base + ".ply")
        if os.path.exists(obj_path) and os.path.exists(ply_path):
            samples.append({
                "filename": base,
                "prompt": extract_prompt(file),
                "image_path": os.path.join(DATA_DIR, file),
                "obj_path": obj_path,
                "ply_path": ply_path
            })

print(f"Found {len(samples)} samples")

#CLIP R-Precision using Hugging Face (Optimized)
prompts = [s["prompt"] for s in samples]
images = [load_image(s["image_path"]) for s in samples]

print("Encoding text prompts...")
text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

print("Encoding images and computing R-Precision...")
clip_r_top1, clip_r_top5, clip_r_top10, clip_r_ranks = [], [], [], []

for i in range(len(images)):
    image_inputs = processor(images=images[i], return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        sims = image_features @ text_features.T
        sorted_indices = torch.argsort(sims.squeeze(0), descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1

        clip_r_ranks.append(rank)
        clip_r_top1.append(1 if rank == 1 else 0)
        clip_r_top5.append(1 if rank <= 5 else 0)
        clip_r_top10.append(1 if rank <= 10 else 0)

#PSNR between .obj and .ply renders
print("Rendering meshes and computing PSNR...")
psnr_scores = []
for s in samples:
    obj_img_path = os.path.join(TMP_RENDER_DIR, s["filename"] + "_obj.png")
    ply_img_path = os.path.join(TMP_RENDER_DIR, s["filename"] + "_ply.png")

    render_mesh(s["obj_path"], obj_img_path)
    render_mesh(s["ply_path"], ply_img_path)

    obj_img = load_image(obj_img_path)
    ply_img = load_image(ply_img_path)

    psnr = compute_psnr(obj_img, ply_img)
    psnr_scores.append(psnr)

#Save full results
df = pd.DataFrame({
    "filename": [s["filename"] for s in samples],
    "prompt": prompts,
    "clip_r_precision_top1": clip_r_top1,
    "clip_r_precision_top5": clip_r_top5,
    "clip_r_precision_top10": clip_r_top10,
    "clip_rank": clip_r_ranks,
    "psnr_obj_vs_ply": psnr_scores
})
df.to_csv(OUT_CSV, index=False)
print(f" Full results saved to: {OUT_CSV}")

# Save summary
summary = {
    "clip_r_precision_top1 (%)": [100 * np.mean(clip_r_top1)],
    "clip_r_precision_top5 (%)": [100 * np.mean(clip_r_top5)],
    "clip_r_precision_top10 (%)": [100 * np.mean(clip_r_top10)],
    "average_clip_rank": [np.mean(clip_r_ranks)],
    "average_psnr": [np.mean(psnr_scores)]
}
df_summary = pd.DataFrame(summary)
df_summary.to_csv(SUMMARY_CSV, index=False)
print(f"✅ Summary saved to: {SUMMARY_CSV}")
