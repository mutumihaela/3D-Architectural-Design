# import os
# import time
# import re
# import numpy as np
# from PIL import Image
# import open3d as o3d
# import torch
# import clip
# import pandas as pd

# # CONFIG
# input_obj_dir = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/fine_tuned_generations"
# output_png_dir = input_obj_dir + "_png"
# os.makedirs(output_png_dir, exist_ok=True)
# csv_output_path = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/clip_scores_finetuned.csv"

# # Load CLIP
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root="/shared_storage/mutumihaela/clip_cache")


# # Helper: render mesh to image
# def render_mesh_open3d(mesh, save_path):
#     try:
#         mesh.compute_vertex_normals()
#         render = o3d.visualization.rendering.OffscreenRenderer(512, 512)
#         material = o3d.visualization.rendering.MaterialRecord()
#         material.shader = "defaultLit"
#         render.scene.add_geometry("mesh", mesh, material)
#         render.scene.scene.enable_sun_light(True)

#         cam_position = [1.5, 1.5, 1.5]
#         look_at = [0, 0, 0]
#         up_vector = [0, 1, 0]
#         render.scene.camera.look_at(look_at, cam_position, up_vector)

#         img = render.render_to_image()
#         img_np = np.asarray(img)
#         img_pil = Image.fromarray(img_np)
#         img_pil.save(save_path)

#         time.sleep(0.2)  # prevent race condition
#         del render
#         print(f"[✓] Saved: {save_path}")
#         return True

#     except Exception as e:
#         print(f"[✗] Render failed for {save_path}: {e}")
#         return False

# # Helper: compute CLIP similarity
# def compute_clip_similarity(image_path, prompt):
#     try:
#         image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
#         text = clip.tokenize([prompt]).to(device)
#         with torch.no_grad():
#             image_features = clip_model.encode_image(image)
#             text_features = clip_model.encode_text(text)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             text_features /= text_features.norm(dim=-1, keepdim=True)
#             similarity = (image_features @ text_features.T).item()
#         return similarity
#     except Exception as e:
#         print(f"[✗] CLIP failed for {image_path}: {e}")
#         return None

# # Helper: extract prompt from filename
# def filename_to_prompt(filename):
#     name = filename.replace(".obj", "").replace("_0", "")
#     prompt = name.replace("_", " ")
#     return prompt.strip()

# # Main loop
# results = []
# for fname in os.listdir(input_obj_dir):
#     if not fname.endswith(".obj"):
#         continue

#     obj_path = os.path.join(input_obj_dir, fname)
#     png_path = os.path.join(output_png_dir, fname.replace(".obj", ".png"))
#     prompt = filename_to_prompt(fname)

#     try:
#         mesh = o3d.io.read_triangle_mesh(obj_path)
#         rendered = render_mesh_open3d(mesh, png_path)
#         if rendered:
#             clip_score = compute_clip_similarity(png_path, prompt)
#             results.append({
#                 "file": fname,
#                 "prompt": prompt,
#                 "clip_score": clip_score
#             })
#     except Exception as e:
#         print(f"[✗] Failed for {fname}: {e}")

# # Save to CSV
# df = pd.DataFrame(results)
# df.to_csv(csv_output_path, index=False)
# print(f"\n[✔] Saved CLIP scores to: {csv_output_path}")


import os
import time
import re
import numpy as np
from PIL import Image
import open3d as o3d
import torch
import clip
import pandas as pd

# CONFIG
input_obj_dir = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/fine_tuned_generations"
output_png_dir = input_obj_dir + "_png"
os.makedirs(output_png_dir, exist_ok=True)
csv_output_path = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/clip_scores_finetuned.csv"

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root="/shared_storage/mutumihaela/clip_cache")

# Helper: render mesh to image
def render_mesh_open3d(mesh, save_path):
    try:
        mesh.compute_vertex_normals()
        render = o3d.visualization.rendering.OffscreenRenderer(512, 512)
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        render.scene.add_geometry("mesh", mesh, material)
        render.scene.scene.enable_sun_light(True)

        cam_position = [1.5, 1.5, 1.5]
        look_at = [0, 0, 0]
        up_vector = [0, 1, 0]
        render.scene.camera.look_at(look_at, cam_position, up_vector)

        img = render.render_to_image()
        img_np = np.asarray(img)
        img_pil = Image.fromarray(img_np)
        img_pil.save(save_path)

        time.sleep(0.2)
        del render
        print(f"[✓] Saved: {save_path}")
        return True

    except Exception as e:
        print(f"[✗] Render failed for {save_path}: {e}")
        return False

# Helper: compute CLIP similarity
def compute_clip_similarity(image_path, prompt):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            text_features = clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
        return similarity
    except Exception as e:
        print(f"[✗] CLIP failed for {image_path}: {e}")
        return None

# Helper: extract prompt from filename
def filename_to_prompt(filename):
    name = filename.replace(".obj", "").replace("_0", "")
    prompt = name.replace("_", " ")
    return prompt.strip()

# Main loop
results = []
for fname in os.listdir(input_obj_dir):
    if not fname.endswith(".obj"):
        continue

    obj_path = os.path.join(input_obj_dir, fname)
    png_path = os.path.join(output_png_dir, fname.replace(".obj", ".png"))
    prompt = filename_to_prompt(fname)

    try:
        mesh = o3d.io.read_triangle_mesh(obj_path)
        rendered = render_mesh_open3d(mesh, png_path)
        if rendered:
            clip_score = compute_clip_similarity(png_path, prompt)
            results.append({
                "file": fname,
                "prompt": prompt,
                "clip_score": clip_score
            })
    except Exception as e:
        print(f"[✗] Failed for {fname}: {e}")

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(csv_output_path, index=False)
print(f"\n[✔] Saved CLIP scores to: {csv_output_path}")

# Compute mean CLIP score
valid_scores = [r['clip_score'] for r in results if r['clip_score'] is not None]
if valid_scores:
    avg_clip_score = sum(valid_scores) / len(valid_scores)
    print(f"\n📊 Average CLIP score: {avg_clip_score:.4f}")
else:
    print("\n⚠️ No valid CLIP scores were computed.")
