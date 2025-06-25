# import os
# import re
# import sys
# import torch
# import clip
# import open3d as o3d
# import numpy as np
# from PIL import Image
# from tqdm import tqdm

# sys.path.append("/shared_storage/mutumihaela/finetune-shape/Cap3D/text-to-3D/shap-e/shap_e/models")

# from shap_e.models.download import load_model, load_config
# from shap_e.diffusion.sample import sample_latents
# from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
# from shap_e.util.notebooks import decode_latent_mesh

# # ========== CONFIG ==========
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# prompt_file = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/prompts_list.txt"
# output_dir = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/finetune3/output"
# result_file = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/generation_results.txt"
# clip_threshold = 0.25
# max_attempts = 3

# # ========== UTILITY ==========
# def prompt_to_filename(prompt):
#     clean = re.sub(r'[^a-zA-Z0-9_]', '_', prompt.strip().lower())
#     return re.sub(r'_+', '_', clean).strip('_')

# def read_prompts(file_path):
#     with open(file_path, 'r') as f:
#         return [line.strip() for line in f if line.strip()]

# def render_mesh_open3d(mesh, save_path):
#     mesh.compute_vertex_normals()
#     width, height = 512, 512
#     render = o3d.visualization.rendering.OffscreenRenderer(width, height)
#     material = o3d.visualization.rendering.MaterialRecord()
#     material.shader = "defaultLit"
#     render.scene.add_geometry("mesh", mesh, material)
#     cam_position = [1.8, 1.8, 1.8]
#     look_at = [0, 0, 0]
#     up_vector = [0, 1, 0]
#     render.scene.camera.look_at(look_at, cam_position, up_vector)
#     img = render.render_to_image()
#     img_np = np.asarray(img)
#     img_pil = Image.fromarray(img_np)
#     img_pil.save(save_path)
#     del render

# def compute_clip_score(image_path, text, clip_model, clip_preprocess):
#     image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
#     tokens = clip.tokenize([text]).to(device)
#     with torch.no_grad():
#         image_features = clip_model.encode_image(image)
#         text_features = clip_model.encode_text(tokens)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#         return (image_features @ text_features.T).item()

# # ========== SETUP ==========
# os.makedirs(output_dir, exist_ok=True)
# xm = load_model('transmitter', device=device).float()
# model = load_model('text300M', device=device).float()
# diffusion = diffusion_from_config(load_config('diffusion'))
# model.eval()

# clip_model, clip_preprocess = clip.load(
#     "ViT-B/32",
#     device=device,
#     download_root="/shared_storage/mutumihaela/clip_cache"
# )

# # ========== GENERATE ==========
# prompts = read_prompts(prompt_file)
# print(f"\n→ Starting generation for {len(prompts)} prompts...\n")

# clip_scores = []

# with open(result_file, 'w') as log_file:
#     for i, prompt in enumerate(prompts):
#         name = prompt_to_filename(prompt)
#         final_obj = os.path.join(output_dir, f"{name}.obj")
#         final_img = os.path.join(output_dir, f"{name}.png")

#         if os.path.exists(final_obj) and os.path.exists(final_img):
#             print(f"[skip] {final_obj} already exists.")
#             continue

#         print(f"[{i+1}/{len(prompts)}] Generating: \"{prompt}\"")
#         best_score = -1
#         best_obj, best_img = None, None

#         for attempt in range(max_attempts):
#             with torch.no_grad():
#                 latents = sample_latents(
#                     batch_size=1,
#                     model=model,
#                     diffusion=diffusion,
#                     guidance_scale=15.0,
#                     model_kwargs={"texts": [prompt]},
#                     progress=True,
#                     clip_denoised=True,
#                     use_fp16=False,
#                     use_karras=True,
#                     karras_steps=32,
#                     sigma_min=1e-3,
#                     sigma_max=160,
#                     s_churn=0,
#                 )

#             mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
#             temp_obj = os.path.join(output_dir, f"{name}_try{attempt+1}.obj")
#             temp_img = os.path.join(output_dir, f"{name}_try{attempt+1}.png")

#             with open(temp_obj, 'w') as f:
#                 mesh.write_obj(f)

#             render_mesh_open3d(mesh, temp_img)
#             score = compute_clip_score(temp_img, prompt, clip_model, clip_preprocess)

#             if score > best_score:
#                 best_score = score
#                 best_obj = temp_obj
#                 best_img = temp_img

#             if score >= clip_threshold:
#                 break

#         if best_obj and best_img:
#             os.rename(best_obj, final_obj)
#             os.rename(best_img, final_img)
#             for f in os.listdir(output_dir):
#                 if f.startswith(name + "_try"):
#                     os.remove(os.path.join(output_dir, f))

#             print(f"[✓] Saved best: {final_obj} | CLIP score: {best_score:.4f}")
#             log_file.write(f"{prompt}\t{final_obj}\t{final_img}\t{best_score:.4f}\n")
#             clip_scores.append(best_score)
#         else:
#             print(f"[x] Failed to generate valid object for prompt: {prompt}")

#     if clip_scores:
#         mean_score = sum(clip_scores) / len(clip_scores)
#         print(f"\n→ Media scorurilor CLIP: {mean_score:.4f}")
#         log_file.write(f"\nAverage CLIP score: {mean_score:.4f}\n")
#     else:
#         print("\n→ Nu s-a generat niciun obiect valid pentru scor CLIP.")
#         log_file.write("\nNo valid CLIP scores were recorded.\n")



import os
import re
import sys
import torch
import clip
import open3d as o3d
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append("/shared_storage/mutumihaela/finetune-shape/Cap3D/text-to-3D/shap-e/shap_e/models")

from shap_e.models.download import load_model, load_config
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import decode_latent_mesh

# ========== CONFIG ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prompt_file = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/prompts_list.txt"
output_dir = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/finetune3/output"
result_file = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/generation_results.txt"
clip_threshold = 0.25
max_attempts = 3

# ========== UTILITY ==========
def prompt_to_filename(prompt):
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', prompt.strip().lower())
    return re.sub(r'_+', '_', clean).strip('_')

def read_prompts(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def shapE_trimesh_to_open3d(tri_mesh):
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(tri_mesh.verts)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    return mesh_o3d

def render_mesh_open3d(mesh, save_path):
    mesh.compute_vertex_normals()
    width, height = 512, 512
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    render.scene.add_geometry("mesh", mesh, material)
    cam_position = [1.8, 1.8, 1.8]
    look_at = [0, 0, 0]
    up_vector = [0, 1, 0]
    render.scene.camera.look_at(look_at, cam_position, up_vector)
    img = render.render_to_image()
    img_np = np.asarray(img)
    img_pil = Image.fromarray(img_np)
    img_pil.save(save_path)
    del render

def compute_clip_score(image_path, text, clip_model, clip_preprocess):
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T).item()

# ========== SETUP ==========
os.makedirs(output_dir, exist_ok=True)
xm = load_model('transmitter', device=device).float()
model = load_model('text300M', device=device).float()
diffusion = diffusion_from_config(load_config('diffusion'))
model.eval()

clip_model, clip_preprocess = clip.load(
    "ViT-B/32",
    device=device,
    download_root="/shared_storage/mutumihaela/clip_cache"
)

# ========== GENERATE ==========
prompts = read_prompts(prompt_file)
print(f"\n→ Starting generation for {len(prompts)} prompts...\n")

clip_scores = []

with open(result_file, 'w') as log_file:
    for i, prompt in enumerate(prompts):
        name = prompt_to_filename(prompt)
        final_obj = os.path.join(output_dir, f"{name}.obj")
        final_img = os.path.join(output_dir, f"{name}.png")

        if os.path.exists(final_obj) and os.path.exists(final_img):
            print(f"[skip] {final_obj} already exists.")
            continue

        print(f"[{i+1}/{len(prompts)}] Generating: \"{prompt}\"")
        best_score = -1
        best_obj, best_img = None, None

        for attempt in range(max_attempts):
            with torch.no_grad():
                latents = sample_latents(
                    batch_size=1,
                    model=model,
                    diffusion=diffusion,
                    guidance_scale=15.0,
                    model_kwargs={"texts": [prompt]},
                    progress=True,
                    clip_denoised=True,
                    use_fp16=False,
                    use_karras=True,
                    karras_steps=32,
                    sigma_min=1e-3,
                    sigma_max=160,
                    s_churn=0,
                )

            tri_mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()

            temp_obj = os.path.join(output_dir, f"{name}_try{attempt+1}.obj")
            temp_img = os.path.join(output_dir, f"{name}_try{attempt+1}.png")

            with open(temp_obj, 'w') as f:
                tri_mesh.write_obj(f)

            mesh_o3d = shapE_trimesh_to_open3d(tri_mesh)
            render_mesh_open3d(mesh_o3d, temp_img)
            score = compute_clip_score(temp_img, prompt, clip_model, clip_preprocess)

            if score > best_score:
                best_score = score
                best_obj = temp_obj
                best_img = temp_img

            if score >= clip_threshold:
                break

        if best_obj and best_img:
            os.rename(best_obj, final_obj)
            os.rename(best_img, final_img)
            for f in os.listdir(output_dir):
                if f.startswith(name + "_try"):
                    os.remove(os.path.join(output_dir, f))

            print(f"[✓] Saved best: {final_obj} | CLIP score: {best_score:.4f}")
            log_file.write(f"{prompt}\t{final_obj}\t{final_img}\t{best_score:.4f}\n")
            log_file.flush()
            clip_scores.append(best_score)
        else:
            print(f"[x] Failed to generate valid object for prompt: {prompt}")

    if clip_scores:
        mean_score = sum(clip_scores) / len(clip_scores)
        print(f"\n→ Media scorurilor CLIP: {mean_score:.4f}")
        log_file.write(f"\nAverage CLIP score: {mean_score:.4f}\n")
    else:
        print("\n→ Nu s-a generat niciun obiect valid pentru scor CLIP.")
        log_file.write("\nNo valid CLIP scores were recorded.\n")
