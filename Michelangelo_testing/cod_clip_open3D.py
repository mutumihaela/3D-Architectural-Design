import os
import re
import torch
import clip
import open3d as o3d
import numpy as np
from PIL import Image
from subprocess import run
import shutil
import glob

# ---------- Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

prompt_file = '/shared_storage/mutumihaela/michelangelo/prompts_list.txt'
output_dir = '/shared_storage/mutumihaela/michelangelo/output'
text2mesh_script = '/shared_storage/mutumihaela/michelangelo/Michelangelo/scripts/inference/text2mesh.sh'
os.makedirs(output_dir, exist_ok=True)

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_score_file = "/shared_storage/mutumihaela/michelangelo/clip_scores.txt "

def sanitize_filename(text):
    return re.sub(r'[^\w\s-]', '', text).strip().replace(' ', '_')

def render_mesh_open3d(mesh_path, save_path):
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
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

        print(f"🖼️ Rendered and saved: {save_path}")
        del render
    except Exception as e:
        print(f"Error rendering with Open3D: {e}")

# ---------- Main loop ----------
with open(prompt_file, 'r') as f:
    prompts = [line.strip() for line in f if line.strip()]

for idx, prompt in enumerate(prompts):
    print(f"\n[{idx+1}/{len(prompts)}] Prompt: {prompt}")
    torch.cuda.empty_cache()
    prompt_safe = sanitize_filename(prompt)

    # === Generate mesh using Michelangelo ===
    run(["bash", text2mesh_script, prompt, output_dir])

    # === Locate the .obj file ===
    mesh_candidates = glob.glob(os.path.join(output_dir, "text2mesh", "*_out_mesh.obj"))
    if not mesh_candidates:
        print(f"❌ No mesh found for prompt: {prompt}")
        continue

    original_obj_path = mesh_candidates[0]
    obj_path = os.path.join(output_dir, f"{prompt_safe}_{idx}.obj")
    ply_path = os.path.join(output_dir, f"{prompt_safe}_{idx}.ply")
    img_path = os.path.join(output_dir, f"{prompt_safe}_{idx}.png")

    # Save renamed .obj and .ply
    shutil.copy(original_obj_path, obj_path)
    print(f"✅ Saved .obj: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)
    o3d.io.write_triangle_mesh(ply_path, mesh)
    print(f"✅ Saved .ply: {ply_path}")

    # === Render ===
    render_mesh_open3d(obj_path, img_path)

    # === CLIP Score ===
    if os.path.exists(img_path):
        image_input = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        text_input = clip.tokenize([prompt]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            clip_score = torch.cosine_similarity(image_features, text_features).item()

        print(f"📈 CLIP Score: {clip_score:.4f}")
        with open(clip_score_file, 'a') as f:
            f.write(f"{prompt_safe}_{idx}: {clip_score:.4f}\n")
    else:
        print(f"⚠️ Image not found for CLIP scoring: {img_path}")

    torch.cuda.empty_cache()
