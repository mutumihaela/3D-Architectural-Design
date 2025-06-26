import os
import torch
import re
import clip
import open3d as o3d
import numpy as np
import time
from PIL import Image
from torchvision import transforms

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

# Ensure GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CLIP Model for Scoring
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Set paths
prompt_file = '/shared_storage/mutumihaela/shape-e-bun/prompts_list.txt'
output_dir = '/shared_storage/mutumihaela/shape-e-bun/output'
os.makedirs(output_dir, exist_ok=True)

# Free up unused GPU memory
torch.cuda.empty_cache()

# Load models (keep everything in float32 to avoid precision mismatch)
xm = load_model('transmitter', device=device).to(device).float()
model = load_model('text300M', device=device).to(device).float()
diffusion = diffusion_from_config(load_config('diffusion'))

# Lower parameters to reduce memory usage
batch_size = 1  # Keep at 1 to avoid VRAM overload
guidance_scale = 10.0  # Lower this value to reduce memory load
karras_steps = 16  # Reduce from 32 to save memory

# Create file for saving CLIP scores
clip_score_file = os.path.join(output_dir, "clip_score.txt")

# Read prompts from file
with open(prompt_file, 'r') as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines

def sanitize_filename(text):
    return re.sub(r'[^\w\s-]', '', text).strip().replace(' ', '_')

def render_mesh_open3d(mesh, save_path):
    """
    Render a 3D mesh using Open3D with basic shading.
    """
    try:
        # Compute vertex normals for shading
        mesh.compute_vertex_normals()

        # Create an offscreen renderer
        width, height = 512, 512
        render = o3d.visualization.rendering.OffscreenRenderer(width, height)

        # **Set a basic material**
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"  # Basic lighting shader

        # Add the mesh to the scene
        render.scene.add_geometry("mesh", mesh, material)

        # **Use Open3D's default settings for better rendering**
        render.scene.scene.enable_sun_light(True)  # Enable default lighting

        # **Adjust camera angle for better visibility**
        cam_position = [1.5, 1.5, 1.5]  # Slightly elevated diagonal view
        look_at = [0, 0, 0]  # Center the object
        up_vector = [0, 1, 0]  # Up direction

        render.scene.camera.look_at(look_at, cam_position, up_vector)

        # **Render the image**
        img = render.render_to_image()

        # **Ensure image saving completes before proceeding**
        img_np = np.asarray(img)  # Convert to NumPy array
        img_pil = Image.fromarray(img_np)
        img_pil.save(save_path)

        # **Check if file exists after saving**
        time.sleep(1)  # Allow disk write
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Rendering failed, image {save_path} was not created.")

        print(f"Rendered and saved improved image: {save_path}")

        # **Free renderer memory**
        del render

    except Exception as e:
        print(f"Error rendering with Open3D: {e}")

for idx, prompt in enumerate(prompts):
    print(f"Processing prompt {idx+1}/{len(prompts)}: {prompt}")

    # Free up memory before each generation
    torch.cuda.empty_cache()

    with torch.no_grad():  # Save VRAM by disabling gradients
        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs={"texts": [prompt] * batch_size},
            progress=True,
            clip_denoised=True,  
            use_fp16=False,  # Ensuring all calculations remain in float32
            use_karras=True,
            karras_steps=karras_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

    # Free memory after sampling
    torch.cuda.empty_cache()

    # Save the latents as meshes without rendering
    for i, latent in enumerate(latents):
        with torch.no_grad():  # Save memory during decoding
            t = decode_latent_mesh(xm, latent.to(device).float()).tri_mesh()  # Ensure tensor is float32

        # Ensure filenames are safe for file systems
        prompt_safe = sanitize_filename(prompt)
        
        # Define output paths
        ply_path = os.path.join(output_dir, f'{prompt_safe}_{idx}_{i}.ply')
        obj_path = os.path.join(output_dir, f'{prompt_safe}_{idx}_{i}.obj')
        img_path = os.path.join(output_dir, f'{prompt_safe}_{idx}_{i}.png')

        # Save mesh as PLY and OBJ files
        with open(ply_path, 'wb') as f:
            t.write_ply(f)
        with open(obj_path, 'w') as f:
            t.write_obj(f)

        print(f"Saved: {ply_path} and {obj_path}")

        # **Render the mesh as an image using Open3D (offscreen)**
        try:
            mesh = o3d.io.read_triangle_mesh(obj_path)  # Load 3D mesh
            render_mesh_open3d(mesh, img_path)
        except Exception as e:
            print(f"Warning: Could not render image for CLIP scoring - {e}")
            continue

        # Ensure image file exists before CLIP processing
        if not os.path.exists(img_path):
            print(f"Skipping CLIP scoring, image file {img_path} not found.")
            continue

        # Compute CLIP score
        image_input = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        text_input = clip.tokenize([prompt]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            clip_score = torch.cosine_similarity(image_features, text_features).item()

        print(f"CLIP Score for {obj_path}: {clip_score}")

        # Save the CLIP score to file
        with open(clip_score_file, "a") as f:
            f.write(f"{prompt_safe}_{idx}_{i}: {clip_score}\n")

    # Free up GPU memory after processing each prompt
    torch.cuda.empty_cache()
