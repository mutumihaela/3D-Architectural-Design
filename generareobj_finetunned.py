
# import torch
# import os
# import re
# import sys
# from shap_e.models.download import load_model, load_config
# from shap_e.models.configs import model_from_config
# from shap_e.diffusion.sample import sample_latents
# from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
# from shap_e.util.notebooks import decode_latent_mesh

# # ========== CONFIG ==========
# # Path spre Shape-E local
# sys.path.append("/shared_storage/mutumihaela/finetune-shape/Cap3D/text-to-3D/shap-e/shap_e/models")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Calea către decoderul fine-tunat (transmitter)
# finetuned_ckpt = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/shapE_finetuned_best.pth"

# # Folder unde salvăm .obj
# outdir = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/fine_tuned_generations"
# os.makedirs(outdir, exist_ok=True)

# # Prompturi de test
# prompts = [
#    "Plastic lounge chair with curved back.",
#    "A modern office desk with storage compartments.",
#    "A vibrant colored bed with frame."
# ]

# # ======== UTILITY ========
# def prompt_to_filename(prompt):
#     clean = re.sub(r'[^a-zA-Z0-9_]', '_', prompt.strip().lower())
#     return re.sub(r'_+', '_', clean).strip('_')

# # ========== LOAD MODELS ==========

# # Encoder (text → latent)
# print("→ Loading original text300M encoder...")
# model = load_model('text300M', device=device)

# # Decoder (latent → mesh) — transmitter
# print("→ Loading fine-tuned decoder (transmitter)...")
# xm = load_model('transmitter', device=device)
# checkpoint = torch.load(finetuned_ckpt, map_location=device)
# xm.load_state_dict(checkpoint, strict=False)
# xm.eval()


# # Diffusion config
# diffusion = diffusion_from_config(load_config('diffusion'))

# # ========== GENERATE ==========
# print("\n→ Starting generation...\n")
# for i, prompt in enumerate(prompts):
#     name = prompt_to_filename(prompt)
#     obj_path = os.path.join(outdir, f"{name}.obj")
#     if os.path.exists(obj_path):
#         print(f"[skip] {obj_path} already exists.")
#         continue

#     print(f"[{i+1}/{len(prompts)}] Generating for prompt: \"{prompt}\"")

#     latents = sample_latents(
#         batch_size=1,
#         model=model,
#         diffusion=diffusion,
#         guidance_scale=15.0,
#         model_kwargs=dict(texts=[prompt]),
#         progress=True,
#         clip_denoised=True,
#         use_fp16=True,
#         use_karras=True,
#         karras_steps=64,
#         sigma_min=1e-3,
#         sigma_max=160,
#         s_churn=0,
#     )

#     with torch.no_grad():
#         mesh = decode_latent_mesh(xm, latents).tri_mesh()
#         with open(obj_path, 'w') as f:
#             mesh.write_obj(f)
#         print(f"[✓] Saved: {obj_path}")


import torch
import os
import re
import sys

from shap_e.models.download import load_model, load_config
from shap_e.models.configs import model_from_config
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import decode_latent_mesh

# ========== CONFIG ==========

sys.path.append("/shared_storage/mutumihaela/finetune-shape/Cap3D/text-to-3D/shap-e/shap_e/models")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

finetuned_ckpt = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/shapE_finetuned_best.pth"
outdir = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/fine_tuned_generations"
os.makedirs(outdir, exist_ok=True)

prompt_file = "/shared_storage/mutumihaela/finetune-shape/finetune_cap3d/prompts_list.txt"

# ======== UTILITY ========
def prompt_to_filename(prompt):
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', prompt.strip().lower())
    return re.sub(r'_+', '_', clean).strip('_')

def read_prompts(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [line.strip().strip('"').strip("'") for line in lines if line.strip()]

# ========== LOAD MODELS ==========

print("→ Loading original text300M encoder...")
model = load_model('text300M', device=device)

print("→ Loading fine-tuned decoder (transmitter)...")
xm = load_model('transmitter', device=device)
checkpoint = torch.load(finetuned_ckpt, map_location=device)
xm.load_state_dict(checkpoint, strict=False)
xm.eval()

diffusion = diffusion_from_config(load_config('diffusion'))

# ========== GENERATE ==========
prompts = read_prompts(prompt_file)
print(f"\n→ Starting generation for {len(prompts)} prompts...\n")

for i, prompt in enumerate(prompts):
    name = prompt_to_filename(prompt)
    obj_path = os.path.join(outdir, f"{name}.obj")

    if os.path.exists(obj_path):
        print(f"[skip] {obj_path} already exists.")
        continue

    print(f"[{i+1}/{len(prompts)}] Generating: \"{prompt}\"")

    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=15.0,
        model_kwargs=dict(texts=[prompt]),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=32,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    with torch.no_grad():
        mesh = decode_latent_mesh(xm, latents).tri_mesh()
        with open(obj_path, 'w') as f:
            mesh.write_obj(f)
        print(f"[✓] Saved: {obj_path}")

