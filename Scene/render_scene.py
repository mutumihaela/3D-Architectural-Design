
import bpy
import json
import os
import mathutils
import math

# CONFIG
JSON_PATH = "/Users/mutumihaela/Desktop/blender/placement_plan.json"
RENDER_DIR = "/Users/mutumihaela/Desktop/blender/renders"
os.makedirs(RENDER_DIR, exist_ok=True)

# Găsește un nume unic de fișier: render_001.png, render_002.png etc.
base_name = "render"
ext = ".png"
i = 1
while True:
    render_name = f"{base_name}_{i:03d}{ext}"
    render_path = os.path.join(RENDER_DIR, render_name)
    if not os.path.exists(render_path):
        break
    i += 1

# Camera 4x4m
POSITION_COORDS = {
    "center": (2.0, 2.0, 0.0),
    "left": (0.7, 2.0, 0.0),
    "right": (3.3, 2.0, 0.0),
    "corner": (0.7, 0.7, 0.0),
    "near wall": (2.0, 0.6, 0.0),
    "near window": (2.0, 3.4, 0.0)
}

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def import_object(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".obj":
        bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=filepath)
    else:
        print(f"[!] Unsupported format: {ext}")
        return []
    return bpy.context.selected_objects

def scale_to_max_height(obj, max_height=1.2):
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    z_min = min(v.z for v in bbox)
    z_max = max(v.z for v in bbox)
    height = z_max - z_min
    if height == 0:
        return
    factor = max_height / height
    obj.scale = (factor, factor, factor)

def align_to_floor(obj):
    bpy.context.view_layer.update()
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_z = min(v.z for v in bbox)
    obj.location.z += (0.0 - min_z)

# MAIN
clear_scene()

with open(JSON_PATH, "r") as f:
    placement_data = json.load(f)

for item in placement_data:
    filepath = item["file"]
    position_name = item["position"]
    location = POSITION_COORDS.get(position_name, (2.0, 2.0, 0.0))

    imported = import_object(filepath)
    for obj in imported:
        obj.rotation_euler = (0, 0, math.radians(180))
        obj.location = location
        scale_to_max_height(obj)
        align_to_floor(obj)
        obj.name = os.path.splitext(os.path.basename(filepath))[0]

# Floor
bpy.ops.mesh.primitive_plane_add(size=4.0, location=(2.0, 2.0, 0))
floor = bpy.context.object
floor.name = "Floor"
floor_mat = bpy.data.materials.new(name="FloorMaterial")
floor_mat.use_nodes = True
bsdf = floor_mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.1, 0.05, 0.02, 1.0)  
floor.data.materials.append(floor_mat)

# Walls
wall_coords = [
    ((2.0, 4.0, 1.5), math.radians(180)),   # Nord
    ((0.0, 2.0, 1.5), math.radians(90)),    # Vest
    ((4.0, 2.0, 1.5), math.radians(-90)),   # Est
]

for i, (loc, rot_z) in enumerate(wall_coords):
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=loc, rotation=(math.radians(90), 0, rot_z))
    wall = bpy.context.object
    wall.name = f"Wall_{i}"

    mat = bpy.data.materials.new(name=f"WallMat_{i}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.96, 0.89, 0.75, 1.0)  # cream
    wall.data.materials.append(mat)

# WINDOW 
bpy.ops.mesh.primitive_plane_add(size=1.6, location=(2.0, 3.99, 1.2), rotation=(math.radians(90), 0, math.radians(180)))
window = bpy.context.object
window.name = "Window"

glass_mat = bpy.data.materials.new(name="GlassMaterial")
glass_mat.use_nodes = True
bsdf = glass_mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Transmission"].default_value = 1.0
bsdf.inputs["Roughness"].default_value = 0.0
window.data.materials.append(glass_mat)

# LIGHT
bpy.ops.object.light_add(type='SUN', location=(5, -5, 6))
light = bpy.context.object
light.data.energy = 4.0

# CAMERA 
bpy.ops.object.camera_add(location=(2.0, -10, 4.5), rotation=(math.radians(75), 0, 0))
cam = bpy.context.object
bpy.context.scene.camera = cam

# Randari 
bpy.context.scene.render.resolution_x = 1280
bpy.context.scene.render.resolution_y = 720
bpy.context.scene.render.filepath = render_path  
bpy.context.scene.render.image_settings.file_format = 'PNG'

# RANDARE FINALĂ
bpy.ops.render.render(write_still=True)
print(f"✅ Imagine salvată: {render_path}")
