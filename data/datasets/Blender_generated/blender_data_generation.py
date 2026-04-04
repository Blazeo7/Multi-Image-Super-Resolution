import bpy
import random
import os
from mathutils import Vector
from pathlib import Path

# Import the dataset library
from datasets import load_dataset

# ==========================================
# CONFIGURATION
# ==========================================
NUM_TEXTURES_TO_PROCESS = 5  # How many textures to pull from the dataset
RENDERS_PER_TEXTURE = 3  # How many random angles to shoot per texture

OUTPUT_DIR = Path(bpy.data.filepath).parent / "blender_dataset_renders/"
TEMP_TEX_DIR = Path(bpy.data.filepath).parent / "blender_temp_textures/"
TARGET_NAME = "Cube"  # The object receiving the texture

X_RANGE = (-8.0, 8.0)
Y_RANGE = (-8.0, 8.0)
Z_RANGE = (2.0, 8.0)


# ==========================================

def setup_material_and_apply_texture(target_obj, image_path):
    """Creates a material, adds an Image Texture node, and links it to the object."""
    mat_name = "MatSynth_AutoMaterial"

    # 1. Create or get the material
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True

    # 2. Assign material to object
    if not target_obj.data.materials:
        target_obj.data.materials.append(mat)
    else:
        target_obj.data.materials[0] = mat

    # 3. Set up the Shader Nodes
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")

    # Check if we already created an Image Texture node previously
    tex_node = nodes.get("Image Texture")
    if not tex_node:
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.location = (-300, 0)
        # Link texture color to BSDF Base Color
        links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])

    # 4. Load the image from disk and assign to the node
    try:
        img = bpy.data.images.load(image_path)
        tex_node.image = img
    except Exception as e:
        print(f"Could not load image {image_path}: {e}")


def render_random_angles(texture_index):
    """Handles the camera movement and rendering for the current texture."""
    scene = bpy.context.scene
    camera = scene.camera
    target_obj = scene.objects.get(TARGET_NAME)

    for i in range(RENDERS_PER_TEXTURE):
        # A. Randomize Camera Location
        camera.location = (
            random.uniform(*X_RANGE),
            random.uniform(*Y_RANGE),
            random.uniform(*Z_RANGE)
        )

        # B. Track Camera to Target
        direction = target_obj.location - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()

        # C. Set Filepath (e.g., tex_001_angle_002.png)
        filename = f"tex_{texture_index:03d}_angle_{i:03d}.png"
        scene.render.filepath = os.path.join(OUTPUT_DIR, filename)

        # D. Render
        print(f"Rendering {filename}...")
        bpy.ops.render.render(write_still=True)


def main():
    scene = bpy.context.scene
    target_obj = scene.objects.get(TARGET_NAME)

    if not target_obj:
        print(f"ERROR: Target object '{TARGET_NAME}' not found.")
        return

    # Ensure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_TEX_DIR, exist_ok=True)

    print("Loading Hugging Face Dataset...")
    # Load dataset in streaming mode
    ds = load_dataset("gvecchio/MatSynth", streaming=True, split="train")
    ds = ds.select_columns(["metadata", "basecolor"])
    ds = ds.filter(lambda x: x["metadata"]["license"] == "CC0")
    ds = ds.filter(lambda x: x["metadata"]["source"] != "deschaintre_2020")

    # Iterate over the dataset stream
    texture_count = 0
    for item in ds:
        if texture_count >= NUM_TEXTURES_TO_PROCESS:
            break

        print(f"--- Processing Texture {texture_count + 1}/{NUM_TEXTURES_TO_PROCESS} ---")

        # 1. Extract the PIL Image from the dataset item
        # The Hugging Face dataset outputs PIL images for image features natively
        pil_image = item["basecolor"]

        # Optional: Resize the PIL image here if you still want the 1024x1024 constraint
        pil_image = pil_image.resize((1024, 1024))

        # 2. Save image to disk temporarily so Blender can read it
        temp_image_path = os.path.join(TEMP_TEX_DIR, f"current_tex_{texture_count}.png")
        pil_image.save(temp_image_path)

        # 3. Apply the texture to our target object in Blender
        setup_material_and_apply_texture(target_obj, temp_image_path)

        # 4. Trigger the multi-angle render loop
        render_random_angles(texture_count)

        texture_count += 1

    print(f"Done! All renders saved to {OUTPUT_DIR}")


# Execute
main()
