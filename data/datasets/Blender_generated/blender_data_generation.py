import json
import os
import random
from pathlib import Path

import bpy
import bpy_extras
import numpy as np
from mathutils import Vector

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = Path(bpy.data.filepath).parent / "blender_dataset_renders/"
TEMP_TEX_DIR = Path(bpy.data.filepath).parent / "blender_temp_textures/"
INPUT_BASE_DIR = Path(bpy.data.filepath).parent / "downloaded_textures"

TARGET_NAME = "Plane"
RENDERS_PER_TEXTURE = 5
UV_SCALE = (3.0, 3.0, 1.0)

BASE_RESOLUTION_X = 1024
BASE_RESOLUTION_Y = 1024

# Amount of Gaussian noise to add to the LQ images (0.05 is usually a good realistic baseline)
NOISE_STRENGTH = 0.05
LOW_RES_PERCENTAGE = 50

X_RANGE = (-2.5, 2.5)
Y_RANGE = (-2.5, 2.5)
Z_RANGE = (5.0, 8.0)
SHIFT_RADIUS = 0.5

PLANE_ANCHORS_3D = [
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (-1.0, -1.0, 0.0),
    (1.0, -1.0, 0.0)
]

# ==========================================
# GLOBALS FOR EVENT-DRIVEN RENDERING
# ==========================================
RENDER_QUEUE = []
CURRENT_DEAD_ON_HOMOGRAPHY = None
CURRENT_MATRIX_DATA = {}
TOTAL_TASKS = 0
CURRENT_TASK_INFO = None  # Used to track the current render for post-processing


# ==========================================
# SYSTEM HELPERS
# ==========================================
def enable_gpu_rendering():
    """Forces Blender to use the GPU for Cycles rendering."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences
    cycles_prefs.refresh_devices()

    compute_types = ['OPTIX', 'CUDA', 'METAL', 'HIP', 'ONEAPI']
    for compute_type in compute_types:
        try:
            cycles_prefs.compute_device_type = compute_type
            break
        except TypeError:
            pass

    gpu_found = False
    for device in cycles_prefs.devices:
        if device.type != 'CPU':
            device.use = True
            gpu_found = True
            print(f"--> Enabled GPU: {device.name} via {cycles_prefs.compute_device_type}")

    if not gpu_found:
        print("--> WARNING: No compatible GPU found. Falling back to CPU.")


# ==========================================
# MATH & GEOMETRY HELPERS
# ==========================================
def get_plane_to_camera_homography(scene, camera):
    """Computes an exact 3x3 homography matrix mapping a Z=0 plane to camera pixels."""
    # Factor in the resolution percentage
    res_x = scene.render.resolution_x * (scene.render.resolution_percentage / 100.0)
    res_y = scene.render.resolution_y * (scene.render.resolution_percentage / 100.0)

    # 1. View Matrix (World to Camera Space)
    V = camera.matrix_world.inverted()

    # 2. Projection Matrix (Camera Space to Normalized Device Coordinates / NDC)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    P = camera.calc_matrix_camera(
        depsgraph,
        x=int(res_x),
        y=int(res_y),
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y
    )

    # 3. Full Transform (World Space to NDC)
    M = P @ V
    # 4. Extract the Z=0 plane
    # By dropping the 3rd column (index 2), we convert the 4x4 3D projection
    # into a 3x3 2D planar projection mapping (X, Y, 1) to (X_ndc, Y_ndc, W)
    M_3x3 = np.array([
        [M[0][0], M[0][1], M[0][3]],
        [M[1][0], M[1][1], M[1][3]],
        [M[3][0], M[3][1], M[3][3]]
    ])
    # 5. NDC to Pixel Coordinates Matrix (Top-Left Origin)
    # Blender NDC goes from -1 to 1. Image pixels go from 0 to Resolution.
    S = np.array([
        [res_x / 2.0, 0.0, res_x / 2.0],
        [0.0, -res_y / 2.0, res_y / 2.0],
        [0.0, 0.0, 1.0]
    ])
    return S @ M_3x3

def setup_tracking_target(camera):
    if "CameraTracker" in bpy.data.objects:
        tracker = bpy.data.objects["CameraTracker"]
    else:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
        tracker = bpy.context.active_object
        tracker.name = "CameraTracker"

    camera.constraints.clear()
    track_constraint = camera.constraints.new(type='TRACK_TO')
    track_constraint.target = tracker
    track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    track_constraint.up_axis = 'UP_Y'
    return tracker


# ==========================================
# POST-PROCESSING
# ==========================================
def apply_gaussian_noise(filepath, strength):
    """Uses Numpy to instantly load the saved image, add noise, and resave it."""
    try:
        img = bpy.data.images.load(str(filepath))

        # Read pixels quickly into a numpy array
        pixels = np.empty(len(img.pixels), dtype=np.float32)
        img.pixels.foreach_get(pixels)

        # Reshape flat array into (Number_of_Pixels, 4) for RGBA
        pixels = pixels.reshape(-1, 4)

        # Generate Gaussian noise matching the RGB channels
        noise = np.random.normal(loc=0.0, scale=strength, size=(pixels.shape[0], 3)).astype(np.float32)

        # Add noise to RGB only (leave Alpha alone) and clip between 0.0 and 1.0
        pixels[:, :3] = np.clip(pixels[:, :3] + noise, 0.0, 1.0)

        # Write pixels back rapidly and save
        img.pixels.foreach_set(pixels.ravel())
        img.save()

        # Free up memory
        bpy.data.images.remove(img)
    except Exception as e:
        print(f"Error applying noise to {filepath}: {e}")


# ==========================================
# MATERIAL SETUP
# ==========================================
def setup_pbr_material(target_obj, tex_dir):
    mat_name = f"Mat_{tex_dir.name}"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
        mat.node_tree.nodes.clear()
    else:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        mat.node_tree.nodes.clear()

    if not target_obj.data.materials:
        target_obj.data.materials.append(mat)
    else:
        target_obj.data.materials[0] = mat

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-1000, 0)

    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-800, 0)
    mapping.inputs['Scale'].default_value = UV_SCALE

    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    output_node = nodes.new('ShaderNodeOutputMaterial')
    output_node.location = (300, 0)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    def add_texture_node(filename, y_loc, is_color_data=False):
        img_path = tex_dir / filename
        if not img_path.exists():
            return None
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.location = (-400, y_loc)
        try:
            img = bpy.data.images.load(str(img_path))
            if not is_color_data:
                img.colorspace_settings.name = 'Non-Color'
            tex_node.image = img

            links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
            return tex_node
        except Exception as e:
            print(f"Failed loading texture {filename}: {e}")
            return None

    # Base Color / Diffuse
    color_node = add_texture_node("basecolor.png", 300, True) or add_texture_node("diffuse.png", 300, True)
    if color_node: links.new(color_node.outputs['Color'], bsdf.inputs['Base Color'])

    # Metallic
    metallic_node = add_texture_node("metallic.png", 0, False)
    if metallic_node: links.new(metallic_node.outputs['Color'], bsdf.inputs['Metallic'])

    # Specular
    specular_node = add_texture_node("specular.png", -150, False)
    if specular_node:
        # Blender 4.0 renamed 'Specular' to 'Specular IOR Level'. This handles both older and newer versions.
        if 'Specular IOR Level' in bsdf.inputs:
            links.new(specular_node.outputs['Color'], bsdf.inputs['Specular IOR Level'])
        elif 'Specular' in bsdf.inputs:
            links.new(specular_node.outputs['Color'], bsdf.inputs['Specular'])

    # Roughness
    roughness_node = add_texture_node("roughness.png", -300, False)
    if roughness_node: links.new(roughness_node.outputs['Color'], bsdf.inputs['Roughness'])

    # Normal
    normal_node = add_texture_node("normal.png", -600, False)
    if normal_node:
        normal_map_node = nodes.new('ShaderNodeNormalMap')
        normal_map_node.location = (-150, -600)
        links.new(normal_node.outputs['Color'], normal_map_node.inputs['Color'])
        links.new(normal_map_node.outputs['Normal'], bsdf.inputs['Normal'])

    # Displacement / Height Fallback
    disp_node = add_texture_node("displacement.png", -900, False) or add_texture_node("height.png", -900, False)
    if disp_node:
        disp_map_node = nodes.new('ShaderNodeDisplacement')
        disp_map_node.location = (0, -300)
        links.new(disp_node.outputs['Color'], disp_map_node.inputs['Height'])
        links.new(disp_map_node.outputs['Displacement'], output_node.inputs['Displacement'])


# ==========================================
# ASYNCHRONOUS EXECUTION & BATCH POST-PROCESSING
# ==========================================
RENDER_QUEUE = []
CURRENT_DEAD_ON_PIXELS = None
LQ_FILES_TO_NOISE = []


def apply_noise_batch():
    """Runs at the very end to apply noise to all generated LQ images safely."""
    total = len(LQ_FILES_TO_NOISE)
    print(f"\n=== APPLYING NOISE TO {total} IMAGES ===")

    for idx, filepath in enumerate(LQ_FILES_TO_NOISE):
        if not os.path.exists(filepath):
            continue

        print(f"Applying noise [{idx + 1}/{total}]: {os.path.basename(filepath)}")
        try:
            img = bpy.data.images.load(str(filepath))
            pixels = np.empty(len(img.pixels), dtype=np.float32)
            img.pixels.foreach_get(pixels)

            pixels = pixels.reshape(-1, 4)
            noise = np.random.normal(loc=0.0, scale=NOISE_STRENGTH, size=(pixels.shape[0], 3)).astype(np.float32)
            pixels[:, :3] = np.clip(pixels[:, :3] + noise, 0.0, 1.0)

            img.pixels.foreach_set(pixels.ravel())
            img.save()
            bpy.data.images.remove(img)
        except Exception as e:
            print(f"  -> Error on {os.path.basename(filepath)}: {e}")

    print("\n=== DATASET GENERATION 100% COMPLETE! ===")
    return None


def trigger_next_render():
    """Pops the next task and starts rendering. Runs outside the handler context."""
    global RENDER_QUEUE, CURRENT_DEAD_ON_HOMOGRAPHY, LQ_FILES_TO_NOISE

    if not RENDER_QUEUE:
        print("\n=== RENDERING COMPLETE! STARTING NOISE PASS ===")
        clear_handlers()
        # Give the OS 1 second to release the final rendered image lock before applying noise
        bpy.app.timers.register(apply_noise_batch, first_interval=1.0)
        return None

    task = RENDER_QUEUE.pop(0)

    scene = bpy.context.scene
    camera = scene.camera
    tracker = bpy.data.objects.get("CameraTracker")
    target_obj = scene.objects.get(TARGET_NAME)

    if task["is_first_of_texture"]:
        setup_pbr_material(target_obj, task["tex_dir"])

    tracker.location = task["tracker_loc"]
    camera.location = task["camera_loc"]
    scene.render.resolution_percentage = task["res_pct"]
    scene.render.filepath = task["filepath"]

    # Force Blender to recalculate the matrix data for the new camera position
    bpy.context.view_layer.update()

    if task["is_dead_on"]:
        # Store the mapping from the 3D Plane to the High-Quality dead-on camera pixels
        CURRENT_DEAD_ON_HOMOGRAPHY = get_plane_to_camera_homography(scene, camera)
    else:
        # Calculate mapping from the 3D Plane to the Low-Quality angled camera pixels
        H_angled = get_plane_to_camera_homography(scene, camera)

        # Mathematically link them:
        # H_matrix maps (LQ Angled Pixels) -> (HQ Dead-on Pixels)
        # Equation: H_deadon * inverse(H_angled)
        H_matrix = CURRENT_DEAD_ON_HOMOGRAPHY @ np.linalg.inv(H_angled)

        # Normalize the matrix (Standardizes the scale so the bottom-right value is 1.0)
        H_matrix = H_matrix / H_matrix[2, 2]

        LQ_FILES_TO_NOISE.append(task["filepath"])

        # Write to JSON
        json_path = os.path.join(OUTPUT_DIR, f"{task['base_filename']}_matrices.json")
        matrix_data = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                matrix_data = json.load(f)

        matrix_data[os.path.basename(task["filepath"])] = {
            "target_hq_file": os.path.basename(task["hq_filepath"]),
            "homography_matrix": H_matrix.tolist()
        }
        with open(json_path, 'w') as f:
            json.dump(matrix_data, f, indent=4)
    print(f"Rendering: {os.path.basename(task['filepath'])}")
    bpy.ops.render.render('INVOKE_DEFAULT', write_still=True)

    return None  # Timers must return None to stop looping


import bpy.app.handlers


@bpy.app.handlers.persistent
def on_render_complete(scene=None, dummy=None):
    """Fires when a frame finishes. Uses a 0.5s timer to safely escape context before starting the next."""
    bpy.app.timers.register(trigger_next_render, first_interval=0.5)


def clear_handlers():
    for handler in list(bpy.app.handlers.render_complete):
        if handler.__name__ == "on_render_complete":
            bpy.app.handlers.render_complete.remove(handler)


def main():
    global RENDER_QUEUE, LQ_FILES_TO_NOISE

    scene = bpy.context.scene
    camera = scene.camera
    target_obj = scene.objects.get(TARGET_NAME)

    if not target_obj:
        print(f"ERROR: Target object '{TARGET_NAME}' not found.")
        return

    enable_gpu_rendering()
    scene.cycles.samples = 128
    scene.render.resolution_x = BASE_RESOLUTION_X
    scene.render.resolution_y = BASE_RESOLUTION_Y
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    setup_tracking_target(camera)

    # 1. Recursively find all manifest.json files in all subdirectories
    manifest_files = list(INPUT_BASE_DIR.rglob("metadata.json"))

    if not manifest_files:
        print(f"ERROR: No manifest.json files found in any subdirectories of {INPUT_BASE_DIR}.")
        return

    RENDER_QUEUE = []
    LQ_FILES_TO_NOISE = []

    print(f"Found {len(manifest_files)} texture manifests. Building queue...")

    # 2. Build the Queue using the found manifests
    for manifest_path in manifest_files:
        # The texture directory is whichever folder contains this specific manifest.json
        tex_dir = manifest_path.parent
        texture_name = tex_dir.name

        # Optional: If you need to read the JSON to get specific metadata, you can do it here:
        # with open(manifest_path, 'r') as f:
        #     manifest_data = json.load(f)

        base_filename = f"tex_{texture_name.replace(' ', '_')}"
        hq_filepath = os.path.join(OUTPUT_DIR, f"{base_filename}_deadon_HQ.png")

        # Add HQ Task
        RENDER_QUEUE.append({
            "is_first_of_texture": True, "tex_dir": tex_dir, "is_dead_on": True,
            "tracker_loc": (0.0, 0.0, 0.0), "camera_loc": (0.0, 0.0, 8.0),
            "res_pct": 100, "filepath": hq_filepath, "base_filename": base_filename,
            "hq_filepath": hq_filepath
        })

        # Add LQ Tasks
        for i in range(RENDERS_PER_TEXTURE):
            lq_filepath = os.path.join(OUTPUT_DIR, f"{base_filename}_angle_{i:03d}_LQ.png")
            RENDER_QUEUE.append({
                "is_first_of_texture": False, "tex_dir": tex_dir, "is_dead_on": False,
                "tracker_loc": (
                random.uniform(-SHIFT_RADIUS, SHIFT_RADIUS), random.uniform(-SHIFT_RADIUS, SHIFT_RADIUS), 0.0),
                "camera_loc": (random.uniform(*X_RANGE), random.uniform(*Y_RANGE), random.uniform(*Z_RANGE)),
                "res_pct": LOW_RES_PERCENTAGE, "filepath": lq_filepath, "base_filename": base_filename,
                "hq_filepath": hq_filepath
            })

    # 3. Hand off to the asynchronous render loop
    clear_handlers()
    bpy.app.handlers.render_complete.append(on_render_complete)

    print(f"\n=== QUEUE BUILT: {len(RENDER_QUEUE)} TASKS ===")
    trigger_next_render()


if __name__ == "__main__":
    main()
