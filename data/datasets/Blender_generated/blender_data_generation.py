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
BASE_DIR = Path(bpy.data.filepath).parent.resolve() if bpy.data.filepath else Path.cwd()
OUTPUT_DIR = BASE_DIR / "blender_dataset_renders"
TEMP_TEX_DIR = BASE_DIR / "blender_temp_textures"
INPUT_BASE_DIR = BASE_DIR / "downloaded_textures"

TARGET_NAME = "Plane"
RENDERS_PER_TEXTURE = 5
UV_SCALE = (3.0, 3.0, 1.0)

BASE_RESOLUTION_X = 1024
BASE_RESOLUTION_Y = 1024

NOISE_STRENGTH = 0.05

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
# SYSTEM HELPERS
# ==========================================
def enable_gpu_rendering():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    # Get the Cycles preferences
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences

    # Force Blender to query the system for all devices
    cycles_prefs.refresh_devices()

    compute_types = [ 'CUDA', 'METAL', 'HIP', 'ONEAPI']
    gpu_found = False

    for compute_type in compute_types:
        # 1. Check if the Blender build even supports this API string
        try:
            cycles_prefs.compute_device_type = compute_type
        except TypeError:
            continue

        # 2. Query the hardware specifically for this compute type
        backend_devices = cycles_prefs.get_devices_for_type(compute_type)

        # 3. Check if the backend actually found a compatible GPU
        has_compatible_gpu = any(d.type != 'CPU' for d in backend_devices)

        if has_compatible_gpu:
            gpu_found = True

            # 4. Enable the GPUs and disable the CPU to prevent hybrid rendering bugs
            for device in cycles_prefs.devices:
                if device.type != 'CPU':
                    device.use = True
                    print(f"--> Enabled GPU: {device.name} via {compute_type}")
                else:
                    device.use = False

            # We found the highest priority working API, stop searching
            break

    # 5. Final fallback if absolutely no GPU works with any API
    if gpu_found:
        scene.cycles.device = 'GPU'
    else:
        print("--> WARNING: No compatible GPU found. Falling back to CPU.")
        scene.cycles.device = 'CPU'
        for device in cycles_prefs.devices:
            device.use = (device.type == 'CPU')

def purge_unused_images():
    """Removes unassigned images from Blender's memory to prevent RAM bloat."""
    for img in list(bpy.data.images):
        if not img.users:
            bpy.data.images.remove(img)


# ==========================================
# MATH & GEOMETRY HELPERS
# ==========================================
def get_2d_pixels(scene, camera, points_3d):
    res_x = scene.render.resolution_x * (scene.render.resolution_percentage / 100.0)
    res_y = scene.render.resolution_y * (scene.render.resolution_percentage / 100.0)
    coords_2d = []
    for p in points_3d:
        co_ndc = bpy_extras.object_utils.world_to_camera_view(scene, camera, Vector(p))
        x_pix = co_ndc.x * res_x
        y_pix = (1.0 - co_ndc.y) * res_y
        coords_2d.append((x_pix, y_pix))
    return coords_2d


def compute_homography(src_pts, dst_pts):
    A = []
    for i in range(4):
        x, y = src_pts[i]
        u, v = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]


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

    color_node = add_texture_node("basecolor.png", 300, True) or add_texture_node("diffuse.png", 300, True)
    if color_node: links.new(color_node.outputs['Color'], bsdf.inputs['Base Color'])

    metallic_node = add_texture_node("metallic.png", 0, False)
    if metallic_node: links.new(metallic_node.outputs['Color'], bsdf.inputs['Metallic'])

    specular_node = add_texture_node("specular.png", -150, False)
    if specular_node:
        if 'Specular IOR Level' in bsdf.inputs:
            links.new(specular_node.outputs['Color'], bsdf.inputs['Specular IOR Level'])
        elif 'Specular' in bsdf.inputs:
            links.new(specular_node.outputs['Color'], bsdf.inputs['Specular'])

    roughness_node = add_texture_node("roughness.png", -300, False)
    if roughness_node: links.new(roughness_node.outputs['Color'], bsdf.inputs['Roughness'])

    normal_node = add_texture_node("normal.png", -600, False)
    if normal_node:
        normal_map_node = nodes.new('ShaderNodeNormalMap')
        normal_map_node.location = (-150, -600)
        links.new(normal_node.outputs['Color'], normal_map_node.inputs['Color'])
        links.new(normal_map_node.outputs['Normal'], bsdf.inputs['Normal'])

    disp_node = add_texture_node("displacement.png", -900, False) or add_texture_node("height.png", -900, False)
    if disp_node:
        disp_map_node = nodes.new('ShaderNodeDisplacement')
        disp_map_node.location = (0, -300)
        links.new(disp_node.outputs['Color'], disp_map_node.inputs['Height'])
        links.new(disp_map_node.outputs['Displacement'], output_node.inputs['Displacement'])


# ==========================================
# POST-PROCESSING
# ==========================================
def apply_noise_batch(lq_files):
    total = len(lq_files)
    print(f"\n=== APPLYING NOISE TO {total} IMAGES ===")

    for idx, filepath in enumerate(lq_files):
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


# ==========================================
# MAIN EXECUTION (SYNCHRONOUS)
# ==========================================
def main():
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

    tracker = setup_tracking_target(camera)

    # Pre-calculate dead-on pixels for homography matrices globally
    tracker.location = (0.0, 0.0, 0.0)
    camera.location = (0.0, 0.0, 8.0)
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()

    current_dead_on_pixels = get_2d_pixels(scene, camera, PLANE_ANCHORS_3D)
    manifest_files = sorted(list(INPUT_BASE_DIR.rglob("metadata.json")), reverse=True)

    if not manifest_files:
        print(f"ERROR: No metadata.json files found in any subdirectories of {INPUT_BASE_DIR}.")
        return

    render_queue = []
    lq_files_to_noise = []

    print(f"Found {len(manifest_files)} texture manifests. Building queue...")

    for manifest_path in manifest_files:
        tex_dir = manifest_path.parent
        texture_name = tex_dir.name
        base_filename = f"tex_{texture_name.replace(' ', '_')}"

        hq_file_path = OUTPUT_DIR / f"{base_filename}_deadon_HQ.png"
        tasks_for_this_texture = 0

        # Check if HQ needs rendering
        if not hq_file_path.exists():
            render_queue.append({
                "is_first_of_texture": (tasks_for_this_texture == 0),
                "tex_dir": tex_dir, "is_dead_on": True,
                "tracker_loc": (0.0, 0.0, 0.0), "camera_loc": (0.0, 0.0, 8.0),
                "res_pct": 100, "filepath": str(hq_file_path), "base_filename": base_filename,
                "hq_filepath": str(hq_file_path)
            })
            tasks_for_this_texture += 1
        else:
            print(f"Skipping existing HQ: {hq_file_path.name}")

        # Check which LQs need rendering
        for i in range(RENDERS_PER_TEXTURE):
            lq_file_path = OUTPUT_DIR / f"{base_filename}_angle_{i:03d}_LQ.png"

            if not lq_file_path.exists():
                render_queue.append({
                    "is_first_of_texture": (tasks_for_this_texture == 0),
                    "tex_dir": tex_dir, "is_dead_on": False,
                    "tracker_loc": (
                        random.uniform(-SHIFT_RADIUS, SHIFT_RADIUS), random.uniform(-SHIFT_RADIUS, SHIFT_RADIUS), 0.0),
                    "camera_loc": (random.uniform(*X_RANGE), random.uniform(*Y_RANGE), random.uniform(*Z_RANGE)),
                    "res_pct": 50, "filepath": str(lq_file_path), "base_filename": base_filename,
                    "hq_filepath": str(hq_file_path)
                })
                tasks_for_this_texture += 1
            else:
                print(f"Skipping existing LQ: {lq_file_path.name}")

    print(f"\n=== QUEUE BUILT: {len(render_queue)} TASKS ===")

    if not render_queue:
        print("All files already generated. Nothing to do!")
        return

    # --- SYNCHRONOUS RENDERING LOOP ---
    print("\n=== STARTING RENDER LOOP ===")
    for idx, task in enumerate(render_queue):
        print(f"[{idx + 1}/{len(render_queue)}] Rendering: {os.path.basename(task['filepath'])}")

        if task["is_first_of_texture"]:
            purge_unused_images()
            setup_pbr_material(target_obj, task["tex_dir"])

        tracker.location = task["tracker_loc"]
        camera.location = task["camera_loc"]
        scene.render.resolution_percentage = task["res_pct"]
        scene.render.filepath = task["filepath"]

        bpy.context.view_layer.update()

        # Handle LQ specific calculations (Homography JSON tracking)
        if not task["is_dead_on"]:
            angled_pixels = get_2d_pixels(scene, camera, PLANE_ANCHORS_3D)
            H_matrix = compute_homography(angled_pixels, current_dead_on_pixels)
            lq_files_to_noise.append(task["filepath"])

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

        # Blocking render call (no 'INVOKE_DEFAULT' string)
        bpy.ops.render.render(write_still=True)

    # Post-process noise synchronously once rendering is complete
    print("\n=== RENDERING COMPLETE! STARTING NOISE PASS ===")
    apply_noise_batch(lq_files_to_noise)


if __name__ == "__main__":
    main()