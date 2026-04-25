import json
import os
import random
from pathlib import Path

import bpy
import bpy_extras
import numpy as np
from mathutils import Vector, Matrix
from numpy.ma.extras import average

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(bpy.data.filepath).parent.resolve() if bpy.data.filepath else Path.cwd()
OUTPUT_DIR = BASE_DIR / "blender_dataset_renders"
TEMP_TEX_DIR = BASE_DIR / "blender_temp_textures"
INPUT_BASE_DIR = BASE_DIR / "downloaded_textures"
RENDERING_INFO_FILE_NAME = "render_info.json"

TARGET_NAME = "Plane"
LIGHT_OBJECT_NAME = "Sun.001"
TRACKER_NAME = "CameraTracker"

RENDERS_PER_TEXTURE = 5
# number of camera/tracker/light position variations to be rendered
NUM_SCENE_SETTINGS = 1
UV_SCALE = (1.0, 1.0, 1.0)

BASE_RESOLUTION_X = 1024
BASE_RESOLUTION_Y = 1024
# the quality compared to the base image
LOW_RES_PERCENTAGE = 50
# the number of rendering samples. Affects the rendering time
RENDER_SAMPLE_CNT = 128

# the height of the camera for the reference image
REF_CAM_Z = 1.4
X_CAMERA_RANGE = (-0.625, 0.625)
Y_CAMERA_RANGE = (-0.625, 0.625)
Z_CAMERA_RANGE = (1.2, 1.6)
X_TRACKER_RANGE = (-0.1, 0.1)
Y_TRACKER_RANGE = (-0.1, 0.1)
Z_TRACKER_RANGE = (0.0, 0.0)
X_LIGHT_RANGE = (-9.0, 9.0)
Y_LIGHT_RANGE = (-9.0, 9.0)
Z_LIGHT_RANGE = (4.0, 9.0)
LIGHT_POWER_RANGE = (45.0, 80.0)
LIGHT_EXPOSURE = 4.0
LIGHT_NORMALIZE = True

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

    compute_types = ['OPTIX','CUDA', 'METAL', 'HIP', 'ONEAPI']
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


# ---------------------------------------------------------------
# 3x4 P matrix from Blender camera
# ---------------------------------------------------------------

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0  # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
         (0, s_v, v_0),
         (0, 0, 1)))
    return K


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
    ))
    return RT


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K @ RT, K, RT


# this should help: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera?noredirect=1&lq=1

def get_plane_homography(cam):
    return cam[:, [0, 1, 3]]


def get_homography(main_cam, second_cam):
    P_main, _, _ = get_3x4_P_matrix_from_blender(main_cam)
    P_rand, _, _ = get_3x4_P_matrix_from_blender(second_cam)
    H_main = get_plane_homography(main_cam)
    H_rand = get_plane_homography(second_cam)
    return H_main @ H_rand.inverted()

# ==========================================
# MATERIAL SETUP
# ==========================================
def setup_pbr_material(target_obj, tex_dir):
    mat_name = f"Mat_{tex_dir.name}"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
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

    disp_node = add_texture_node("displacement.png", -900, False)
    if disp_node:
        disp_map_node = nodes.new('ShaderNodeDisplacement')
        disp_map_node.location = (0, -300)
        disp_map_node.inputs['Midlevel'] = 0.5
        disp_map_node.inputs['Scale'] = 1.0
        links.new(normal_map_node.outputs['Normal'], disp_map_node.inputs['Normal'])
        links.new(disp_node.outputs['Color'], disp_map_node.inputs['Height'])
        links.new(disp_map_node.outputs['Displacement'], output_node.inputs['Displacement'])


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
    scene.cycles.samples = RENDER_SAMPLE_CNT
    scene.render.resolution_x = BASE_RESOLUTION_X
    scene.render.resolution_y = BASE_RESOLUTION_Y

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tracker = bpy.data.objects[TRACKER_NAME]
    # Pre-calculate dead-on pixels for homography matrices globally
    tracker.location = (0.0, 0.0, 0.0)
    camera.location = (0.0, 0.0, REF_CAM_Z)
    # light_object should be in scene already and have constraints set
    light_object = bpy.data.objects[LIGHT_OBJECT_NAME]
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
        tex_out_dir = OUTPUT_DIR / tex_dir.name
        tasks_for_this_texture = 0
        for i_setting in range(0, NUM_SCENE_SETTINGS):
            setting_out_dir = tex_out_dir / f"{i_setting}"
            setting_base_filename = f"{base_filename}_{i_setting}"
            hq_file_path = setting_out_dir / f"{setting_base_filename}_deadon_HQ.png"

            if (setting_out_dir / RENDERING_INFO_FILE_NAME).exists():
                print(f"{tex_out_dir} ")

            render_queue.append({
                "is_first_of_texture": (tasks_for_this_texture == 0),
                "tex_dir": tex_dir, "is_dead_on": True,
                "tracker_loc": (0.0, 0.0, 0.0), "camera_loc": (0.0, 0.0, REF_CAM_Z),
                "res_pct": 100, "filepath": str(hq_file_path), "base_filename": setting_base_filename,
                "hq_filepath": str(hq_file_path),
                "light_location": (random.uniform(*X_LIGHT_RANGE), random.uniform(*Y_LIGHT_RANGE),
                                   random.uniform(*Z_LIGHT_RANGE)),
                "light_power": random.uniform(*LIGHT_POWER_RANGE),
            })
            tasks_for_this_texture += 1

            # Check which LQs need rendering
            for i in range(RENDERS_PER_TEXTURE):
                lq_file_path = setting_out_dir / f"{setting_base_filename}_angle_{i:03d}_LQ.png"

                render_queue.append({
                    "is_first_of_texture": (tasks_for_this_texture == 0),
                    "tex_dir": tex_dir, "is_dead_on": False,
                    "tracker_loc": (
                        random.uniform(*X_TRACKER_RANGE), random.uniform(*Y_TRACKER_RANGE),
                        random.uniform(*Z_TRACKER_RANGE)),
                    "camera_loc": (random.uniform(*X_CAMERA_RANGE), random.uniform(*Y_CAMERA_RANGE),
                                   random.uniform(*Z_CAMERA_RANGE)),
                    "res_pct": 50, "filepath": str(lq_file_path), "base_filename": setting_base_filename,
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
        light_object.location = task["light_location"]
        light_object.energy = task["light_power"]
        scene.render.resolution_percentage = task["res_pct"]
        scene.render.filepath = task["filepath"]

        bpy.context.view_layer.update()

        # Handle LQ specific calculations (Homography JSON tracking)
        if not task["is_dead_on"]:

            json_path = os.path.join(OUTPUT_DIR, f"{task['base_filename']}_matrices.json")
            matrix_data = {}
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    matrix_data = json.load(f)

            with open(json_path, 'w') as f:
                json.dump(matrix_data, f, indent=4)

        # Blocking render call (no 'INVOKE_DEFAULT' string)
        bpy.ops.render.render(write_still=True)

    # Post-process noise synchronously once rendering is complete
    print("\n=== RENDERING COMPLETE! ===")


if __name__ == "__main__":
    main()
