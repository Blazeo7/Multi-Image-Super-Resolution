import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector, Matrix

# ==========================================
# CONSTANTS & IMPORTS
# ==========================================
TARGET_NAME = "Plane"
LIGHT_OBJECT_NAME = "Sun.001"
TRACKER_NAME = "CameraTracker"
BASE_RESOLUTION_X = 1024
BASE_RESOLUTION_Y = 1024
RENDER_SAMPLE_CNT = 128
EXPECTED_TEX_WIDTH = 4096
EXPECTED_TEX_HEGHT = 4096


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


def get_plane_homography(P_matrix):
    # Converts a mathutils 3x4 Matrix to a 3x3 Numpy array dropping the Z column
    import numpy as np
    P_np = np.array(P_matrix)
    return P_np[:, [0, 1, 3]]


def setup_pbr_material(target_obj, tex_dir):
    mat_name = f"Mat_{tex_dir.name}"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(name=mat_name)
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
    basecolor_size = bpy.data.images.load(str(tex_dir / 'basecolor.png')).size
    # scale the texture accordingly to pixel size of the texture
    mapping.inputs['Scale'].default_value = (
        basecolor_size[0] / EXPECTED_TEX_WIDTH, basecolor_size[1] / EXPECTED_TEX_HEGHT, 1.0
    )

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
        links.new(normal_map_node.outputs['Normal'], disp_map_node.inputs['Normal'])
        links.new(disp_node.outputs['Color'], disp_map_node.inputs['Height'])
        links.new(disp_map_node.outputs['Displacement'], output_node.inputs['Displacement'])


def enable_gpu_rendering():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    # Get the Cycles preferences
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences

    # Force Blender to query the system for all devices
    cycles_prefs.refresh_devices()

    compute_types = ['ONEAPI', 'OPTIX', 'CUDA', 'METAL', 'HIP']
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


def main():
    # 1. Parse command line arguments for the job JSON
    # Blender passes arguments after "--" to the python script
    try:
        argv = sys.argv
        argv = argv[argv.index("--") + 1:]
        job_filepath = argv[0]
    except (ValueError, IndexError):
        print("ERROR: No job JSON provided. Use: blender -b file.blend -P render_job.py -- path/to/job.json")
        sys.exit(1)

    with open(job_filepath, 'r') as f:
        job = json.load(f)

    # Blender Setup
    enable_gpu_rendering()
    scene = bpy.context.scene
    scene.cycles.samples = RENDER_SAMPLE_CNT
    scene.render.resolution_x = BASE_RESOLUTION_X
    scene.render.resolution_y = BASE_RESOLUTION_Y

    camera = scene.camera
    tracker = bpy.data.objects[TRACKER_NAME]
    light = bpy.data.objects[LIGHT_OBJECT_NAME]
    target_obj = scene.objects.get(TARGET_NAME)

    tex_dir = Path(job["texture_dir"])
    out_dir = Path(job["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    setup_pbr_material(target_obj, tex_dir)

    print(f"=== Starting Job: {job['texture_name']} ===")

    # Execution Loop
    for scene_data in job["scenes"]:
        setting_out_dir = out_dir
        matrices_json_path = setting_out_dir / f"tex_{job['texture_name']}_{scene_data['setting_id']}_metadata.json"

        # --- SIMPLIFIED ALL-OR-NOTHING CHECK ---
        skip_scene = True
        if not matrices_json_path.exists():
            skip_scene = False
        else:
            # Check if every single image file for this scene already exists
            for render_task in scene_data["renders"]:
                target_filepath = setting_out_dir / render_task["filename"]
                if not target_filepath.exists():
                    skip_scene = False
                    break

        if skip_scene:
            print(f"Skipping scene {scene_data['setting_id']}: All renders and matrices exist.")
            continue  # Skip to the next scene entirely!
        # ---------------------------------------

        os.makedirs(setting_out_dir, exist_ok=True)

        # Apply Scene Lighting
        light.location = scene_data["light_loc"]
        light.data.energy = scene_data["light_power"]

        # Initialize the metadata with a COPY of the scene_data from the job
        # We use a dict copy so we don't accidentally modify the original job object
        full_metadata = scene_data.copy()
        # Add texture info for completeness
        full_metadata["texture_name"] = job["texture_name"]

        P_main_matrix = None

        # Pass 1: Calculate Matrices (We do this first so we can save the JSON once)
        for render_task in full_metadata["renders"]:
            scene.render.resolution_percentage = render_task["res_pct"]
            scene.render.filepath = str(target_filepath)

            # Force Blender to update matrices before doing math
            bpy.context.view_layer.update()

            # Homography Logic
            P_matrix, _, _ = get_3x4_P_matrix_from_blender(camera)

            if render_task["type"] == "HQ":
                P_main_matrix = P_matrix
                render_task["homography_matrix"] = np.identity(3).tolist()

            elif render_task["type"] == "LQ" and P_main_matrix is not None:
                H_main = get_plane_homography(P_main_matrix)
                H_rand = get_plane_homography(P_matrix)

                H_rand_inv = np.linalg.inv(H_rand)
                H_rand_to_main = np.matmul(H_main, H_rand_inv)
                render_task["homography_matrix"] = H_rand_to_main.tolist()
            else:
                render_task["homography_matrix"] = None

        # Save the full metadata (Scene data + All Matrices)
        with open(matrices_json_path, 'w') as mf:
            json.dump(full_metadata, mf, indent=4)

        # Pass 2: Actual Rendering
        for render_task in full_metadata["renders"]:
            target_filepath = setting_out_dir / render_task["filename"]

            # Setup scene state for render
            tracker.location = render_task["tracker_loc"]
            camera.location = render_task["camera_loc"]
            scene.render.resolution_percentage = render_task["res_pct"]
            scene.render.filepath = str(target_filepath)
            bpy.context.scene.view_settings.view_transform = 'Khronos PBR Neutral'
            bpy.context.scene.render.image_settings.color_mode = 'RGB'
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.image_settings.color_depth = '16'
            bpy.context.scene.render.image_settings.compression = 15

            print(f"Rendering {render_task['filename']}...")
            bpy.ops.render.render(write_still=True)

    print(f"=== Completed Job: {job['texture_name']} ===")


if __name__ == "__main__":
    main()
