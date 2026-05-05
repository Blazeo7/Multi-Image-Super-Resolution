import math
import random

import bpy
from mathutils import Vector

# ==========================================
# CONFIGURATION
# ==========================================
target_name = "CameraTracker"  # Name of your target surface
light_name = "Sun.001"  # Name of your light (Spot, Area, or Point)

# Set your desired exposure range (you will need to test these numbers
# with False Color to find the sweet spot for your specific scene scale)
min_exposure = 50.0
max_exposure = 200.0

L_X_RANGE = (-9.0, 9.0)
L_Y_RANGE = (-9.0, 9.0)
L_Z_RANGE = (3.0, 9.0)


# ==========================================
# EXECUTION
# ==========================================
def randomize_light():
    target = bpy.data.objects.get(target_name)
    light = bpy.data.objects.get(light_name)

    if not target or not light:
        print("Error: Target or Light not found. Check the names.")
        return

    # Convert spherical to Cartesian (XYZ) coordinates
    x = random.uniform(*L_X_RANGE)
    y = random.uniform(*L_Y_RANGE)
    z = random.uniform(*L_Z_RANGE)

    # Move the light to the new random location relative to the target
    light.location = target.location + Vector((x, y, z))

    # 2. Calculate the Angle of Incidence (cos(theta))
    # Assuming the target is a flat plane pointing straight up (Z-axis)
    surface_normal = Vector((0, 0, 1))

    # Calculate the normalized direction vector from target to light
    light_direction = (light.location - target.location).normalized()

    # The dot product of the normal and direction gives us cos(theta)
    cos_theta = surface_normal.dot(light_direction)

    # 3. Calculate the required power to achieve the target exposure
    # Pick a random target exposure for this specific scene
    target_exposure = random.uniform(min_exposure, max_exposure)

    # Apply the inverted formula: Power = (Exposure * Distance^2) / cos(theta)
    required_power = (target_exposure * (9 ** 2)) / cos_theta

    # Apply the calculated power to the light
    light.data.energy = required_power

    print(f"New Setup -> Dist: {9:.2f}m, Angle Cos: {cos_theta:.2f}, Set Power: {required_power:.2f}W")


# Run the function
randomize_light()
