import bpy
import json
import math
import os
import sys
from mathutils import Vector


CAMERAS = {
    "front_wide": {"location": (0, -28, 11), "lens": 28},
    "front_center": {"location": (0, -20, 8), "lens": 35},
    "left_perspective": {"location": (-18, -18, 10), "lens": 32},
    "right_perspective": {"location": (18, -18, 10), "lens": 32},
    "top_plan": {"location": (0, 0, 55), "lens": 24},
    "audience_view": {"location": (0, -40, 6), "lens": 50},
}


def get_json_path() -> str:
    argv = sys.argv
    if "--" not in argv:
        raise RuntimeError("Missing JSON path. Usage: blender -b -P blender_script.py -- /path/to/scene.json")
    idx = argv.index("--") + 1
    if idx >= len(argv):
        raise RuntimeError("JSON path not provided after --")
    return argv[idx]


def load_payload(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block, do_unlink=True)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block, do_unlink=True)
    for block in list(bpy.data.images):
        if block.users == 0:
            bpy.data.images.remove(block, do_unlink=True)


def hex_to_rgba(hex_color: str):
    hex_color = (hex_color or "#FFFFFF").lstrip("#")
    if len(hex_color) != 6:
        return (1.0, 1.0, 1.0, 1.0)
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, 1.0)


def apply_material(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def make_principled_material(name: str, base_color=(0.1, 0.1, 0.1, 1.0), roughness=0.35, metallic=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def make_emission_material(name: str, color=(0.2, 0.5, 1.0, 1.0), strength=4.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = strength
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def build_floor(size=120):
    bpy.ops.mesh.primitive_plane_add(location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "Floor"
    floor.scale = (size, size, 1)
    return floor


def build_stage(stage_data: dict):
    width = float(stage_data.get("width", 60))
    depth = float(stage_data.get("depth", 24))
    height = float(stage_data.get("height", 4))
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, height / 2))
    stage = bpy.context.object
    stage.name = "Stage"
    stage.scale = (width / 2, depth / 2, height / 2)
    return stage


def build_led_wall(stage_data: dict, led_data: dict):
    stage_depth = float(stage_data.get("depth", 24))
    led_width = float(led_data.get("width", 40))
    led_height = float(led_data.get("height", 12))
    bpy.ops.mesh.primitive_plane_add(location=(0, stage_depth / 2, led_height / 2 + 1))
    wall = bpy.context.object
    wall.name = "LED_Wall"
    wall.scale = (led_width / 2, led_height / 2, 1)
    wall.rotation_euler[0] = math.radians(90)
    return wall


def build_side_panels(colors: dict):
    secondary = hex_to_rgba(colors.get("secondary", "#A855F7"))
    panel_mat = make_emission_material("AccentPanel", secondary, 2.5)
    panels = []
    for x in (-25, 25):
        bpy.ops.mesh.primitive_plane_add(location=(x, 8, 8))
        panel = bpy.context.object
        panel.name = f"SidePanel_{'L' if x < 0 else 'R'}"
        panel.scale = (3, 8, 1)
        panel.rotation_euler[0] = math.radians(90)
        apply_material(panel, panel_mat)
        panels.append(panel)
    return panels


def build_audience(audience: dict):
    rows = int(audience.get("rows", 6))
    cols = int(audience.get("cols", 10))
    spacing_x = 2.4
    spacing_y = 2.8
    start_y = -8
    chair_mat = make_principled_material("ChairMat", (0.15, 0.15, 0.18, 1), 0.5, 0.0)
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols - 1) / 2) * spacing_x
            y = start_y - (r * spacing_y)
            bpy.ops.mesh.primitive_cube_add(location=(x, y, 0.45))
            chair = bpy.context.object
            chair.scale = (0.45, 0.45, 0.45)
            apply_material(chair, chair_mat)


def add_lights(lighting_data: dict, colors: dict):
    primary = hex_to_rgba(colors.get("primary", "#1A5DFF"))
    secondary = hex_to_rgba(colors.get("secondary", "#A855F7"))
    style = lighting_data.get("style", "futuristic")

    bpy.ops.object.light_add(type="AREA", location=(0, -10, 18))
    key = bpy.context.object
    key.data.energy = 3000
    key.data.color = primary[:3]

    bpy.ops.object.light_add(type="AREA", location=(-15, -8, 14))
    left = bpy.context.object
    left.data.energy = 1500
    left.data.color = secondary[:3]

    bpy.ops.object.light_add(type="AREA", location=(15, -8, 14))
    right = bpy.context.object
    right.data.energy = 1500
    right.data.color = secondary[:3]

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 30))
    sun = bpy.context.object
    sun.data.energy = 2.5

    if style == "futuristic":
        for x in (-10, -5, 5, 10):
            bpy.ops.object.light_add(type="SPOT", location=(x, -6, 14))
            spot = bpy.context.object
            spot.data.energy = 1200
            spot.data.spot_size = math.radians(35)
            spot.rotation_euler = (math.radians(60), 0, 0)


def add_camera(name: str, location, lens=35):
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.object
    cam.name = name
    cam.data.lens = lens
    return cam


def point_camera_to(cam, target):
    direction = Vector(target) - cam.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()


def setup_cycles(width: int, height: int):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.cycles.use_adaptive_sampling = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    prefs = bpy.context.preferences
    if "cycles" in prefs.addons:
        cycles_prefs = prefs.addons["cycles"].preferences
        for device_type in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                cycles_prefs.compute_device_type = device_type
                break
            except Exception:
                continue


def render_all(output_dir: str, target, width: int, height: int):
    setup_cycles(width, height)
    outputs = {}
    for name, cfg in CAMERAS.items():
        cam = add_camera(name, cfg["location"], cfg["lens"])
        point_camera_to(cam, target)
        bpy.context.scene.camera = cam
        out_path = os.path.join(output_dir, f"{name}.png")
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        outputs[name] = out_path
    return outputs


def export_glb(output_dir: str):
    glb_path = os.path.join(output_dir, "scene.glb")
    bpy.ops.export_scene.gltf(filepath=glb_path, export_format="GLB")
    return glb_path


def write_manifest(output_dir: str, scene_data: dict, render_outputs: dict, glb_path: str):
    manifest = {
        "scene": scene_data,
        "renders": render_outputs,
        "glb": glb_path,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main():
    payload_path = get_json_path()
    payload = load_payload(payload_path)
    scene_data = payload["scene"]
    render_data = payload["render"]

    output_dir = render_data["output_dir"]
    width = int(render_data.get("width", 1920))
    height = int(render_data.get("height", 1080))
    os.makedirs(output_dir, exist_ok=True)

    clear_scene()

    stage_data = scene_data.get("stage", {"width": 60, "depth": 24, "height": 4})
    led_data = scene_data.get("led_wall", {"type": "curved", "width": 40, "height": 12})
    audience_data = scene_data.get("audience", {"rows": 6, "cols": 10})
    colors = scene_data.get("colors", {"primary": "#1A5DFF", "secondary": "#A855F7"})
    lighting = scene_data.get("lighting", {"style": "futuristic"})
    camera_target = scene_data.get("camera_target", [0, 0, 6])

    floor = build_floor()
    stage = build_stage(stage_data)
    led_wall = build_led_wall(stage_data, led_data)
    build_side_panels(colors)
    build_audience(audience_data)
    add_lights(lighting, colors)

    floor_mat = make_principled_material("FloorMat", (0.03, 0.03, 0.04, 1), 0.7, 0.0)
    stage_mat = make_principled_material("StageMat", (0.08, 0.08, 0.1, 1), 0.35, 0.0)
    led_mat = make_emission_material("LEDMat", hex_to_rgba(colors.get("primary", "#1A5DFF")), 4.0)

    apply_material(floor, floor_mat)
    apply_material(stage, stage_mat)
    apply_material(led_wall, led_mat)

    render_outputs = render_all(output_dir, camera_target, width, height)
    glb_path = export_glb(output_dir)
    write_manifest(output_dir, scene_data, render_outputs, glb_path)

    print("Blender render pipeline completed successfully")


if __name__ == "__main__":
    main()
