import bpy
import json
import math
import os
import sys
from mathutils import Vector

"""
BriefCraft AI — Hybrid 3D Stage + Sketch Environment Blender Worker

CLI:
  blender -b -P blender_script.py -- /path/to/scene.json

Expected JSON:
{
  "scene": {
    "stage": {"width": 60, "depth": 24, "height": 4},
    "led_wall": {"width": 40, "height": 12},
    "audience": {"rows": 6, "cols": 10},
    "colors": {"primary": "#D7A94B", "secondary": "#FFD487"},
    "lighting": {"style": "premium"},
    "camera_target": [0, 0, 6]
  },
  "render": {
    "output_dir": "/tmp/render",
    "width": 1920,
    "height": 1080,
    "samples": 96,
    "engine": "CYCLES"
  }
}
"""


# --------------------------
# Basic utilities
# --------------------------

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


def hex_to_rgba(hex_color: str, alpha: float = 1.0):
    hex_color = (hex_color or "#FFFFFF").lstrip("#")
    if len(hex_color) != 6:
        return (1.0, 1.0, 1.0, alpha)
    return (
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0,
        alpha,
    )


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for collection in list(bpy.data.collections):
        if collection.name not in ("Collection",):
            bpy.data.collections.remove(collection)

    for datablock_group in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.textures,
        bpy.data.curves,
        bpy.data.lights,
        bpy.data.cameras,
    ):
        for block in list(datablock_group):
            if block.users == 0:
                datablock_group.remove(block)


def ensure_collection(name: str):
    col = bpy.data.collections.get(name)
    if not col:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def link_to_collection(obj, col):
    try:
        for c in obj.users_collection:
            c.objects.unlink(obj)
    except Exception:
        pass
    col.objects.link(obj)


def apply_material(obj, mat):
    if hasattr(obj.data, "materials"):
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)


def set_node_input(node, name, value):
    if name in node.inputs:
        node.inputs[name].default_value = value


def make_principled_material(name: str, base=(0.1, 0.1, 0.1, 1.0), roughness=0.35, metallic=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = "BLEND" if alpha < 1 else "OPAQUE"
    mat.use_screen_refraction = alpha < 1 if hasattr(mat, "use_screen_refraction") else False
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        set_node_input(bsdf, "Base Color", base)
        set_node_input(bsdf, "Roughness", roughness)
        set_node_input(bsdf, "Metallic", metallic)
        set_node_input(bsdf, "Alpha", alpha)
    return mat


def make_emission_material(name: str, color=(1, 1, 1, 1), strength=2.0, alpha=1.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = "BLEND" if alpha < 1 else "OPAQUE"
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new(type="ShaderNodeOutputMaterial")
    if alpha < 1:
        transparent = nodes.new(type="ShaderNodeBsdfTransparent")
        emission = nodes.new(type="ShaderNodeEmission")
        mix = nodes.new(type="ShaderNodeMixShader")
        emission.inputs["Color"].default_value = color
        emission.inputs["Strength"].default_value = strength
        mix.inputs[0].default_value = 1.0 - alpha
        links.new(transparent.outputs["BSDF"], mix.inputs[1])
        links.new(emission.outputs["Emission"], mix.inputs[2])
        links.new(mix.outputs["Shader"], out.inputs["Surface"])
    else:
        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Color"].default_value = color
        emission.inputs["Strength"].default_value = strength
        links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def add_cube(name, location, scale, mat, col):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    apply_material(obj, mat)
    obj.cast_shadow = True
    obj.receive_shadow = True
    link_to_collection(obj, col)
    return obj


def add_cylinder(name, location, radius, depth, mat, col, vertices=24, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    apply_material(obj, mat)
    obj.cast_shadow = True
    obj.receive_shadow = True
    link_to_collection(obj, col)
    return obj


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_line(name, points, mat, col, bevel=0.035):
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = bevel
    curve.bevel_resolution = 2
    spl = curve.splines.new("POLY")
    spl.points.add(len(points) - 1)
    for p, co in zip(spl.points, points):
        p.co = (co[0], co[1], co[2], 1)
    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def add_wire_box(name, center, size, mat, col, bevel=0.025):
    x, y, z = center
    w, d, h = size
    xs = [x - w / 2, x + w / 2]
    ys = [y - d / 2, y + d / 2]
    zs = [z - h / 2, z + h / 2]
    corners = [(a, b, c) for a in xs for b in ys for c in zs]
    def c(ix, iy, iz): return (xs[ix], ys[iy], zs[iz])
    edges = [
        (c(0,0,0), c(1,0,0)), (c(0,1,0), c(1,1,0)), (c(0,0,1), c(1,0,1)), (c(0,1,1), c(1,1,1)),
        (c(0,0,0), c(0,1,0)), (c(1,0,0), c(1,1,0)), (c(0,0,1), c(0,1,1)), (c(1,0,1), c(1,1,1)),
        (c(0,0,0), c(0,0,1)), (c(1,0,0), c(1,0,1)), (c(0,1,0), c(0,1,1)), (c(1,1,0), c(1,1,1)),
    ]
    group = []
    for i, (a, b) in enumerate(edges):
        group.append(add_line(f"{name}_edge_{i}", [a, b], mat, col, bevel))
    return group


# --------------------------
# Materials
# --------------------------

def build_materials(colors):
    primary = hex_to_rgba(colors.get("primary", "#D7A94B"))
    secondary = hex_to_rgba(colors.get("secondary", "#FFD487"))
    return {
        "floor": make_principled_material("dark_reflective_floor", (0.015, 0.016, 0.02, 1), 0.18, 0.65),
        "stage": make_principled_material("matte_black_stage", (0.045, 0.047, 0.055, 1), 0.34, 0.45),
        "gold": make_principled_material("brushed_gold_trim", primary, 0.24, 0.9),
        "black_metal": make_principled_material("black_metal_truss", (0.006, 0.007, 0.009, 1), 0.28, 0.85),
        "seating": make_principled_material("premium_black_seating", (0.02, 0.022, 0.028, 1), 0.48, 0.2),
        "screen": make_emission_material("led_screen_emission", primary, 3.8),
        "accent": make_emission_material("accent_light_emission", secondary, 2.8),
        "beam": make_emission_material("soft_spotlight_beam", secondary, 1.1, 0.17),
        "sketch": make_emission_material("sketch_white_lines", (0.86, 0.9, 0.96, 1), 0.9, 0.58),
        "sketch_dim": make_emission_material("dimension_lines", (1, 0.88, 0.58, 1), 1.15, 0.82),
        "transparent": make_principled_material("soft_transparent_shell", (0.7, 0.85, 1, 0.08), 0.8, 0, 0.08),
    }


# --------------------------
# Realistic stage
# --------------------------

def build_realistic_stage(scene_data, mats, col):
    stage = scene_data.get("stage", {})
    led = scene_data.get("led_wall", {})
    width = float(stage.get("width", 60))
    depth = float(stage.get("depth", 24))
    height = float(stage.get("height", 4))
    led_w = float(led.get("width", 40))
    led_h = float(led.get("height", 12))

    # Main stage and risers
    add_cube("main_stage_platform", (0, 0, height / 2), (width / 2, depth / 2, height / 2), mats["stage"], col)
    add_cube("front_gold_trim", (0, -depth / 2 - 0.08, height + 0.08), (width / 2, 0.12, 0.08), mats["gold"], col)
    add_cube("left_gold_trim", (-width / 2 - 0.08, 0, height + 0.08), (0.08, depth / 2, 0.08), mats["gold"], col)
    add_cube("right_gold_trim", (width / 2 + 0.08, 0, height + 0.08), (0.08, depth / 2, 0.08), mats["gold"], col)

    for i in range(3):
        add_cube(f"front_step_{i+1}", (0, -depth / 2 - 2.0 - i * 1.15, 0.28 + i * 0.22),
                 (width * 0.34 - i * 3, 0.55, 0.22), mats["stage"], col)

    # LED wall and scenic panels
    bpy.ops.mesh.primitive_cube_add(location=(0, depth / 2 + 0.28, height + led_h / 2))
    wall = bpy.context.object
    wall.name = "realistic_led_wall"
    wall.scale = (led_w / 2, 0.16, led_h / 2)
    apply_material(wall, mats["screen"])
    link_to_collection(wall, col)

    # Text on LED screen
    for txt, z, size in [("VISION", height + led_h * 0.62, 3.8), ("IGNITED", height + led_h * 0.42, 3.8), ("CREATE WITHOUT LIMITS", height + led_h * 0.22, 1.25)]:
        bpy.ops.object.text_add(location=(0, depth / 2 - 0.02, z), rotation=(math.radians(90), 0, 0))
        t = bpy.context.object
        t.name = "led_text_" + txt.lower().replace(" ", "_")
        t.data.body = txt
        t.data.align_x = "CENTER"
        t.data.align_y = "CENTER"
        t.data.size = size
        t.data.extrude = 0.015
        apply_material(t, mats["gold"])
        link_to_collection(t, col)

    panel_w = max(4, width * 0.075)
    for x, side in [(-(led_w / 2 + panel_w * 0.8), "left"), ((led_w / 2 + panel_w * 0.8), "right")]:
        add_cube(f"{side}_scenic_panel", (x, depth / 2 + 0.18, height + led_h / 2),
                 (panel_w / 2, 0.13, led_h * 0.48), mats["black_metal"], col)
        for k in range(4):
            add_cube(f"{side}_gold_slash_{k}", (x + (-0.9 + k * 0.55), depth / 2 - 0.02, height + 2.8 + k * 1.6),
                     (0.055, 0.08, 2.1), mats["gold"], col).rotation_euler[1] = math.radians(18 if side == "left" else -18)

    # Truss rectangle
    truss_z = height + led_h + 2.2
    truss_y = depth / 2 - 1.8
    add_cube("truss_front", (0, truss_y - 4.2, truss_z), (width * 0.42, 0.10, 0.10), mats["black_metal"], col)
    add_cube("truss_back", (0, truss_y + 1.0, truss_z), (width * 0.42, 0.10, 0.10), mats["black_metal"], col)
    add_cube("truss_left", (-width * 0.42, truss_y - 1.6, truss_z), (0.10, 2.6, 0.10), mats["black_metal"], col)
    add_cube("truss_right", (width * 0.42, truss_y - 1.6, truss_z), (0.10, 2.6, 0.10), mats["black_metal"], col)

    # Moving heads + beams
    for i, x in enumerate([-20, -14, -8, -2, 4, 10, 16, 22]):
        add_cylinder(f"moving_head_{i}", (x, truss_y - 4.2, truss_z - 0.55), 0.38, 0.75, mats["black_metal"], col, 18, (math.radians(90), 0, 0))
        bpy.ops.object.light_add(type="SPOT", location=(x, truss_y - 4.2, truss_z - 0.9))
        spot = bpy.context.object
        spot.name = f"stage_spot_{i}"
        spot.data.energy = 950
        spot.data.color = (1.0, 0.78, 0.43)
        spot.data.spot_size = math.radians(32)
        look_at(spot, (x * 0.18, -depth / 2 + 2, height + 0.4))
        link_to_collection(spot, col)

        bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=2.0, radius2=0.22, depth=9.5, location=(x * 0.55, truss_y - 6.7, truss_z - 4.8))
        beam = bpy.context.object
        beam.name = f"visible_light_beam_{i}"
        beam.rotation_euler[0] = math.radians(16)
        apply_material(beam, mats["beam"])
        link_to_collection(beam, col)

    # Podium
    add_cube("premium_podium", (-width * 0.34, -depth * 0.16, height + 1.25), (1.4, 0.75, 1.25), mats["stage"], col)
    add_cube("podium_gold_cap", (-width * 0.34, -depth * 0.16, height + 2.55), (1.55, 0.85, 0.12), mats["gold"], col)

    # Foreground lounge seating
    sofa_y = -depth / 2 - 8
    for x in [-12, 0, 12]:
        add_cube(f"vip_sofa_base_{x}", (x, sofa_y, 0.6), (2.8, 0.75, 0.35), mats["seating"], col)
        add_cube(f"vip_sofa_back_{x}", (x, sofa_y + 0.65, 1.25), (2.8, 0.20, 0.75), mats["seating"], col)
        add_cube(f"vip_table_{x}", (x + 3.4, sofa_y - 0.2, 0.35), (0.55, 0.55, 0.12), mats["gold"], col)

    return {"stage_width": width, "stage_depth": depth, "stage_height": height}


def build_audience(scene_data, mats, col):
    aud = scene_data.get("audience", {})
    rows = int(aud.get("rows", 6))
    cols = int(aud.get("cols", 10))
    spacing_x = 2.3
    spacing_y = 2.45
    start_y = -22
    for r in range(rows):
        count = cols + (1 if r % 2 else 0)
        for c in range(count):
            x = (c - (count - 1) / 2) * spacing_x
            y = start_y - r * spacing_y
            add_cube(f"seat_{r}_{c}_base", (x, y, 0.42), (0.44, 0.44, 0.22), mats["seating"], col)
            add_cube(f"seat_{r}_{c}_back", (x, y + 0.35, 0.92), (0.44, 0.08, 0.48), mats["seating"], col)


# --------------------------
# Sketch environment
# --------------------------

def build_sketch_environment(scene_data, mats, col):
    stage = scene_data.get("stage", {})
    width = float(stage.get("width", 60))
    depth = float(stage.get("depth", 24))

    # Venue shell and floor planning grid
    add_wire_box("venue_shell_wireframe", (0, -8, 13), (width + 28, depth + 54, 26), mats["sketch"], col, 0.025)
    for x in range(-42, 43, 6):
        add_line(f"floor_grid_x_{x}", [(x, -58, 0.04), (x, 28, 0.04)], mats["sketch"], col, 0.012)
    for y in range(-58, 29, 6):
        add_line(f"floor_grid_y_{y}", [(-42, y, 0.04), (42, y, 0.04)], mats["sketch"], col, 0.012)

    # Far seating blocks as sketch/wire blocks
    for r in range(4):
        for s, x in enumerate([-28, 28]):
            add_wire_box(f"sketch_far_seating_{r}_{s}", (x, -22 - r * 7, 1.0), (12, 4.2, 2.0), mats["sketch"], col, 0.018)

    # Venue architectural outlines
    for i, x in enumerate([-40, -32, -24, 24, 32, 40]):
        h = 7 + (i % 3) * 2.4
        add_wire_box(f"context_tower_{i}", (x, 20 - (i % 2) * 5, h / 2), (4 + i % 2, 4, h), mats["sketch"], col, 0.018)

    # Trees/context shapes
    for i, (x, y) in enumerate([(-36, -2), (-41, -16), (36, -2), (41, -16), (-34, 18), (34, 18)]):
        add_line(f"tree_trunk_{i}", [(x, y, 0), (x, y, 2.5)], mats["sketch"], col, 0.018)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=4, radius=1.6, location=(x, y, 4.0))
        crown = bpy.context.object
        crown.name = f"tree_crown_wire_{i}"
        apply_material(crown, mats["transparent"])
        link_to_collection(crown, col)
        crown.display_type = "WIRE"
        add_wire_box(f"tree_hint_box_{i}", (x, y, 4), (3.2, 3.2, 3.2), mats["sketch"], col, 0.013)

    # Directional circulation arrows
    for i, y in enumerate([-36, -30, -24]):
        add_line(f"flow_arrow_line_{i}", [(-36, y, 0.08), (-18, y + 4, 0.08)], mats["sketch_dim"], col, 0.025)
        add_line(f"flow_arrow_head_a_{i}", [(-18, y + 4, 0.08), (-20, y + 2.2, 0.08)], mats["sketch_dim"], col, 0.025)
        add_line(f"flow_arrow_head_b_{i}", [(-18, y + 4, 0.08), (-20.4, y + 5.2, 0.08)], mats["sketch_dim"], col, 0.025)


def build_dimensions(scene_data, mats, col):
    stage = scene_data.get("stage", {})
    width = float(stage.get("width", 60))
    depth = float(stage.get("depth", 24))
    led = scene_data.get("led_wall", {})
    led_h = float(led.get("height", 12))

    z = 0.12
    y = -depth / 2 - 5.2
    add_line("dim_stage_width", [(-width / 2, y, z), (width / 2, y, z)], mats["sketch_dim"], col, 0.035)
    add_line("dim_width_tick_l", [(-width / 2, y - 1.2, z), (-width / 2, y + 1.2, z)], mats["sketch_dim"], col, 0.03)
    add_line("dim_width_tick_r", [(width / 2, y - 1.2, z), (width / 2, y + 1.2, z)], mats["sketch_dim"], col, 0.03)

    x = width / 2 + 5.4
    add_line("dim_led_height", [(x, depth / 2 + 0.4, 4), (x, depth / 2 + 0.4, 4 + led_h)], mats["sketch_dim"], col, 0.035)
    add_line("dim_venue_depth", [(width / 2 + 7, -52, z), (width / 2 + 7, depth / 2, z)], mats["sketch_dim"], col, 0.025)

    # Text labels as 3D text
    for body, loc, rot, size in [
        (f"{width:.0f}.00 m", (0, y - 1.2, z + 0.15), (math.radians(90), 0, 0), 1.0),
        (f"{led_h:.0f}.00 m", (x + 1.1, depth / 2 + 0.4, 4 + led_h / 2), (math.radians(90), 0, math.radians(90)), 0.8),
        ("36.00 m", (width / 2 + 8.4, -25, z + 0.15), (math.radians(90), 0, math.radians(90)), 0.8),
    ]:
        bpy.ops.object.text_add(location=loc, rotation=rot)
        t = bpy.context.object
        t.name = "dimension_label_" + body.replace(" ", "_")
        t.data.body = body
        t.data.align_x = "CENTER"
        t.data.align_y = "CENTER"
        t.data.size = size
        t.data.extrude = 0.01
        apply_material(t, mats["sketch_dim"])
        link_to_collection(t, col)


# --------------------------
# Camera, render, export
# --------------------------

def setup_lighting(scene_data, mats, stage_col):
    colors = scene_data.get("colors", {})
    primary = hex_to_rgba(colors.get("primary", "#D7A94B"))
    secondary = hex_to_rgba(colors.get("secondary", "#FFD487"))

    bpy.ops.object.light_add(type="AREA", location=(0, -18, 22))
    key = bpy.context.object
    key.name = "large_warm_key_light"
    key.data.energy = 2800
    key.data.size = 18
    key.data.color = primary[:3]
    link_to_collection(key, stage_col)

    bpy.ops.object.light_add(type="AREA", location=(-24, -10, 15))
    left = bpy.context.object
    left.name = "left_soft_fill"
    left.data.energy = 900
    left.data.size = 10
    left.data.color = secondary[:3]
    link_to_collection(left, stage_col)

    bpy.ops.object.light_add(type="AREA", location=(24, -10, 15))
    right = bpy.context.object
    right.name = "right_soft_fill"
    right.data.energy = 900
    right.data.size = 10
    right.data.color = secondary[:3]
    link_to_collection(right, stage_col)

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 30))
    sun = bpy.context.object
    sun.name = "soft_global_sun"
    sun.data.energy = 0.5
    link_to_collection(sun, stage_col)


def add_camera(name, location, target, lens):
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.object
    cam.name = name
    cam.data.lens = lens
    look_at(cam, target)
    return cam


def setup_render(width, height, samples=96, engine="CYCLES"):
    scene = bpy.context.scene
    engine = (engine or "CYCLES").upper()
    if engine not in ("CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        engine = "CYCLES"

    try:
        scene.render.engine = engine
    except Exception:
        scene.render.engine = "CYCLES"

    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(samples)
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.max_bounces = 6
        scene.cycles.diffuse_bounces = 3
        scene.cycles.glossy_bounces = 3
        try:
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
            scene.cycles.device = "GPU"
        except Exception:
            scene.cycles.device = "CPU"

    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1


def set_collection_render_state(stage_col, env_col, dim_col, mode):
    stage_col.hide_render = mode == "sketch"
    env_col.hide_render = mode == "realistic"
    dim_col.hide_render = mode == "realistic"


def render_view(output_dir, name, cam_cfg, target, width, height, samples, engine, stage_col, env_col, dim_col, mode):
    set_collection_render_state(stage_col, env_col, dim_col, mode)
    setup_render(width, height, samples, engine)
    cam = add_camera("CAM_" + name, cam_cfg["location"], target, cam_cfg["lens"])
    bpy.context.scene.camera = cam
    out_path = os.path.join(output_dir, f"{name}.png")
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    return out_path


def export_glb(output_dir, stage_col, env_col, dim_col):
    # Export merged scene so frontend can show a real rotatable model if it loads GLB.
    stage_col.hide_render = False
    env_col.hide_render = False
    dim_col.hide_render = False
    glb_path = os.path.join(output_dir, "stage_sketch_environment.glb")
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format="GLB",
        export_apply=True,
        export_cameras=True,
        export_lights=True,
    )
    return glb_path


def write_manifest(output_dir, scene_data, renders, glb_path):
    manifest = {
        "ok": True,
        "type": "briefcraft_hybrid_3d_stage",
        "scene": scene_data,
        "renders": renders,
        "outputs": {
            "realistic_view": renders.get("realistic_view"),
            "sketch_concept": renders.get("sketch_concept"),
            "merged_hybrid": renders.get("merged_hybrid"),
            "stage_closeup": renders.get("stage_closeup"),
            "top_plan": renders.get("top_plan"),
            "glb_model": glb_path,
        },
        "glb": glb_path,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main():
    payload_path = get_json_path()
    payload = load_payload(payload_path)
    scene_data = payload.get("scene", {})
    render_data = payload.get("render", {})

    output_dir = render_data.get("output_dir", os.path.join(os.getcwd(), "briefcraft_render_output"))
    width = int(render_data.get("width", 1920))
    height = int(render_data.get("height", 1080))
    samples = int(render_data.get("samples", 96))
    engine = render_data.get("engine", "CYCLES")
    os.makedirs(output_dir, exist_ok=True)

    clear_scene()

    stage_col = ensure_collection("STAGE_REALISTIC")
    env_col = ensure_collection("ENVIRONMENT_SKETCH")
    dim_col = ensure_collection("DIMENSIONS")

    mats = build_materials(scene_data.get("colors", {"primary": "#D7A94B", "secondary": "#FFD487"}))

    # Global floor, linked to stage collection because it belongs to the realistic render too.
    floor_size = 120
    bpy.ops.mesh.primitive_plane_add(size=floor_size, location=(0, -10, 0))
    floor = bpy.context.object
    floor.name = "premium_dark_floor"
    apply_material(floor, mats["floor"])
    floor.receive_shadow = True
    link_to_collection(floor, stage_col)

    info = build_realistic_stage(scene_data, mats, stage_col)
    build_audience(scene_data, mats, stage_col)
    build_sketch_environment(scene_data, mats, env_col)
    build_dimensions(scene_data, mats, dim_col)
    setup_lighting(scene_data, mats, stage_col)

    target = tuple(scene_data.get("camera_target", [0, -2, 7]))
    cameras = {
        "realistic_view": {"location": (0, -58, 17), "lens": 34},
        "sketch_concept": {"location": (-42, -46, 27), "lens": 31},
        "merged_hybrid": {"location": (0, -62, 22), "lens": 30},
        "stage_closeup": {"location": (18, -34, 13), "lens": 46},
        "top_plan": {"location": (0, -10, 82), "lens": 28},
    }

    renders = {
        "realistic_view": render_view(output_dir, "realistic_view", cameras["realistic_view"], target, width, height, samples, engine, stage_col, env_col, dim_col, "realistic"),
        "sketch_concept": render_view(output_dir, "sketch_concept", cameras["sketch_concept"], target, width, height, samples, engine, stage_col, env_col, dim_col, "sketch"),
        "merged_hybrid": render_view(output_dir, "merged_hybrid", cameras["merged_hybrid"], target, width, height, samples, engine, stage_col, env_col, dim_col, "merged"),
        "stage_closeup": render_view(output_dir, "stage_closeup", cameras["stage_closeup"], target, width, height, samples, engine, stage_col, env_col, dim_col, "realistic"),
        "top_plan": render_view(output_dir, "top_plan", cameras["top_plan"], (0, -10, 0), width, height, max(32, samples // 2), engine, stage_col, env_col, dim_col, "merged"),
    }

    glb_path = export_glb(output_dir, stage_col, env_col, dim_col)
    manifest_path = write_manifest(output_dir, scene_data, renders, glb_path)

    print(json.dumps({
        "ok": True,
        "manifest": manifest_path,
        "renders": renders,
        "glb": glb_path,
    }, indent=2))


if __name__ == "__main__":
    main()
