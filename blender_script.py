import bpy
import json
import math
import os
import sys
from mathutils import Vector

"""
BriefCraft AI — Realistic Hybrid Stage Renderer v3

Goal:
- Realistic premium stage / LED / truss / lights / seating
- Sketchy 3D venue environment around it
- Separate render outputs:
  realistic_view.png
  sketch_concept.png
  merged_hybrid.png
  stage_closeup.png
  top_plan.png
  stage_sketch_environment.glb

CLI:
  blender -b -P blender_script.py -- /path/to/scene.json
"""


# -----------------------------
# Input / scene reset
# -----------------------------

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

    for datablocks in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.curves,
        bpy.data.lights,
        bpy.data.cameras,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)

    for col in list(bpy.data.collections):
        if col.name != "Collection":
            bpy.data.collections.remove(col)


def ensure_collection(name: str):
    col = bpy.data.collections.get(name)
    if not col:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def move_to_collection(obj, col):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)


def hex_to_rgba(hex_color: str, alpha: float = 1.0):
    h = (hex_color or "#D7A94B").replace("#", "")
    if len(h) != 6:
        return (1, 1, 1, alpha)
    return (
        int(h[0:2], 16) / 255,
        int(h[2:4], 16) / 255,
        int(h[4:6], 16) / 255,
        alpha,
    )


def apply_material(obj, mat):
    if hasattr(obj.data, "materials"):
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def shade_smooth(obj):
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
        obj.select_set(False)
    except Exception:
        pass


def add_bevel(obj, amount=0.04, segments=2):
    try:
        mod = obj.modifiers.new("soft_beveled_edges", "BEVEL")
        mod.width = amount
        mod.segments = segments
        mod.affect = "EDGES"
        obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    except Exception:
        pass


# -----------------------------
# Materials
# -----------------------------

def node_mat(name, base=(1,1,1,1), metallic=0.0, roughness=0.35, alpha=1.0, emission=None, strength=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = "BLEND" if alpha < 1 else "OPAQUE"
    mat.use_screen_refraction = False
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        def setv(inp, val):
            if inp in bsdf.inputs:
                bsdf.inputs[inp].default_value = val
        setv("Base Color", base)
        setv("Metallic", metallic)
        setv("Roughness", roughness)
        setv("Alpha", alpha)
        if emission:
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = emission
            elif "Emission" in bsdf.inputs:
                bsdf.inputs["Emission"].default_value = emission
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = strength
    return mat


def emission_mat(name, color=(1,1,1,1), strength=2.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = "BLEND" if alpha < 1 else "OPAQUE"
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    if alpha < 1:
        tr = nodes.new("ShaderNodeBsdfTransparent")
        em = nodes.new("ShaderNodeEmission")
        mix = nodes.new("ShaderNodeMixShader")
        em.inputs["Color"].default_value = color
        em.inputs["Strength"].default_value = strength
        mix.inputs[0].default_value = 1 - alpha
        links.new(tr.outputs["BSDF"], mix.inputs[1])
        links.new(em.outputs["Emission"], mix.inputs[2])
        links.new(mix.outputs["Shader"], out.inputs["Surface"])
    else:
        em = nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = color
        em.inputs["Strength"].default_value = strength
        links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_materials(colors):
    primary = hex_to_rgba(colors.get("primary", "#D7A94B"))
    secondary = hex_to_rgba(colors.get("secondary", "#FFD487"))
    blue = hex_to_rgba(colors.get("blue", "#4F89FF"))

    mats = {
        "floor": node_mat("black_polished_reflective_floor", (0.012,0.012,0.014,1), metallic=.35, roughness=.16),
        "stage": node_mat("black_laminate_stage", (0.025,0.026,0.030,1), metallic=.22, roughness=.28),
        "stage_side": node_mat("dark_stage_side_panels", (0.018,0.018,0.020,1), metallic=.38, roughness=.32),
        "gold": node_mat("brushed_champagne_gold", primary, metallic=1.0, roughness=.20, emission=primary, strength=.08),
        "dark_metal": node_mat("powder_coated_black_metal", (0.006,0.007,0.009,1), metallic=.9, roughness=.25),
        "screen": node_mat("active_led_screen", (0.03,0.028,0.02,1), metallic=.05, roughness=.18, emission=primary, strength=1.8),
        "screen_glow": emission_mat("led_screen_glow", primary, 4.2),
        "accent": emission_mat("gold_accent_emission", secondary, 2.4),
        "beam": emission_mat("warm_volumetric_beam_proxy", secondary, 0.9, 0.13),
        "seat_fabric": node_mat("black_velvet_seating", (0.018,0.018,0.022,1), metallic=.05, roughness=.72),
        "seat_metal": node_mat("gold_seat_legs", primary, metallic=.95, roughness=.23),
        "speaker": node_mat("speaker_black_mesh", (0.006,0.006,0.007,1), metallic=.15, roughness=.62),
        "sketch": emission_mat("white_sketch_wire_lines", (0.78,0.84,0.94,1), .8, .50),
        "sketch_bright": emission_mat("bright_sketch_dimension_lines", (1.0,.86,.52,1), 1.2, .85),
        "glass": node_mat("soft_transparent_venue_hint", (0.45,0.65,1,.08), metallic=0, roughness=.85, alpha=.08),
    }
    return mats


# -----------------------------
# Geometry helpers
# -----------------------------

def cube(name, loc, scale, mat, col, bevel=0.03):
    bpy.ops.mesh.primitive_cube_add(location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    apply_material(obj, mat)
    add_bevel(obj, bevel, 2)
    obj.cast_shadow = True
    obj.receive_shadow = True
    move_to_collection(obj, col)
    return obj


def cyl(name, loc, radius, depth, mat, col, vertices=32, rotation=(0,0,0), bevel=False):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=loc, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    apply_material(obj, mat)
    shade_smooth(obj)
    if bevel:
        add_bevel(obj, 0.015, 1)
    obj.cast_shadow = True
    obj.receive_shadow = True
    move_to_collection(obj, col)
    return obj


def plane(name, loc, scale, mat, col, rot=(0,0,0)):
    bpy.ops.mesh.primitive_plane_add(location=loc, rotation=rot)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    apply_material(obj, mat)
    obj.receive_shadow = True
    move_to_collection(obj, col)
    return obj


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def line_obj(name, pts, mat, col, bevel=.025):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = bevel
    curve.bevel_resolution = 2
    spl = curve.splines.new("POLY")
    spl.points.add(len(pts) - 1)
    for p, co in zip(spl.points, pts):
        p.co = (co[0], co[1], co[2], 1)
    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def wire_box(name, center, size, mat, col, bevel=.018):
    x,y,z = center
    w,d,h = size
    xs = [x-w/2, x+w/2]
    ys = [y-d/2, y+d/2]
    zs = [z-h/2, z+h/2]
    def c(a,b,c_): return (xs[a], ys[b], zs[c_])
    edges = [
        (c(0,0,0),c(1,0,0)),(c(0,1,0),c(1,1,0)),(c(0,0,1),c(1,0,1)),(c(0,1,1),c(1,1,1)),
        (c(0,0,0),c(0,1,0)),(c(1,0,0),c(1,1,0)),(c(0,0,1),c(0,1,1)),(c(1,0,1),c(1,1,1)),
        (c(0,0,0),c(0,0,1)),(c(1,0,0),c(1,0,1)),(c(0,1,0),c(0,1,1)),(c(1,1,0),c(1,1,1)),
    ]
    for i,(a,b) in enumerate(edges):
        line_obj(f"{name}_{i}", [a,b], mat, col, bevel)


def make_text(name, body, loc, size, mat, col, rot=(math.radians(90),0,0), align="CENTER"):
    bpy.ops.object.text_add(location=loc, rotation=rot)
    obj = bpy.context.object
    obj.name = name
    obj.data.body = body
    obj.data.align_x = align
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.extrude = 0.018
    apply_material(obj, mat)
    move_to_collection(obj, col)
    return obj


# -----------------------------
# Realistic stage construction
# -----------------------------

def build_led_texture():
    img = bpy.data.images.new("procedural_led_wall_graphic", 1600, 900, alpha=False)
    pixels = []
    for y in range(900):
        v = y / 899
        for x in range(1600):
            u = x / 1599
            wave = 0.18 * math.sin(u * 18 + v * 8) + 0.12 * math.sin(u * 6 - v * 14)
            glow = max(0, 1 - ((u - .5)**2 * 5 + (v - .55)**2 * 8))
            r = 0.025 + glow * 0.55 + wave * 0.08
            g = 0.020 + glow * 0.38 + wave * 0.06
            b = 0.018 + glow * 0.15
            if abs(v - (.68 + 0.045 * math.sin(u * 8))) < .01:
                r += .5; g += .35; b += .1
            pixels.extend([min(1,r), min(1,g), min(1,b), 1])
    img.pixels = pixels
    return img


def make_led_material(primary):
    img = build_led_texture()
    mat = bpy.data.materials.new("procedural_real_led_wall")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if "Emission Color" in bsdf.inputs:
        links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 2.0
    bsdf.inputs["Roughness"].default_value = .22
    return mat


def build_stage(scene_data, mats, stage_col):
    stage_data = scene_data.get("stage", {})
    led_data = scene_data.get("led_wall", {})
    colors = scene_data.get("colors", {})
    w = float(stage_data.get("width", 60))
    d = float(stage_data.get("depth", 24))
    h = float(stage_data.get("height", 4))
    led_w = float(led_data.get("width", 40))
    led_h = float(led_data.get("height", 12))
    primary = hex_to_rgba(colors.get("primary", "#D7A94B"))

    # Main layered stage with realistic bevels
    cube("main_stage_black_laminate", (0, 0, h/2), (w/2, d/2, h/2), mats["stage"], stage_col, .08)
    cube("stage_front_face_dark", (0, -d/2-.08, h/2), (w/2, .09, h/2), mats["stage_side"], stage_col, .03)
    cube("stage_gold_front_lip", (0, -d/2-.24, h+.12), (w/2+.35, .13, .12), mats["gold"], stage_col, .035)
    cube("stage_gold_left_lip", (-w/2-.18, 0, h+.12), (.12, d/2+.25, .12), mats["gold"], stage_col, .035)
    cube("stage_gold_right_lip", (w/2+.18, 0, h+.12), (.12, d/2+.25, .12), mats["gold"], stage_col, .035)

    # Steps and risers
    for i in range(4):
        cube(f"real_step_layer_{i+1}", (0, -d/2 - 1.1 - i*1.05, .22 + i*.18),
             (w*.26 - i*2.1, .48, .20), mats["stage"], stage_col, .045)
        cube(f"real_step_gold_nose_{i+1}", (0, -d/2 - .62 - i*1.05, .45 + i*.18),
             (w*.26 - i*2.0, .055, .04), mats["gold"], stage_col, .018)

    # Curved / segmented LED wall illusion
    led_mat = make_led_material(primary)
    segments = 9
    radius = 38
    center_y = d/2 + 6.4
    arc = math.radians(42)
    for i in range(segments):
        t = (i - (segments-1)/2) / ((segments-1)/2)
        theta = t * arc/2
        seg_w = led_w / segments * 1.08
        x = math.sin(theta) * radius
        y = center_y - math.cos(theta) * radius + radius
        panel = cube(f"curved_led_panel_{i+1}", (x, y, h + led_h/2), (seg_w/2, .12, led_h/2), led_mat, stage_col, .025)
        panel.rotation_euler[2] = -theta

    # LED overlay text
    make_text("led_title_vision", "VISION", (0, d/2+0.08, h + led_h*.66), 3.2, mats["gold"], stage_col)
    make_text("led_title_ignited", "IGNITED", (0, d/2+0.06, h + led_h*.45), 3.2, mats["gold"], stage_col)
    make_text("led_subtitle", "CREATE WITHOUT LIMITS", (0, d/2+0.04, h + led_h*.24), 1.05, mats["accent"], stage_col)

    # Scenic side towers and gold diagonal trims
    tower_w = max(3.6, w*.065)
    for side, x, sign in [("left", -led_w/2 - tower_w*1.15, -1), ("right", led_w/2 + tower_w*1.15, 1)]:
        cube(f"{side}_tall_scenic_tower", (x, d/2+0.2, h+led_h/2), (tower_w/2, .45, led_h*.52), mats["dark_metal"], stage_col, .06)
        for k in range(5):
            bar = cube(f"{side}_diagonal_gold_light_{k}", (x + sign*(-1.1+k*.55), d/2-.22, h+2.0+k*1.55),
                       (.055, .09, 1.55), mats["gold"], stage_col, .014)
            bar.rotation_euler[1] = math.radians(sign*23)

    # Truss: cylinders instead of cubes
    truss_z = h + led_h + 2.2
    y_front = d/2 - 4.3
    y_back = d/2 + 1.8
    x_left = -w*.43
    x_right = w*.43
    cyl("truss_front_tube", (0, y_front, truss_z), .11, x_right-x_left, mats["dark_metal"], stage_col, 32, (0, math.radians(90), 0))
    cyl("truss_back_tube", (0, y_back, truss_z), .11, x_right-x_left, mats["dark_metal"], stage_col, 32, (0, math.radians(90), 0))
    cyl("truss_left_tube", (x_left, (y_front+y_back)/2, truss_z), .11, y_back-y_front, mats["dark_metal"], stage_col, 32, (math.radians(90), 0, 0))
    cyl("truss_right_tube", (x_right, (y_front+y_back)/2, truss_z), .11, y_back-y_front, mats["dark_metal"], stage_col, 32, (math.radians(90), 0, 0))
    for x in [x_left, x_right]:
        for yy in [y_front, y_back]:
            cyl(f"truss_drop_{x}_{yy}", (x, yy, truss_z-5.4), .09, 10.8, mats["dark_metal"], stage_col, 24)
    for x in [x_left + i*4 for i in range(int((x_right-x_left)/4)+1)]:
        cyl(f"truss_cross_x_{x}", (x, (y_front+y_back)/2, truss_z), .055, y_back-y_front, mats["dark_metal"], stage_col, 18, (math.radians(90), 0, 0))

    # Moving head lights and physical beams
    for i, x in enumerate([-22, -16, -10, -4, 4, 10, 16, 22]):
        cyl(f"moving_head_yoke_{i}", (x, y_front, truss_z-.48), .34, .38, mats["dark_metal"], stage_col, 24, (math.radians(90),0,0))
        lamp = cyl(f"moving_head_lens_{i}", (x, y_front-.22, truss_z-.86), .26, .32, mats["accent"], stage_col, 32, (math.radians(90),0,0))
        look_target = (x*.16, -d/2+2.2, h+.55)

        bpy.ops.object.light_add(type="SPOT", location=(x, y_front-.35, truss_z-.9))
        spot = bpy.context.object
        spot.name = f"real_spotlight_{i}"
        spot.data.energy = 1450
        spot.data.color = (1, .76, .42)
        spot.data.spot_size = math.radians(28)
        spot.data.spot_blend = .55
        look_at(spot, look_target)
        move_to_collection(spot, stage_col)

        bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=2.4, radius2=.18, depth=10.5, location=(x*.58, y_front-4.9, truss_z-5.0))
        beam = bpy.context.object
        beam.name = f"soft_visible_light_cone_{i}"
        beam.rotation_euler[0] = math.radians(14)
        apply_material(beam, mats["beam"])
        move_to_collection(beam, stage_col)

    # Speaker stacks
    for side, x in [("left", -w/2-3.2), ("right", w/2+3.2)]:
        for z in [h+1.0, h+2.45, h+3.9]:
            box = cube(f"{side}_speaker_box_{z:.1f}", (x, -d/2+3, z), (1.1, .75, .6), mats["speaker"], stage_col, .035)
            cyl(f"{side}_speaker_driver_{z:.1f}_a", (x, -d/2+2.22, z+.13), .22, .045, mats["dark_metal"], stage_col, 24, (math.radians(90),0,0))
            cyl(f"{side}_speaker_driver_{z:.1f}_b", (x, -d/2+2.22, z-.18), .18, .045, mats["dark_metal"], stage_col, 24, (math.radians(90),0,0))

    # Premium podium
    cube("premium_black_podium_body", (-w*.34, -d*.16, h+1.05), (1.25,.78,1.05), mats["stage_side"], stage_col, .06)
    cube("podium_gold_top", (-w*.34, -d*.16, h+2.18), (1.44,.92,.10), mats["gold"], stage_col, .025)
    cube("podium_gold_logo_plate", (-w*.34, -d*.55, h+1.05), (.42,.035,.42), mats["gold"], stage_col, .018)

    # VIP foreground lounge furniture with legs
    sofa_y = -d/2 - 9.2
    for idx, x in enumerate([-13, 0, 13]):
        cube(f"vip_sofa_{idx}_seat", (x, sofa_y, .58), (2.7,.76,.32), mats["seat_fabric"], stage_col, .10)
        cube(f"vip_sofa_{idx}_back", (x, sofa_y+.68, 1.22), (2.7,.18,.78), mats["seat_fabric"], stage_col, .09)
        cube(f"vip_sofa_{idx}_arm_l", (x-2.85, sofa_y, .90), (.18,.74,.62), mats["seat_fabric"], stage_col, .05)
        cube(f"vip_sofa_{idx}_arm_r", (x+2.85, sofa_y, .90), (.18,.74,.62), mats["seat_fabric"], stage_col, .05)
        for lx in [-1.8, 1.8]:
            for ly in [-.45, .45]:
                cyl(f"vip_sofa_{idx}_leg_{lx}_{ly}", (x+lx, sofa_y+ly, .24), .055, .48, mats["seat_metal"], stage_col, 14)

        # Cocktail table
        cyl(f"round_table_top_{idx}", (x+4.2, sofa_y-.12, .62), .78, .07, mats["gold"], stage_col, 36)
        cyl(f"round_table_stem_{idx}", (x+4.2, sofa_y-.12, .34), .055, .58, mats["gold"], stage_col, 18)
        cyl(f"round_table_base_{idx}", (x+4.2, sofa_y-.12, .06), .48, .06, mats["dark_metal"], stage_col, 36)

    return {"width": w, "depth": d, "height": h, "led_h": led_h}


def build_audience(scene_data, mats, stage_col):
    aud = scene_data.get("audience", {})
    rows = int(aud.get("rows", 7))
    cols = int(aud.get("cols", 12))
    start_y = -28
    for r in range(rows):
        count = cols + (1 if r % 2 else 0)
        for c in range(count):
            x = (c - (count - 1)/2) * 2.05
            y = start_y - r * 2.25
            cube(f"chair_{r}_{c}_seat", (x,y,.48), (.44,.44,.18), mats["seat_fabric"], stage_col, .045)
            cube(f"chair_{r}_{c}_back", (x,y+.35,.98), (.44,.075,.42), mats["seat_fabric"], stage_col, .035)
            for lx in [-.29,.29]:
                for ly in [-.25,.25]:
                    cyl(f"chair_{r}_{c}_leg_{lx}_{ly}", (x+lx,y+ly,.25), .025, .45, mats["seat_metal"], stage_col, 8)


# -----------------------------
# Sketchy environment
# -----------------------------

def build_sketch_environment(scene_data, mats, env_col):
    stage = scene_data.get("stage", {})
    w = float(stage.get("width", 60))
    d = float(stage.get("depth", 24))

    wire_box("venue_shell_sketch", (0,-12,14), (w+36, d+64, 28), mats["sketch"], env_col, .018)
    # Floor grid
    for x in range(-50, 51, 5):
        line_obj(f"sketch_floor_x_{x}", [(x,-66,.06),(x,32,.06)], mats["sketch"], env_col, .008)
    for y in range(-66, 33, 5):
        line_obj(f"sketch_floor_y_{y}", [(-50,y,.06),(50,y,.06)], mats["sketch"], env_col, .008)

    # Far seating blocks as architecture diagram
    for r in range(5):
        for side, x in [("L",-36),("R",36)]:
            wire_box(f"sketch_tier_{side}_{r}", (x, -22-r*7, 1.25), (14,4.2,2.5), mats["sketch"], env_col, .015)

    # Venue columns / skyline / context
    for i,x in enumerate([-45,-37,-30,-24,24,30,37,45]):
        h = 7 + (i % 4)*2.8
        wire_box(f"context_architecture_{i}", (x, 22-(i%2)*5, h/2), (3.8+(i%2),3.2,h), mats["sketch"], env_col, .014)

    # Tree/context objects
    for i,(x,y,s) in enumerate([(-43,-5,1.0),(-45,-20,1.3),(43,-5,1.0),(45,-20,1.3),(-38,20,1.1),(38,20,1.1)]):
        line_obj(f"tree_trunk_{i}", [(x,y,0),(x,y,2.8*s)], mats["sketch"], env_col, .015)
        wire_box(f"tree_crown_wire_{i}", (x,y,4.1*s), (3.2*s,3.2*s,3.2*s), mats["sketch"], env_col, .011)

    # Flow arrows
    for i,y in enumerate([-46,-39,-32]):
        line_obj(f"flow_arrow_{i}", [(-44,y,.1),(-22,y+5,.1)], mats["sketch_bright"], env_col, .018)
        line_obj(f"flow_arrow_head_a_{i}", [(-22,y+5,.1),(-24.5,y+3,.1)], mats["sketch_bright"], env_col, .018)
        line_obj(f"flow_arrow_head_b_{i}", [(-22,y+5,.1),(-25,y+6.5,.1)], mats["sketch_bright"], env_col, .018)


def build_dimensions(scene_data, mats, dim_col):
    stage = scene_data.get("stage", {})
    led = scene_data.get("led_wall", {})
    w = float(stage.get("width", 60))
    d = float(stage.get("depth", 24))
    led_h = float(led.get("height", 12))
    z=.14
    y=-d/2-5.5
    x=w/2+6.2
    line_obj("dim_stage_width", [(-w/2,y,z),(w/2,y,z)], mats["sketch_bright"], dim_col, .03)
    line_obj("dim_stage_width_tick_l", [(-w/2,y-1,z),(-w/2,y+1,z)], mats["sketch_bright"], dim_col, .025)
    line_obj("dim_stage_width_tick_r", [(w/2,y-1,z),(w/2,y+1,z)], mats["sketch_bright"], dim_col, .025)
    make_text("dim_stage_width_label", f"{w:.0f}.00 m", (0,y-1.6,z+.15), .9, mats["sketch_bright"], dim_col, (math.radians(90),0,0))

    line_obj("dim_led_height", [(x,d/2+.4,4),(x,d/2+.4,4+led_h)], mats["sketch_bright"], dim_col, .03)
    make_text("dim_led_height_label", f"{led_h:.0f}.00 m", (x+1.2,d/2+.4,4+led_h/2), .75, mats["sketch_bright"], dim_col, (math.radians(90),0,math.radians(90)))

    line_obj("dim_depth", [(w/2+8,-52,z),(w/2+8,d/2,z)], mats["sketch_bright"], dim_col, .022)
    make_text("dim_depth_label", "36.00 m", (w/2+9.2,-25,z+.15), .75, mats["sketch_bright"], dim_col, (math.radians(90),0,math.radians(90)))


# -----------------------------
# Lighting, rendering, outputs
# -----------------------------

def setup_lights(scene_data, mats, col):
    colors = scene_data.get("colors", {})
    primary = hex_to_rgba(colors.get("primary", "#D7A94B"))
    secondary = hex_to_rgba(colors.get("secondary", "#FFD487"))

    # Big soft key
    bpy.ops.object.light_add(type="AREA", location=(0,-30,24))
    key = bpy.context.object
    key.name = "cinematic_large_front_softbox"
    key.data.energy = 2500
    key.data.size = 24
    key.data.color = primary[:3]
    move_to_collection(key, col)

    bpy.ops.object.light_add(type="AREA", location=(-24,-10,18))
    l = bpy.context.object
    l.name = "warm_left_fill"
    l.data.energy = 950
    l.data.size = 12
    l.data.color = secondary[:3]
    move_to_collection(l, col)

    bpy.ops.object.light_add(type="AREA", location=(24,-10,18))
    r = bpy.context.object
    r.name = "warm_right_fill"
    r.data.energy = 950
    r.data.size = 12
    r.data.color = secondary[:3]
    move_to_collection(r, col)

    bpy.ops.object.light_add(type="POINT", location=(0,-12,6))
    p = bpy.context.object
    p.name = "floor_reflection_glow"
    p.data.energy = 420
    p.data.color = primary[:3]
    move_to_collection(p, col)


def setup_world():
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (0.002, 0.003, 0.005)


def setup_render(width, height, samples=112):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.samples = int(samples)
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.max_bounces = 8
        scene.cycles.diffuse_bounces = 3
        scene.cycles.glossy_bounces = 4
        scene.cycles.transparent_max_bounces = 4
        scene.cycles.use_denoising = True
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for dtype in ("OPTIX","CUDA","HIP","METAL","ONEAPI"):
                try:
                    prefs.compute_device_type = dtype
                    scene.cycles.device = "GPU"
                    break
                except Exception:
                    pass
        except Exception:
            scene.cycles.device = "CPU"
    except Exception:
        pass

    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "High Contrast"
    scene.view_settings.exposure = -0.05
    scene.view_settings.gamma = 1


def camera(name, loc, target, lens):
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.object
    cam.name = name
    cam.data.lens = lens
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = (Vector(target) - Vector(loc)).length
    cam.data.dof.aperture_fstop = 6.5
    look_at(cam, target)
    return cam


def set_collection_visibility(stage_col, env_col, dim_col, mode):
    # viewport + render
    show_stage = mode != "sketch"
    show_env = mode != "realistic"
    show_dim = mode != "realistic"
    stage_col.hide_render = not show_stage
    env_col.hide_render = not show_env
    dim_col.hide_render = not show_dim
    for col, show in [(stage_col, show_stage), (env_col, show_env), (dim_col, show_dim)]:
        col.hide_viewport = not show


def render_one(output_dir, name, loc, target, lens, mode, width, height, samples, stage_col, env_col, dim_col):
    set_collection_visibility(stage_col, env_col, dim_col, mode)
    setup_render(width, height, samples)
    cam = camera("CAM_" + name, loc, target, lens)
    bpy.context.scene.camera = cam
    path = os.path.join(output_dir, name + ".png")
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def export_glb(output_dir, stage_col, env_col, dim_col):
    set_collection_visibility(stage_col, env_col, dim_col, "merged")
    path = os.path.join(output_dir, "stage_sketch_environment.glb")
    try:
        bpy.ops.export_scene.gltf(
            filepath=path,
            export_format="GLB",
            export_apply=True,
            export_cameras=True,
            export_lights=True,
            export_yup=True,
        )
    except TypeError:
        bpy.ops.export_scene.gltf(filepath=path, export_format="GLB")
    return path


def write_manifest(output_dir, scene_data, renders, glb_path):
    data = {
        "ok": True,
        "type": "briefcraft_realistic_stage_sketch_environment_v3",
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
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


# -----------------------------
# Main
# -----------------------------

def main():
    payload = load_payload(get_json_path())
    scene_data = payload.get("scene", {})
    render_data = payload.get("render", {})

    output_dir = render_data.get("output_dir", os.path.join(os.getcwd(), "briefcraft_render_output"))
    width = int(render_data.get("width", 1920))
    height = int(render_data.get("height", 1080))
    samples = int(render_data.get("samples", 112))
    os.makedirs(output_dir, exist_ok=True)

    clear_scene()
    setup_world()

    stage_col = ensure_collection("STAGE_REALISTIC")
    env_col = ensure_collection("ENVIRONMENT_SKETCH")
    dim_col = ensure_collection("DIMENSIONS")

    colors = scene_data.get("colors", {"primary": "#D7A94B", "secondary": "#FFD487"})
    mats = make_materials(colors)

    # Real reflective floor
    plane("large_black_reflective_floor", (0, -14, 0), (72, 72, 1), mats["floor"], stage_col)

    stage_info = build_stage(scene_data, mats, stage_col)
    build_audience(scene_data, mats, stage_col)
    build_sketch_environment(scene_data, mats, env_col)
    build_dimensions(scene_data, mats, dim_col)
    setup_lights(scene_data, mats, stage_col)

    target = tuple(scene_data.get("camera_target", [0, -1.5, 7.5]))
    cams = {
        "realistic_view": ((0, -62, 20), target, 32, "realistic"),
        "sketch_concept": ((-42, -48, 30), target, 30, "sketch"),
        "merged_hybrid": ((0, -68, 24), target, 29, "merged"),
        "stage_closeup": ((18, -34, 14), (0, 2, 8), 48, "realistic"),
        "top_plan": ((0, -12, 86), (0, -12, 0), 28, "merged"),
    }

    renders = {}
    for name, (loc, tgt, lens, mode) in cams.items():
        renders[name] = render_one(output_dir, name, loc, tgt, lens, mode, width, height, samples, stage_col, env_col, dim_col)

    glb = export_glb(output_dir, stage_col, env_col, dim_col)
    manifest = write_manifest(output_dir, scene_data, renders, glb)

    print(json.dumps({"ok": True, "manifest": manifest, "renders": renders, "glb": glb}, indent=2))


if __name__ == "__main__":
    main()
