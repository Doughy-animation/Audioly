bl_info = {
    "name": "Audioly",
    "author": "Patrick Daugherty - Doughy Animation Studio",
    "version": (2, 0, 0),
    "blender": (4, 3, 0),
    "location": "View3D > Sidebar > Audioly",
    "description": ("Bake Shape keys based on user inputted audio tracks."),
    "category": "3D View",
    "doc_url": "https://www.doughyanimation.com/audioly",
}

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
import bpy
import os
import sys
import subprocess
import tempfile
import wave
import random
import math
import shutil
import numpy as np
import urllib.request
import zipfile
import tarfile
from mathutils import Vector
from bpy.props import (
    StringProperty,
    FloatProperty,
    IntProperty,
    BoolProperty,
    EnumProperty,
)

# -------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------
def download_and_extract(url, extract_to, members_filter=None):
    """Download an archive and extract selected members to extract_to."""
    os.makedirs(extract_to, exist_ok=True)
    tmp_path = tempfile.mktemp(suffix=os.path.splitext(url)[1])
    urllib.request.urlretrieve(url, tmp_path)

    extracted = []
    if url.endswith(".zip"):
        with zipfile.ZipFile(tmp_path, "r") as zf:
            for member in zf.namelist():
                if members_filter and not members_filter(member):
                    continue
                zf.extract(member, extract_to)
                extracted.append(os.path.join(extract_to, member))
    elif url.endswith((".tar.xz", ".tar.gz", ".tgz")):
        with tarfile.open(tmp_path, "r:*") as tf:
            for member in tf.getmembers():
                if members_filter and not members_filter(member.name):
                    continue
                tf.extract(member, extract_to)
                extracted.append(os.path.join(extract_to, member.name))

    os.remove(tmp_path)
    return extracted


def get_ffmpeg_exe():
    """
    Locate an ffmpeg executable.

    Search order:
        1. System PATH
        2. imageio-ffmpeg helper
        3. Download a static build for the current platform
    """
    # 1. PATH
    exe = shutil.which("ffmpeg")
    if exe and os.path.isfile(exe):
        return exe

    # 2. imageio-ffmpeg
    try:
        import imageio_ffmpeg as _i

        exe = _i.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass

    # 3. Fetch a static build
    addon_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(addon_dir, "ffmpeg_cache")
    os.makedirs(cache_dir, exist_ok=True)

    if sys.platform.startswith("win"):
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        try:
            extracted = download_and_extract(
                url,
                cache_dir,
                members_filter=lambda n: n.lower().endswith("ffmpeg.exe"),
            )
            for path in extracted:
                if path.lower().endswith("ffmpeg.exe"):
                    return path
        except Exception as e:
            print(f"[Audioly] FFmpeg download failed (Windows): {e}")

    elif sys.platform.startswith("linux"):
        url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        try:
            extracted = download_and_extract(
                url,
                cache_dir,
                members_filter=lambda n: os.path.basename(n) == "ffmpeg",
            )
            for path in extracted:
                if os.path.basename(path) == "ffmpeg":
                    os.chmod(path, 0o755)
                    return path
        except Exception as e:
            print(f"[Audioly] FFmpeg download failed (Linux): {e}")

    elif sys.platform == "darwin":
        url = "https://evermeet.cx/ffmpeg/ffmpeg-6.0.zip"
        try:
            extracted = download_and_extract(
                url,
                cache_dir,
                members_filter=lambda n: os.path.basename(n) == "ffmpeg",
            )
            for path in extracted:
                if os.path.basename(path) == "ffmpeg":
                    os.chmod(path, 0o755)
                    return path
        except Exception as e:
            print(f"[Audioly] FFmpeg download failed (macOS): {e}")

    return None


FFMPEG_EXE = get_ffmpeg_exe()


def mod_collection(obj):
    """Return the correct modifier collection for meshes and grease-pencil objects."""
    return getattr(obj, "grease_pencil_modifiers", None) or obj.modifiers


# -------------------------------------------------------------------------
# Mesh static-shape helpers
# -------------------------------------------------------------------------
def update_static_keys(self, context):
    """Refresh static deformation keys when strength / gains change."""
    obj = context.object
    P = context.scene.audio_shape_props
    if not obj or obj.type != "MESH":
        return

    mesh = obj.data
    if not mesh.shape_keys:
        obj.shape_key_add(name="Basis")
    kb = mesh.shape_keys.key_blocks

    # Ensure keys exist
    for nm in ("Grow", "Shrink", "Distort", "Twist", "Noise"):
        if nm not in kb:
            obj.shape_key_add(name=nm)

    normals = [v.normal.copy() for v in mesh.vertices]
    rnd = random.Random(42)
    c, s = math.cos(P.twist_angle), math.sin(P.twist_angle)

    for i, nrm in enumerate(normals):
        base = kb["Basis"].data[i].co

        kb["Grow"].data[i].co = base + nrm * P.strength * P.low_gain
        kb["Shrink"].data[i].co = base - nrm * P.strength * P.low_gain
        kb["Distort"].data[i].co = (
            base + nrm * P.strength * P.mid_gain * P.distort_ratio
        )

        xr = base.x * c - base.y * s
        yr = base.x * s + base.y * c
        kb["Twist"].data[i].co = base + (
            Vector((xr, yr, base.z)) - base
        ) * P.high_gain

        kb["Noise"].data[i].co = (
            base + nrm * P.noise_strength * rnd.uniform(-1, 1) * P.randomizer
        )


# -------------------------------------------------------------------------
# Audio-loading helpers
# -------------------------------------------------------------------------
def load_audio(props, context):
    """
    Load or reload props.audio_path into props.audio_sound and,
    if requested, add a strip to the Sequencer via the direct API.
    """
    path = bpy.path.abspath(props.audio_path)
    if not path or not os.path.isfile(path):
        props.audio_sound = None
        return

    # Find or load Sound datablock
    snd = next(
        (s for s in bpy.data.sounds if bpy.path.abspath(s.filepath) == path), None
    )
    if snd is None:
        try:
            snd = bpy.data.sounds.load(path)
        except RuntimeError as e:
            print(f"[Audioly] Sound load failed: {e}")
            snd = None

    props.audio_sound = snd

    # Automatically import to Sequencer using the direct API
    if snd and props.import_to_sequencer:
        scene = context.scene
        seq = scene.sequence_editor or scene.sequence_editor_create()
        exists = any(
            st.type == 'SOUND' and bpy.path.abspath(getattr(st, 'filepath', '')) == path
            for st in seq.sequences_all
        )
        if not exists:
            try:
                name = os.path.basename(path)
                seq.sequences.new_sound(name, path, 1, props.frame_start)
            except Exception as e:
                print(f"[Audioly] Failed to add sound strip via API: {e}")


def audio_path_update(self, context):
    load_audio(self, context)
    update_static_keys(self, context)


# -------------------------------------------------------------------------
# Particle & color helpers / preset driver updates
# -------------------------------------------------------------------------
def update_particles(self, context):
    obj = context.object
    P = context.scene.audio_shape_props
    if not obj or obj.type != "MESH" or not P.baked_object:
        return

    if P.use_particles:
        if "AudioParticles" not in obj.modifiers:
            mod = obj.modifiers.new("AudioParticles", type="PARTICLE_SYSTEM")
            psys = obj.particle_systems[-1]
            psys.name = "AudioParticles"
            settings = bpy.data.particles.new("AudioParticlesSettings")
            psys.settings = settings
            settings.render_type = "OBJECT"
            settings.instance_object = context.scene.objects.get(P.baked_object)
    else:
        if "AudioParticles" in obj.modifiers:
            obj.modifiers.remove(obj.modifiers["AudioParticles"])


def update_selected_preset(self, context):
    P = context.scene.audio_shape_props
    baked = context.scene.objects.get(P.baked_object)
    if not baked:
        return

    obj = context.object
    # Update all drivers on the object (mesh, curve, grease pencil, material nodes)
    # to use the selected preset shape key from the baked object

    # Mesh shape key drivers
    if obj and obj.type == "MESH" and obj.data.shape_keys and obj.data.shape_keys.animation_data:
        for fc in obj.data.shape_keys.animation_data.drivers:
            for var in fc.driver.variables:
                if var.type == "SINGLE_PROP" and var.targets and var.targets[0].id == baked:
                    var.targets[0].data_path = f'data.shape_keys.key_blocks["{P.selected_preset}"].value'

    # Object-level drivers (modifiers, etc)
    if obj and obj.animation_data:
        for fc in obj.animation_data.drivers:
            for var in fc.driver.variables:
                if var.type == "SINGLE_PROP" and var.targets and var.targets[0].id == baked:
                    var.targets[0].data_path = f'data.shape_keys.key_blocks["{P.selected_preset}"].value'

    # Curve data drivers
    if obj and obj.type == "CURVE" and obj.data.animation_data:
        for fc in obj.data.animation_data.drivers:
            for var in fc.driver.variables:
                if var.type == "SINGLE_PROP" and var.targets and var.targets[0].id == baked:
                    var.targets[0].data_path = f'data.shape_keys.key_blocks["{P.selected_preset}"].value'

    # Grease pencil modifiers
    if obj and hasattr(obj, "grease_pencil_modifiers"):
        for m in obj.grease_pencil_modifiers:
            if hasattr(m, "animation_data") and m.animation_data:
                for fc in m.animation_data.drivers:
                    for var in fc.driver.variables:
                        if var.type == "SINGLE_PROP" and var.targets and var.targets[0].id == baked:
                            var.targets[0].data_path = f'data.shape_keys.key_blocks["{P.selected_preset}"].value'

    # Material node tree drivers
    mat = getattr(obj, "active_material", None)
    if mat and mat.use_nodes and mat.node_tree and mat.node_tree.animation_data:
        for fc in mat.node_tree.animation_data.drivers:
            for var in fc.driver.variables:
                if var.type == "SINGLE_PROP" and var.targets and var.targets[0].id == baked:
                    var.targets[0].data_path = f'data.shape_keys.key_blocks["{P.selected_preset}"].value'


def update_colors_section(self, context):
    P, obj = context.scene.audio_shape_props, context.object
    if not obj or obj.type != "MESH":
        return

    mat = obj.active_material
    if mat and mat.use_nodes:
        nt = mat.node_tree
        for n in list(nt.nodes):
            if n.get("audio_node"):
                nt.nodes.remove(n)

    if not mat:
        mat = bpy.data.materials.new("AudioColorMat")
        mat.use_nodes = True
        obj.active_material = mat
    elif not mat.use_nodes:
        mat.use_nodes = True

    nt = mat.node_tree
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp["audio_node"] = True
    ramp.label = "Audio ColorRamp"
    ramp.location = (0, 200)

    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf:
        for sock in ("Base Color", "Emission", "Emission Color"):
            if sock in bsdf.inputs:
                nt.links.new(ramp.outputs["Color"], bsdf.inputs[sock])
        if "Emission Strength" in bsdf.inputs:
            es = bsdf.inputs["Emission Strength"]
            es.default_value = min(max(es.default_value, 0.0), 10.0)

    if P.effect_type != "NONE":
        cls_map = {
            "MAGIC": "ShaderNodeTexMagic",
            "WAVE": "ShaderNodeTexWave",
            "NOISE": "ShaderNodeTexNoise",
        }
        node = nt.nodes.new(cls_map[P.effect_type])
        node["audio_node"] = True
        node["audio_effect"] = True
        node.label = f"Audio {P.effect_type}"
        node.location = (-200, 200)
        out = node.outputs.get("Fac") or node.outputs[0]
        nt.links.new(out, ramp.inputs["Fac"])


# -------------------------------------------------------------------------
# Modifier dictionaries and util
# -------------------------------------------------------------------------
MESH_MODS = {"Wave": "WAVE", "Displace": "DISPLACE"}
CURVE_MODS = {"Wave": "WAVE"}
GP_MODS = {
    "Opacity": "GREASE_PENCIL_OPACITY",
    "Noise": "GREASE_PENCIL_NOISE",
    "Thickness": "GREASE_PENCIL_THICKNESS",
    "Simplify": "GREASE_PENCIL_SIMPLIFY",
    "Envelope": "GREASE_PENCIL_ENVELOPE",
    "Length": "GREASE_PENCIL_LENGTH",
    "Outline": "GREASE_PENCIL_OUTLINE",
    "MultiStroke": "GREASE_PENCIL_MULTIPLY",
}


def allowed_mod_dict(obj):
    if not obj:
        return {}
    if obj.type == "MESH":
        return MESH_MODS
    if obj.type == "CURVE":
        return CURVE_MODS
    if obj.type in {"GPENCIL", "GREASEPENCIL"}:
        return GP_MODS
    return {}


def addable_items(self, context):
    return [(c, name, "") for name, c in allowed_mod_dict(context.object).items()]


# -------------------------------------------------------------------------
# Operators
# -------------------------------------------------------------------------
class AUDIO_OT_install_deps(bpy.types.Operator):
    bl_idname = "audio_shape.install_deps"
    bl_label = "Fetch FFmpeg"

    def execute(self, context):
        global FFMPEG_EXE
        FFMPEG_EXE = get_ffmpeg_exe()
        if FFMPEG_EXE:
            self.report({"INFO"}, "FFmpeg downloaded / found and ready")
            return {"FINISHED"}
        self.report({"ERROR"}, "Failed to download or locate FFmpeg")
        return {"CANCELLED"}


class AUDIO_OT_ensure_color_material(bpy.types.Operator):
    bl_idname = "audio_shape.ensure_color_material"
    bl_label = "Setup Color & Effects"

    def execute(self, context):
        update_colors_section(None, context)
        for area in context.window.screen.areas:
            if area.type in {"NODE_EDITOR", "VIEW_3D"}:
                area.tag_redraw()
        self.report({"INFO"}, "Color & Effects ready")
        return {"FINISHED"}


class AUDIO_OT_move_modifier(bpy.types.Operator):
    bl_idname = "audio_shape.move_modifier"
    bl_label = "Move Modifier"

    mod_name: StringProperty()
    direction: IntProperty()

    def execute(self, context):
        mc = mod_collection(context.object)
        if mc and self.mod_name in mc:
            i = list(mc).index(mc[self.mod_name])
            j = i + self.direction
            if 0 <= j < len(mc):
                mc.move(i, j)
        return {"FINISHED"}


class AUDIO_OT_reset_driver_prop(bpy.types.Operator):
    bl_idname = "audio_shape.reset_driver_prop"
    bl_label = "Reset & Unlink"

    mod_name: StringProperty()
    prop_name: StringProperty()

    def execute(self, context):
        obj = context.object
        mc = mod_collection(obj)
        m = mc.get(self.mod_name)
        if m:
            path = f'modifiers["{m.name}"].{self.prop_name}'
            if obj.animation_data:
                try:
                    obj.driver_remove(path)
                except Exception:
                    pass
            default = type(m).bl_rna.properties[self.prop_name].default
            setattr(m, self.prop_name, default)
        return {"FINISHED"}


class AUDIO_OT_reset_node_driver(bpy.types.Operator):
    bl_idname = "audio_shape.reset_node_driver"
    bl_label = "Reset & Unlink Node"

    node_name: StringProperty()
    input_idx: IntProperty()

    def execute(self, context):
        mat = context.object.active_material
        if mat and mat.use_nodes:
            nt = mat.node_tree
            node = nt.nodes.get(self.node_name)
            if node:
                if node.type == "VALTORGB":
                    elems = node.color_ramp.elements
                    if 0 <= self.input_idx < len(elems):
                        elems[self.input_idx].position = (
                            type(elems[self.input_idx])
                            .bl_rna.properties["position"]
                            .default
                        )
                        path = f'nodes["{node.name}"].color_ramp.elements[{self.input_idx}].position'
                else:
                    inp = node.inputs[self.input_idx]
                    inp.default_value = (
                        type(inp).bl_rna.properties["default_value"].default
                    )
                    path = f'nodes["{node.name}"].inputs[{self.input_idx}].default_value'

                if nt.animation_data:
                    try:
                        nt.driver_remove(path)
                    except Exception:
                        pass
        return {"FINISHED"}


class AUDIO_OT_toggle_node_driver(bpy.types.Operator):
    bl_idname = "audio_shape.toggle_node_driver"
    bl_label = "Toggle Node Driver"

    node_name: StringProperty()
    input_idx: IntProperty()

    def execute(self, context):
        P = context.scene.audio_shape_props
        mat = context.object.active_material
        if not mat or not mat.use_nodes:
            return {"CANCELLED"}

        nt = mat.node_tree
        node = nt.nodes.get(self.node_name)
        if not node:
            return {"CANCELLED"}

        if node.type == "VALTORGB":
            path = f'nodes["{node.name}"].color_ramp.elements[{self.input_idx}].position'
        else:
            path = f'nodes["{node.name}"].inputs[{self.input_idx}].default_value'

        removed = False
        if nt.animation_data and nt.animation_data.drivers:
            for fc in list(nt.animation_data.drivers):
                if fc.data_path == path:
                    try:
                        nt.driver_remove(path)
                    except Exception:
                        pass
                    removed = True
                    break

        if not removed:
            fcu = nt.driver_add(path)
            drv = fcu.driver
            drv.type = "SCRIPTED"

            v1 = drv.variables.new()
            v1.name = "var1"
            v1.type = "SINGLE_PROP"
            t1 = v1.targets[0]
            t1.id = context.scene.objects.get(P.baked_object)
            t1.data_path = (
                f'data.shape_keys.key_blocks["{P.selected_preset}"].value'
            )

            v2 = drv.variables.new()
            v2.name = "var2"
            v2.type = "SINGLE_PROP"
            t2 = v2.targets[0]
            t2.id_type = "SCENE"
            t2.id = context.scene
            t2.data_path = "audio_shape_props.driver_strength"

            drv.expression = "var1 * var2"

        for area in context.window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"FINISHED"}


class AUDIO_OT_toggle_driver(bpy.types.Operator):
    bl_idname = "audio_shape.toggle_driver"
    bl_label = "Toggle Driver"

    mod_name: StringProperty()
    prop_name: StringProperty()

    def execute(self, context):
        P = context.scene.audio_shape_props
        obj = context.object
        mc = mod_collection(obj)
        m = mc.get(self.mod_name)
        if not m:
            return {"CANCELLED"}

        path = f'modifiers["{m.name}"].{self.prop_name}'
        removed = False

        if obj.animation_data and obj.animation_data.drivers:
            for fc in list(obj.animation_data.drivers):
                if fc.data_path == path:
                    try:
                        obj.driver_remove(path)
                    except Exception:
                        pass
                    removed = True
                    break

        if not removed:
            fcu = m.driver_add(self.prop_name)
            drv = fcu.driver
            drv.type = "SCRIPTED"

            v1 = drv.variables.new()
            v1.name = "var1"
            v1.type = "SINGLE_PROP"
            t1 = v1.targets[0]
            t1.id = context.scene.objects.get(P.baked_object)
            t1.data_path = (
                f'data.shape_keys.key_blocks["{P.selected_preset}"].value'
            )

            v2 = drv.variables.new()
            v2.name = "var2"
            v2.type = "SINGLE_PROP"
            t2 = v2.targets[0]
            t2.id_type = "SCENE"
            t2.id = context.scene
            t2.data_path = "audio_shape_props.driver_strength"

            drv.expression = "var1 * var2"

        for area in context.window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"FINISHED"}


class AUDIO_OT_toggle_mod_expand(bpy.types.Operator):
    bl_idname = "audio_shape.toggle_mod_expand"
    bl_label = ""

    mod_name: StringProperty()

    def execute(self, context):
        mc = mod_collection(context.object)
        if mc and mc.get(self.mod_name):
            mc[self.mod_name].show_expanded = not mc[self.mod_name].show_expanded
        return {"FINISHED"}


class AUDIO_OT_reset_curve_prop(bpy.types.Operator):
    bl_idname = "audio_shape.reset_curve_prop"
    bl_label = "Reset Curve Prop"

    prop_name: StringProperty()

    def execute(self, context):
        data = context.object.data
        default = type(data).bl_rna.properties[self.prop_name].default
        setattr(data, self.prop_name, default)
        if data.animation_data:
            try:
                data.driver_remove(self.prop_name)
            except Exception:
                pass

        for area in context.window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"FINISHED"}


class AUDIO_OT_toggle_curve_driver(bpy.types.Operator):
    bl_idname = "audio_shape.toggle_curve_driver"
    bl_label = "Toggle Curve Driver"

    prop_name: StringProperty()

    def execute(self, context):
        P = context.scene.audio_shape_props
        mesh_obj = context.scene.objects.get(P.baked_object)
        data = context.object.data
        path = self.prop_name
        removed = False

        if data.animation_data and data.animation_data.drivers:
            for fc in list(data.animation_data.drivers):
                if fc.data_path == path:
                    try:
                        data.driver_remove(path)
                    except Exception:
                        pass
                    removed = True
                    break

        if not removed:
            fcu = data.driver_add(path)
            drv = fcu.driver
            drv.type = "SCRIPTED"

            v1 = drv.variables.new()
            v1.name = "var1"
            v1.type = "SINGLE_PROP"
            t1 = v1.targets[0]
            t1.id = mesh_obj
            t1.data_path = (
                f'data.shape_keys.key_blocks["{P.selected_preset}"].value'
            )

            v2 = drv.variables.new()
            v2.name = "var2"
            v2.type = "SINGLE_PROP"
            t2 = v2.targets[0]
            t2.id_type = "SCENE"
            t2.id = context.scene
            t2.data_path = "audio_shape_props.driver_strength"

            drv.expression = "var1 * var2"

        for area in context.window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"FINISHED"}


class AUDIO_OT_generate_shape_keys(bpy.types.Operator):
    bl_idname = "audio_shape.generate_keys"
    bl_label = "Bake Audio on Mesh"

    def execute(self, context):
        P, obj = context.scene.audio_shape_props, context.object

        # Ensure audio_sound is loaded
        if P.audio_sound is None and P.audio_path:
            load_audio(P, context)

        snd = P.audio_sound
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh")
            return {"CANCELLED"}
        if not snd:
            self.report({"ERROR"}, "No audio loaded")
            return {"CANCELLED"}
        if not FFMPEG_EXE:
            self.report(
                {"ERROR"},
                "FFmpeg executable not found. Click 'Fetch FFmpeg' or install FFmpeg in PATH.",
            )
            return {"CANCELLED"}

        mesh = obj.data
        static = ["Grow", "Shrink", "Distort", "Twist", "Noise"]
        presets = ["ALL", "VOICE", "PIANO", "GUITAR", "DRUM", "BASS"]

        if mesh.shape_keys:
            for kb in list(mesh.shape_keys.key_blocks):
                if kb.name in static + presets:
                    obj.shape_key_remove(kb)
        else:
            obj.shape_key_add(name="Basis")
        for nm in static + presets:
            if nm not in mesh.shape_keys.key_blocks:
                obj.shape_key_add(name=nm)

        BANDS = {
            "ALL": (None, None),
            "VOICE": (300, 3400),
            "PIANO": (27, 4186),
            "GUITAR": (82, 1175),
            "DRUM": (50, 3000),
            "BASS": (20, 250),
        }

        tmp = tempfile.mktemp(".wav")
        subprocess.run(
            [
                FFMPEG_EXE,
                "-y",
                "-i",
                bpy.path.abspath(snd.filepath),
                "-ac",
                "2",
                "-ar",
                "44100",
                tmp,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wf = wave.open(tmp, "rb")
        raw = wf.readframes(wf.getnframes())
        sw = wf.getsampwidth()
        wf.close()
        os.remove(tmp)

        stereo = (
            np.frombuffer(raw, {1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(
                np.float32
            )
            / float(2 ** (8 * sw - 1))
        )
        left, right = stereo[0::2], stereo[1::2]
        mono = (left + right) * 0.5

        N = len(mono)
        freqs = np.fft.rfftfreq(N, 1 / 44100)
        spec = np.fft.rfft(mono)

        band_sq = {}
        for nm, (lo, hi) in BANDS.items():
            if nm == "ALL":
                band_sq[nm] = mono ** 2
            else:
                mask = (freqs >= lo) & (freqs < hi)
                sig = np.fft.irfft(spec * mask, n=N)
                band_sq[nm] = sig ** 2

        scene, fps = context.scene, context.scene.render.fps
        total = int(N / 44100 * fps)
        step = N / total
        rnd = random.Random(42)

        for i in range(total):
            frame = P.frame_start + i
            scene.frame_set(frame)
            kb = mesh.shape_keys.key_blocks

            low = math.sqrt(
                band_sq["ALL"][int(i * step) : int((i + 1) * step)].mean()
            )
            noise = rnd.uniform(0, P.noise_strength)

            kb["Grow"].value = low * P.low_gain
            kb["Shrink"].value = 1.0 - low * P.low_gain
            kb["Distort"].value = low * P.mid_gain
            kb["Twist"].value = low * P.high_gain
            kb["Noise"].value = noise

            for nm in static:
                kb[nm].keyframe_insert("value", frame=frame)

            for nm in presets:
                bake = True
                if nm == "ALL" and not P.bake_preset_all:
                    bake = False
                if nm == "VOICE" and not P.bake_preset_voice:
                    bake = False
                if nm == "PIANO" and not P.bake_preset_piano:
                    bake = False
                if nm == "GUITAR" and not P.bake_preset_guitar:
                    bake = False
                if nm == "DRUM" and not P.bake_preset_drum:
                    bake = False
                if nm == "BASS" and not P.bake_preset_bass:
                    bake = False
                if not bake:
                    continue

                sq = band_sq[nm][int(i * step) : int((i + 1) * step)]
                val = math.sqrt(sq.mean()) if sq.size else 0.0
                if nm == "VOICE":
                    thr = P.voice_threshold
                    val = (val - thr) / (1 - thr) if val > thr else 0.0

                kb[nm].value = val
                kb[nm].keyframe_insert("value", frame=frame)

        scene.frame_end = P.frame_start + total - 1
        obj["_audio_shape_baked"] = True
        P.baked_object = obj.name

        # --- Auto-increment strength to force mesh refresh ---
        P.strength = min(P.strength + 0.01, 1.0)
        update_static_keys(P, context)

        self.report({"INFO"}, "Baked with band-pass & voice clamp")
        return {"FINISHED"}


class AUDIO_OT_unbake(bpy.types.Operator):
    bl_idname = "audio_shape.unbake"
    bl_label = "Unbake Audio"

    def execute(self, context):
        obj = context.object
        if not obj or not obj.get("_audio_shape_baked"):
            return {"CANCELLED"}

        mesh = obj.data
        if mesh.shape_keys:
            for kb in list(mesh.shape_keys.key_blocks):
                if kb.name != "Basis":
                    obj.shape_key_remove(kb)

        if obj.data.animation_data:
            obj.data.animation_data_clear()
        if obj.animation_data:
            obj.animation_data.clear()

        mat = obj.active_material
        if mat and mat.use_nodes:
            nt = mat.node_tree
            for n in list(nt.nodes):
                if n.get("audio_node"):
                    nt.nodes.remove(n)

        obj.pop("_audio_shape_baked", None)
        if context.scene.audio_shape_props.baked_object == obj.name:
            context.scene.audio_shape_props.baked_object = ""

        self.report({"INFO"}, "Unbaked & restored")
        return {"FINISHED"}


class AUDIO_OT_prev_baked(bpy.types.Operator):
    bl_idname = "audio_shape.prev_baked"
    bl_label = ""

    def execute(self, context):
        names = [o.name for o in context.scene.objects if o.get("_audio_shape_baked")]
        if not names:
            return {"CANCELLED"}

        P = context.scene.audio_shape_props
        idx = names.index(P.baked_object) if P.baked_object in names else 0
        P.baked_object = names[(idx - 1) % len(names)]
        return {"FINISHED"}


class AUDIO_OT_next_baked(bpy.types.Operator):
    bl_idname = "audio_shape.next_baked"
    bl_label = ""

    def execute(self, context):
        names = [o.name for o in context.scene.objects if o.get("_audio_shape_baked")]
        if not names:
            return {"CANCELLED"}

        P = context.scene.audio_shape_props
        idx = names.index(P.baked_object) if P.baked_object in names else -1
        P.baked_object = names[(idx + 1) % len(names)]
        return {"FINISHED"}


class AUDIO_OT_select_baked(bpy.types.Operator):
    bl_idname = "audio_shape.select_baked"
    bl_label = "Select"

    def execute(self, context):
        name = context.scene.audio_shape_props.baked_object
        o = context.scene.objects.get(name)
        if o:
            bpy.ops.object.select_all(action="DESELECT")
            o.select_set(True)
            context.view_layer.objects.active = o
        return {"FINISHED"}


class AUDIO_OT_new_baked(bpy.types.Operator):
    bl_idname = "audio_shape.new_baked"
    bl_label = "Bake Mesh"

    def execute(self, context):
        return bpy.ops.audio_shape.generate_keys()


class AUDIO_OT_link_baked(bpy.types.Operator):
    bl_idname = "audio_shape.link_baked"
    bl_label = "Link Object"

    def execute(self, context):
        P = context.scene.audio_shape_props
        src = context.scene.objects.get(P.baked_object)
        dst = context.object
        if not src or not dst or dst == src:
            return {"CANCELLED"}

        if dst.type == "MESH":
            if not dst.data.shape_keys:
                dst.shape_key_add(name="Basis")
            for nm in src.data.shape_keys.key_blocks.keys():
                if nm not in dst.data.shape_keys.key_blocks:
                    dst.shape_key_add(name=nm)

                fcu = dst.data.shape_keys.key_blocks[nm].driver_add("value")
                drv = fcu.driver
                drv.type = "SCRIPTED"

                v1 = drv.variables.new()
                v1.name = "var1"
                v1.type = "SINGLE_PROP"
                t1 = v1.targets[0]
                t1.id = src
                t1.data_path = f'data.shape_keys.key_blocks["{nm}"].value'

                v2 = drv.variables.new()
                v2.name = "var2"
                v2.type = "SINGLE_PROP"
                t2 = v2.targets[0]
                t2.id_type = "SCENE"
                t2.id = context.scene
                t2.data_path = "audio_shape_props.driver_strength"

                drv.expression = "var1 * var2"

        elif dst.type == "CURVE":
            fcu = dst.data.driver_add("bevel_factor_end")
            drv = fcu.driver
            drv.type = "SCRIPTED"

            v1 = drv.variables.new()
            v1.name = "var1"
            v1.type = "SINGLE_PROP"
            t1 = v1.targets[0]
            t1.id = src
            t1.data_path = (
                f'data.shape_keys.key_blocks["{P.selected_preset}"].value'
            )

            v2 = drv.variables.new()
            v2.name = "var2"
            v2.type = "SINGLE_PROP"
            t2 = v2.targets[0]
            t2.id_type = "SCENE"
            t2.id = context.scene
            t2.data_path = "audio_shape_props.driver_strength"

            drv.expression = "var1 * var2"

        else:
            mc2 = mod_collection(dst)
            for code in (
                "GREASE_PENCIL_THICKNESS",
                "GREASE_PENCIL_OPACITY",
                "GREASE_PENCIL_NOISE",
            ):
                if code not in [m.type for m in mc2]:
                    mc2.new(name=code, type=code).show_expanded = True
                m2 = next(m for m in mc2 if m.type == code)

                pid = [
                    p.identifier
                    for p in m2.bl_rna.properties
                    if p.type in {"FLOAT", "INT"}
                ][-1]

                fcu = m2.driver_add(pid)
                drv = fcu.driver
                drv.type = "SCRIPTED"

                v1 = drv.variables.new()
                v1.name = "var1"
                v1.type = "SINGLE_PROP"
                t1 = v1.targets[0]
                t1.id = src
                t1.data_path = (
                    f'data.shape_keys.key_blocks["{P.selected_preset}"].value'
                )

                v2 = drv.variables.new()
                v2.name = "var2"
                v2.type = "SINGLE_PROP"
                t2 = v2.targets[0]
                t2.id_type = "SCENE"
                t2.id = context.scene
                t2.data_path = "audio_shape_props.driver_strength"

                drv.expression = "var1 * var2"

        dst["_audio_shape_linked_to"] = src.name

        # --- Auto-increment strength to force mesh refresh ---
        if hasattr(P, "strength"):
            P.strength = min(P.strength + 0.01, 1.0)
            update_static_keys(P, context)

        for A in context.window.screen.areas:
            if A.type == "VIEW_3D":
                A.tag_redraw()

        self.report({"INFO"}, f"Linked {dst.name} → {src.name}")
        return {"FINISHED"}


class AUDIO_OT_unlink_baked(bpy.types.Operator):
    bl_idname = "audio_shape.unlink_baked"
    bl_label = "Unlink Object"

    def execute(self, context):
        obj = context.object
        if not obj or not obj.get("_audio_shape_linked_to"):
            return {"CANCELLED"}

        # Remove drivers from shape keys if present
        if obj.data.shape_keys and obj.data.shape_keys.animation_data:
            for fc in list(obj.data.shape_keys.animation_data.drivers):
                try:
                    obj.data.shape_keys.driver_remove(fc.data_path)
                except Exception:
                    pass

        # Remove object-level drivers
        if obj.animation_data:
            obj.animation_data.clear()

        obj.pop("_audio_shape_linked_to", None)

        for area in context.window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()

        self.report({"INFO"}, "Unlinked object")
        return {"FINISHED"}


class AUDIO_OT_reset_particle_prop(bpy.types.Operator):
    bl_idname = "audio_shape.reset_particle_prop"
    bl_label = "Reset Particle Prop"

    prop_name: StringProperty()

    def execute(self, context):
        ps = context.object.particle_systems.get("AudioParticles")
        if not ps:
            return {"CANCELLED"}
        s = ps.settings

        if "." in self.prop_name:
            root, attr = self.prop_name.split(".", 1)
            obj_attr = getattr(s, root)
            default = type(obj_attr).bl_rna.properties[attr].default
            setattr(obj_attr, attr, default)
            # EffectorWeights has no animation_data, so skip driver removal
        else:
            default = type(s).bl_rna.properties[self.prop_name].default
            setattr(s, self.prop_name, default)
            if s.animation_data:
                try:
                    s.driver_remove(self.prop_name)
                except Exception:
                    pass

        for area in context.window.screen.areas:
            area.tag_redraw()
        return {"FINISHED"}


class AUDIO_OT_toggle_particle_driver(bpy.types.Operator):
    bl_idname = "audio_shape.toggle_particle_driver"
    bl_label = "Toggle Particle Driver"

    prop_name: StringProperty()

    def execute(self, context):
        P = context.scene.audio_shape_props
        ps = context.object.particle_systems.get("AudioParticles")
        if not ps:
            return {"CANCELLED"}

        s = ps.settings
        path = self.prop_name

        if s.animation_data and s.animation_data.drivers:
            for d in list(s.animation_data.drivers):
                if d.data_path == path:
                    try:
                        s.driver_remove(path)
                    except Exception:
                        pass
                    return {"FINISHED"}

        fcu = s.driver_add(path)
        drv = fcu.driver
        drv.type = "SCRIPTED"

        v1 = drv.variables.new()
        v1.name = "var1"
        v1.type = "SINGLE_PROP"
        t1 = v1.targets[0]
        t1.id = context.scene.objects.get(P.baked_object)
        t1.data_path = (
            f'data.shape_keys.key_blocks["{P.selected_preset}"].value'
        )

        v2 = drv.variables.new()
        v2.name = "var2"
        v2.type = "SINGLE_PROP"
        t2 = v2.targets[0]
        t2.id_type = "SCENE"
        t2.id = context.scene
        t2.data_path = "audio_shape_props.driver_strength"

        drv.expression = "var1 * var2"

        for area in context.window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"FINISHED"}


class AUDIO_OT_toggle_strength_driver(bpy.types.Operator):
    bl_idname = "audio_shape.toggle_strength_driver"
    bl_label = "Toggle Strength Driver"

    prop_name: StringProperty()

    def execute(self, context):
        scene = context.scene
        path = f"audio_shape_props.{self.prop_name}"

        if scene.animation_data and scene.animation_data.drivers:
            for d in list(scene.animation_data.drivers):
                if d.data_path == path:
                    try:
                        scene.driver_remove(path)
                    except Exception:
                        pass
                    return {"FINISHED"}

        fcu = scene.driver_add(path)
        drv = fcu.driver
        drv.type = "SCRIPTED"

        v1 = drv.variables.new()
        v1.name = "var1"
        v1.type = "SINGLE_PROP"
        t1 = v1.targets[0]
        t1.id = scene.objects.get(scene.audio_shape_props.baked_object)
        t1.data_path = (
            f'data.shape_keys.key_blocks["{scene.audio_shape_props.selected_preset}"].value'
        )

        v2 = drv.variables.new()
        v2.name = "var2"
        v2.type = "SINGLE_PROP"
        t2 = v2.targets[0]
        t2.id_type = "SCENE"
        t2.id = scene
        t2.data_path = "audio_shape_props.driver_strength"

        drv.expression = "var1 * var2"
        return {"FINISHED"}


# -------------------------------------------------------------------------
# Audioly – Preferences Quality-of-Life Buttons (from Painterly)
# -------------------------------------------------------------------------
import bpy
from bpy.types import Operator, AddonPreferences
from bpy.props import BoolProperty

class AudiolySavePreferencesOperator(Operator):
    """Forces Blender to write current add-on prefs to disk"""
    bl_idname = "audioly.save_preferences"
    bl_label  = "Save Preferences"

    def execute(self, context):
        bpy.ops.wm.save_userpref()
        self.report({'INFO'}, "Preferences saved to userpref.blend")
        return {'FINISHED'}

class AudiolyOpenFeedbackOperator(Operator):
    """Opens Doughy Animation’s ‘Contact / Feedback’ web-page"""
    bl_idname = "audioly.open_feedback"
    bl_label  = "Open Feedback Page"

    def execute(self, context):
        bpy.ops.wm.url_open(url="https://www.doughyanimation.com/contact")
        return {'FINISHED'}

class AudiolyOpenYouTubeOperator(Operator):
    """Launches the YouTube playlist of Audioly tutorials"""
    bl_idname = "audioly.open_youtube"
    bl_label  = "Open YouTube Channel"

    def execute(self, context):
        bpy.ops.wm.url_open(url="https://www.youtube.com/@doughyanimation")
        return {'FINISHED'}

class AudiolyAddonPreferences(AddonPreferences):
    """
    Minimalist preferences panel containing only the
    generic utility buttons kept from the full Painterly code.
    """
    bl_idname = __name__

    show_tips: BoolProperty(
        name   = "Show Usage Tips on Startup",
        default= True
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "show_tips")
        row = layout.row()
        row.operator("audioly.save_preferences", icon='FILE_TICK')
        row = layout.row()
        row.operator("audioly.open_feedback", icon='HELP')
        row.operator("audioly.open_youtube", icon='URL')


# -------------------------------------------------------------------------
# Audioly – Automatic Updater System (adapted from Painterly)
# -------------------------------------------------------------------------

import json
from bpy.types import Operator, AddonPreferences
from bpy.app.handlers import persistent

# --- Updater static config ---
AUDIOLY_CURRENT_VERSION = (1, 23, 7)
AUDIOLY_UPDATE_JSON_URL = (
    "https://drive.google.com/uc?export=download&id=1cDVI9vFpD2GZ5QMlzFthylRHOxvA-rJB"
)

def audioly_compare_versions(v1, v2):
    for a, b in zip(v1, v2):
        if a < b:
            return -1
        if a > b:
            return 1
    if len(v1) < len(v2):
        return -1
    if len(v1) > len(v2):
        return  1
    return 0

def audioly_get_addon_key():
    return __name__

class AudiolyAddonUpdaterProperties(bpy.types.PropertyGroup):
    update_checked:    BoolProperty(default=False)
    update_available:  BoolProperty(default=False)
    update_downloaded: BoolProperty(default=False)
    latest_version:   StringProperty(default="")
    latest_changelog: StringProperty(default="")
    latest_url:       StringProperty(default="")

class AudiolyAddonPreferences(AddonPreferences):
    bl_idname = __name__
    enable_auto_update: BoolProperty(
        name   = "Enable Automatic Updates",
        default= True
    )
    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Automatic Updater", icon='FILE_REFRESH')
        box.prop(self, "enable_auto_update")
        props = context.scene.audioly_updater_props
        if self.enable_auto_update:
            if not props.update_checked:
                box.operator("audioly.check_for_updates", text="Check for Updates")
            else:
                if props.update_available and not props.update_downloaded:
                    box.label(text=f"New Version: {props.latest_version}", icon='INFO')
                    box.label(text=props.latest_changelog)
                    box.operator("audioly.download_update", text="Download Update", icon='IMPORT')
                elif props.update_downloaded:
                    box.label(text="Update Downloaded! Please restart Blender.", icon='INFO')
                    box.operator("audioly.restart_blender", text="Restart Blender", icon='RECOVER_LAST')
                else:
                    box.label(text="No Updates Available", icon='CHECKMARK')
                    box.operator("audioly.check_for_updates", text="Check Again")
        layout.separator()
        row = layout.row(align=True)
        row.operator("audioly.save_preferences", icon='FILE_TICK', text="Save Preferences")
        row.operator("audioly.open_feedback", icon='HELP', text="Feedback")
        row.operator("audioly.open_youtube", icon='URL', text="Youtube Tutorials")


class AUDIO_OT_check_for_updates(Operator):
    bl_idname = "audioly.check_for_updates"
    bl_label  = "Check for Updates"
    def execute(self, context):
        prefs = context.preferences.addons[audioly_get_addon_key()].preferences
        if not prefs.enable_auto_update:
            self.report({'INFO'}, "Auto updates disabled.")
            return {'CANCELLED'}
        props = context.scene.audioly_updater_props
        try:
            with urllib.request.urlopen(AUDIOLY_UPDATE_JSON_URL) as response:
                data = response.read().decode('utf-8')
            info = json.loads(data)
            version_list   = info.get("version", [])
            latest_url     = info.get("url", "")
            latest_changes = info.get("changelog", "")
            if not isinstance(version_list, list) or len(version_list) < 3:
                raise ValueError("Invalid version in JSON.")
            latest_version_tuple = tuple(version_list)
            if audioly_compare_versions(AUDIOLY_CURRENT_VERSION, latest_version_tuple) < 0:
                props.update_checked     = True
                props.update_available   = True
                props.update_downloaded  = False
                props.latest_version     = ".".join(map(str, latest_version_tuple))
                props.latest_changelog   = latest_changes
                props.latest_url         = latest_url
                self.report({'INFO'}, f"New version {props.latest_version} available!")
            else:
                props.update_checked     = True
                props.update_available   = False
                props.update_downloaded  = False
                props.latest_version     = ""
                props.latest_changelog   = ""
                props.latest_url         = ""
                self.report({'INFO'}, "No updates available.")
        except Exception as e:
            props.update_checked     = True
            props.update_available   = False
            props.update_downloaded  = False
            self.report({'ERROR'}, f"Failed to check updates: {e}")
        return {'FINISHED'}

class AUDIO_OT_download_update(Operator):
    bl_idname = "audioly.download_update"
    bl_label  = "Download Update"
    def execute(self, context):
        prefs = context.preferences.addons[audioly_get_addon_key()].preferences
        if not prefs.enable_auto_update:
            self.report({'INFO'}, "Auto updates disabled.")
            return {'CANCELLED'}
        props = context.scene.audioly_updater_props
        download_url = props.latest_url
        if not download_url:
            self.report({'ERROR'}, "No download URL found.")
            return {'CANCELLED'}
        try:
            with urllib.request.urlopen(download_url) as response:
                new_code = response.read()
            current_addon_dir = os.path.dirname(os.path.abspath(__file__))
            init_path   = os.path.join(current_addon_dir, "__init__.py")
            backup_path = init_path + ".bak"
            if os.path.exists(backup_path):
                os.remove(backup_path)
            if os.path.exists(init_path):
                os.rename(init_path, backup_path)
            with open(init_path, "wb") as f:
                f.write(new_code)
            props.update_downloaded = True
            self.report({'INFO'}, "Update downloaded. Please restart Blender.")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to download update: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

class AUDIO_OT_restart_blender(Operator):
    bl_idname = "audioly.restart_blender"
    bl_label  = "Restart Blender"
    def execute(self, context):
        try:
            binary     = bpy.app.binary_path
            blend_file = bpy.data.filepath
            args = [binary]
            if blend_file:
                args.append(blend_file)
            if os.name == 'nt':
                DETACHED_PROCESS = 0x00000008
                subprocess.Popen(args, creationflags=DETACHED_PROCESS)
            else:
                subprocess.Popen(args)
            bpy.ops.wm.quit_blender()
        except Exception as e:
            self.report({'WARNING'}, f"Failed to automatically restart Blender: {e}")
        return {'FINISHED'}

@persistent
def audioly_load_post_handler(dummy):
    prefs = bpy.context.preferences.addons[audioly_get_addon_key()].preferences
    props = bpy.context.scene.audioly_updater_props
    props.update_checked     = False
    props.update_available   = False
    props.update_downloaded  = False
    props.latest_version     = ""
    props.latest_changelog   = ""
    props.latest_url         = ""
    if prefs.enable_auto_update:
        bpy.ops.audioly.check_for_updates()


# -------------------------------------------------------------------------
# Property Group
# -------------------------------------------------------------------------
class AudioShapeProperties(bpy.types.PropertyGroup):
    # Audio
    audio_path: StringProperty(
        name="Audio Path", subtype="FILE_PATH", update=audio_path_update
    )
    audio_sound: bpy.props.PointerProperty(type=bpy.types.Sound)
    import_to_sequencer: BoolProperty(
        default=True, name="Import to Sequencer"
    )
    frame_start: IntProperty(default=1, min=1, name="Frame Start")

    # Baked object handling
    baked_object: EnumProperty(
        items=lambda s, c: [
            (o.name, o.name, "") for o in c.scene.objects if o.get("_audio_shape_baked")
        ]
        or [("", "(none)", "")]
    )
    selected_preset: EnumProperty(
        items=[
            ("ALL", "All", ""),
            ("VOICE", "Voice", ""),
            ("PIANO", "Piano", ""),
            ("GUITAR", "Guitar", ""),
            ("DRUM", "Drum", ""),
            ("BASS", "Bass", ""),
        ],
        default="ALL",
        update=update_selected_preset,
        name="Active Preset",
    )

    # Bake preset toggles
    bake_preset_all: BoolProperty(default=True, name="All")
    bake_preset_voice: BoolProperty(default=True, name="Voice")
    bake_preset_piano: BoolProperty(default=True, name="Piano")
    bake_preset_guitar: BoolProperty(default=True, name="Guitar")
    bake_preset_drum: BoolProperty(default=True, name="Drum")
    bake_preset_bass: BoolProperty(default=True, name="Bass")

    # Static-shape and gains
    keyframe_threshold: FloatProperty(
        default=0.01, min=0, max=1, name="Keyframe Threshold"
    )
    strength: FloatProperty(
        default=0.5, min=0, max=1, update=update_static_keys, name="Strength"
    )
    low_gain: FloatProperty(
        default=1.0, min=0, max=2, update=update_static_keys, name="Low Gain"
    )
    mid_gain: FloatProperty(
        default=1.0, min=0, max=2, update=update_static_keys, name="Mid Gain"
    )
    high_gain: FloatProperty(
        default=1.0, min=0, max=2, update=update_static_keys, name="High Gain"
    )
    distort_ratio: FloatProperty(
        default=0.3, min=0, max=1, update=update_static_keys, name="Distort"
    )
    twist_angle: FloatProperty(
        default=0.5, min=0, max=math.pi, update=update_static_keys, name="Twist"
    )
    noise_strength: FloatProperty(
        default=0.2, min=0, max=1, update=update_static_keys, name="Noise"
    )
    randomizer: FloatProperty(
        default=0.0, min=0, max=1, update=update_static_keys, name="Randomizer"
    )

    # Drivers / voice clamp
    driver_strength: FloatProperty(
        name="Driver Strength", default=1.0, min=0.1, max=10.0
    )
    voice_threshold: FloatProperty(
        name="Voice Clamp", default=0.1, min=0, max=1.0
    )

    # Color & effects
    effect_type: EnumProperty(
        items=[
            ("NONE", "None", ""),
            ("MAGIC", "Magic", ""),
            ("WAVE", "Wave", ""),
            ("NOISE", "Noise", ""),
        ],
        default="NONE",
        update=update_colors_section,
        name="Effect Type",
    )

    # Modifiers / particles
    add_mod: EnumProperty(items=addable_items)
    mesh_expand: BoolProperty(default=True)
    curve_expand: BoolProperty(default=True)
    gp_expand: BoolProperty(default=True)
    use_particles: BoolProperty(
        default=False, update=update_particles, name="Enable Particles"
    )


# -------------------------------------------------------------------------
# UI Panel
# -------------------------------------------------------------------------
class AUDIO_PT_shape_panel(bpy.types.Panel):
    bl_label = "Audioly"
    bl_idname = "AUDIO_PT_shape_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Audioly"

    def draw(self, context):
        P = context.scene.audio_shape_props
        obj = context.object
        L = self.layout

        # Dependency notice
        if not FFMPEG_EXE:
            L.operator("audio_shape.install_deps", icon="ERROR")
            return

        # ------------------  Audio Input  ------------------
        box = L.box()
        box.label(text="Audio Input", icon="SOUND")
        box.prop(P, "audio_path", text="Audio Path")
        box.prop(P, "import_to_sequencer")
        box.prop(P, "frame_start")

        # ------------------  Bake / Link  -------------------
        row = L.row(align=True)
        row.label(text="Baked Objects")
        row.operator("audio_shape.prev_baked", text="", icon="TRIA_LEFT")
        row.prop(P, "baked_object", text="")
        row.operator("audio_shape.next_baked", text="", icon="TRIA_RIGHT")
        row.operator(
            "audio_shape.select_baked", text="Select", icon="RESTRICT_SELECT_OFF"
        )

        if obj and obj.type == "MESH":
            if obj.get("_audio_shape_baked"):
                L.operator("audio_shape.unbake", text="Unbake Audio", icon="LOOP_BACK")
            elif obj.get("_audio_shape_linked_to"):
                L.operator("audio_shape.unlink_baked", text="Unlink Object", icon="UNLINKED")
            else:
                r = L.row(align=True)
                r.operator(
                    "audio_shape.new_baked", text="Bake Mesh", icon="RENDER_ANIMATION"
                )
                if P.baked_object:
                    r.operator("audio_shape.link_baked", text="Link Object", icon="LINKED")
        elif obj and not obj.get("_audio_shape_baked") and P.baked_object:
            L.operator("audio_shape.link_baked", text="Link Object", icon="LINKED")

        # Active preset & driver strength
        if obj and (obj.get("_audio_shape_baked") or obj.get("_audio_shape_linked_to")):
            L.prop(P, "selected_preset", text="Active Preset")
            r2 = L.row(align=True)
            r2.prop(P, "driver_strength", text="Driver Strength", slider=True)
            has_drv = (
                context.scene.animation_data
                and any(
                    d.data_path == "audio_shape_props.driver_strength"
                    for d in context.scene.animation_data.drivers
                )
            )
            op = r2.operator(
                "audio_shape.toggle_strength_driver",
                icon=("LINKED" if has_drv else "UNLINKED"),
                text="",
            )
            op.prop_name = "driver_strength"

        # ------------------  Bake Presets  ------------------
        bp = L.box()
        bp.label(text="Bake Presets", icon="RNDCURVE")
        row1 = bp.row(align=True)
        row1.prop(P, "bake_preset_all", text="All", toggle=True)
        row1.prop(P, "bake_preset_voice", text="Voice", toggle=True)
        row1.prop(P, "bake_preset_piano", text="Piano", toggle=True)
        row2 = bp.row(align=True)
        row2.prop(P, "bake_preset_guitar", text="Guitar", toggle=True)
        row2.prop(P, "bake_preset_drum", text="Drum", toggle=True)
        row2.prop(P, "bake_preset_bass", text="Bass", toggle=True)
        bp.prop(P, "keyframe_threshold", text="Keyframe Threshold", slider=True)
        bp.prop(P, "voice_threshold", text="Voice Clamp", slider=True)

        # ------------------  Mesh Deform  -------------------
        if obj and obj.type == "MESH":
            md = L.box()
            md.label(text="Mesh Deform", icon="MOD_MESHDEFORM")
            md.prop(P, "strength", slider=True)
            md.prop(P, "distort_ratio", text="Distort", slider=True)
            md.prop(P, "noise_strength", text="Noise", slider=True)
            md.prop(P, "twist_angle", text="Twist", slider=True)
            md.prop(P, "randomizer", text="Randomizer", slider=True)
            gm = md.column(align=True)
            gm.label(text="Gains:")
            gm.prop(P, "low_gain", text="Low Gain", slider=True)
            gm.prop(P, "mid_gain", text="Mid Gain", slider=True)
            gm.prop(P, "high_gain", text="High Gain", slider=True)

        # ------------------  Color & Effects  ---------------
        if obj and obj.type == "MESH":
            ce = L.box()
            ce.label(text="Color & Effects", icon="COLOR")

            mat = obj.active_material
            if not mat or not mat.use_nodes:
                ce.operator("audio_shape.ensure_color_material", text="Setup")

            if mat and mat.use_nodes:
                nt = mat.node_tree
                ramp = next(
                    (
                        n
                        for n in nt.nodes
                        if n.type == "VALTORGB" and n.get("audio_node")
                    ),
                    None,
                )
                bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)

                if ramp and bsdf:
                    col = ce.column(align=True)
                    col.template_color_ramp(ramp, "color_ramp", expand=True)
                    adn = nt.animation_data

                    for idx, ele in enumerate(ramp.color_ramp.elements):
                        drv_path = f'nodes["{ramp.name}"].color_ramp.elements[{idx}].position'
                        has_drv = adn and adn.drivers and any(
                            d.data_path == drv_path for d in adn.drivers
                        )
                        row = col.row(align=True)
                        row.prop(ele, "color", text="")
                        op_reset = row.operator(
                            "audio_shape.reset_node_driver", icon="FILE_REFRESH", text=""
                        )
                        op_reset.node_name = ramp.name
                        op_reset.input_idx = idx
                        op_toggle = row.operator(
                            "audio_shape.toggle_node_driver",
                            icon=("LINKED" if has_drv else "UNLINKED"),
                            text="",
                        )
                        op_toggle.node_name = ramp.name
                        op_toggle.input_idx = idx

                    # Emission strength
                    if "Emission Strength" in bsdf.inputs:
                        idx_em = [
                            i for i, n in enumerate(bsdf.inputs) if n.name == "Emission Strength"
                        ][0]
                        row = ce.row(align=True)
                        row.prop(
                            bsdf.inputs["Emission Strength"],
                            "default_value",
                            text="Emission Strength",
                            slider=True,
                        )
                        has_drv = adn and adn.drivers and any(
                            d.data_path
                            == f'nodes["{bsdf.name}"].inputs[{idx_em}].default_value'
                            for d in adn.drivers
                        )
                        op_toggle = row.operator(
                            "audio_shape.toggle_node_driver",
                            icon=("LINKED" if has_drv else "UNLINKED"),
                            text="",
                        )
                        op_toggle.node_name = bsdf.name
                        op_toggle.input_idx = idx_em

                ce.prop(P, "effect_type", text="Effect Type")

                if P.effect_type != "NONE":
                    node = next((n for n in nt.nodes if n.get("audio_effect")), None)
                    if node:
                        adn = nt.animation_data
                        for idx, inp in enumerate(node.inputs):
                            if inp.type != "VALUE":
                                continue
                            path = f'nodes["{node.name}"].inputs[{idx}].default_value'
                            has_drv = adn and adn.drivers and any(
                                fc.data_path == path for fc in adn.drivers
                            )
                            row = ce.row(align=True)
                            row.prop(
                                inp,
                                "default_value",
                                text=inp.name,
                                slider=True,
                            )
                            op_reset = row.operator(
                                "audio_shape.reset_node_driver", icon="FILE_REFRESH", text=""
                            )
                            op_reset.node_name = node.name
                            op_reset.input_idx = idx
                            op_toggle = row.operator(
                                "audio_shape.toggle_node_driver",
                                icon=("LINKED" if has_drv else "UNLINKED"),
                                text="",
                            )
                            op_toggle.node_name = node.name
                            op_toggle.input_idx = idx

        # ------------------  Curve Controls  ----------------
        if obj and obj.type == "CURVE":
            cb = L.box()
            cb.label(text="Curve Controls", icon="CURVE_DATA")
            for prop, label in (
                ("extrude", "Extrude"),
                ("bevel_depth", "Depth"),
                ("bevel_factor_start", "Start"),
                ("bevel_factor_end", "End"),
            ):
                row = cb.row(align=True)
                op_reset = row.operator(
                    "audio_shape.reset_curve_prop", icon="FILE_REFRESH", text=""
                )
                op_reset.prop_name = prop
                row.prop(obj.data, prop, text=label, slider=True)
                has_drv = obj.data.animation_data and any(
                    fc.data_path == prop for fc in obj.data.animation_data.drivers
                )
                op_toggle = row.operator(
                    "audio_shape.toggle_curve_driver",
                    icon=("LINKED" if has_drv else "UNLINKED"),
                    text="",
                )
                op_toggle.prop_name = prop

        # ------------------ Beat-Reactive Modifiers ---------
        banned_ids = {
            "seed",
            "step",
            "spread",
            "mat_nr",
            "layer_pass",
            "layer_pass_filter",
            "material_pass",
            "material_pass_filter",
        }

        mb = L.box()
        mb.label(text="Beat Modifiers", icon="MODIFIER")
        adict = allowed_mod_dict(obj)
        if obj and adict:
            mc = mod_collection(obj)
            ad = obj.animation_data
            for m in mc:
                if m.type not in adict.values():
                    continue
                row = mb.row(align=True)
                up = row.operator(
                    "audio_shape.move_modifier", icon="TRIA_UP", emboss=False
                )
                up.mod_name, up.direction = m.name, -1
                dn = row.operator(
                    "audio_shape.move_modifier", icon="TRIA_DOWN", emboss=False
                )
                dn.mod_name, dn.direction = m.name, 1
                row.operator(
                    "audio_shape.toggle_mod_expand",
                    icon=("TRIA_DOWN" if m.show_expanded else "TRIA_RIGHT"),
                    emboss=False,
                ).mod_name = m.name
                row.label(text=m.name.replace("_", " ").title())
                row.operator("audio_shape.remove_audio_modifier", icon="TRASH").mod_name = m.name

                if m.show_expanded:
                    sub = mb.box()
                    for prop in m.bl_rna.properties:
                        if (
                            prop.is_readonly
                            or prop.identifier in {"rna_type", "name"}
                            or prop.type not in {"FLOAT", "INT"}
                            or prop.identifier in banned_ids
                        ):
                            continue
                        ident = prop.identifier
                        path = f'modifiers["{m.name}"].{ident}'
                        has_drv = ad and ad.drivers and any(
                            fc.data_path == path for fc in ad.drivers
                        )
                        r2 = sub.row(align=True)
                        op_reset = r2.operator(
                            "audio_shape.reset_driver_prop", icon="FILE_REFRESH", text=""
                        )
                        op_reset.mod_name = m.name
                        op_reset.prop_name = ident
                        r2.prop(m, ident, text=ident.replace("_", " ").title(), slider=True)
                        op_toggle = r2.operator(
                            "audio_shape.toggle_driver",
                            icon=("LINKED" if has_drv else "UNLINKED"),
                            text="",
                        )
                        op_toggle.mod_name = m.name
                        op_toggle.prop_name = ident
        else:
            mb.label(text="No compatible modifiers.")

        add = mb.row(align=True)
        add.prop(P, "add_mod", text="")
        if P.add_mod:
            op = add.operator("audio_shape.add_audio_modifier", text="Add", icon="ADD")
            op.mod_code = P.add_mod

        # ------------------  Particles  ---------------------
        if obj and obj.type == "MESH" and obj.get("_audio_shape_linked_to"):
            psb = L.box()
            psb.label(text="Particles", icon="PARTICLES")
            psb.prop(P, "use_particles", text="Enable Particles", toggle=True)

            if P.use_particles and obj.particle_systems:
                psys = obj.particle_systems.get("AudioParticles")
                if psys:
                    s = psys.settings
                    row = psb.row(align=True)
                    op = row.operator(
                        "audio_shape.reset_particle_prop", icon="FILE_REFRESH", text=""
                    )
                    op.prop_name = "count"
                    row.prop(s, "count", text="Number", slider=True)
                    has_drv = s.animation_data and s.animation_data.drivers and any(
                        d.data_path == "count" for d in s.animation_data.drivers
                    )
                    lk = row.operator(
                        "audio_shape.toggle_particle_driver",
                        icon=("LINKED" if has_drv else "UNLINKED"),
                        text="",
                    )
                    lk.prop_name = "count"

                    psb.prop(s, "frame_start", text="Frame Start")
                    psb.prop(s, "frame_end", text="Frame End")
                    psb.prop(s, "lifetime", text="Lifetime")
                    psb.prop(s, "lifetime_randomness", text="Lifetime Rand")

                    def p_row(label, attr):
                        rw = psb.row(align=True)
                        op2 = rw.operator(
                            "audio_shape.reset_particle_prop", icon="FILE_REFRESH", text=""
                        )
                        op2.prop_name = attr
                        rw.prop(s, attr, text=label, slider=True)
                        hd = (
                            s.animation_data
                            and s.animation_data.drivers
                            and any(d.data_path == attr for d in s.animation_data.drivers)
                        )
                        lk2 = rw.operator(
                            "audio_shape.toggle_particle_driver",
                            icon=("LINKED" if hd else "UNLINKED"),
                            text="",
                        )
                        lk2.prop_name = attr

                    p_row("Normal Velocity", "normal_factor")
                    # Removed "Object Velocity" row as requested
                    # p_row("Object Velocity", "object_align_factor")
                    p_row("Scale", "particle_size")
                    p_row("Scale Randomness", "size_random")

                    rowg = psb.row(align=True)
                    opg = rowg.operator(
                        "audio_shape.reset_particle_prop", icon="FILE_REFRESH", text=""
                    )
                    opg.prop_name = "effector_weights.gravity"
                    rowg.prop(
                        s.effector_weights, "gravity", text="Gravity", slider=True
                    )
                    hgd = (
                        s.animation_data
                        and s.animation_data.drivers
                        and any(
                            d.data_path == "effector_weights.gravity"
                            for d in s.animation_data.drivers
                        )
                    )
                    lk3 = rowg.operator(
                        "audio_shape.toggle_particle_driver",
                        icon=("LINKED" if hgd else "UNLINKED"),
                        text="",
                    )
                    lk3.prop_name = "effector_weights.gravity"


# -------------------------------------------------------------------------
# Registration
# -------------------------------------------------------------------------
# Insert new operator classes for audio modifiers

class AUDIO_OT_add_audio_modifier(bpy.types.Operator):
    """Add a new audio‐reactive modifier to the active object"""
    bl_idname = "audio_shape.add_audio_modifier"
    bl_label  = "Add Modifier"

    mod_code: StringProperty()

    def execute(self, context):
        obj = context.object
        if not obj:
            return {'CANCELLED'}
        mc = mod_collection(obj)
        if self.mod_code not in {m.type for m in mc}:
            mod = mc.new(name=self.mod_code, type=self.mod_code)
            mod.show_expanded = True
        return {'FINISHED'}

class AUDIO_OT_remove_audio_modifier(bpy.types.Operator):
    """Remove an audio‐reactive modifier from the active object"""
    bl_idname = "audio_shape.remove_audio_modifier"
    bl_label  = "Remove Modifier"

    mod_name: StringProperty()

    def execute(self, context):
        obj = context.object
        if not obj:
            return {'CANCELLED'}
        mc = mod_collection(obj)
        m = mc.get(self.mod_name)
        if m:
            mc.remove(m)
        return {'FINISHED'}


classes = (
    AudioShapeProperties,
    AUDIO_OT_install_deps,
    AUDIO_OT_ensure_color_material,
    AUDIO_OT_move_modifier,
    AUDIO_OT_reset_driver_prop,
    AUDIO_OT_reset_node_driver,
    AUDIO_OT_toggle_node_driver,
    AUDIO_OT_toggle_driver,
    AUDIO_OT_toggle_mod_expand,
    AUDIO_OT_reset_curve_prop,
    AUDIO_OT_toggle_curve_driver,
    AUDIO_OT_generate_shape_keys,
    AUDIO_OT_unbake,
    AUDIO_OT_prev_baked,
    AUDIO_OT_next_baked,
    AUDIO_OT_select_baked,
    AUDIO_OT_new_baked,
    AUDIO_OT_link_baked,
    AUDIO_OT_unlink_baked,
    AUDIO_OT_add_audio_modifier,
    AUDIO_OT_remove_audio_modifier,
    AUDIO_OT_reset_particle_prop,
    AUDIO_OT_toggle_particle_driver,
    AUDIO_OT_toggle_strength_driver,
    # Updater classes
    AudiolyAddonUpdaterProperties,
    AudiolyAddonPreferences,
    AUDIO_OT_check_for_updates,
    AUDIO_OT_download_update,
    AUDIO_OT_restart_blender,
    AUDIO_PT_shape_panel,
    AudiolySavePreferencesOperator,
    AudiolyOpenFeedbackOperator,
    AudiolyOpenYouTubeOperator,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.audio_shape_props = bpy.props.PointerProperty(
        type=AudioShapeProperties
    )
    # Updater property group
    bpy.types.Scene.audioly_updater_props = bpy.props.PointerProperty(
        type=AudiolyAddonUpdaterProperties
    )
    # Add handler for auto-update
    if audioly_load_post_handler not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(audioly_load_post_handler)

def unregister():
    del bpy.types.Scene.audio_shape_props
    # Updater property group
    del bpy.types.Scene.audioly_updater_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    # Remove handler
    if audioly_load_post_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(audioly_load_post_handler)

if __name__ == "__main__":
    register()
