bl_info = {
    "name": "CryEngine 1 Asset Importer/Exporter (Far Cry)",
    "author": "Ported from Takaro CryImporter for 3ds Max",
    "version": (1, 4, 31),
    "blender": (4, 0, 0),
    "location": "File > Import/Export > CryEngine",
    "description": "Import/Export CryEngine 1 / Far Cry asset, geometry, skeleton, and animation files",
    "category": "Import-Export",
}

import bpy
import os
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import AddonPreferences, Panel, PropertyGroup
from bpy_extras.io_utils import ImportHelper, ExportHelper
from . import cry_chunk_reader
from . import cry_asset_builder
from . import cry_exporter


ADDON_BUILD_TAG = "v244-next-fixes-readme"


PLAYBACK_MODE_ITEMS = [
    ('MAXSPACE', "Maxspace Preview", "Bake mesh preview directly from Cry max-space pose evaluation"),
    ('PROXY', "Proxy-First Preview", "Drive animation through Cry-style proxy transforms and bake mesh preview"),
    ('ARMATURE', "Armature Keys", "Apply animation directly to Blender pose bones"),
    ('RAWMAX', "Raw Max Keys", "Apply animation from raw Max-style controller evaluation without the current Cry evaluator path"),
]


# ── CryEngine Material Properties ─────────────────────────────────────────────

# Common Far Cry shaders
SHADER_ITEMS = [
    ('custom',                       "Custom...",                    "Enter shader name manually"),
    ('TemplModelCommon',             "TemplModelCommon",             "Standard model — no bump"),
    ('TemplBumpDiffuse',             "TemplBumpDiffuse",             "Diffuse + bump/normal map"),
    ('TemplBumpSpec',                "TemplBumpSpec",                "Diffuse + bump + specular"),
    ('TemplBumpSpec_GlossAlpha',     "TemplBumpSpec_GlossAlpha",     "Diffuse + bump + spec + gloss in alpha"),
    ('TemplBumpSpec_HP_GlossAlpha',  "TemplBumpSpec_HP_GlossAlpha",  "Hi-poly bump + spec + gloss in alpha"),
    ('Phong',                        "Phong",                        "Simple Phong shading"),
    ('NoDraw',                       "NoDraw",                       "Invisible — collision/physics only"),
    ('Glass',                        "Glass",                        "Glass / transparent surface"),
    ('Vegetation',                   "Vegetation",                   "Vegetation / foliage"),
    ('Terrain',                      "Terrain",                      "Terrain layer blend"),
    ('Metal',                        "Metal",                        "Metal surface shader"),
]

SURFACE_ITEMS = [
    ('mat_default',      "mat_default",      "Default — general purpose"),
    ('mat_metal',        "mat_metal",        "Metal — generic"),
    ('mat_metal_plate',  "mat_metal_plate",  "Metal plate"),
    ('mat_metal_pipe',   "mat_metal_pipe",   "Metal pipe"),
    ('mat_concrete',     "mat_concrete",     "Concrete"),
    ('mat_rock',         "mat_rock",         "Rock / stone"),
    ('mat_wood',         "mat_wood",         "Wood"),
    ('mat_grass',        "mat_grass",        "Grass / ground"),
    ('mat_sand',         "mat_sand",         "Sand / dirt"),
    ('mat_water',        "mat_water",        "Water"),
    ('mat_glass',        "mat_glass",        "Glass"),
    ('mat_flesh',        "mat_flesh",        "Flesh / organic (characters)"),
    ('mat_head',         "mat_head",         "Head (characters)"),
    ('mat_helmet',       "mat_helmet",       "Helmet / hard hat"),
    ('mat_armor',        "mat_armor",        "Armor / hard protection"),
    ('mat_arm',          "mat_arm",          "Arm (characters)"),
    ('mat_leg',          "mat_leg",          "Leg (characters)"),
    ('mat_cloth',        "mat_cloth",        "Cloth / fabric"),
    ('mat_rubber',       "mat_rubber",       "Rubber"),
]


class CryMaterialProperties(PropertyGroup):
    shader_preset: EnumProperty(
        name="Shader",
        description="CryEngine shader preset",
        items=SHADER_ITEMS,
        default='custom',
        update=lambda self, ctx: _update_cgf_full_name(self, ctx),
    )
    shader_custom: StringProperty(
        name="Custom Shader",
        description="Custom shader name (used when Shader = Custom)",
        default="",
        update=lambda self, ctx: _update_cgf_full_name(self, ctx),
    )
    surface: EnumProperty(
        name="Surface",
        description="Physics surface type",
        items=SURFACE_ITEMS,
        default='mat_default',
        update=lambda self, ctx: _update_cgf_full_name(self, ctx),
    )


def _update_cgf_full_name(self, context):
    """Rebuild cgf_full_name whenever shader or surface changes."""
    mat = getattr(context, "material", None)
    if mat is None:
        mat = getattr(self, "id_data", None)
    if mat is None or not isinstance(mat, bpy.types.Material):
        return
    cry = mat.cry
    shader = cry.shader_custom if cry.shader_preset == 'custom' else cry.shader_preset
    base_name = mat.name.split('(')[0].split('/')[0]
    full = base_name
    if shader:
        full += f"({shader})"
    full += f"/{cry.surface}"
    mat['cgf_full_name']    = full
    mat['cgf_shader_name']  = shader
    mat['cgf_surface_name'] = cry.surface


class VIEW3D_PT_cryengine(Panel):
    bl_label       = "CryEngine 1"
    bl_idname      = "VIEW3D_PT_cryengine"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = 'CryEngine'

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        # Active material from selected mesh
        mat = None
        if obj and obj.type == 'MESH' and obj.active_material:
            mat = obj.active_material

        if mat is None:
            layout.label(text="Select a mesh object", icon='INFO')
            return

        cry = mat.cry

        box = layout.box()
        box.label(text=f"Material: {mat.name}", icon='MATERIAL')

        # Show current full CGF name
        full = mat.get('cgf_full_name', '')
        if full:
            box.label(text=full, icon='INFO')

        col = layout.column()
        col.label(text="Shader:")
        col.prop(cry, "shader_preset", text="")
        if cry.shader_preset == 'custom':
            col.prop(cry, "shader_custom", text="Name")

        col.separator()
        col.label(text="Surface / Physics:")
        col.prop(cry, "surface", text="")


# ── Addon Preferences ─────────────────────────────────────────────────────────

class CGFAddonPreferences(AddonPreferences):
    bl_idname = __name__

    game_root_path: StringProperty(
        name="Game Data/Textures Root",
        description="Root folder used to resolve CryEngine texture paths (Objects/Textures/Levels). "
                    "Point this to your Far Cry game data root.",
        default="",
        subtype='DIR_PATH',
    )
    skip_collision_geometry: BoolProperty(
        name="Skip Collision-Like Geometry",
        description="Globally skip collision-like helper geometry such as NoDraw or obstruct meshes during geometry import",
        default=False,
    )
    enable_scene_setup: BoolProperty(
        name="Enable Full Scene Setup",
        description="Single master switch for asset root, node transforms, helper/controller targets, and producer cameras",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Far Cry / CryEngine 1 Settings:", icon='SETTINGS')
        layout.prop(self, "game_root_path")
        layout.prop(self, "skip_collision_geometry")
        layout.prop(self, "enable_scene_setup")
        if not self.game_root_path:
            layout.label(text="Set this to your Far Cry data root (e.g. C:\\FarCry)",
                         icon='ERROR')


def get_game_root_path():
    """Get the globally configured game root path from addon preferences."""
    prefs = bpy.context.preferences.addons.get(__name__)
    if prefs:
        return prefs.preferences.game_root_path
    return ""


def get_skip_collision_geometry():
    prefs = bpy.context.preferences.addons.get(__name__)
    if prefs:
        return bool(getattr(prefs.preferences, "skip_collision_geometry", False))
    return False


def _get_pref_bool(name, default=False):
    prefs = bpy.context.preferences.addons.get(__name__)
    if prefs:
        return bool(getattr(prefs.preferences, name, default))
    return bool(default)


def _scene_meshes(context, selected_only=False):
    source = context.selected_objects if selected_only else context.view_layer.objects
    return [o for o in source if o.type == 'MESH' and not o.hide_get()]


def _find_export_armature(context, meshes=None):
    meshes = meshes or []
    active = context.active_object
    if active and active.type == 'ARMATURE' and not active.hide_get():
        return active
    for obj in context.view_layer.objects:
        if obj.type == 'ARMATURE' and not obj.hide_get():
            return obj
    for obj in meshes:
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object:
                return mod.object
    return None


def _has_skinned_meshes(meshes, arm_obj):
    if not arm_obj:
        return False
    for obj in meshes:
        if any(mod.type == 'ARMATURE' and mod.object == arm_obj for mod in obj.modifiers):
            return True
    return False


def _actions_for_armature(arm_obj):
    if arm_obj is None:
        return []
    result = []
    for action in bpy.data.actions:
        fcurves = getattr(action, 'fcurves', None)
        if not fcurves:
            continue
        if any(fc.data_path.startswith('pose.bones[') for fc in fcurves):
            result.append(action)
    return result


# ── CGF / CGA geometry importer ───────────────────────────────────────────────

class ImportCGF(bpy.types.Operator, ImportHelper):
    """Import CryEngine 1 CGF/CGA geometry file (Far Cry)"""
    bl_idname  = "import_scene.cgf"
    bl_label   = "Import CGF/CGA"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".cgf"
    filter_glob: StringProperty(default="*.cgf;*.cga;*.bld", options={'HIDDEN'})

    import_materials: BoolProperty(name="Import Materials",
        description="Create Principled BSDF materials", default=True)
    import_normals: BoolProperty(name="Import Normals",
        description="Use normals from file", default=True)
    import_uvs: BoolProperty(name="Import UVs",
        description="Import texture coordinates", default=True)
    import_skeleton: BoolProperty(name="Import Skeleton",
        description="Build armature from bone chunks", default=True)
    import_weights: BoolProperty(name="Import Vertex Weights",
        description="Assign bone weights for skinned meshes", default=True)
    game_root_override: StringProperty(
        name="Override Data/Textures Root",
        description="Override the global Game Data/Textures Root for this import only. "
                    "Leave empty to use the path from Addon Preferences.",
        default="",
        subtype='DIR_PATH',
    )

    def execute(self, context):
        # Use per-import override if set, otherwise fall back to global prefs
        game_root = self.game_root_override.strip() or get_game_root_path()
        full_scene_setup = _get_pref_bool("enable_scene_setup", True)
        result = cry_asset_builder.load(
            self, context,
            filepath         = self.filepath,
            import_materials = self.import_materials,
            import_normals   = self.import_normals,
            import_uvs       = self.import_uvs,
            import_skeleton  = self.import_skeleton,
            import_weights   = self.import_weights,
            game_root_path   = game_root,
            skip_collision_geometry = get_skip_collision_geometry(),
            create_asset_root_empty = full_scene_setup,
            apply_armature_node_transform = True,
            apply_mesh_node_transform = True,
            preserve_mesh_world_on_armature_parent = True,
            create_helper_nodes = full_scene_setup,
            create_controller_targets = full_scene_setup,
            create_producer_cameras = full_scene_setup,
        )
        if result == {'FINISHED'}:
            for obj in context.scene.objects:
                if obj.type == 'ARMATURE' and not obj.get('cgf_source_path'):
                    obj['cgf_source_path'] = self.filepath
                    _store_ctrl_ids(obj)
        return result

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Geometry", icon='MESH_DATA')
        box.prop(self, "import_uvs")
        box.prop(self, "import_normals")
        box.prop(self, "import_materials")
        box = layout.box()
        box.label(text="Skinning", icon='ARMATURE_DATA')
        box.prop(self, "import_skeleton")
        box.prop(self, "import_weights")
        box = layout.box()
        box.label(text="Textures", icon='TEXTURE')
        global_root = get_game_root_path()
        if global_root:
            box.label(text=f"Global data root: {global_root}", icon='CHECKMARK')
        else:
            box.label(text="No global data root set (see Addon Preferences)", icon='ERROR')
        box.prop(self, "game_root_override")


def _store_ctrl_ids(arm_obj):
    source_path = arm_obj.get('cgf_source_path', '')
    if not source_path:
        return
    try:
        reader = cry_chunk_reader.ChunkReader()
        archive = reader.read_file(source_path)
        if archive.bone_anim_chunks:
            name_list = archive.bone_name_list_chunks[0].name_list \
                        if archive.bone_name_list_chunks else []
            for bone in archive.bone_anim_chunks[0].bones:
                bid   = bone.bone_id
                bname = name_list[bid] if bid < len(name_list) else f"Bone_{bid}"
                if bname in arm_obj.pose.bones:
                    arm_obj.pose.bones[bname]['cry_ctrl_id'] = bone.ctrl_id
    except Exception as e:
        print(f"[CGF] Could not store ctrl_ids: {e}")


# ── CAF animation importer ────────────────────────────────────────────────────

class ImportCAF(bpy.types.Operator, ImportHelper):
    """Import CryEngine 1 CAF animation file onto the active armature"""
    bl_idname  = "import_scene.caf"
    bl_label   = "Import CAF Animation"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".caf"
    filter_glob: StringProperty(default="*.caf", options={'HIDDEN'})

    append: BoolProperty(name="Append to Timeline",
        description="Add after existing animation range", default=True)
    debug_caf: BoolProperty(
        name="Debug CAF (log transforms)",
        description="Print detailed CAF transform diagnostics to Blender console",
        default=False,
    )
    playback_mode: EnumProperty(
        name="Animation Playback Mode",
        description="Choose how Cry bone animation is evaluated inside Blender",
        items=PLAYBACK_MODE_ITEMS,
        default='MAXSPACE',
    )

    def execute(self, context):
        return cry_asset_builder.load_caf(
            self, context, self.filepath, self.append, self.debug_caf, self.playback_mode
        )

    def draw(self, context):
        self.layout.prop(self, "append")
        self.layout.prop(self, "debug_caf")
        self.layout.prop(self, "playback_mode")


class ImportANM(bpy.types.Operator, ImportHelper):
    """Import CryEngine 1 ANM animation file onto the active armature"""
    bl_idname  = "import_scene.anm"
    bl_label   = "Import ANM Animation"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".anm"
    filter_glob: StringProperty(default="*.anm", options={'HIDDEN'})

    append: BoolProperty(name="Append to Timeline",
        description="Add after existing animation range", default=True)
    debug_caf: BoolProperty(
        name="Debug ANM (log transforms)",
        description="Print detailed ANM/CAF transform diagnostics to Blender console",
        default=False,
    )
    playback_mode: EnumProperty(
        name="Animation Playback Mode",
        description="Choose how Cry bone animation is evaluated inside Blender",
        items=PLAYBACK_MODE_ITEMS,
        default='MAXSPACE',
    )

    def execute(self, context):
        # CE1 ANM is treated here as the same animation container class as CAF.
        return cry_asset_builder.load_caf(
            self, context, self.filepath, self.append, self.debug_caf, self.playback_mode
        )

    def draw(self, context):
        self.layout.prop(self, "append")
        self.layout.prop(self, "debug_caf")
        self.layout.prop(self, "playback_mode")


# ── CAL animation list importer ───────────────────────────────────────────────

class ImportCAL(bpy.types.Operator, ImportHelper):
    """Import all animations from a CryEngine 1 CAL file"""
    bl_idname  = "import_scene.cal"
    bl_label   = "Import CAL Animation List"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".cal"
    filter_glob: StringProperty(default="*.cal", options={'HIDDEN'})
    debug_caf: BoolProperty(
        name="Debug CAL/CAF (log transforms)",
        description="Print detailed CAF transform diagnostics for animations loaded from CAL",
        default=False,
    )
    playback_mode: EnumProperty(
        name="Animation Playback Mode",
        description="Choose how Cry bone animation is evaluated inside Blender",
        items=PLAYBACK_MODE_ITEMS,
        default='MAXSPACE',
    )

    def execute(self, context):
        return cry_asset_builder.load_cal(self, context, self.filepath, self.debug_caf, self.playback_mode)

    def draw(self, context):
        self.layout.prop(self, "debug_caf")
        self.layout.prop(self, "playback_mode")


# ── CGF exporter ──────────────────────────────────────────────────────────────

class ExportCGF(bpy.types.Operator, ExportHelper):
    """Export to CryEngine 1 CGF geometry file (Far Cry)"""
    bl_idname  = "export_scene.cgf"
    bl_label   = "Export CGF"
    bl_options = {'PRESET'}

    filename_ext = ".cgf"
    filter_glob: StringProperty(default="*.cgf", options={'HIDDEN'})

    export_materials: BoolProperty(name="Export Materials",
        description="Write material chunks", default=True)
    export_skeleton: BoolProperty(name="Export Skeleton",
        description="Write bone chunks from active armature", default=True)
    export_weights: BoolProperty(name="Export Vertex Weights",
        description="Write physique (bone weights)", default=True)
    selected_only: BoolProperty(name="Selected Only",
        description="Export only selected mesh objects", default=False)

    def execute(self, context):
        return cry_exporter.export_cgf_scene(
            self, context, self.filepath,
            export_materials = self.export_materials,
            export_skeleton  = self.export_skeleton,
            export_weights   = self.export_weights,
            selected_only    = self.selected_only,
        )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Geometry", icon='MESH_DATA')
        box.prop(self, "selected_only")
        box.prop(self, "export_materials")
        box = layout.box()
        box.label(text="Skinning", icon='ARMATURE_DATA')
        box.prop(self, "export_skeleton")
        box.prop(self, "export_weights")


class ExportCGA(bpy.types.Operator, ExportHelper):
    """Export to CryEngine 1 CGA animated geometry file (Far Cry)"""
    bl_idname  = "export_scene.cga"
    bl_label   = "Export CGA"
    bl_options = {'PRESET'}

    filename_ext = ".cga"
    filter_glob: StringProperty(default="*.cga", options={'HIDDEN'})

    export_materials: BoolProperty(name="Export Materials",
        description="Write material chunks", default=True)
    export_skeleton: BoolProperty(name="Export Skeleton",
        description="Write bone chunks from active armature", default=True)
    export_weights: BoolProperty(name="Export Vertex Weights",
        description="Write physique (bone weights)", default=True)
    selected_only: BoolProperty(name="Selected Only",
        description="Export only selected mesh objects", default=False)

    def execute(self, context):
        return cry_exporter.export_cgf_scene(
            self, context, self.filepath,
            export_materials = self.export_materials,
            export_skeleton  = self.export_skeleton,
            export_weights   = self.export_weights,
            selected_only    = self.selected_only,
        )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Geometry", icon='MESH_DATA')
        box.prop(self, "selected_only")
        box.prop(self, "export_materials")
        box = layout.box()
        box.label(text="Skinning", icon='ARMATURE_DATA')
        box.prop(self, "export_skeleton")
        box.prop(self, "export_weights")


class ExportBLD(bpy.types.Operator, ExportHelper):
    """Export to CryEngine 1 BLD building geometry file (Far Cry)"""
    bl_idname  = "export_scene.bld"
    bl_label   = "Export BLD"
    bl_options = {'PRESET'}

    filename_ext = ".bld"
    filter_glob: StringProperty(default="*.bld", options={'HIDDEN'})

    export_materials: BoolProperty(name="Export Materials",
        description="Write material chunks", default=True)
    export_skeleton: BoolProperty(name="Export Skeleton",
        description="Write bone chunks from active armature", default=False)
    export_weights: BoolProperty(name="Export Vertex Weights",
        description="Write physique (bone weights)", default=False)
    selected_only: BoolProperty(name="Selected Only",
        description="Export only selected mesh objects", default=False)

    def execute(self, context):
        return cry_exporter.export_cgf_scene(
            self, context, self.filepath,
            export_materials = self.export_materials,
            export_skeleton  = self.export_skeleton,
            export_weights   = self.export_weights,
            selected_only    = self.selected_only,
        )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Geometry", icon='MESH_DATA')
        box.prop(self, "selected_only")
        box.prop(self, "export_materials")
        box = layout.box()
        box.label(text="Skinning", icon='ARMATURE_DATA')
        box.prop(self, "export_skeleton")
        box.prop(self, "export_weights")


class ExportCryAuto(bpy.types.Operator, ExportHelper):
    """Auto export CryEngine assets based on scene contents"""
    bl_idname  = "export_scene.cry_auto"
    bl_label   = "Auto Export CryEngine Asset"
    bl_options = {'PRESET'}

    filename_ext = ".cgf"
    filter_glob: StringProperty(default="*.cgf;*.cga;*.bld;*.caf;*.cal;*.anm", options={'HIDDEN'})

    export_materials: BoolProperty(name="Export Materials",
        description="Write material chunks", default=True)
    selected_only: BoolProperty(name="Selected Only",
        description="Export only selected mesh objects", default=False)
    export_animation_set: BoolProperty(name="Export CAL/CAF When Present",
        description="If the scene has armature actions, also write CAL+CAF next to geometry", default=True)
    prefer_cga_for_skinned: BoolProperty(name="Use CGA For Skinned Meshes",
        description="Skinned geometry is exported as .cga instead of .cgf", default=True)

    def execute(self, context):
        meshes = _scene_meshes(context, self.selected_only)
        if not meshes:
            self.report({'ERROR'}, "No visible mesh objects found")
            return {'CANCELLED'}

        arm_obj = _find_export_armature(context, meshes)
        has_skinned = _has_skinned_meshes(meshes, arm_obj)
        actions = _actions_for_armature(arm_obj)

        base_dir = os.path.dirname(self.filepath)
        base_name = os.path.splitext(os.path.basename(self.filepath))[0]

        geom_ext = ".cga" if (has_skinned and self.prefer_cga_for_skinned) else ".cgf"
        geom_path = os.path.join(base_dir, base_name + geom_ext)

        active_before = context.view_layer.objects.active
        selected_before = list(context.selected_objects)

        if arm_obj:
            try:
                context.view_layer.objects.active = arm_obj
            except Exception:
                pass

        result = cry_exporter.export_cgf_scene(
            self, context, geom_path,
            export_materials=self.export_materials,
            export_skeleton=bool(arm_obj),
            export_weights=has_skinned,
            selected_only=self.selected_only,
        )
        if result != {'FINISHED'}:
            return result

        exported = [os.path.basename(geom_path)]

        if self.export_animation_set and arm_obj and actions:
            cal_path = os.path.join(base_dir, base_name + ".cal")
            result = cry_exporter.export_cal(self, context, cal_path)
            if result == {'FINISHED'}:
                exported.append(os.path.basename(cal_path))

        try:
            context.view_layer.objects.active = active_before
            for obj in context.view_layer.objects:
                obj.select_set(False)
            for obj in selected_before:
                if obj.name in context.view_layer.objects:
                    obj.select_set(True)
        except Exception:
            pass

        self.report({'INFO'}, "Auto exported: " + ", ".join(exported))
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Detection", icon='FILE_TICK')
        box.prop(self, "selected_only")
        box.prop(self, "prefer_cga_for_skinned")
        box.prop(self, "export_animation_set")
        box = layout.box()
        box.label(text="Geometry", icon='MESH_DATA')
        box.prop(self, "export_materials")


# ── CAF exporter ──────────────────────────────────────────────────────────────

class ExportCAF(bpy.types.Operator, ExportHelper):
    """Export active action to CryEngine 1 CAF animation file"""
    bl_idname  = "export_scene.caf"
    bl_label   = "Export CAF Animation"
    bl_options = {'PRESET'}

    filename_ext = ".caf"
    filter_glob: StringProperty(default="*.caf", options={'HIDDEN'})
    debug_export: BoolProperty(
        name="Debug Export CAF",
        description="Log action range and per-bone exported key counts to the Blender console",
        default=False,
    )

    def execute(self, context):
        return cry_exporter.export_caf(self, context, self.filepath, debug_export=self.debug_export)


class ExportANM(bpy.types.Operator, ExportHelper):
    """Export active action to CryEngine 1 ANM animation file"""
    bl_idname  = "export_scene.anm"
    bl_label   = "Export ANM Animation"
    bl_options = {'PRESET'}

    filename_ext = ".anm"
    filter_glob: StringProperty(default="*.anm", options={'HIDDEN'})
    debug_export: BoolProperty(
        name="Debug Export ANM",
        description="Log action range and per-bone exported key counts to the Blender console",
        default=False,
    )

    def execute(self, context):
        # Current backend writes the generic CE1 animation container used by CAF/ANM.
        return cry_exporter.export_caf(self, context, self.filepath, debug_export=self.debug_export)


# ── CAL exporter ──────────────────────────────────────────────────────────────

class ExportCAL(bpy.types.Operator, ExportHelper):
    """Export all actions to CAF files and write a CAL list"""
    bl_idname  = "export_scene.cal"
    bl_label   = "Export CAL Animation List"
    bl_options = {'PRESET'}

    filename_ext = ".cal"
    filter_glob: StringProperty(default="*.cal", options={'HIDDEN'})

    def execute(self, context):
        return cry_exporter.export_cal(self, context, self.filepath)


# ── Menu entries ──────────────────────────────────────────────────────────────

def menu_import(self, context):
    self.layout.operator(ImportCGF.bl_idname, text="CryEngine Geometry (.cgf, .cga, .bld)")
    self.layout.operator(ImportCAF.bl_idname, text="CryEngine Animation (.caf)")
    self.layout.operator(ImportANM.bl_idname, text="CryEngine Animation (.anm)")
    self.layout.operator(ImportCAL.bl_idname, text="CryEngine Animation List (.cal)")


def menu_export(self, context):
    self.layout.operator(ExportCryAuto.bl_idname, text="CryEngine Auto Export")
    self.layout.operator(ExportCGF.bl_idname, text="CryEngine Geometry (.cgf)")
    self.layout.operator(ExportCGA.bl_idname, text="CryEngine Animated Geometry (.cga)")
    self.layout.operator(ExportBLD.bl_idname, text="CryEngine Building (.bld)")
    self.layout.operator(ExportCAF.bl_idname, text="CryEngine Animation (.caf)")
    self.layout.operator(ExportANM.bl_idname, text="CryEngine Animation (.anm)")
    self.layout.operator(ExportCAL.bl_idname, text="CryEngine Animation List (.cal)")


def register():
    print(f"[CGF] Addon build: {ADDON_BUILD_TAG}")
    bpy.utils.register_class(CGFAddonPreferences)
    bpy.utils.register_class(CryMaterialProperties)
    bpy.utils.register_class(VIEW3D_PT_cryengine)
    bpy.utils.register_class(ImportCGF)
    bpy.utils.register_class(ImportCAF)
    bpy.utils.register_class(ImportANM)
    bpy.utils.register_class(ImportCAL)
    bpy.utils.register_class(ExportCGF)
    bpy.utils.register_class(ExportCGA)
    bpy.utils.register_class(ExportBLD)
    bpy.utils.register_class(ExportCryAuto)
    bpy.utils.register_class(ExportCAF)
    bpy.utils.register_class(ExportANM)
    bpy.utils.register_class(ExportCAL)
    bpy.types.Material.cry = bpy.props.PointerProperty(type=CryMaterialProperties)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    bpy.utils.unregister_class(CGFAddonPreferences)
    bpy.utils.unregister_class(CryMaterialProperties)
    bpy.utils.unregister_class(VIEW3D_PT_cryengine)
    bpy.utils.unregister_class(ImportCGF)
    bpy.utils.unregister_class(ImportCAF)
    bpy.utils.unregister_class(ImportANM)
    bpy.utils.unregister_class(ImportCAL)
    bpy.utils.unregister_class(ExportCGF)
    bpy.utils.unregister_class(ExportCGA)
    bpy.utils.unregister_class(ExportBLD)
    bpy.utils.unregister_class(ExportCryAuto)
    bpy.utils.unregister_class(ExportCAF)
    bpy.utils.unregister_class(ExportANM)
    bpy.utils.unregister_class(ExportCAL)
    del bpy.types.Material.cry
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)


if __name__ == "__main__":
    register()
