bl_info = {
    "name": "CryEngine 1 CGF Importer/Exporter (Far Cry)",
    "author": "Ported from Takaro CryImporter for 3ds Max",
    "version": (1, 2, 0),
    "blender": (4, 0, 0),
    "location": "File > Import/Export > CryEngine",
    "description": "Import/Export CryEngine 1 / Far Cry geometry and animation files",
    "category": "Import-Export",
}

import bpy
import os
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import AddonPreferences, Panel, PropertyGroup
from bpy_extras.io_utils import ImportHelper, ExportHelper
from . import cgf_reader
from . import cgf_builder
from . import cgf_exporter


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
    mat = context.material
    if mat is None:
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
        name="Game Root Path",
        description="Root folder of Far Cry installation (where Objects/, Textures/ etc. are). "
                    "Used globally for all CGF imports to find textures automatically.",
        default="",
        subtype='DIR_PATH',
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Far Cry / CryEngine 1 Settings:", icon='SETTINGS')
        layout.prop(self, "game_root_path")
        if not self.game_root_path:
            layout.label(text="⚠  Set this to your Far Cry install folder (e.g. C:\\FarCry)",
                         icon='ERROR')


def get_game_root_path():
    """Get the globally configured game root path from addon preferences."""
    prefs = bpy.context.preferences.addons.get(__name__)
    if prefs:
        return prefs.preferences.game_root_path
    return ""


# ── CGF / CGA geometry importer ───────────────────────────────────────────────

class ImportCGF(bpy.types.Operator, ImportHelper):
    """Import CryEngine 1 CGF/CGA geometry file (Far Cry)"""
    bl_idname  = "import_scene.cgf"
    bl_label   = "Import CGF/CGA"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".cgf"
    filter_glob: StringProperty(default="*.cgf;*.cga", options={'HIDDEN'})

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
        name="Override Game Root",
        description="Override the global Game Root Path for this import only. "
                    "Leave empty to use the path from Addon Preferences.",
        default="",
        subtype='DIR_PATH',
    )

    def execute(self, context):
        # Use per-import override if set, otherwise fall back to global prefs
        game_root = self.game_root_override.strip() or get_game_root_path()
        result = cgf_builder.load(
            self, context,
            filepath         = self.filepath,
            import_materials = self.import_materials,
            import_normals   = self.import_normals,
            import_uvs       = self.import_uvs,
            import_skeleton  = self.import_skeleton,
            import_weights   = self.import_weights,
            game_root_path   = game_root,
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
            box.label(text=f"Global root: {global_root}", icon='CHECKMARK')
        else:
            box.label(text="No global root set (see Addon Preferences)", icon='ERROR')
        box.prop(self, "game_root_override")


def _store_ctrl_ids(arm_obj):
    source_path = arm_obj.get('cgf_source_path', '')
    if not source_path:
        return
    try:
        reader = cgf_reader.ChunkReader()
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

    def execute(self, context):
        return cgf_builder.load_caf(self, context, self.filepath, self.append)

    def draw(self, context):
        self.layout.prop(self, "append")


# ── CAL animation list importer ───────────────────────────────────────────────

class ImportCAL(bpy.types.Operator, ImportHelper):
    """Import all animations from a CryEngine 1 CAL file"""
    bl_idname  = "import_scene.cal"
    bl_label   = "Import CAL Animation List"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".cal"
    filter_glob: StringProperty(default="*.cal", options={'HIDDEN'})

    def execute(self, context):
        return cgf_builder.load_cal(self, context, self.filepath)


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
        return cgf_exporter.export_cgf(
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


# ── CAF exporter ──────────────────────────────────────────────────────────────

class ExportCAF(bpy.types.Operator, ExportHelper):
    """Export active action to CryEngine 1 CAF animation file"""
    bl_idname  = "export_scene.caf"
    bl_label   = "Export CAF Animation"
    bl_options = {'PRESET'}

    filename_ext = ".caf"
    filter_glob: StringProperty(default="*.caf", options={'HIDDEN'})

    def execute(self, context):
        return cgf_exporter.export_caf(self, context, self.filepath)


# ── CAL exporter ──────────────────────────────────────────────────────────────

class ExportCAL(bpy.types.Operator, ExportHelper):
    """Export all actions to CAF files and write a CAL list"""
    bl_idname  = "export_scene.cal"
    bl_label   = "Export CAL Animation List"
    bl_options = {'PRESET'}

    filename_ext = ".cal"
    filter_glob: StringProperty(default="*.cal", options={'HIDDEN'})

    def execute(self, context):
        return cgf_exporter.export_cal(self, context, self.filepath)


# ── Menu entries ──────────────────────────────────────────────────────────────

def menu_import(self, context):
    self.layout.operator(ImportCGF.bl_idname, text="CryEngine Geometry (.cgf, .cga)")
    self.layout.operator(ImportCAF.bl_idname, text="CryEngine Animation (.caf)")
    self.layout.operator(ImportCAL.bl_idname, text="CryEngine Animation List (.cal)")


def menu_export(self, context):
    self.layout.operator(ExportCGF.bl_idname, text="CryEngine Geometry (.cgf)")
    self.layout.operator(ExportCAF.bl_idname, text="CryEngine Animation (.caf)")
    self.layout.operator(ExportCAL.bl_idname, text="CryEngine Animation List (.cal)")


def register():
    bpy.utils.register_class(CGFAddonPreferences)
    bpy.utils.register_class(CryMaterialProperties)
    bpy.utils.register_class(VIEW3D_PT_cryengine)
    bpy.utils.register_class(ImportCGF)
    bpy.utils.register_class(ImportCAF)
    bpy.utils.register_class(ImportCAL)
    bpy.utils.register_class(ExportCGF)
    bpy.utils.register_class(ExportCAF)
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
    bpy.utils.unregister_class(ImportCAL)
    bpy.utils.unregister_class(ExportCGF)
    bpy.utils.unregister_class(ExportCAF)
    bpy.utils.unregister_class(ExportCAL)
    del bpy.types.Material.cry
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)


if __name__ == "__main__":
    register()
