bl_info = {
    "name": "CryEngine 1 CGF Importer (Far Cry)",
    "author": "Ported from Takaro CryImporter for 3ds Max",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "location": "File > Import > CryEngine CGF/CAF/CAL",
    "description": "Import CryEngine 1 / Far Cry geometry and animation files",
    "category": "Import-Export",
}

import bpy
import os
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper
from . import cgf_reader
from . import cgf_builder


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

    def execute(self, context):
        result = cgf_builder.load(
            self, context,
            filepath         = self.filepath,
            import_materials = self.import_materials,
            import_normals   = self.import_normals,
            import_uvs       = self.import_uvs,
            import_skeleton  = self.import_skeleton,
            import_weights   = self.import_weights,
        )
        # Store source path on armature for later CAF import
        if result == {'FINISHED'}:
            for obj in context.scene.objects:
                if obj.type == 'ARMATURE' and not obj.get('cgf_source_path'):
                    obj['cgf_source_path'] = self.filepath
                    # Store ctrl_ids on pose bones
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


def _store_ctrl_ids(arm_obj):
    """Store bone ctrl_ids as custom properties for later CAF import."""
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
    """Import all animations from a CryEngine 1 CAL file (list of CAF files)"""
    bl_idname  = "import_scene.cal"
    bl_label   = "Import CAL Animation List"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".cal"
    filter_glob: StringProperty(default="*.cal", options={'HIDDEN'})

    def execute(self, context):
        return cgf_builder.load_cal(self, context, self.filepath)


# ── Menu entries ──────────────────────────────────────────────────────────────

def menu_import(self, context):
    self.layout.operator(ImportCGF.bl_idname, text="CryEngine Geometry (.cgf, .cga)")
    self.layout.operator(ImportCAF.bl_idname, text="CryEngine Animation (.caf)")
    self.layout.operator(ImportCAL.bl_idname, text="CryEngine Animation List (.cal)")


def register():
    bpy.utils.register_class(ImportCGF)
    bpy.utils.register_class(ImportCAF)
    bpy.utils.register_class(ImportCAL)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)


def unregister():
    bpy.utils.unregister_class(ImportCGF)
    bpy.utils.unregister_class(ImportCAF)
    bpy.utils.unregister_class(ImportCAL)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)


if __name__ == "__main__":
    register()
