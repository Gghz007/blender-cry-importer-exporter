bl_info = {
    "name": "CryEngine 1 CGF Importer (Far Cry)",
    "author": "Ported from Takaro CryImporter for 3ds Max",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "File > Import > CryEngine CGF (.cgf, .cga)",
    "description": "Import CryEngine 1 / Far Cry geometry files (.cgf, .cga)",
    "category": "Import-Export",
}

import bpy
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper
from . import cgf_reader
from . import cgf_builder


class ImportCGF(bpy.types.Operator, ImportHelper):
    """Import a CryEngine 1 CGF/CGA file (Far Cry)"""
    bl_idname = "import_scene.cgf"
    bl_label = "Import CGF/CGA"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".cgf"
    filter_glob: StringProperty(
        default="*.cgf;*.cga",
        options={'HIDDEN'},
    )

    import_materials: BoolProperty(
        name="Import Materials",
        description="Create materials from chunk data",
        default=True,
    )

    import_normals: BoolProperty(
        name="Import Normals",
        description="Use normals from file (otherwise recalculate)",
        default=True,
    )

    import_uvs: BoolProperty(
        name="Import UVs",
        description="Import texture coordinates",
        default=True,
    )

    import_skeleton: BoolProperty(
        name="Import Skeleton",
        description="Import bones and armature (for skinned meshes)",
        default=True,
    )

    import_weights: BoolProperty(
        name="Import Vertex Weights",
        description="Import bone weights for skinned meshes",
        default=True,
    )

    def execute(self, context):
        keywords = self.as_keywords(ignore=("filter_glob",))
        return cgf_builder.load(self, context, **keywords)

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


def menu_func_import(self, context):
    self.layout.operator(ImportCGF.bl_idname, text="CryEngine CGF (.cgf, .cga)")


def register():
    bpy.utils.register_class(ImportCGF)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.utils.unregister_class(ImportCGF)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()
