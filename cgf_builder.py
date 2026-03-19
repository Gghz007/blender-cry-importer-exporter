"""
cgf_builder.py - Converts parsed CGF chunks into Blender scene objects.

Ported from SceneBuilder (CryImporter for 3ds Max by Takaro Pty. Ltd.)

=== COORDINATE SYSTEM ANALYSIS ===

CryEngine 1 (Far Cry) uses a RIGHT-HANDED, Z-UP system — same as 3ds Max default.
  X = right, Y = forward (into screen), Z = up

Blender uses a RIGHT-HANDED, Z-UP system too — BUT:
  X = right, Y = forward (into screen), Z = up
  … the difference is Blender's viewport shows Y going INTO screen,
  while Max shows it going UP in front view. For data purposes they're the same layout.

So the axes themselves are already compatible! The only real difference is:
  - 3ds Max Matrix3 is ROW-major: rows are basis vectors (row0=X, row1=Y, row2=Z, row3=translation)
  - Blender mathutils.Matrix is also row-major in constructor, but COLUMN-major internally
  - CryEngine stores matrix as 4 rows of 4 floats (row0..row3), same Max convention

The original Max script does:
    Matrix3 [m.row1[1..3]] [m.row2[1..3]] [m.row3[1..3]] [m.row4[1..3]]
    node.transform = m3     ← applied directly, no axis swap

This means CGF matrices are already in Max/world space — no axis conversion needed
for the matrix itself. The vertex positions are also in the same space.

QUATERNIONS in CGF (BoneKey.relquat):
  - Stored as x,y,z,w
  - Applied directly by Max as local bone rotation
  - Max quaternion is also x,y,z,w
  - Blender quaternion is w,x,y,z  ← just reorder

QUATERNION LOG (v827 controller, CryKey.rotLog):
  - Stored as logarithm: rotLog = (rx, ry, rz)
  - Max reconstructs via: exp(quat rx ry rz 0)
  - This is the standard quaternion exponential map

GLOBAL ORIENTATION:
  Max and Blender both show Z-up by default, but Blender's viewport
  is typically oriented so that -Y is "front". For Far Cry assets
  exported from Max (which uses Y-forward), we apply a single
  -90° X-axis rotation so assets face correctly in Blender.
"""

import bpy
import bmesh
import math
import os
import mathutils

from . import cgf_reader


# ---- Coordinate conversion ----
# CryEngine/Max Z-up, row-major → Blender Z-up, column-major
# The matrix layout is compatible; we just need to handle the
# row-major vs Blender's Matrix() constructor correctly.

# 3ds Max default units are INCHES. CryEngine 1 / Far Cry stores coordinates
# in Max units (inches). Blender uses meters. 1 inch = 0.0254 m.
INCHES_TO_METERS = 0.0254


def cry_vec_to_blender(v):
    """
    Convert a CryEngine/Max position vector to Blender, applying inch→meter scale.
    """
    s = INCHES_TO_METERS
    return mathutils.Vector((v[0] * s, v[1] * s, v[2] * s))


def cry_matrix_to_blender(m44):
    """
    Convert a CGF/Max 4x4 matrix (flat list of 16 floats) to Blender Matrix4x4.
    Applies inch→meter scale to the translation component.
    Rotation/scale components are dimensionless — no scaling needed there.
    """
    r0 = m44[0:4]
    r1 = m44[4:8]
    r2 = m44[8:12]
    r3 = m44[12:16]
    m = mathutils.Matrix((r0, r1, r2, r3)).transposed()
    # Scale only the translation column
    s = INCHES_TO_METERS
    m.translation *= s
    return m


def cry_matrix43_to_blender(m43):
    """
    Convert a CGF 4x3 bone matrix (flat list of 12 floats) to Blender Matrix4x4.
    Applies inch→meter scale to translation.
    """
    rot = mathutils.Matrix((
        (m43[0], m43[1], m43[2]),
        (m43[3], m43[4], m43[5]),
        (m43[6], m43[7], m43[8]),
    )).transposed()
    m = rot.to_4x4()
    s = INCHES_TO_METERS
    m.translation = mathutils.Vector((m43[9] * s, m43[10] * s, m43[11] * s))
    return m


def cry_quat_to_blender(xyzw):
    """
    Convert CryEngine quaternion stored as (x, y, z, w) to Blender Quaternion (w, x, y, z).
    No axis remapping needed — same coordinate system.
    """
    x, y, z, w = xyzw
    return mathutils.Quaternion((w, x, y, z))


def quat_exp(log_xyz):
    """
    Reconstruct quaternion from its logarithm (x, y, z) where w=0.
    This is what Max does: exp(quat rx ry rz 0)
    Standard formula: if θ = |v|, q = (v/θ * sin(θ), cos(θ))
    """
    rx, ry, rz = log_xyz
    theta = math.sqrt(rx*rx + ry*ry + rz*rz)
    if theta < 1e-10:
        return mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    s = math.sin(theta) / theta
    return mathutils.Quaternion((math.cos(theta), rx*s, ry*s, rz*s))


# ---- Material builder ----

def _set_bsdf_input(bsdf_node, *names, value):
    """Try input names in order — handles API renames across Blender versions."""
    for name in names:
        if name in bsdf_node.inputs:
            inp = bsdf_node.inputs[name]
            # Scalar inputs need a plain float; color inputs need a tuple
            try:
                inp.default_value = value
            except Exception:
                pass
            return


def build_material(mat_chunk, archive, filepath, import_materials):
    """Create a Blender material from a CryMaterialChunk."""
    if not import_materials:
        return None

    mat = bpy.data.materials.get(mat_chunk.name)
    if mat:
        return mat

    mat = bpy.data.materials.new(name=mat_chunk.name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node  = nodes.new('ShaderNodeOutputMaterial')
    bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    out_node.location  = (400, 0)
    bsdf_node.location = (0, 0)
    links.new(bsdf_node.outputs['BSDF'], out_node.inputs['Surface'])

    # Base color
    d = mat_chunk.diffuse
    _set_bsdf_input(bsdf_node, 'Base Color', value=(d[0], d[1], d[2], 1.0))

    # Specular — renamed in Blender 4.0: 'Specular' → 'Specular IOR Level'
    s = mat_chunk.specular
    spec_level = ((s[0] + s[1] + s[2]) / 3.0) * mat_chunk.specular_level
    _set_bsdf_input(bsdf_node, 'Specular IOR Level', 'Specular', value=min(spec_level, 1.0))

    # Roughness (inverse of shininess, 0-100 range in CryEngine)
    if mat_chunk.specular_shininess > 0:
        roughness = 1.0 - min(mat_chunk.specular_shininess / 100.0, 1.0)
        _set_bsdf_input(bsdf_node, 'Roughness', value=roughness)

    # Opacity / Alpha
    if mat_chunk.opacity < 1.0:
        _set_bsdf_input(bsdf_node, 'Alpha', value=mat_chunk.opacity)
        # blend_method was removed in Blender 4.2 (alpha mode is now automatic)
        if hasattr(mat, 'blend_method'):
            mat.blend_method = 'BLEND'

    # Diffuse texture
    if mat_chunk.tex_diffuse and mat_chunk.tex_diffuse.name:
        tex_path = find_texture(mat_chunk.tex_diffuse.name, filepath)
        if tex_path:
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (-400, 0)
            try:
                img = bpy.data.images.load(tex_path, check_existing=True)
                tex_node.image = img
            except Exception:
                pass
            links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])

    # Bump / normal map texture
    if mat_chunk.tex_bump and mat_chunk.tex_bump.name:
        tex_path = find_texture(mat_chunk.tex_bump.name, filepath)
        if tex_path:
            bump_tex  = nodes.new('ShaderNodeTexImage')
            bump_node = nodes.new('ShaderNodeBump')
            bump_tex.location  = (-400, -300)
            bump_node.location = (-100, -300)
            try:
                img = bpy.data.images.load(tex_path, check_existing=True)
                img.colorspace_settings.name = 'Non-Color'
                bump_tex.image = img
            except Exception:
                pass
            links.new(bump_tex.outputs['Color'], bump_node.inputs['Height'])
            links.new(bump_node.outputs['Normal'], bsdf_node.inputs['Normal'])

    return mat


def find_texture(tex_name, cgf_filepath):
    """Try to find a texture file relative to the CGF file."""
    if not tex_name:
        return None

    base_dir = os.path.dirname(cgf_filepath)
    basename = os.path.basename(tex_name)

    # Extension candidates
    extensions = ['', '.dds', '.tga', '.png', '.jpg', '.bmp']
    tex_without_ext = os.path.splitext(tex_name)[0]

    candidates = [
        tex_name,                                    # as given
        os.path.join(base_dir, basename),            # same dir
        os.path.join(base_dir, tex_name),            # relative to cgf dir
        os.path.join(base_dir, tex_without_ext),     # without ext
    ]

    for candidate in candidates:
        for ext in extensions:
            p = candidate + ext if not os.path.splitext(candidate)[1] else candidate
            if os.path.isfile(p):
                return p
            p2 = candidate + ext
            if os.path.isfile(p2):
                return p2

    return None


# ---- Mesh builder ----

def build_mesh(mesh_chunk, node_chunk, archive, collection,
               import_materials, import_normals, import_uvs,
               import_weights, blender_materials, filepath):
    """Build a Blender mesh object from a CryMeshChunk."""

    name = node_chunk.name if node_chunk else f"Mesh_{mesh_chunk.header.chunk_id}"
    mc = mesh_chunk

    if not mc.vertices or not mc.faces:
        return None

    # Create mesh data
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    bm = bmesh.new()

    # Add vertices — no per-vertex axis swap; axis correction applied globally later
    for cv in mc.vertices:
        p = cry_vec_to_blender(cv.pos)
        bm.verts.new(p)
    bm.verts.ensure_lookup_table()

    # Add faces (triangles)
    face_to_mat_id = {}
    skipped = 0
    for fi, cf in enumerate(mc.faces):
        v0, v1, v2 = cf.v0, cf.v1, cf.v2
        if v0 >= len(bm.verts) or v1 >= len(bm.verts) or v2 >= len(bm.verts):
            skipped += 1
            continue
        try:
            face = bm.faces.new((bm.verts[v0], bm.verts[v1], bm.verts[v2]))
            face.smooth = True
            face_to_mat_id[face.index] = cf.mat_id
        except ValueError:
            # Duplicate face
            skipped += 1

    bm.faces.ensure_lookup_table()

    # UV layer
    uv_layer = None
    if import_uvs and mc.tex_vertices and mc.tex_faces:
        uv_layer = bm.loops.layers.uv.new("UVMap")

    # Assign UVs
    if uv_layer and mc.tex_faces:
        bm.faces.ensure_lookup_table()
        face_real_idx = 0
        for fi, cf in enumerate(mc.faces):
            if fi >= len(mc.faces):
                break
            if face_real_idx >= len(bm.faces):
                break

            tface = mc.tex_faces[fi] if fi < len(mc.tex_faces) else None
            if tface is None:
                face_real_idx += 1
                continue

            face = bm.faces[face_real_idx]
            face_real_idx += 1

            tv_indices = [tface.t0, tface.t1, tface.t2]
            for li, loop in enumerate(face.loops):
                tvi = tv_indices[li]
                if tvi < len(mc.tex_vertices):
                    u, v = mc.tex_vertices[tvi]
                    loop[uv_layer].uv = (u, 1.0 - v)  # flip V for Blender

    bm.to_mesh(mesh)
    bm.free()

    # Custom normals
    # use_auto_smooth was removed in Blender 4.1; normals_split_custom_set
    # still works but the mesh must have valid loop count first.
    if import_normals and mc.vertices:
        normals = []
        for cf in mc.faces:
            for vi in [cf.v0, cf.v1, cf.v2]:
                if vi < len(mc.vertices):
                    n = mc.vertices[vi].normal
                    bn = cry_vec_to_blender(n)
                    if bn.length > 1e-6:
                        bn.normalize()
                    normals.append(bn)
                else:
                    normals.append(mathutils.Vector((0, 0, 1)))
        try:
            # use_auto_smooth only exists in Blender < 4.1
            if hasattr(mesh, 'use_auto_smooth'):
                mesh.use_auto_smooth = True
            if len(normals) == len(mesh.loops):
                mesh.normals_split_custom_set([(n.x, n.y, n.z) for n in normals])
        except Exception:
            pass

    # Materials
    if import_materials and blender_materials:
        mat_chunk_id = node_chunk.material_id if node_chunk else -1
        mat_chunk = archive.get_material_chunk(mat_chunk_id) if mat_chunk_id >= 0 else None

        if mat_chunk:
            # Collect unique mat IDs from faces
            used_mat_ids = sorted(set(cf.mat_id for cf in mc.faces))
            mat_id_to_slot = {}

            if mat_chunk.children:
                # Multi-material
                for slot_idx, child_id in enumerate(mat_chunk.children):
                    child = archive.get_material_chunk(child_id)
                    if child and child.name in blender_materials:
                        bmat = blender_materials[child.name]
                        if bmat.name not in [m.name for m in mesh.materials]:
                            mesh.materials.append(bmat)
                        mat_id_to_slot[slot_idx] = list(mesh.materials).index(bmat)
            else:
                # Single material
                if mat_chunk.name in blender_materials:
                    bmat = blender_materials[mat_chunk.name]
                    mesh.materials.append(bmat)
                    for mid in used_mat_ids:
                        mat_id_to_slot[mid] = 0

            # Assign material slots to polygons
            for pi, poly in enumerate(mesh.polygons):
                fi = pi  # polygon index matches face index
                if fi < len(mc.faces):
                    mid = mc.faces[fi].mat_id
                    slot = mat_id_to_slot.get(mid, 0)
                    poly.material_index = slot

    # Transformation from node chunk.
    # Max applies transMatrix directly (no axis swap) — we do the same,
    # the global AXIS_CORRECTION handles the viewport orientation difference.
    if node_chunk and node_chunk.trans_matrix:
        obj.matrix_world = cry_matrix_to_blender(node_chunk.trans_matrix)
    elif node_chunk:
        obj.location = cry_vec_to_blender(node_chunk.position)

    # Vertex groups for skinning
    if import_weights and mc.physique and archive.bone_anim_chunks:
        _assign_vertex_weights(obj, mc, archive)

    return obj


def _assign_vertex_weights(obj, mesh_chunk, archive):
    """Create vertex groups for bone skinning."""
    mc = mesh_chunk

    # Build bone name map
    bone_names = {}
    if archive.bone_name_list_chunks:
        for i, name in enumerate(archive.bone_name_list_chunks[0].name_list):
            bone_names[i] = name

    for bone_links in mc.physique:
        vid = bone_links.vertex_id
        for link in bone_links.links:
            bone_name = bone_names.get(link.bone_id, f"Bone_{link.bone_id}")
            if bone_name not in obj.vertex_groups:
                obj.vertex_groups.new(name=bone_name)
            vg = obj.vertex_groups[bone_name]
            vg.add([vid], link.blending, 'REPLACE')


# ---- Armature builder ----

def build_armature(archive, collection):
    """Build a Blender armature from BoneAnim chunks."""
    if not archive.bone_anim_chunks:
        return None, None

    bone_anim = archive.bone_anim_chunks[0]
    if not bone_anim.bones:
        return None, None

    name_list = []
    if archive.bone_name_list_chunks:
        name_list = archive.bone_name_list_chunks[0].name_list

    arm_data = bpy.data.armatures.new("Armature")
    arm_obj  = bpy.data.objects.new("Armature", arm_data)
    collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = arm_data.edit_bones

    bone_edit_map = {}  # bone_id → edit bone

    for bone in bone_anim.bones:
        bone_id = bone.bone_id
        bone_name = (name_list[bone_id] if bone_id < len(name_list)
                     else bone.name or f"Bone_{bone_id}")

        eb = edit_bones.new(bone_name)
        eb.head = (0, 0, 0)
        eb.tail = (0, 0.05, 0)  # default tiny length, will be adjusted

        # Use initial position if available.
        # BoneInitialPos is a 4x3 matrix: the Max script does
        #   b.transform = chunkArchive.getBoneInitialPos x.boneID
        # meaning it's applied directly as a world transform in Max space.
        init_mat = archive.get_bone_initial_pos(bone_id)
        if init_mat:
            try:
                mx = cry_matrix43_to_blender(init_mat)
                head = mx.translation
                # In Max/Cry bones point along local X axis (BoneSys convention).
                # Use local X as the bone direction.
                local_x = mx.col[0].xyz.normalized() * 0.05
                eb.head = head
                eb.tail = head + local_x
                eb.roll = 0
            except Exception as e:
                print(f"[CGF] Bone matrix error for {bone_name}: {e}")

        bone_edit_map[bone_id] = eb

    # Set parent relationships
    for bone in bone_anim.bones:
        if bone.parent_id >= 0 and bone.parent_id in bone_edit_map:
            child_eb  = bone_edit_map[bone.bone_id]
            parent_eb = bone_edit_map[bone.parent_id]
            child_eb.parent = parent_eb
            # Connect if child head is close to parent tail
            if (child_eb.head - parent_eb.tail).length < 0.001:
                child_eb.use_connect = True

    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj, arm_data


def apply_armature_to_meshes(arm_obj, mesh_objects, archive):
    """Parent all skinned mesh objects to the armature."""
    if arm_obj is None:
        return

    for mobj in mesh_objects:
        if mobj and mobj.vertex_groups:
            mobj.parent = arm_obj
            mod = mobj.modifiers.new(name="Armature", type='ARMATURE')
            mod.object = arm_obj
            mod.use_vertex_groups = True


# ---- Shape keys (morph targets) ----

def build_shape_keys(obj, mesh_chunk, archive):
    """Add shape keys from MeshMorphTarget chunks."""
    morphs = archive.get_morphs_for_mesh(mesh_chunk.header.chunk_id)
    if not morphs:
        return

    # Add basis shape key
    obj.shape_key_add(name="Basis", from_mix=False)

    for morph in morphs:
        sk = obj.shape_key_add(name=morph.name, from_mix=False)
        for mv in morph.target_vertices:
            vid = mv.vertex_id
            if vid < len(sk.data):
                p = cry_vec_to_blender(mv.target_point)
                sk.data[vid].co = p


# ---- Main load function ----

def load(operator, context, filepath,
         import_materials=True, import_normals=True, import_uvs=True,
         import_skeleton=True, import_weights=True):
    """Main entry point called by the import operator."""

    print(f"[CGF Importer] Loading: {filepath}")

    # Parse the file
    reader = cgf_reader.ChunkReader(filepath)
    try:
        archive = reader.read_file(filepath)
    except ValueError as e:
        operator.report({'ERROR'}, str(e))
        return {'CANCELLED'}

    print(f"[CGF Importer] Parsed {archive.num_chunks} chunks")
    print(f"  Meshes: {len(archive.mesh_chunks)}")
    print(f"  Nodes: {len(archive.node_chunks)}")
    print(f"  Materials: {len(archive.material_chunks)}")
    print(f"  Bones: {len(archive.bone_anim_chunks)}")
    print(f"  Bone meshes: {len(archive.bone_mesh_chunks)}")

    # Create a collection for the imported objects
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    collection = bpy.data.collections.new(file_name)
    context.scene.collection.children.link(collection)

    # Build all materials first
    blender_materials = {}
    if import_materials:
        for mat_chunk in archive.material_chunks:
            bmat = build_material(mat_chunk, archive, filepath, import_materials)
            if bmat:
                blender_materials[mat_chunk.name] = bmat
            # Also process sub-materials
            for child_id in mat_chunk.children:
                child = archive.get_material_chunk(child_id)
                if child:
                    bmat_child = build_material(child, archive, filepath, import_materials)
                    if bmat_child:
                        blender_materials[child.name] = bmat_child

    # Build armature (skeleton) if present
    arm_obj = None
    if import_skeleton and archive.bone_anim_chunks:
        arm_obj, arm_data = build_armature(archive, collection)

    # Build meshes
    mesh_objects = []

    for mesh_chunk in archive.mesh_chunks:
        # Find corresponding node chunk
        node_chunk = archive.get_node(mesh_chunk.header.chunk_id)

        mobj = build_mesh(
            mesh_chunk, node_chunk, archive, collection,
            import_materials, import_normals, import_uvs,
            import_weights, blender_materials, filepath
        )

        if mobj:
            mesh_objects.append(mobj)

            # Shape keys
            if archive.mesh_morph_target_chunks:
                build_shape_keys(mobj, mesh_chunk, archive)

    # If no node chunks gave us transforms, check if there's just one mesh
    # and it has no transform data — keep it at origin

    # Parent meshes to armature
    if arm_obj and import_skeleton and import_weights:
        apply_armature_to_meshes(arm_obj, mesh_objects, archive)

    # No global axis correction — the node chunk transform matrix already
    # encodes the correct world orientation as exported from 3ds Max.

    # Deselect all, then select imported objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in collection.objects:
        obj.select_set(True)
    if mesh_objects:
        context.view_layer.objects.active = mesh_objects[0]

    print(f"[CGF Importer] Done. Created {len(mesh_objects)} mesh(es).")
    operator.report(
        {'INFO'},
        f"Imported {len(mesh_objects)} mesh(es) from {os.path.basename(filepath)}"
    )
    return {'FINISHED'}
