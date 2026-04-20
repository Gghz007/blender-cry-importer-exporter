"""
cgf_exporter.py — Exports Blender scene objects to CryEngine 1 CGF/CGA/CAF files.
"""

import bpy
import os
import math
import mathutils
import struct

from .cry_chunk_writer import (
    CGFWriter,
    ctrl_id_from_name,
    build_source_info_chunk,
    build_timing_chunk,
    build_bone_name_list_chunk,
    build_bone_anim_chunk,
    build_bone_initial_pos_chunk,
    build_mesh_chunk,
    build_node_chunk,
    build_material_chunk,
    build_controller_chunk_v827,
    pack_u32,
    CHUNK_TYPE_MESH, CHUNK_TYPE_NODE, CHUNK_TYPE_MATERIAL,
    CHUNK_TYPE_BONE_ANIM, CHUNK_TYPE_BONE_NAME_LIST,
    CHUNK_TYPE_BONE_INITIAL_POS, CHUNK_TYPE_TIMING,
    CHUNK_TYPE_SOURCE_INFO, CHUNK_TYPE_CONTROLLER,
)

from .cry_chunk_reader import (
    ChunkReader,
    CHUNK_TYPE_MESH       as CT_MESH,
    CHUNK_TYPE_NODE       as CT_NODE,
    CHUNK_TYPE_MATERIAL   as CT_MAT,
    CHUNK_TYPE_BONE_ANIM  as CT_BANIM,
    CHUNK_TYPE_BONE_NAME_LIST as CT_BNAMES,
    CHUNK_TYPE_BONE_INITIAL_POS as CT_BIPOS,
    CHUNK_TYPE_TIMING     as CT_TIMING,
    CHUNK_TYPE_SOURCE_INFO as CT_SRCINFO,
    CHUNK_TYPE_CONTROLLER as CT_CTRL,
)

INCHES_TO_METERS = 0.0254
METERS_TO_INCHES = 1.0 / INCHES_TO_METERS


# ── Coordinate conversion (Blender → CryEngine/Max) ──────────────────────────

def blender_vec_to_cry(v):
    """Blender meters → CryEngine inches."""
    return (v[0] * METERS_TO_INCHES,
            v[1] * METERS_TO_INCHES,
            v[2] * METERS_TO_INCHES)


def blender_matrix_to_cry(mat):
    """
    Blender Matrix4x4 → CGF flat 16-float row-major matrix.
    Blender columns = basis vectors → transpose → rows = basis vectors (Max convention).
    Scale translation from meters to inches.
    """
    m = mat.transposed()
    result = []
    for row_i in range(4):
        for col_i in range(4):
            v = m[row_i][col_i]
            if row_i == 3 and col_i < 3:
                v *= METERS_TO_INCHES  # scale translation
            result.append(v)
    return result


def blender_matrix_to_cry43(mat):
    """
    Blender Matrix4x4 → CGF flat 12-float 4x3 matrix (bone initial pos).
    """
    m = mat.transposed()
    result = []
    for row_i in range(3):
        for col_i in range(3):
            result.append(m[row_i][col_i])
    # Translation row
    result.append(mat.translation.x * METERS_TO_INCHES)
    result.append(mat.translation.y * METERS_TO_INCHES)
    result.append(mat.translation.z * METERS_TO_INCHES)
    return result


def blender_quat_to_cry(q):
    """Blender Quaternion (w,x,y,z) → CryEngine (x,y,z,w)."""
    return (q.x, q.y, q.z, q.w)


def build_material_key(mat):
    """Use the preserved CGF material identity when available."""
    return mat.get('cgf_full_name', mat.name)


def _build_cgf_mat_name(name, shader_name, surface_name):
    """Reconstruct full CGF material name: 'name(shader)/surface'."""
    result = name
    if shader_name:
        result += f"({shader_name})"
    if surface_name:
        result += f"/{surface_name}"
    return result


def quat_log(q):
    """
    Quaternion logarithm — inverse of quat_exp used in reader.
    Used for v827 controller keys.
    """
    bq = mathutils.Quaternion((q[3], q[0], q[1], q[2]))  # w,x,y,z
    # log(q) = (v/|v|) * acos(w) where v = (x,y,z)
    vec = mathutils.Vector((bq.x, bq.y, bq.z))
    half_angle = math.acos(max(-1.0, min(1.0, bq.w)))
    sin_half = math.sin(half_angle)
    if abs(sin_half) < 1e-10:
        return (0.0, 0.0, 0.0)
    scale = half_angle / sin_half
    return (vec.x * scale, vec.y * scale, vec.z * scale)


# ── Mesh extraction ───────────────────────────────────────────────────────────

def triangulate_mesh(obj):
    """Get a triangulated copy of the mesh."""
    import bmesh as bm_mod
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval  = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(obj_eval)

    bm = bm_mod.new()
    bm.from_mesh(mesh)
    bm_mod.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    return mesh


def extract_mesh_data(obj, arm_obj=None, bone_name_list=None):
    """
    Extract vertices, faces, UVs, normals, and bone weights from a Blender mesh.
    Returns dict with all data ready for build_mesh_chunk.
    """
    mesh = triangulate_mesh(obj)
    # calc_normals_split removed in Blender 4.1 — normals are automatic
    if hasattr(mesh, 'calc_normals_split'):
        mesh.calc_normals_split()

    # Vertex data stays in object local space; the Node chunk carries object placement.
    # Bone link offsets, however, need armature-local positions for skinned meshes.
    if arm_obj:
        armature_local_mat = arm_obj.matrix_world.inverted() @ obj.matrix_world
    else:
        armature_local_mat = obj.matrix_world

    vertices   = []  # (pos_cry, normal_cry)
    faces      = []  # (v0, v1, v2, mat_id, smooth_group)
    tex_verts  = []  # (u, v)
    tex_faces  = []  # (t0, t1, t2)

    # We need to split vertices by UV — same position can have different UVs
    uv_layer = mesh.uv_layers.active
    source_vert_ids = obj.get('_cgf_source_vert_ids')
    source_smoothing_groups = obj.get('_cgf_face_smoothing_groups')

    # Build vertex/UV table
    vert_map = {}    # (vert_idx, uv_tuple) → new_idx
    new_verts = []   # (pos, normal)
    new_uvs   = []   # (u, v)

    for poly in mesh.polygons:
        face_tv = []
        face_vv = []

        for li in poly.loop_indices:
            loop = mesh.loops[li]
            vi   = loop.vertex_index
            src_vi = vi
            if source_vert_ids and vi < len(source_vert_ids):
                try:
                    src_vi = int(source_vert_ids[vi])
                except Exception:
                    src_vi = vi

            # Position
            pos_bl = mesh.vertices[vi].co
            pos_cry = (pos_bl.x * METERS_TO_INCHES,
                       pos_bl.y * METERS_TO_INCHES,
                       pos_bl.z * METERS_TO_INCHES)

            # Normal (split normal)
            n_bl = loop.normal.copy()
            if n_bl.length > 1e-6:
                n_bl.normalize()
            nor_cry = (n_bl.x, n_bl.y, n_bl.z)

            # UV
            if uv_layer:
                uv = uv_layer.data[li].uv
                uv_key = (round(uv[0], 6), round(uv[1], 6))
            else:
                uv_key = (0.0, 0.0)

            key = (src_vi, uv_key)
            if key not in vert_map:
                vert_map[key] = len(new_verts)
                new_verts.append((pos_cry, nor_cry))
                new_uvs.append(uv_key)

            new_vi = vert_map[key]
            face_vv.append(new_vi)
            face_tv.append(new_vi)  # tex face uses same index

        mat_id = poly.material_index
        if source_smoothing_groups and poly.index < len(source_smoothing_groups):
            try:
                smooth_group = int(source_smoothing_groups[poly.index])
            except Exception:
                smooth_group = 2 if poly.use_smooth else 0
        else:
            smooth_group = 2 if poly.use_smooth else 0
        # CryEngine uses same winding as Blender
        faces.append((face_vv[0], face_vv[1], face_vv[2], mat_id, smooth_group))
        tex_faces.append((face_tv[0], face_tv[1], face_tv[2]))

    vertices  = new_verts
    tex_verts = new_uvs

    # Bone weights
    physique = None
    has_bone_info = False
    if arm_obj and obj.vertex_groups:
        # Build bone name → index map from armature
        bone_names = bone_name_list or [b.name for b in arm_obj.data.bones]
        bone_idx   = {name: i for i, name in enumerate(bone_names)}

        physique = []
        has_bone_info = True

        for key, new_vi in sorted(vert_map.items(), key=lambda x: x[1]):
            vi = key[0]
            vert = mesh.vertices[vi]
            links = []
            total_w = 0.0
            for g in vert.groups:
                vg = obj.vertex_groups[g.group]
                if vg.name in bone_idx and g.weight > 0.0:
                    bid = bone_idx[vg.name]
                    # CryEngine stores the offset in rest-bone local space.
                    bone = arm_obj.data.bones.get(vg.name)
                    if bone is None:
                        continue
                    pos_arm = armature_local_mat @ vert.co
                    offset_bl = bone.matrix_local.inverted() @ pos_arm
                    offset_cry = (offset_bl.x * METERS_TO_INCHES,
                                  offset_bl.y * METERS_TO_INCHES,
                                  offset_bl.z * METERS_TO_INCHES)
                    links.append((bid, offset_cry, g.weight))
                    total_w += g.weight
            # Normalize weights
            if total_w > 0 and links:
                links = [(b, o, w/total_w) for b, o, w in links]
            physique.append(links)

    bpy.data.meshes.remove(mesh)

    return {
        'vertices':  vertices,
        'faces':     faces,
        'tex_verts': tex_verts,
        'tex_faces': tex_faces,
        'physique':  physique,
        'has_bone_info': has_bone_info,
    }


# ── Armature extraction ───────────────────────────────────────────────────────

def extract_armature_data(arm_obj):
    """
    Extract bone data from a Blender armature.
    Returns list of bone dicts in topological order (parents before children).
    """
    arm = arm_obj.data
    bones = arm.bones

    # Prefer original imported Cry bone ids when they are available.
    imported_pose_bones = []
    if arm_obj.pose:
        for bone in bones:
            if bone.name not in arm_obj.pose.bones:
                continue
            pbone = arm_obj.pose.bones[bone.name]
            if 'cry_bone_id' not in pbone:
                imported_pose_bones = []
                break
            imported_pose_bones.append((int(pbone['cry_bone_id']), bone, pbone))

    if imported_pose_bones and len(imported_pose_bones) == len(bones):
        imported_pose_bones.sort(key=lambda item: item[0])
        sorted_bones = [bone for _, bone, _ in imported_pose_bones]
        bone_idx = {b.name: i for i, b in enumerate(sorted_bones)}

        result = []
        for i, bone in enumerate(sorted_bones):
            pbone = arm_obj.pose.bones.get(bone.name)
            parent_id = int(pbone.get('cry_parent_id', -1)) if pbone else -1
            num_children = sum(
                1 for other in arm_obj.pose.bones
                if int(other.get('cry_parent_id', -999999)) == i
            ) if arm_obj.pose else len(bone.children)

            ctrl_id = ctrl_id_from_name(bone.name)
            custom_property = ''
            mesh_id = -1
            flags = 0xFFFFFFFF

            if pbone:
                stored = pbone.get('cry_ctrl_id')
                if stored:
                    try:
                        ctrl_id = int(stored, 16)
                    except Exception:
                        pass
                custom_property = pbone.get('cry_custom_property', '')
                mesh_id = int(pbone.get('cry_bone_mesh_id', -1))
                stored_flags = pbone.get('cry_bone_flags', 'FFFFFFFF')
                if isinstance(stored_flags, str):
                    try:
                        flags = int(stored_flags, 16)
                    except Exception:
                        flags = 0xFFFFFFFF
                else:
                    flags = int(stored_flags)

            result.append({
                'bone_id':        i,
                'name':           bone.name,
                'parent_id':      parent_id,
                'num_children':   num_children,
                'ctrl_id':        ctrl_id,
                'custom_property': custom_property,
                'bone_physics': {
                    'mesh_id': mesh_id,
                    'flags': flags,
                },
                'bone':           bone,
            })

        return result, bone_idx

    # Topological sort: parents before children
    sorted_bones = []
    visited = set()

    def visit(bone):
        if bone.name in visited:
            return
        if bone.parent:
            visit(bone.parent)
        visited.add(bone.name)
        sorted_bones.append(bone)

    for bone in bones:
        visit(bone)

    bone_idx = {b.name: i for i, b in enumerate(sorted_bones)}

    result = []
    for i, bone in enumerate(sorted_bones):
        parent_id = bone_idx[bone.parent.name] if bone.parent else -1
        num_children = len(bone.children)
        ctrl_id = ctrl_id_from_name(bone.name)

        # Check if original ctrl_id was stored
        if arm_obj.pose and bone.name in arm_obj.pose.bones:
            pbone = arm_obj.pose.bones[bone.name]
            stored = pbone.get('cry_ctrl_id')
            if stored:
                try:
                    ctrl_id = int(stored, 16)
                except Exception:
                    pass
            custom_property = pbone.get('cry_custom_property', '')
            mesh_id = int(pbone.get('cry_bone_mesh_id', -1))
            stored_flags = pbone.get('cry_bone_flags', 'FFFFFFFF')
            if isinstance(stored_flags, str):
                try:
                    flags = int(stored_flags, 16)
                except Exception:
                    flags = 0xFFFFFFFF
            else:
                flags = int(stored_flags)
        else:
            custom_property = ''
            mesh_id = -1
            flags = 0xFFFFFFFF

        result.append({
            'bone_id':        i,
            'name':           bone.name,
            'parent_id':      parent_id,
            'num_children':   num_children,
            'ctrl_id':        ctrl_id,
            'custom_property': custom_property,
            'bone_physics': {
                'mesh_id': mesh_id,
                'flags': flags,
            },
            'bone':           bone,  # keep reference for matrix
        })

    return result, bone_idx


def extract_bone_matrices(arm_obj, bone_data_list):
    """
    Extract 4x3 rest pose world-space matrices for BoneInitialPos.
    Uses original CGF matrices stored on armature if available (round-trip).
    Falls back to computing from current bone world-space transform.
    """
    import json

    # Try to load stored original matrices
    stored_json = arm_obj.get('cgf_bone_matrices')
    stored = {}
    if stored_json:
        try:
            stored = json.loads(stored_json)
        except Exception:
            pass

    matrices = []
    for bd in bone_data_list:
        bname = bd['name']
        if bname in stored:
            matrices.append(stored[bname])
        else:
            # Compute world-space matrix from current bone state
            bone = arm_obj.data.bones[bname]
            world_mat = arm_obj.matrix_world @ bone.matrix_local
            m = blender_matrix_to_cry43(world_mat)
            matrices.append(m)
    return matrices


# ── Material extraction ───────────────────────────────────────────────────────

def extract_materials(obj):
    """
    Extract material data from a Blender object.
    Returns list of material dicts.
    """
    result = []
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None:
            result.append({'name': 'default', 'diffuse': (0.8, 0.8, 0.8),
                           'specular': (0, 0, 0), 'opacity': 1.0,
                           'tex_diffuse': '', 'tex_bump': ''})
            continue

        diffuse  = (0.8, 0.8, 0.8)
        specular = (0, 0, 0)
        opacity  = 1.0
        tex_diff = ''
        tex_bump   = ''  # slot 4 — DDN normal map (_ddn)
        tex_detail = ''  # slot 9 — heightmap (_bump)

        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bc = node.inputs.get('Base Color')
                    if bc:
                        diffuse = (bc.default_value[0],
                                   bc.default_value[1],
                                   bc.default_value[2])
                    al = node.inputs.get('Alpha')
                    if al:
                        opacity = al.default_value
                    # Diffuse texture
                    if bc and bc.links:
                        tex_node = bc.links[0].from_node
                        if tex_node.type == 'TEX_IMAGE' and tex_node.image:
                            tex_diff = bpy.path.abspath(
                                tex_node.image.filepath_raw or
                                tex_node.image.filepath or
                                tex_node.image.name
                            )
                    # Normal/bump textures via Normal input
                    normal_input = node.inputs.get('Normal')
                    if normal_input and normal_input.links:
                        norm_node = normal_input.links[0].from_node
                        if norm_node.type == 'NORMAL_MAP':
                            # Normal Map node → DDN → slot 4 (bump)
                            color_in = norm_node.inputs.get('Color')
                            if color_in and color_in.links:
                                t = color_in.links[0].from_node
                                if t.type == 'TEX_IMAGE' and t.image:
                                    tex_bump = bpy.path.abspath(
                                        t.image.filepath_raw or
                                        t.image.filepath or
                                        t.image.name
                                    )
                        elif norm_node.type == 'BUMP':
                            # Bump node → heightmap → slot 9 (detail)
                            height_in = norm_node.inputs.get('Height')
                            if height_in and height_in.links:
                                t = height_in.links[0].from_node
                                if t.type == 'TEX_IMAGE' and t.image:
                                    tex_detail = bpy.path.abspath(
                                        t.image.filepath_raw or
                                        t.image.filepath or
                                        t.image.name
                                    )

        # Build a robust Cry material identity for export.
        # If explicit Cry metadata is missing (common on brand-new Blender materials),
        # fall back to safe visible defaults instead of exporting a bare name.
        source_name = mat.get('cgf_source_name', mat.name)
        full_name = mat.get('cgf_full_name')
        shader_name = mat.get('cgf_shader_name', '')
        surface_name = mat.get('cgf_surface_name', '')

        if not shader_name and hasattr(mat, "cry"):
            try:
                cry = mat.cry
                if getattr(cry, "shader_preset", "") == "custom":
                    shader_name = (getattr(cry, "shader_custom", "") or "").strip()
                else:
                    shader_name = (getattr(cry, "shader_preset", "") or "").strip()
                surface_name = surface_name or (getattr(cry, "surface", "") or "").strip()
            except Exception:
                pass

        if not shader_name:
            shader_name = "TemplModelCommon"
        if not surface_name:
            surface_name = "mat_default"

        if not full_name:
            full_name = _build_cgf_mat_name(source_name, shader_name, surface_name)

        result.append({
            'name':        full_name,
            'source_name': source_name,
            'chunk_id':    int(mat.get('cgf_chunk_id', -1)),
            'diffuse':     diffuse,
            'specular':    specular,
            'opacity':     opacity,
            'tex_diffuse': tex_diff,
            'tex_bump':    tex_bump,
            'tex_detail':  tex_detail,
        })
        print(f"[CGF Export] Material '{mat.name}': diffuse='{tex_diff}' bump(ddn)='{tex_bump}' detail(bump)='{tex_detail}'")

    return result


# ── CGF export ────────────────────────────────────────────────────────────────

def _to_game_relative(path, game_root_path):
    """
    Convert absolute texture path to game-relative path with backslashes.
    Always writes .dds extension — CryEngine 1 expects .dds in CGF files.
    """
    if not path:
        return ''
    path = os.path.normpath(path)
    if game_root_path:
        root = os.path.normpath(game_root_path)
        if path.lower().startswith(root.lower()):
            path = path[len(root):]
            if path.startswith(os.sep):
                path = path[1:]
    # Force .dds extension regardless of actual file format
    base = os.path.splitext(path)[0]
    path = base + '.dds'
    # Convert to backslashes (CryEngine convention)
    return path.replace('/', '\\')


def _get_node_local_matrix(obj, arm_obj=None):
    """Node transform in armature-local space for skinned meshes, world space otherwise."""
    if arm_obj:
        return arm_obj.matrix_world.inverted() @ obj.matrix_world
    return obj.matrix_world.copy()


def _read_chunk_headers_raw(data):
    if data[:6] != b'CryTek':
        return None, []
    chunk_table_pos, = struct.unpack_from('<I', data, 16)
    num_chunks, = struct.unpack_from('<I', data, chunk_table_pos)
    table_start = chunk_table_pos + 4
    headers = []
    for i in range(num_chunks):
        chunk_type, = struct.unpack_from('<H', data, table_start + i * 16)
        version, file_offset, chunk_id = struct.unpack_from(
            '<III', data, table_start + i * 16 + 4
        )
        headers.append({
            'type': chunk_type,
            'version': version,
            'file_offset': file_offset,
            'chunk_id': chunk_id,
        })
    headers.sort(key=lambda h: (h['file_offset'], h['type'], h['chunk_id']))
    return chunk_table_pos, headers


def _load_preserved_source_chunks(source_path):
    """
    Preserve original character-support chunks that current exporter does not rebuild yet.
    This is primarily for round-trip compatibility with imported CE1 character assets.
    """
    if not source_path or not os.path.isfile(source_path):
        return []

    with open(source_path, 'rb') as f:
        data = f.read()

    chunk_table_pos, headers = _read_chunk_headers_raw(data)
    if chunk_table_pos is None:
        return []

    preserve_types = {
        0x0009,  # LIGHT
        0x000F,  # BONE_MESH
        0x0010,  # BONE_LIGHT_BIND
        0x0011,  # MESH_MORPH_TARGET
    }
    preserved = []
    for i, h in enumerate(headers):
        next_pos = headers[i + 1]['file_offset'] if i + 1 < len(headers) else chunk_table_pos
        chunk_data = data[h['file_offset'] + 16:next_pos]

        if h['type'] == 0x000B:
            # Preserve helper/hitspot nodes from original files.
            try:
                name_raw = chunk_data[:64]
                nul = name_raw.find(b'\x00')
                if nul >= 0:
                    name_raw = name_raw[:nul]
                name = name_raw.decode('latin1', errors='replace')
                object_id, parent_id, num_children, material_id = struct.unpack_from('<iiIi', chunk_data, 64)
                if material_id == -1 or name.startswith('_hs_'):
                    preserved.append((h['type'], h['version'], h['chunk_id'], chunk_data))
            except Exception:
                pass
            continue

        if h['type'] in preserve_types:
            preserved.append((h['type'], h['version'], h['chunk_id'], chunk_data))
    return preserved


def _load_source_roundtrip_metadata(source_path):
    if not source_path or not os.path.isfile(source_path):
        return {
            'max_chunk_id': 0,
            'source_info_id': None,
            'timing_id': None,
            'bone_anim_id': None,
            'bone_name_list_id': None,
            'bone_initial_pos_id': None,
            'node_ids_by_name': {},
            'mesh_ids_by_name': {},
            'multi_material_id_by_name': {},
            'material_ids_by_full_name': {},
            'material_ids_by_source_name': {},
        }

    with open(source_path, 'rb') as f:
        data = f.read()
    _, headers = _read_chunk_headers_raw(data)

    meta = {
        'max_chunk_id': max((h['chunk_id'] for h in headers), default=0),
        'source_info_id': None,
        'timing_id': None,
        'bone_anim_id': None,
        'bone_name_list_id': None,
        'bone_initial_pos_id': None,
        'node_ids_by_name': {},
        'mesh_ids_by_name': {},
        'multi_material_id_by_name': {},
        'material_ids_by_full_name': {},
        'material_ids_by_source_name': {},
    }

    for h in headers:
        if h['type'] == CT_SRCINFO and meta['source_info_id'] is None:
            meta['source_info_id'] = h['chunk_id']
        elif h['type'] == CT_TIMING and meta['timing_id'] is None:
            meta['timing_id'] = h['chunk_id']
        elif h['type'] == CT_BANIM and meta['bone_anim_id'] is None:
            meta['bone_anim_id'] = h['chunk_id']
        elif h['type'] == CT_BNAMES and meta['bone_name_list_id'] is None:
            meta['bone_name_list_id'] = h['chunk_id']
        elif h['type'] == CT_BIPOS and meta['bone_initial_pos_id'] is None:
            meta['bone_initial_pos_id'] = h['chunk_id']

    try:
        archive = ChunkReader().read_file(source_path)
    except Exception:
        return meta

    for node in archive.node_chunks:
        meta['node_ids_by_name'].setdefault(node.name, node.header.chunk_id)
        if node.object_id is not None and node.object_id >= 0:
            meta['mesh_ids_by_name'].setdefault(node.name, int(node.object_id))
        if node.material_id is not None and node.material_id >= 0:
            meta['multi_material_id_by_name'].setdefault(node.name, int(node.material_id))

    for mat in archive.material_chunks:
        full_name = _build_cgf_mat_name(mat.name, mat.shader_name, mat.surface_name)
        meta['material_ids_by_full_name'].setdefault(full_name, mat.header.chunk_id)
        meta['material_ids_by_source_name'].setdefault(mat.name, mat.header.chunk_id)

    return meta


def _load_source_archive(source_path):
    if not source_path or not os.path.isfile(source_path):
        return None
    try:
        return ChunkReader().read_file(source_path)
    except Exception:
        return None


def _safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def _resolve_action_datablock(action_candidate, arm_obj=None):
    if action_candidate is not None and hasattr(action_candidate, 'fcurves'):
        return action_candidate

    nested = getattr(action_candidate, 'action', None)
    if nested is not None and hasattr(nested, 'fcurves'):
        return nested

    ad = getattr(arm_obj, 'animation_data', None) if arm_obj else None
    if ad:
        direct = getattr(ad, 'action', None)
        if direct is not None and hasattr(direct, 'fcurves'):
            return direct
        slot = getattr(ad, 'action_slot', None)
        slot_action = getattr(slot, 'action', None) if slot else None
        if slot_action is not None and hasattr(slot_action, 'fcurves'):
            return slot_action

    if action_candidate is not None:
        name = getattr(action_candidate, 'name', None)
        if name:
            found = bpy.data.actions.get(name)
            if found is not None and hasattr(found, 'fcurves'):
                return found

    return None


def _action_fcurves(action_candidate):
    action = _resolve_action_datablock(action_candidate)
    if action is None:
        return []
    fcurves = getattr(action, 'fcurves', None)
    return fcurves if fcurves is not None else []


def _action_has_pose_fcurves(action_candidate):
    return any(
        getattr(fc, 'data_path', '').startswith('pose.bones[')
        for fc in _action_fcurves(action_candidate)
    )


def _assign_action_to_armature(arm_obj, action):
    resolved = _resolve_action_datablock(action, arm_obj)
    if arm_obj is None or resolved is None:
        return False
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    ad = arm_obj.animation_data
    try:
        ad.action = resolved
        if _resolve_action_datablock(getattr(ad, 'action', None), arm_obj) is resolved:
            return True
    except Exception:
        pass
    slot = getattr(ad, 'action_slot', None)
    if slot is not None:
        try:
            slot.action = resolved
            if _resolve_action_datablock(getattr(slot, 'action', None), arm_obj) is resolved:
                return True
        except Exception:
            pass
    return False


def _pick_export_action(arm_obj):
    ad = getattr(arm_obj, 'animation_data', None)
    if ad:
        resolved = _resolve_action_datablock(getattr(ad, 'action', None), arm_obj)
        if resolved is not None:
            return resolved
    for action in bpy.data.actions:
        resolved = _resolve_action_datablock(action, arm_obj)
        if resolved is not None and _action_has_pose_fcurves(resolved):
            return resolved
    return None


def _action_frame_range(action, scene):
    resolved = _resolve_action_datablock(action)
    if resolved is not None:
        try:
            frame_range = resolved.frame_range
            return int(frame_range[0]), int(frame_range[1])
        except Exception:
            pass
    return int(scene.frame_start), int(scene.frame_end)


def _make_chunk_id_allocator(source_meta=None):
    used = set()
    next_auto = max(1, int((source_meta or {}).get('max_chunk_id', 0)) + 1)

    def allocate(preferred=None):
        nonlocal next_auto
        if preferred is not None and preferred > 0 and preferred not in used:
            used.add(preferred)
            return preferred
        while next_auto in used:
            next_auto += 1
        cid = next_auto
        used.add(cid)
        next_auto += 1
        return cid

    return allocate


def export_cgf(operator, context, filepath,
               export_materials=True, export_skeleton=True,
               export_weights=True, selected_only=False):
    """Export selected/active mesh(es) to CGF."""

    # Get game root path from addon preferences
    game_root_path = ""
    prefs = context.preferences.addons.get('io_import_cgf')
    if prefs:
        game_root_path = prefs.preferences.game_root_path

    if selected_only:
        objects = [o for o in context.selected_objects
                   if o.type == 'MESH' and not o.hide_get()]
    else:
        # Only visible objects in the current view layer
        objects = [o for o in context.view_layer.objects
                   if o.type == 'MESH' and not o.hide_get() and o.visible_get()]

    if not objects:
        operator.report({'ERROR'}, "No visible mesh objects found")
        return {'CANCELLED'}

    # Find armature — only visible ones
    arm_obj = None
    if export_skeleton:
        for obj in context.view_layer.objects:
            if obj.type == 'ARMATURE' and not obj.hide_get():
                arm_obj = obj
                break
        if arm_obj is None:
            for obj in objects:
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE' and mod.object:
                        arm_obj = mod.object
                        break

    writer = CGFWriter(is_anim=False)
    chunk_id = 0

    def next_id():
        nonlocal chunk_id
        chunk_id += 1
        return chunk_id

    # ── Pre-extract all data ──────────────────────────────────────────────────

    # Armature
    bone_data_list = []
    bone_idx = {}
    bone_name_list = []
    if arm_obj and export_skeleton:
        print(f"[CGF Export] Extracting armature: {arm_obj.name}")
        bone_data_list, bone_idx = extract_armature_data(arm_obj)
        bone_name_list = [b['name'] for b in bone_data_list]
        source_path = arm_obj.get('cgf_source_path', '')
    else:
        source_path = ''

    # Assign chunk IDs upfront so Node can reference Mesh and Material
    # Original order: SourceInfo, Timing, Node, MultiMat, Mesh, StandardMats, BoneAnim, BoneNames
    mat_chunk_ids   = {}   # mat_name → chunk_id
    all_standard_mats = []
    multi_mat_ids   = {}   # obj.name → chunk_id

    if export_materials:
        for obj in objects:
            mats = extract_materials(obj)
            for mat in mats:
                if mat['name'] not in mat_chunk_ids:
                    cid = next_id()
                    mat_chunk_ids[mat['name']] = cid
                    all_standard_mats.append((cid, mat))

    # Pre-assign mesh and node IDs
    obj_mesh_ids = {}   # obj → mesh_cid
    obj_node_ids = {}   # obj → node_cid
    obj_multi_ids = {}  # obj → multi_mat_cid
    obj_bipos_ids = {}  # obj → bipos_cid

    for obj in objects:
        obj_mesh_ids[obj.name] = next_id()
        obj_node_ids[obj.name] = next_id()
        # Multi-material — use same key as mat_chunk_ids (cgf_full_name or mat.name)
        if export_materials and len(obj.material_slots) > 1:
            children = []
            for s in obj.material_slots:
                if s.material:
                    key = build_material_key(s.material)
                    if key in mat_chunk_ids:
                        children.append(mat_chunk_ids[key])
            if children:
                obj_multi_ids[obj.name] = next_id()

    # ── Write chunks in correct order ─────────────────────────────────────────

    # 1. SourceInfo
    import getpass, datetime
    data, ver, cid = build_source_info_chunk(
        next_id(),
        source_file=filepath,
        date=datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y"),
        user=getpass.getuser()
    )
    writer.add_chunk(CT_SRCINFO, ver, cid, data)

    # 2. Timing
    scene = context.scene
    fps = scene.render.fps
    ticks_per_frame = 160
    secs_per_tick = 1.0 / (fps * ticks_per_frame)
    data, ver, cid = build_timing_chunk(
        next_id(), ticks_per_frame, secs_per_tick,
        scene.frame_start, scene.frame_end
    )
    writer.add_chunk(CT_TIMING, ver, cid, data)

    # 3. Node chunks (before mesh — original order)
    for obj in objects:
        mesh_cid = obj_mesh_ids[obj.name]
        node_cid = obj_node_ids[obj.name]
        node_mat = _get_node_local_matrix(obj, arm_obj if export_weights else None)
        loc, rot, scale = node_mat.decompose()

        if export_materials:
            if obj.name in obj_multi_ids:
                mat_id = obj_multi_ids[obj.name]
            elif obj.material_slots and obj.material_slots[0].material:
                mat_id = mat_chunk_ids.get(
                    build_material_key(obj.material_slots[0].material), -1
                )
            else:
                mat_id = -1
        else:
            mat_id = -1

        data, ver, _ = build_node_chunk(
            node_cid, obj.name,
            object_id    = mesh_cid,
            parent_id    = -1,
            material_id  = mat_id,
            trans_matrix = blender_matrix_to_cry(node_mat),
            position     = blender_vec_to_cry(loc),
            rotation     = blender_quat_to_cry(rot),
            scale        = (scale.x, scale.y, scale.z),
            pos_ctrl_id   = 0xFFFFFFFF,
            rot_ctrl_id   = 0xFFFFFFFF,
            scale_ctrl_id = 0xFFFFFFFF,
        )
        writer.add_chunk(CT_NODE, ver, node_cid, data)

    # 4. Multi-material chunks (before mesh — original order)
    if export_materials and not (source_archive and source_archive.material_chunks):
        for obj in objects:
            if obj.name in obj_multi_ids:
                children = []
                for s in obj.material_slots:
                    if s.material:
                        key = build_material_key(s.material)
                        if key in mat_chunk_ids:
                            children.append(mat_chunk_ids[key])
                cid = obj_multi_ids[obj.name]
                # Multi-material name = base name of first material
                first_mat = obj.material_slots[0].material if obj.material_slots else None
                base_name = ''
                if first_mat:
                    full = build_material_key(first_mat)
                    base_name = full.split('(')[0].split('/')[0]
                data, ver, _ = build_material_chunk(
                    cid, base_name, mat_type=2, children=children
                )
                writer.add_chunk(CT_MAT, ver, cid, data)

    # 5. Mesh chunks (with embedded BoneInitialPos)
    for obj in objects:
        print(f"[CGF Export] Extracting mesh: {obj.name}")
        md = extract_mesh_data(
            obj,
            arm_obj if export_weights else None,
            bone_name_list if export_weights else None,
        )

        mesh_cid = obj_mesh_ids[obj.name]

        bone_matrices = None
        bipos_cid = None
        if md['has_bone_info'] and arm_obj and bone_data_list:
            bone_matrices = extract_bone_matrices(arm_obj, bone_data_list)
            bipos_cid = next_id()

        data, ver, _, bipos_offset = build_mesh_chunk(
            mesh_cid,
            vertices      = md['vertices'],
            faces         = md['faces'],
            tex_vertices  = md['tex_verts'],
            tex_faces     = md['tex_faces'],
            physique      = md['physique'],
            has_bone_info = md['has_bone_info'],
            bone_matrices = bone_matrices,
        )
        mesh_chunk_idx = len(writer.chunks)
        writer.add_chunk(CT_MESH, ver, mesh_cid, data)

        if bone_matrices and bipos_offset is not None:
            writer.add_embedded_chunk_entry(
                CT_BIPOS, 0x0001, bipos_cid,
                mesh_chunk_idx, bipos_offset
            )

    # 6. Standard material chunks (after mesh — original order)
    if export_materials:
        for cid, mat in all_standard_mats:
            data, ver, _ = build_material_chunk(
                cid, mat['name'],
                mat_type=1,
                diffuse=mat['diffuse'],
                specular=mat['specular'],
                opacity=mat['opacity'],
                tex_diffuse=_to_game_relative(mat.get('tex_diffuse', ''), game_root_path),
                tex_bump=_to_game_relative(mat.get('tex_bump', ''), game_root_path),
                tex_detail=_to_game_relative(mat.get('tex_detail', ''), game_root_path),
            )
            writer.add_chunk(CT_MAT, ver, cid, data)

    # 7. BoneAnim + BoneNameList (after mesh — original order)
    if arm_obj and export_skeleton and bone_data_list:
        data, ver, cid = build_bone_anim_chunk(next_id(), bone_data_list)
        writer.add_chunk(CT_BANIM, ver, cid, data)

        data, ver, cid = build_bone_name_list_chunk(next_id(), bone_name_list)
        writer.add_chunk(CT_BNAMES, ver, cid, data)

    # 8. Preserve original CE1 character-support chunks when round-tripping from source.
    for p_type, p_ver, p_cid, p_data in _load_preserved_source_chunks(source_path):
        writer.add_chunk(p_type, p_ver, p_cid, p_data)

    writer.write(filepath)
    print(f"[CGF Export] Written: {filepath}")
    operator.report({'INFO'}, f"Exported {len(objects)} mesh(es) to {os.path.basename(filepath)}")
    return {'FINISHED'}


def export_cgf_scene(operator, context, filepath,
                     export_materials=True, export_skeleton=True,
                     export_weights=True, selected_only=False):
    """
    Updated geometry exporter.
    For skinned .cga exports, writes an extra export-only root helper at 0,0,0
    and parents skinned mesh nodes to it.
    """
    game_root_path = ""
    prefs = context.preferences.addons.get('io_import_cgf')
    if prefs:
        game_root_path = prefs.preferences.game_root_path

    if selected_only:
        objects = [o for o in context.selected_objects if o.type == 'MESH' and not o.hide_get()]
    else:
        objects = [o for o in context.view_layer.objects if o.type == 'MESH' and not o.hide_get() and o.visible_get()]

    if not objects:
        operator.report({'ERROR'}, "No visible mesh objects found")
        return {'CANCELLED'}

    arm_obj = None
    if export_skeleton:
        for obj in context.view_layer.objects:
            if obj.type == 'ARMATURE' and not obj.hide_get():
                arm_obj = obj
                break
        if arm_obj is None:
            for obj in objects:
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE' and mod.object:
                        arm_obj = mod.object
                        break

    writer = CGFWriter(is_anim=False)
    export_ext = os.path.splitext(filepath)[1].lower()
    chunk_id = 0

    def next_id():
        nonlocal chunk_id
        chunk_id += 1
        return chunk_id

    bone_data_list = []
    bone_name_list = []
    if arm_obj and export_skeleton:
        print(f"[CGF Export] Extracting armature: {arm_obj.name}")
        bone_data_list, _ = extract_armature_data(arm_obj)
        bone_name_list = [b['name'] for b in bone_data_list]
        source_path = arm_obj.get('cgf_source_path', '')
    else:
        source_path = ''

    source_meta = _load_source_roundtrip_metadata(source_path)
    source_archive = _load_source_archive(source_path)
    allocate_chunk_id = _make_chunk_id_allocator(source_meta)

    mat_chunk_ids = {}
    all_standard_mats = []
    obj_multi_ids = {}
    if export_materials:
        for obj in objects:
            mats = extract_materials(obj)
            for mat in mats:
                if mat['name'] not in mat_chunk_ids:
                    preferred_mat_id = _safe_int(mat.get('chunk_id'), None)
                    if preferred_mat_id is not None and preferred_mat_id <= 0:
                        preferred_mat_id = None
                    cid = allocate_chunk_id(
                        preferred_mat_id
                        or source_meta['material_ids_by_full_name'].get(mat['name'])
                        or source_meta['material_ids_by_source_name'].get(mat.get('source_name', ''))
                    )
                    mat_chunk_ids[mat['name']] = cid
                    all_standard_mats.append((cid, mat))

    obj_mesh_ids = {}
    obj_node_ids = {}
    skinned_objects = []
    for obj in objects:
        is_skinned_obj = bool(
            arm_obj and export_weights and any(
                mod.type == 'ARMATURE' and mod.object == arm_obj
                for mod in obj.modifiers
            )
        )
        if is_skinned_obj:
            skinned_objects.append(obj)
        obj_mesh_ids[obj.name] = allocate_chunk_id(
            _safe_int(obj.get('cgf_chunk_id'), None) or
            source_meta['mesh_ids_by_name'].get(obj.name)
        )
        obj_node_ids[obj.name] = allocate_chunk_id(
            source_meta['node_ids_by_name'].get(obj.name)
        )
        if export_materials and len(obj.material_slots) > 1:
            children = []
            for s in obj.material_slots:
                if s.material:
                    key = build_material_key(s.material)
                    if key in mat_chunk_ids:
                        children.append(mat_chunk_ids[key])
            if children:
                obj_multi_ids[obj.name] = allocate_chunk_id(
                    source_meta['multi_material_id_by_name'].get(obj.name)
                )

    use_root_helper = bool(skinned_objects) and export_ext == '.cga'
    root_helper_id = allocate_chunk_id() if use_root_helper else None

    import getpass, datetime
    data, ver, cid = build_source_info_chunk(
        allocate_chunk_id(source_meta.get('source_info_id')),
        source_file=filepath,
        date=datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y"),
        user=getpass.getuser()
    )
    writer.add_chunk(CT_SRCINFO, ver, cid, data)

    scene = context.scene
    fps = scene.render.fps
    ticks_per_frame = 160
    secs_per_tick = 1.0 / (fps * ticks_per_frame)
    data, ver, cid = build_timing_chunk(
        allocate_chunk_id(source_meta.get('timing_id')), ticks_per_frame, secs_per_tick,
        scene.frame_start, scene.frame_end
    )
    writer.add_chunk(CT_TIMING, ver, cid, data)

    if use_root_helper:
        ident44 = (
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        root_helper_name = "Fbx_Root"
        data, ver, _ = build_node_chunk(
            root_helper_id,
            root_helper_name,
            object_id=-1,
            parent_id=-1,
            material_id=-1,
            trans_matrix=ident44,
            position=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0),
            pos_ctrl_id=0xFFFFFFFF,
            rot_ctrl_id=0xFFFFFFFF,
            scale_ctrl_id=0xFFFFFFFF,
            child_nodes=[obj_node_ids[obj.name] for obj in skinned_objects],
        )
        writer.add_chunk(CT_NODE, ver, root_helper_id, data)

    for obj in objects:
        mesh_cid = obj_mesh_ids[obj.name]
        node_cid = obj_node_ids[obj.name]
        node_mat = _get_node_local_matrix(obj, arm_obj if export_weights else None)
        loc, rot, scale = node_mat.decompose()
        is_skinned_obj = obj in skinned_objects

        if export_materials:
            if obj.name in obj_multi_ids:
                mat_id = obj_multi_ids[obj.name]
            elif obj.material_slots and obj.material_slots[0].material:
                mat_id = mat_chunk_ids.get(build_material_key(obj.material_slots[0].material), -1)
            else:
                mat_id = -1
        else:
            mat_id = -1

        data, ver, _ = build_node_chunk(
            node_cid, obj.name,
            object_id=mesh_cid,
            parent_id=root_helper_id if (use_root_helper and is_skinned_obj) else -1,
            material_id=mat_id,
            trans_matrix=blender_matrix_to_cry(node_mat),
            position=blender_vec_to_cry(loc),
            rotation=blender_quat_to_cry(rot),
            scale=(scale.x, scale.y, scale.z),
            pos_ctrl_id=0xFFFFFFFF,
            rot_ctrl_id=0xFFFFFFFF,
            scale_ctrl_id=0xFFFFFFFF,
        )
        writer.add_chunk(CT_NODE, ver, node_cid, data)

    if export_materials:
        for obj in objects:
            if obj.name in obj_multi_ids:
                children = []
                for s in obj.material_slots:
                    if s.material:
                        key = build_material_key(s.material)
                        if key in mat_chunk_ids:
                            children.append(mat_chunk_ids[key])
                cid = obj_multi_ids[obj.name]
                first_mat = obj.material_slots[0].material if obj.material_slots else None
                base_name = ''
                if first_mat:
                    full = build_material_key(first_mat)
                    base_name = full.split('(')[0].split('/')[0]
                data, ver, _ = build_material_chunk(cid, base_name, mat_type=2, children=children)
                writer.add_chunk(CT_MAT, ver, cid, data)

    for obj in objects:
        print(f"[CGF Export] Extracting mesh: {obj.name}")
        md = extract_mesh_data(
            obj,
            arm_obj if export_weights else None,
            bone_name_list if export_weights else None,
        )
        mesh_cid = obj_mesh_ids[obj.name]
        bone_matrices = None
        bipos_cid = None
        if md['has_bone_info'] and arm_obj and bone_data_list:
            bone_matrices = extract_bone_matrices(arm_obj, bone_data_list)
            bipos_cid = allocate_chunk_id(source_meta.get('bone_initial_pos_id'))

        data, ver, _, bipos_offset = build_mesh_chunk(
            mesh_cid,
            vertices=md['vertices'],
            faces=md['faces'],
            tex_vertices=md['tex_verts'],
            tex_faces=md['tex_faces'],
            physique=md['physique'],
            has_bone_info=md['has_bone_info'],
            bone_matrices=bone_matrices,
        )
        mesh_chunk_idx = len(writer.chunks)
        writer.add_chunk(CT_MESH, ver, mesh_cid, data)

        if bone_matrices and bipos_offset is not None:
            writer.add_embedded_chunk_entry(
                CT_BIPOS, 0x0001, bipos_cid,
                mesh_chunk_idx, bipos_offset
            )

    if export_materials:
        written_ids = set()
        scene_mats_by_full = {mat['name']: mat for _, mat in all_standard_mats}
        scene_mats_by_source = {
            mat.get('source_name', mat['name']): mat
            for _, mat in all_standard_mats
        }

        if source_archive and source_archive.material_chunks:
            for src_mat in source_archive.material_chunks:
                cid = int(src_mat.header.chunk_id)
                if src_mat.type == 2:
                    data, ver, _ = build_material_chunk(
                        cid,
                        src_mat.name,
                        mat_type=2,
                        children=list(src_mat.children),
                        alpha_test=src_mat.alpha_test,
                    )
                else:
                    full_name = _build_cgf_mat_name(
                        src_mat.name,
                        src_mat.shader_name,
                        src_mat.surface_name,
                    )
                    override = (
                        scene_mats_by_full.get(full_name) or
                        scene_mats_by_source.get(src_mat.name)
                    )
                    tex_diffuse = (
                        _to_game_relative(override.get('tex_diffuse', ''), game_root_path)
                        if override else
                        (src_mat.tex_diffuse.name if src_mat.tex_diffuse else '')
                    )
                    tex_bump = (
                        _to_game_relative(override.get('tex_bump', ''), game_root_path)
                        if override else
                        (src_mat.tex_bump.name if src_mat.tex_bump else '')
                    )
                    tex_detail = (
                        _to_game_relative(override.get('tex_detail', ''), game_root_path)
                        if override else
                        (src_mat.tex_detail.name if src_mat.tex_detail else '')
                    )
                    material_full_name = (
                        (override.get('name') or full_name)
                        if override else
                        full_name
                    )
                    data, ver, _ = build_material_chunk(
                        cid,
                        material_full_name,
                        mat_type=1,
                        diffuse=override.get('diffuse', src_mat.diffuse) if override else src_mat.diffuse,
                        specular=override.get('specular', src_mat.specular) if override else src_mat.specular,
                        ambient=src_mat.ambient,
                        specular_level=src_mat.specular_level,
                        specular_shininess=src_mat.specular_shininess,
                        self_illumination=src_mat.self_illumination,
                        opacity=override.get('opacity', src_mat.opacity) if override else src_mat.opacity,
                        tex_diffuse=tex_diffuse,
                        tex_bump=tex_bump,
                        tex_detail=tex_detail,
                        flags=src_mat.flags,
                        alpha_test=src_mat.alpha_test,
                    )
                writer.add_chunk(CT_MAT, ver, cid, data)
                written_ids.add(cid)

        for cid, mat in all_standard_mats:
            if cid in written_ids:
                continue
            data, ver, _ = build_material_chunk(
                cid, mat.get('name', mat.get('source_name', 'material')),
                mat_type=1,
                diffuse=mat['diffuse'],
                specular=mat['specular'],
                opacity=mat['opacity'],
                tex_diffuse=_to_game_relative(mat.get('tex_diffuse', ''), game_root_path),
                tex_bump=_to_game_relative(mat.get('tex_bump', ''), game_root_path),
                tex_detail=_to_game_relative(mat.get('tex_detail', ''), game_root_path),
            )
            writer.add_chunk(CT_MAT, ver, cid, data)

    if arm_obj and export_skeleton and bone_data_list:
        data, ver, cid = build_bone_anim_chunk(
            allocate_chunk_id(source_meta.get('bone_anim_id')),
            bone_data_list
        )
        writer.add_chunk(CT_BANIM, ver, cid, data)

        data, ver, cid = build_bone_name_list_chunk(
            allocate_chunk_id(source_meta.get('bone_name_list_id')),
            bone_name_list
        )
        writer.add_chunk(CT_BNAMES, ver, cid, data)

    for p_type, p_ver, p_cid, p_data in _load_preserved_source_chunks(source_path):
        writer.add_chunk(p_type, p_ver, p_cid, p_data)

    writer.write(filepath)
    print(f"[CGF Export] Written: {filepath}")
    operator.report({'INFO'}, f"Exported {len(objects)} mesh(es) to {os.path.basename(filepath)}")
    return {'FINISHED'}


# ── CAF export ────────────────────────────────────────────────────────────────

def export_caf(operator, context, filepath, action=None, debug_export=False):
    """Export an animation Action to CAF."""

    arm_obj = context.active_object
    if arm_obj is None or arm_obj.type != 'ARMATURE':
        operator.report({'ERROR'}, "Select an armature first")
        return {'CANCELLED'}

    if action is None:
        action = _pick_export_action(arm_obj)
        if action is None:
            if getattr(arm_obj, 'animation_data', None) is None:
                operator.report({'ERROR'}, "No action found on armature")
                return {'CANCELLED'}

    action = _resolve_action_datablock(action, arm_obj)
    action_name = None
    if action is not None:
        action_name = getattr(action, 'name', None)
    if not action_name:
        ad = getattr(arm_obj, 'animation_data', None)
        if ad is not None and getattr(ad, 'action', None) is not None:
            action_name = getattr(ad.action, 'name', None)
    if not action_name:
        action_name = os.path.splitext(os.path.basename(filepath))[0]

    scene = context.scene
    fps = scene.render.fps
    ticks_per_frame = 160
    secs_per_tick = 1.0 / (fps * ticks_per_frame)

    writer = CGFWriter(is_anim=True)
    chunk_id = 0

    def next_id():
        nonlocal chunk_id
        chunk_id += 1
        return chunk_id

    # Source info
    import getpass, datetime
    data, ver, cid = build_source_info_chunk(
        next_id(),
        date=datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y"),
        user=getpass.getuser()
    )
    writer.add_chunk(CT_SRCINFO, ver, cid, data)

    # Timing
    frame_start, frame_end = _action_frame_range(action, scene)
    if debug_export:
        print(f"[CAF-EXPORT-DEBUG] action={action_name} frame_range=({frame_start}, {frame_end})")
    data, ver, cid = build_timing_chunk(
        next_id(), ticks_per_frame, secs_per_tick,
        frame_start * ticks_per_frame,
        frame_end   * ticks_per_frame
    )
    writer.add_chunk(CT_TIMING, ver, cid, data)

    # One controller chunk per bone that has animation
    bone_data_list, _ = extract_armature_data(arm_obj)

    for bd in bone_data_list:
        bone_name = bd['name']
        ctrl_id   = bd['ctrl_id']

        # Find F-Curves for this bone
        path_loc = f'pose.bones["{bone_name}"].location'
        path_rot = f'pose.bones["{bone_name}"].rotation_quaternion'

        fcurves = _action_fcurves(action)
        fc_loc = [fcurves.find(path_loc, index=i) if fcurves else None for i in range(3)]
        fc_rot = [fcurves.find(path_rot, index=i) if fcurves else None for i in range(4)]

        # Collect all keyframe times for this bone
        frame_set = set()
        for fc in fc_loc + fc_rot:
            if fc:
                for kp in fc.keyframe_points:
                    frame_set.add(int(kp.co[0]))

        if not frame_set:
            if action is None:
                frame_set = set(range(frame_start, frame_end + 1))
            elif getattr(arm_obj, 'animation_data', None) is not None:
                frame_set = set(range(frame_start, frame_end + 1))
        if not frame_set:
            continue

        if debug_export:
            loc_counts = [len(fc.keyframe_points) if fc else 0 for fc in fc_loc]
            rot_counts = [len(fc.keyframe_points) if fc else 0 for fc in fc_rot]
            print(
                f"[CAF-EXPORT-DEBUG] bone={bone_name} "
                f"loc_counts={loc_counts} rot_counts={rot_counts} "
                f"frame_set_count={len(frame_set)} first={min(frame_set)} last={max(frame_set)}"
            )

        # Sample pose at each keyframe
        keys = []
        orig_frame = scene.frame_current

        for frame in sorted(frame_set):
            scene.frame_set(frame)
            bpy.context.view_layer.update()

            pbone = arm_obj.pose.bones.get(bone_name)
            if pbone is None:
                continue

            # Position in armature local space → inches
            pos_bl = pbone.location
            pos_cry = (pos_bl.x * METERS_TO_INCHES,
                       pos_bl.y * METERS_TO_INCHES,
                       pos_bl.z * METERS_TO_INCHES)

            # Rotation as quaternion log
            rot_bl = pbone.rotation_quaternion
            rot_xyzw = (rot_bl.x, rot_bl.y, rot_bl.z, rot_bl.w)
            rot_log = quat_log(rot_xyzw)

            time_ticks = frame * ticks_per_frame
            keys.append((time_ticks, pos_cry, rot_log))

        scene.frame_set(orig_frame)

        if keys:
            data, ver, cid = build_controller_chunk_v827(next_id(), ctrl_id, keys)
            writer.add_chunk(CT_CTRL, ver, cid, data)

    writer.write(filepath)
    print(f"[CGF Export] CAF written: {filepath}")
    operator.report({'INFO'}, f"Exported action '{action_name}' to {os.path.basename(filepath)}")
    return {'FINISHED'}


# ── CAL export ────────────────────────────────────────────────────────────────

def export_cal(operator, context, filepath):
    """
    Export all actions on the active armature as CAF files
    and write a CAL list file.
    """
    arm_obj = context.active_object
    if arm_obj is None or arm_obj.type != 'ARMATURE':
        operator.report({'ERROR'}, "Select an armature first")
        return {'CANCELLED'}

    cal_dir  = os.path.dirname(filepath)
    cal_name = os.path.splitext(os.path.basename(filepath))[0]

    cal_lines = []
    exported  = 0

    for action in bpy.data.actions:
        # Check if this action has curves for bones of this armature
        if not _action_has_pose_fcurves(action):
            continue

        caf_name = action.name + ".caf"
        caf_path = os.path.join(cal_dir, caf_name)

        # Temporarily assign action
        _assign_action_to_armature(arm_obj, action)

        result = export_caf(operator, context, caf_path, action=action)
        if result == {'FINISHED'}:
            cal_lines.append(f"{action.name} {caf_name}")
            exported += 1

    # Write CAL file
    with open(filepath, 'w') as f:
        f.write('\n'.join(cal_lines))

    operator.report({'INFO'}, f"Exported {exported} animations to CAL: {os.path.basename(filepath)}")
    return {'FINISHED'}
