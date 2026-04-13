"""
cgf_builder.py — Converts parsed CGF/CAF chunks into Blender scene objects.

Coordinate system:
  CryEngine 1 / 3ds Max: Z-up, right-handed, units = inches
  Blender: Z-up, right-handed, units = meters
  Scale: INCHES_TO_METERS = 0.0254

Matrix convention:
  Max Matrix3 stores BASIS VECTORS AS ROWS → need .transposed() for Blender
  (Blender matrix_world stores basis vectors as COLUMNS)
"""

import bpy
import bmesh
import math
import os
import mathutils

from . import cry_chunk_reader
from .cry_chunk_reader import (CTRL_CRY_BONE, CTRL_LINEAR1, CTRL_LINEAR3, CTRL_LINEAR_Q,
                         CTRL_BEZIER1, CTRL_BEZIER3, CTRL_BEZIER_Q,
                         CTRL_TCB1, CTRL_TCB3, CTRL_TCBQ)

# ── Scale ─────────────────────────────────────────────────────────────────────
# 3ds Max default units = inches. 1 inch = 0.0254 meters.
INCHES_TO_METERS = 0.0254
V827_POS_AMPLITUDE = 1.0
V827_ROT_AMPLITUDE = 1.0
V827_SCALE_AMPLITUDE = 1.0
V827_ABSOLUTE_BONES = set()


# ── Coordinate helpers ────────────────────────────────────────────────────────

def cry_vec(v):
    """Scale a CryEngine/Max vector (inches) to Blender (meters)."""
    s = INCHES_TO_METERS
    return mathutils.Vector((v[0]*s, v[1]*s, v[2]*s))


def cry_matrix_to_blender(m44):
    """
    Convert CGF 4x4 row-major matrix (flat list of 16 floats) to Blender Matrix.
    Max stores basis vectors as ROWS → .transposed() makes them COLUMNS for Blender.
    Only translation is scaled (inches→meters); rotation/scale are dimensionless.
    """
    m = mathutils.Matrix((m44[0:4], m44[4:8], m44[8:12], m44[12:16])).transposed()
    m.translation *= INCHES_TO_METERS
    return m


def cry_matrix43_to_blender(m43):
    """Convert CGF 4x3 bone matrix (flat 12 floats) to Blender Matrix4x4."""
    rot = mathutils.Matrix((
        (m43[0], m43[1], m43[2]),
        (m43[3], m43[4], m43[5]),
        (m43[6], m43[7], m43[8]),
    )).transposed()
    m = rot.to_4x4()
    m.translation = mathutils.Vector((m43[9]*INCHES_TO_METERS,
                                       m43[10]*INCHES_TO_METERS,
                                       m43[11]*INCHES_TO_METERS))
    return m


def cry_bone_matrix43_to_blender(m43):
    """
    BoneInitialPos comes from a Max Matrix3-style row basis.
    The legacy non-transposed path masked upper-chain bones that were close
    to identity, but it breaks rotated bind bones in the weapon branch.
    Use the same row->column conversion as other Max matrices.
    """
    return cry_matrix43_to_blender(m43)


def _raw_matrix44(m44):
    return mathutils.Matrix((
        (m44[0], m44[1], m44[2], m44[3]),
        (m44[4], m44[5], m44[6], m44[7]),
        (m44[8], m44[9], m44[10], m44[11]),
        (m44[12], m44[13], m44[14], m44[15]),
    ))


def _raw_matrix43(m43):
    return mathutils.Matrix((
        (m43[0], m43[1], m43[2], 0.0),
        (m43[3], m43[4], m43[5], 0.0),
        (m43[6], m43[7], m43[8], 0.0),
        (m43[9], m43[10], m43[11], 1.0),
    ))


def _raw_max_matrix_to_blender(raw_m):
    rot = mathutils.Matrix((
        (float(raw_m[0][0]), float(raw_m[0][1]), float(raw_m[0][2])),
        (float(raw_m[1][0]), float(raw_m[1][1]), float(raw_m[1][2])),
        (float(raw_m[2][0]), float(raw_m[2][1]), float(raw_m[2][2])),
    )).transposed().to_4x4()
    rot.translation = mathutils.Vector((
        float(raw_m[3][0]) * INCHES_TO_METERS,
        float(raw_m[3][1]) * INCHES_TO_METERS,
        float(raw_m[3][2]) * INCHES_TO_METERS,
    ))
    return rot


def _compose_raw_max_trs(loc, rot):
    rot_rows = rot.to_matrix().transposed()
    return mathutils.Matrix((
        (float(rot_rows[0][0]), float(rot_rows[0][1]), float(rot_rows[0][2]), 0.0),
        (float(rot_rows[1][0]), float(rot_rows[1][1]), float(rot_rows[1][2]), 0.0),
        (float(rot_rows[2][0]), float(rot_rows[2][1]), float(rot_rows[2][2]), 0.0),
        (float(loc[0]), float(loc[1]), float(loc[2]), 1.0),
    ))


def _mul_point(matrix, vec3):
    # Base reconstruction path: keep the Blender-space bind result that was
    # giving correct arms/hands, then handle the remaining weapon flip as a
    # narrower follow-up instead of changing all skinned vertices at once.
    v = matrix @ mathutils.Vector((vec3[0], vec3[1], vec3[2], 1.0))
    return mathutils.Vector((v.x, v.y, v.z))


def _mul_point_max_row(matrix, vec3):
    # Legacy Max CryImporter path for rigid skin pieces:
    # point * Matrix3 (row-vector style).
    v = matrix.transposed() @ mathutils.Vector((vec3[0], vec3[1], vec3[2], 1.0))
    return mathutils.Vector((v.x, v.y, v.z))


def _matrix_str(m):
    try:
        rows = []
        for row in m:
            rows.append("(" + ",".join(f"{float(v):.4f}" for v in row) + ")")
        return "[" + " ".join(rows) + "]"
    except Exception:
        return str(m)


def _mesh_is_fully_rigid_skin(mesh_chunk):
    if not mesh_chunk.physique:
        return False
    for bone_links in mesh_chunk.physique:
        if len(bone_links.links) != 1:
            return False
    return True


def _build_skinned_bind_positions(mesh_chunk, archive, node_chunk):
    if not mesh_chunk.physique or not archive.bone_initial_pos_chunks:
        return {}

    obj_world_raw = mathutils.Matrix.Identity(4)
    if node_chunk and node_chunk.trans_matrix:
        try:
            obj_world_raw = _raw_matrix44(node_chunk.trans_matrix)
        except Exception:
            obj_world_raw = mathutils.Matrix.Identity(4)
    obj_world_inv_raw = obj_world_raw.inverted_safe()

    bind_mats_raw = {}
    if archive.bone_anim_chunks:
        for bone in archive.bone_anim_chunks[0].bones:
            init = archive.get_bone_initial_pos(bone.bone_id)
            if init:
                bind_mats_raw[bone.bone_id] = _raw_matrix43(init)

    rebuilt = {}
    for bone_links in mesh_chunk.physique:
        src_vid = bone_links.vertex_id
        if not bone_links.links:
            continue
        pos_raw = mathutils.Vector((0.0, 0.0, 0.0))
        total = 0.0
        for link in bone_links.links:
            bone_mx = bind_mats_raw.get(link.bone_id)
            if bone_mx is None:
                continue
            offset = mathutils.Vector((link.offset[0], link.offset[1], link.offset[2]))
            # Follow the Max CryImporter order:
            # local_pos = offset * bone_tm * inverse(node_tm)
            local_tm = bone_mx @ obj_world_inv_raw
            pos_raw += _mul_point_max_row(local_tm, offset) * float(link.blending)
            total += float(link.blending)
        if total > 1e-8:
            rebuilt[src_vid] = cry_vec(pos_raw / total)
    return rebuilt


def cry_quat(xyzw):
    """CryEngine quat (x,y,z,w) → Blender Quaternion (w,x,y,z)."""
    return mathutils.Quaternion((xyzw[3], xyzw[0], xyzw[1], xyzw[2]))


def quat_exp(rot_log):
    """
    Reconstruct quaternion from logarithm (x,y,z).
    Max: exp(quat rx ry rz 0)  ←  standard quaternion exponential map.
    """
    rx, ry, rz = rot_log
    theta = math.sqrt(rx*rx + ry*ry + rz*rz)
    if theta < 1e-10:
        return mathutils.Quaternion((1, 0, 0, 0))
    s = math.sin(theta) / theta
    return mathutils.Quaternion((math.cos(theta), rx*s, ry*s, rz*s))


def quat_exp_half(rot_log):
    rx, ry, rz = rot_log
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-10:
        return mathutils.Quaternion((1, 0, 0, 0))
    inv_theta = 1.0 / theta
    ax, ay, az = rx * inv_theta, ry * inv_theta, rz * inv_theta
    half = theta * 0.5
    s = math.sin(half)
    return mathutils.Quaternion((math.cos(half), ax * s, ay * s, az * s))


def _v827_local_from_key(key, *, half_rot=False, bone_name=None):
    pos_vec = _cry_v827_pos_to_blender((
        float(key.pos[0]) * INCHES_TO_METERS,
        float(key.pos[1]) * INCHES_TO_METERS,
        float(key.pos[2]) * INCHES_TO_METERS,
    ), bone_name=bone_name)
    quat_raw = quat_exp_half(tuple(float(v) for v in key.rot_log)) if half_rot else quat_exp(tuple(float(v) for v in key.rot_log))
    quat = _cry_v827_quat_to_blender(quat_raw)
    return _compose_trs_matrix(
        pos_vec,
        quat,
        mathutils.Vector((1.0, 1.0, 1.0)),
    )


def _scale_delta_trs(loc, rot, scale,
                     pos_factor=1.0, rot_factor=1.0, scale_factor=1.0):
    scaled_loc = mathutils.Vector(loc) * float(pos_factor)
    identity = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    try:
        scaled_rot = identity.slerp(rot, float(rot_factor))
    except Exception:
        scaled_rot = rot.copy()
    scaled_scale = mathutils.Vector((
        1.0 + (float(scale.x) - 1.0) * float(scale_factor),
        1.0 + (float(scale.y) - 1.0) * float(scale_factor),
        1.0 + (float(scale.z) - 1.0) * float(scale_factor),
    ))
    return scaled_loc, scaled_rot, scaled_scale


def _v_len3(v):
    return math.sqrt(float(v[0]) * float(v[0]) + float(v[1]) * float(v[1]) + float(v[2]) * float(v[2]))


def _v827_keys_close(a, b, pos_eps=0.2, rot_eps=0.02):
    try:
        dp = (
            float(a.pos[0]) - float(b.pos[0]),
            float(a.pos[1]) - float(b.pos[1]),
            float(a.pos[2]) - float(b.pos[2]),
        )
        dr = (
            float(a.rot_log[0]) - float(b.rot_log[0]),
            float(a.rot_log[1]) - float(b.rot_log[1]),
            float(a.rot_log[2]) - float(b.rot_log[2]),
        )
        return _v_len3(dp) <= pos_eps and _v_len3(dr) <= rot_eps
    except Exception:
        return False


def _effective_ctrl_keys(ctrl_chunk):
    keys = list(getattr(ctrl_chunk, 'keys', None) or [])
    if len(keys) < 3:
        return keys
    if hasattr(keys[0], 'rel_pos'):
        return keys

    trimmed = list(keys)

    # Preserve the leading v827 key. Root/root1 were effectively starting from
    # the second key, which shifted the whole chain and made frame 0 line up
    # with the next pose instead of the actual first pose from the file.
    # Only collapse duplicated terminal closing poses.
    while len(trimmed) >= 2 and _v827_keys_close(trimmed[-1], trimmed[-2]):
        trimmed.pop(-2)
    return trimmed


# ── Material ──────────────────────────────────────────────────────────────────

def _build_cgf_mat_name(name, shader_name, surface_name):
    """Reconstruct full CGF material name: 'name(shader)/surface'"""
    result = name
    if shader_name:
        result += f"({shader_name})"
    if surface_name:
        result += f"/{surface_name}"
    return result


def _round_tuple(values, digits=6):
    return tuple(round(float(v), digits) for v in values)


def _normalize_material_texture_key(tex_data, filepath, game_root_path=""):
    if not tex_data or not tex_data.name:
        return ""

    resolved = _find_texture(tex_data.name, filepath, game_root_path)
    if resolved:
        return os.path.normcase(os.path.abspath(resolved))

    raw = tex_data.name.replace('\\', os.sep).replace('/', os.sep)
    return os.path.normcase(os.path.splitext(raw)[0])


def _material_signature(mat_chunk, filepath, game_root_path=""):
    return (
        (mat_chunk.shader_name or '').strip().lower(),
        (mat_chunk.surface_name or '').strip().lower(),
        _normalize_material_texture_key(mat_chunk.tex_diffuse, filepath, game_root_path),
        _normalize_material_texture_key(mat_chunk.tex_bump, filepath, game_root_path),
        _normalize_material_texture_key(mat_chunk.tex_detail, filepath, game_root_path),
        _normalize_material_texture_key(mat_chunk.tex_specular, filepath, game_root_path),
        _normalize_material_texture_key(mat_chunk.tex_reflection, filepath, game_root_path),
        _round_tuple(mat_chunk.diffuse),
        _round_tuple(mat_chunk.specular),
        _round_tuple(mat_chunk.ambient),
        round(float(mat_chunk.specular_level), 6),
        round(float(mat_chunk.specular_shininess), 6),
        round(float(mat_chunk.self_illumination), 6),
        round(float(mat_chunk.opacity), 6),
        round(float(mat_chunk.alpha_test), 6),
        int(mat_chunk.type),
        int(mat_chunk.flags),
    )


def _is_nodraw_material(mat_chunk):
    shader = (mat_chunk.shader_name or '').strip().lower()
    surface = (mat_chunk.surface_name or '').strip().lower()
    diffuse = ((getattr(mat_chunk.tex_diffuse, 'name', '') or '')
               .replace('\\', '/').strip().lower())

    if shader in {'nodraw', 'no_draw'}:
        return True
    if surface in {'mat_obstruct', 'mat_nodraw'}:
        return True
    if diffuse.endswith('/nodraw.dds') or diffuse.endswith('common/nodraw.dds'):
        return True
    return False


def _global_standard_material_chunks(archive):
    result = []
    for mc in archive.material_chunks:
        if mc.type == 2:
            continue
        result.append(mc)
    return result


def _mesh_is_collision_like(mesh_chunk, archive):
    global_chunks = _global_standard_material_chunks(archive)
    face_mat_ids = sorted({face.mat_id for face in mesh_chunk.faces if face.mat_id >= 0})
    if not face_mat_ids:
        return False

    resolved = []
    for face_mat_id in face_mat_ids:
        if face_mat_id >= len(global_chunks):
            return False
        resolved.append(global_chunks[face_mat_id])

    return bool(resolved) and all(_is_nodraw_material(mat) for mat in resolved)


def _global_collision_material_ids(archive):
    result = set()
    for idx, mat in enumerate(_global_standard_material_chunks(archive)):
        if _is_nodraw_material(mat):
            result.add(idx)
    return result


def _uses_diffuse_alpha_as_opacity(mat_chunk):
    shader = (mat_chunk.shader_name or '').strip().lower()
    diffuse = ((getattr(mat_chunk.tex_diffuse, 'name', '') or '')
               .replace('\\', '/').strip().lower())

    if 'glossalpha' in shader:
        return False
    if shader in {'glass', 'vegetation'}:
        return True
    if mat_chunk.alpha_test > 0.0:
        return True
    if any(token in diffuse for token in ('chainlink', 'fence', 'grate', 'wire', 'mesh', 'net')):
        return True
    return False


def _configure_diffuse_image_alpha(img, mat_chunk):
    if img is None or not hasattr(img, 'alpha_mode'):
        return
    try:
        if _uses_diffuse_alpha_as_opacity(mat_chunk):
            if img.alpha_mode == 'NONE':
                img.alpha_mode = 'STRAIGHT'
        else:
            img.alpha_mode = 'NONE'
    except Exception:
        pass


def _set_input(node, *names, value):
    for name in names:
        if name in node.inputs:
            try: node.inputs[name].default_value = value
            except Exception: pass
            return


def _assign_action_to_armature(arm_obj, action):
    if arm_obj is None or action is None:
        return False
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    ad = arm_obj.animation_data
    try:
        ad.action = action
        if getattr(ad, 'action', None) == action:
            return True
    except Exception:
        pass
    slot = getattr(ad, 'action_slot', None)
    if slot is not None:
        try:
            slot.action = action
            if getattr(slot, 'action', None) == action:
                return True
        except Exception:
            pass
    return False


def _clear_action_from_armature(arm_obj):
    if arm_obj is None or arm_obj.animation_data is None:
        return
    ad = arm_obj.animation_data
    try:
        ad.action = None
    except Exception:
        pass
    slot = getattr(ad, 'action_slot', None)
    if slot is not None:
        try:
            slot.action = None
        except Exception:
            pass


def _assign_action_to_object(obj, action):
    if obj is None or action is None:
        return False
    if obj.animation_data is None:
        obj.animation_data_create()
    try:
        obj.animation_data.action = action
        return getattr(obj.animation_data, 'action', None) == action
    except Exception:
        return False


def _ensure_action_fcurve(action, datablock, data_path, index):
    if action is None:
        return None
    ensure_for_datablock = getattr(action, "fcurve_ensure_for_datablock", None)
    if ensure_for_datablock is not None and datablock is not None:
        try:
            return ensure_for_datablock(datablock, data_path, index=index)
        except Exception:
            pass

    fcurves = getattr(action, 'fcurves', None)
    if fcurves is None:
        return None
    fc = fcurves.find(data_path, index=index)
    if fc is None:
        fc = fcurves.new(data_path, index=index)
    return fc


def _action_fcurves_view(action, datablock=None):
    if action is None:
        return None
    fcurves = getattr(action, 'fcurves', None)
    if fcurves is not None:
        return fcurves
    if datablock is None:
        return None
    ad = getattr(datablock, "animation_data", None)
    slot = getattr(ad, "action_slot", None) if ad is not None else None
    if slot is None:
        return None
    layers = getattr(action, "layers", None)
    if not layers:
        return None
    try:
        strip = layers[0].strips[0]
        channelbag = strip.channelbag(slot)
        return getattr(channelbag, "fcurves", None)
    except Exception:
        return None


def _insert_action_key(action, datablock, data_path, index, frame, value):
    fc = _ensure_action_fcurve(action, datablock, data_path, index)
    if fc is None:
        return
    kp = fc.keyframe_points
    kp.add(1)
    kp[-1].co = (float(frame), float(value))
    try:
        kp[-1].interpolation = 'LINEAR'
    except Exception:
        pass
    try:
        fc.update()
    except Exception:
        pass


def _insert_posebone_keys_into_action(action, pbone, frame):
    if action is None or pbone is None:
        return
    pbone.rotation_mode = 'QUATERNION'
    loc_path = pbone.path_from_id("location")
    rot_path = pbone.path_from_id("rotation_quaternion")
    scale_path = pbone.path_from_id("scale")
    for index, value in enumerate(pbone.location):
        _insert_action_key(action, pbone.id_data, loc_path, index, frame, value)
    for index, value in enumerate(pbone.rotation_quaternion):
        _insert_action_key(action, pbone.id_data, rot_path, index, frame, value)
    for index, value in enumerate(pbone.scale):
        _insert_action_key(action, pbone.id_data, scale_path, index, frame, value)


def _insert_posebone_basis_keys_into_action(action, pbone, frame, loc, rot, scale):
    if action is None or pbone is None:
        return
    pbone.rotation_mode = 'QUATERNION'
    loc_path = pbone.path_from_id("location")
    rot_path = pbone.path_from_id("rotation_quaternion")
    scale_path = pbone.path_from_id("scale")
    for index, value in enumerate(loc):
        _insert_action_key(action, pbone.id_data, loc_path, index, frame, value)
    for index, value in enumerate(rot):
        _insert_action_key(action, pbone.id_data, rot_path, index, frame, value)
    for index, value in enumerate(scale):
        _insert_action_key(action, pbone.id_data, scale_path, index, frame, value)


def _set_object_from_anim_local(obj, pos=None, quat=None, scale=None):
    if obj is None:
        return
    obj.rotation_mode = 'QUATERNION'
    if pos is not None:
        obj.location = mathutils.Vector(pos)
    if quat is not None:
        obj.rotation_quaternion = quat.copy()
    if scale is not None:
        obj.scale = mathutils.Vector(scale)


def _keyframe_object_transform(obj, frame):
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    obj.keyframe_insert(data_path="scale", frame=frame)


def _compose_trs_matrix(loc, rot, scale):
    loc_m = mathutils.Matrix.Translation(loc)
    rot_m = rot.to_matrix().to_4x4()
    scale_m = mathutils.Matrix.Diagonal((scale.x, scale.y, scale.z, 1.0))
    return loc_m @ rot_m @ scale_m


def _fmt_matrix4(m):
    if m is None:
        return "None"
    rows = []
    for row in m:
        rows.append(
            "(" + ",".join(f"{float(v):.4f}" for v in row) + ")"
        )
    return "[" + " ".join(rows) + "]"


def _cry_anim_pos_to_blender(pos):
    return mathutils.Vector((pos[0], -pos[1], -pos[2]))


def _cry_anim_quat_to_blender(quat):
    return mathutils.Quaternion((quat.w, quat.x, -quat.y, -quat.z))


def _cry_v827_pos_to_blender(pos, bone_name=None):
    # v827 CryBone keys do not follow the same axis contract as the generic
    # rel_pos/rel_quat animation path. For the root/root1 chain, the stable
    # Max-like local mapping is:
    #   Max/Cry (x, y, z) -> Blender (-z, y, x)
    if bone_name == "root1":
        return mathutils.Vector((pos[0], pos[1], 0.0))
    return mathutils.Vector((-pos[2], pos[1], pos[0]))


def _cry_v827_quat_to_blender(quat):
    # Keep the v827 position remap, but do not mirror the quaternion through
    # the generic anim path. For root/root1 the older raw exponential-map
    # orientation is closer to Max than the fully remapped variant.
    return quat.copy()


def _bone_rest_local_matrix(pbone):
    bone = getattr(pbone, 'bone', None)
    if bone is None:
        return mathutils.Matrix.Identity(4)
    try:
        if bone.parent:
            return bone.parent.matrix_local.inverted() @ bone.matrix_local
        return bone.matrix_local.copy()
    except Exception:
        return mathutils.Matrix.Identity(4)


def _bone_name_map(archive):
    names = archive.bone_name_list_chunks[0].name_list if archive.bone_name_list_chunks else []
    result = {}
    if not archive.bone_anim_chunks:
        return result
    for bone in archive.bone_anim_chunks[0].bones:
        bid = bone.bone_id
        result[bid] = names[bid] if bid < len(names) else (bone.name or f"Bone_{bid}")
    return result


def _build_cry_bind_pose(archive, arm_obj=None):
    result = {}
    if not archive.bone_anim_chunks or not archive.bone_anim_chunks[0].bones:
        return result

    armature_world = arm_obj.matrix_world.copy() if arm_obj is not None else mathutils.Matrix.Identity(4)
    armature_world_inv = armature_world.inverted_safe()
    name_map = _bone_name_map(archive)
    bones = archive.bone_anim_chunks[0].bones

    world_by_id = {}
    for bone in bones:
        init = archive.get_bone_initial_pos(bone.bone_id)
        if init is None:
            continue
        world_by_id[bone.bone_id] = armature_world_inv @ cry_bone_matrix43_to_blender(init)

    for bone in bones:
        bid = bone.bone_id
        world_m = world_by_id.get(bid)
        if world_m is None:
            continue
        parent_name = None
        if bone.parent_id >= 0:
            parent_world = world_by_id.get(bone.parent_id)
            local_m = parent_world.inverted_safe() @ world_m if parent_world is not None else world_m.copy()
            parent_name = name_map.get(bone.parent_id)
        else:
            local_m = world_m.copy()
        result[name_map[bid]] = {
            "bone_id": bid,
            "parent_id": bone.parent_id,
            "parent_name": parent_name,
            "bind_world": world_m,
            "bind_local": local_m,
        }
    return result


def _build_cry_bind_pose_raw(archive):
    result = {}
    if not archive.bone_anim_chunks or not archive.bone_anim_chunks[0].bones:
        return result

    name_map = _bone_name_map(archive)
    bones = archive.bone_anim_chunks[0].bones
    world_by_id = {}
    for bone in bones:
        init = archive.get_bone_initial_pos(bone.bone_id)
        if init is None:
            continue
        world_by_id[bone.bone_id] = _raw_matrix43(init)

    for bone in bones:
        bid = bone.bone_id
        world_m = world_by_id.get(bid)
        if world_m is None:
            continue
        parent_name = None
        if bone.parent_id >= 0:
            parent_world = world_by_id.get(bone.parent_id)
            local_m = world_m @ parent_world.inverted_safe() if parent_world is not None else world_m.copy()
            parent_name = name_map.get(bone.parent_id)
        else:
            local_m = world_m.copy()
        result[name_map[bid]] = {
            "bone_id": bid,
            "parent_id": bone.parent_id,
            "parent_name": parent_name,
            "bind_world": world_m,
            "bind_local": local_m,
        }
    return result


def _raw_max_pos_from_key(key):
    if hasattr(key, 'rel_pos'):
        return mathutils.Vector((float(key.rel_pos[0]), float(key.rel_pos[1]), float(key.rel_pos[2])))
    return mathutils.Vector((float(key.pos[0]), float(key.pos[1]), float(key.pos[2])))


def _raw_max_quat_from_key(key):
    if hasattr(key, 'rel_quat'):
        return cry_quat(key.rel_quat)
    return quat_exp(tuple(float(v) for v in key.rot_log))


def _raw_max_local_from_key(key):
    return _compose_raw_max_trs(_raw_max_pos_from_key(key), _raw_max_quat_from_key(key))


def _evaluate_raw_max_controller_at_time(ctrl_chunk, time_tick, default_local=None):
    keys = _effective_ctrl_keys(ctrl_chunk)
    if not keys:
        return default_local.copy() if default_local is not None else mathutils.Matrix.Identity(4)
    if len(keys) == 1 or time_tick <= keys[0].time:
        return _raw_max_local_from_key(keys[0])
    if time_tick >= keys[-1].time:
        return _raw_max_local_from_key(keys[-1])

    prev_key = keys[0]
    next_key = keys[-1]
    for idx in range(1, len(keys)):
        if time_tick <= keys[idx].time:
            prev_key = keys[idx - 1]
            next_key = keys[idx]
            break

    if prev_key.time == next_key.time:
        return _raw_max_local_from_key(prev_key)

    alpha = (time_tick - prev_key.time) / (next_key.time - prev_key.time)
    loc = _raw_max_pos_from_key(prev_key).lerp(_raw_max_pos_from_key(next_key), alpha)
    rot = _raw_max_quat_from_key(prev_key).slerp(_raw_max_quat_from_key(next_key), alpha)
    return _compose_raw_max_trs(loc, rot)


def _evaluate_cry_skeleton_pose_raw(bind_pose_raw, ctrl_by_bone, time_tick):
    pose = {}
    ordered = sorted(bind_pose_raw.items(), key=lambda kv: kv[1]["bone_id"])
    for bone_name, item in ordered:
        local_m = item["bind_local"].copy()
        ctrl_chunk = ctrl_by_bone.get(bone_name)
        if ctrl_chunk is not None:
            local_m = _evaluate_raw_max_controller_at_time(ctrl_chunk, time_tick, default_local=item["bind_local"])
        parent_name = item["parent_name"]
        if parent_name and parent_name in pose:
            world_m = local_m @ pose[parent_name]["world"]
        else:
            world_m = local_m.copy()
        pose[bone_name] = {
            "local": local_m,
            "world": world_m,
            "bind_local": item["bind_local"],
            "bind_world": item["bind_world"],
            "parent_name": parent_name,
            "bone_id": item["bone_id"],
        }
    return pose


def _pose_basis_from_anim_local(pbone, pos=None, quat=None, scale=None):
    rest_local = _bone_rest_local_matrix(pbone)
    rest_loc, rest_rot, rest_scale = rest_local.decompose()

    loc = mathutils.Vector(pos) if pos is not None else rest_loc
    rot = quat.copy() if quat is not None else rest_rot.copy()
    scl = mathutils.Vector(scale) if scale is not None else rest_scale

    anim_local = _compose_trs_matrix(loc, rot, scl)
    delta = rest_local.inverted_safe() @ anim_local
    d_loc, d_rot, d_scale = delta.decompose()
    return d_loc, d_rot, d_scale, delta


def _set_pose_from_anim_local(pbone, pos=None, quat=None, scale=None):
    delta = None
    d_loc = d_rot = d_scale = None
    try:
        d_loc, d_rot, d_scale, delta = _pose_basis_from_anim_local(pbone, pos, quat, scale)
        pbone.rotation_mode = 'QUATERNION'
        pbone.matrix_basis = delta
    except Exception:
        try:
            if delta is not None:
                pbone.matrix_basis = delta
            elif d_loc is not None and d_rot is not None and d_scale is not None:
                pbone.location = d_loc
                pbone.rotation_quaternion = d_rot
                pbone.scale = d_scale
            else:
                raise
        except Exception:
            pbone.location = (0.0, 0.0, 0.0)
            pbone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pbone.scale = (1.0, 1.0, 1.0)


def _quat_in_rest_basis(pbone, quat):
    if pbone is None or quat is None:
        return quat
    try:
        rest_local = _bone_rest_local_matrix(pbone)
        _, rest_rot, _ = rest_local.decompose()
        return rest_rot @ quat @ rest_rot.inverted()
    except Exception:
        return quat


def _set_pose_from_anim_basis(pbone, pos=None, quat=None, scale=None):
    if pbone is None:
        return
    loc = mathutils.Vector(pos) if pos is not None else mathutils.Vector((0.0, 0.0, 0.0))
    rot = quat.copy() if quat is not None else mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    scl = mathutils.Vector(scale) if scale is not None else mathutils.Vector((1.0, 1.0, 1.0))
    try:
        pbone.rotation_mode = 'QUATERNION'
        pbone.location = loc
        pbone.rotation_quaternion = rot
        pbone.scale = scl
    except Exception:
        try:
            pbone.matrix_basis = _compose_trs_matrix(loc, rot, scl)
        except Exception:
            pass


def _set_pose_from_anim_pose_matrix(pbone, local_m):
    if pbone is None or local_m is None:
        return
    try:
        rest_local = _bone_rest_local_matrix(pbone)
        pbone.matrix_basis = rest_local.inverted_safe() @ local_m
        return
    except Exception:
        pass
    try:
        rest_local = _bone_rest_local_matrix(pbone)
        pbone.matrix_basis = rest_local.inverted_safe() @ local_m
    except Exception:
        pass


def _ensure_cry_proxy_collection(arm_obj):
    if arm_obj is None:
        return None
    collection_name = f"{arm_obj.name}_CRYPOSE_PROXY"
    coll = bpy.data.collections.get(collection_name)
    if coll is None:
        coll = bpy.data.collections.new(collection_name)
        parent_coll = arm_obj.users_collection[0] if arm_obj.users_collection else bpy.context.scene.collection
        parent_coll.children.link(coll)
    coll.hide_viewport = True
    try:
        coll.hide_render = True
    except Exception:
        pass
    return coll


def _clear_cry_proxy_data(arm_obj):
    if arm_obj is None:
        return
    try:
        for pbone in getattr(arm_obj.pose, "bones", []) or []:
            removable = [con for con in pbone.constraints if con.name == "CRYPOSE_COPY"]
            for con in removable:
                pbone.constraints.remove(con)
    except Exception:
        pass

    collection_name = f"{arm_obj.name}_CRYPOSE_PROXY"
    coll = bpy.data.collections.get(collection_name)
    if coll is None:
        return

    try:
        proxy_objects = list(coll.objects)
    except Exception:
        proxy_objects = []

    for obj in proxy_objects:
        try:
            coll.objects.unlink(obj)
        except Exception:
            pass
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass

    try:
        parent_coll = arm_obj.users_collection[0] if arm_obj.users_collection else bpy.context.scene.collection
        if coll.name in parent_coll.children:
            parent_coll.children.unlink(coll)
    except Exception:
        pass
    try:
        bpy.data.collections.remove(coll)
    except Exception:
        pass


def _ensure_cry_proxy_targets(arm_obj, bone_names):
    proxies = {}
    coll = _ensure_cry_proxy_collection(arm_obj)
    if coll is None:
        return proxies
    for bone_name in bone_names:
        obj_name = f"{arm_obj.name}__CRYPROXY__{bone_name}"
        proxy = bpy.data.objects.get(obj_name)
        if proxy is None:
            proxy = bpy.data.objects.new(obj_name, None)
            proxy.empty_display_type = 'PLAIN_AXES'
            proxy.empty_display_size = 0.01
        if proxy.name not in coll.objects:
            coll.objects.link(proxy)
        proxy.parent = None
        proxy.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        proxy.hide_viewport = True
        proxy.hide_select = True
        try:
            proxy.hide_render = True
        except Exception:
            pass
        proxies[bone_name] = proxy
    return proxies


def _ensure_cry_proxy_constraint(pbone, proxy_obj):
    if pbone is None or proxy_obj is None:
        return
    con = next((c for c in pbone.constraints if c.name == "CRYPOSE_COPY"), None)
    if con is None:
        con = pbone.constraints.new('COPY_TRANSFORMS')
        con.name = "CRYPOSE_COPY"
    con.target = proxy_obj
    try:
        con.target_space = 'WORLD'
        con.owner_space = 'WORLD'
    except Exception:
        pass
    con.mute = False


def _restore_mesh_armature_playback(arm_obj):
    if arm_obj is None:
        return
    for obj in _mesh_objects_for_armature(arm_obj):
        _remove_cry_preview_shape_keys(obj)
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object == arm_obj:
                mod.show_viewport = True
                mod.show_render = True


def _drive_armature_from_proxies(arm_obj, proxies):
    if arm_obj is None or arm_obj.pose is None or not proxies:
        return
    for bone_name, proxy in proxies.items():
        pbone = arm_obj.pose.bones.get(bone_name)
        if pbone is None:
            continue
        _ensure_cry_proxy_constraint(pbone, proxy)


def _set_proxy_world_matrix(proxy_obj, world_m):
    if proxy_obj is None or world_m is None:
        return
    proxy_obj.rotation_mode = 'QUATERNION'
    proxy_obj.matrix_world = world_m


def _set_proxy_local_matrix(proxy_obj, local_m):
    if proxy_obj is None or local_m is None:
        return
    proxy_obj.rotation_mode = 'QUATERNION'
    loc, rot, scl = local_m.decompose()
    proxy_obj.location = loc
    proxy_obj.rotation_quaternion = rot
    proxy_obj.scale = scl


def _set_pose_from_anim_direct(pbone, pos=None, quat=None, scale=None):
    if pbone is None:
        return
    try:
        if pos is not None:
            pbone.location = mathutils.Vector(pos)
        if quat is not None:
            pbone.rotation_mode = 'QUATERNION'
            pbone.rotation_quaternion = quat.copy()
        if scale is not None:
            pbone.scale = mathutils.Vector(scale)
    except Exception:
        pass


def _reset_pose_bones_to_rest(arm_obj, bone_names=None):
    if arm_obj is None or arm_obj.pose is None:
        return
    names = bone_names or [pb.name for pb in arm_obj.pose.bones]
    for bone_name in names:
        pbone = arm_obj.pose.bones.get(bone_name)
        if pbone is None:
            continue
        pbone.rotation_mode = 'QUATERNION'
        pbone.location = (0.0, 0.0, 0.0)
        pbone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pbone.scale = (1.0, 1.0, 1.0)


def _crybone_local_transform_from_key(ctrl_chunk, key):
    # Keep one Max-compatible transform path for both v826 (rel_pos/rel_quat)
    # and v827 (pos/rotLog): build raw Max local PRS, then convert matrix
    # to Blender. This mirrors the legacy Max CryImporter semantics and avoids
    # per-format axis hacks that can distort child bone chains.
    return _raw_max_matrix_to_blender(_raw_max_local_from_key(key))


def _v827_hybrid_local_transform(bone_name, ctrl_chunk, key, bind_local, first_key=None):
    if key is None:
        return bind_local.copy()
    bind_loc, _, _ = bind_local.decompose()
    raw_pos = cry_vec(tuple(float(v) for v in key.pos))
    if bone_name in {'root', 'root1', 'Bone18', 'Bone20'}:
        return None
    if bone_name == 'Bone19':
        if first_key is None:
            first_key = key
        first_raw = quat_exp_half(tuple(float(v) for v in first_key.rot_log))
        raw_rot = quat_exp_half(tuple(float(v) for v in key.rot_log))
        return _compose_trs_matrix(raw_pos, first_raw.inverted() @ raw_rot, mathutils.Vector((1.0, 1.0, 1.0)))
    return None


def _evaluate_v827_hybrid_at_time(bone_name, ctrl_chunk, time_tick, bind_local):
    keys = getattr(ctrl_chunk, 'keys', None) or []
    if not keys:
        return bind_local.copy()
    if len(keys) == 1:
        hybrid = _v827_hybrid_local_transform(bone_name, ctrl_chunk, keys[0], bind_local, keys[0])
        return hybrid if hybrid is not None else _crybone_local_transform_from_key(ctrl_chunk, keys[0])

    if time_tick <= keys[0].time:
        hybrid = _v827_hybrid_local_transform(bone_name, ctrl_chunk, keys[0], bind_local, keys[0])
        return hybrid if hybrid is not None else _crybone_local_transform_from_key(ctrl_chunk, keys[0])
    if time_tick >= keys[-1].time:
        hybrid = _v827_hybrid_local_transform(bone_name, ctrl_chunk, keys[-1], bind_local, keys[0])
        return hybrid if hybrid is not None else _crybone_local_transform_from_key(ctrl_chunk, keys[-1])

    prev_key = keys[0]
    next_key = keys[-1]
    for idx in range(1, len(keys)):
        if time_tick <= keys[idx].time:
            prev_key = keys[idx - 1]
            next_key = keys[idx]
            break

    prev_m = _v827_hybrid_local_transform(bone_name, ctrl_chunk, prev_key, bind_local, keys[0])
    next_m = _v827_hybrid_local_transform(bone_name, ctrl_chunk, next_key, bind_local, keys[0])
    if prev_m is None or next_m is None:
        return _evaluate_crybone_controller_at_time(ctrl_chunk, time_tick, default_local=bind_local)

    if prev_key.time == next_key.time:
        return prev_m

    prev_loc, prev_rot, prev_scale = prev_m.decompose()
    next_loc, next_rot, next_scale = next_m.decompose()
    alpha = (time_tick - prev_key.time) / (next_key.time - prev_key.time)
    loc = prev_loc.lerp(next_loc, alpha)
    rot = prev_rot.slerp(next_rot, alpha)
    scl = prev_scale.lerp(next_scale, alpha)
    return _compose_trs_matrix(loc, rot, scl)


def _evaluate_v827_absolute_at_time(ctrl_chunk, time_tick, *, half_rot=False, bone_name=None):
    keys = _effective_ctrl_keys(ctrl_chunk)
    if not keys:
        return mathutils.Matrix.Identity(4)
    if len(keys) == 1:
        return _v827_local_from_key(keys[0], half_rot=half_rot, bone_name=bone_name)
    if time_tick <= keys[0].time:
        return _v827_local_from_key(keys[0], half_rot=half_rot, bone_name=bone_name)
    if time_tick >= keys[-1].time:
        return _v827_local_from_key(keys[-1], half_rot=half_rot, bone_name=bone_name)

    prev_key = keys[0]
    next_key = keys[-1]
    for idx in range(1, len(keys)):
        if time_tick <= keys[idx].time:
            prev_key = keys[idx - 1]
            next_key = keys[idx]
            break

    prev_m = _v827_local_from_key(prev_key, half_rot=half_rot, bone_name=bone_name)
    next_m = _v827_local_from_key(next_key, half_rot=half_rot, bone_name=bone_name)
    if prev_key.time == next_key.time:
        return prev_m

    prev_loc, prev_rot, prev_scale = prev_m.decompose()
    next_loc, next_rot, next_scale = next_m.decompose()
    alpha = (time_tick - prev_key.time) / (next_key.time - prev_key.time)
    loc = prev_loc.lerp(next_loc, alpha)
    rot = prev_rot.slerp(next_rot, alpha)
    scl = prev_scale.lerp(next_scale, alpha)
    return _compose_trs_matrix(loc, rot, scl)


def _evaluate_crybone_controller_at_time(ctrl_chunk, time_tick, default_local=None, bone_name=None, evaluator_mode="DEFAULT"):
    keys = _effective_ctrl_keys(ctrl_chunk)
    if not keys:
        return default_local.copy() if default_local is not None else mathutils.Matrix.Identity(4)

    mode = str(evaluator_mode or "DEFAULT").strip().upper()
    if mode == "RAWMAX":
        return _raw_max_matrix_to_blender(
            _evaluate_raw_max_controller_at_time(ctrl_chunk, time_tick)
        )

    if len(keys) == 1:
        local_m = _crybone_local_transform_from_key(ctrl_chunk, keys[0])
        return local_m

    if time_tick <= keys[0].time:
        local_m = _crybone_local_transform_from_key(ctrl_chunk, keys[0])
        return local_m
    if time_tick >= keys[-1].time:
        local_m = _crybone_local_transform_from_key(ctrl_chunk, keys[-1])
        return local_m

    prev_key = keys[0]
    next_key = keys[-1]
    for idx in range(1, len(keys)):
        if time_tick <= keys[idx].time:
            prev_key = keys[idx - 1]
            next_key = keys[idx]
            break

    if prev_key.time == next_key.time:
        local_m = _crybone_local_transform_from_key(ctrl_chunk, prev_key)
        return local_m

    prev_m = _crybone_local_transform_from_key(ctrl_chunk, prev_key)
    next_m = _crybone_local_transform_from_key(ctrl_chunk, next_key)
    prev_loc, prev_rot, prev_scale = prev_m.decompose()
    next_loc, next_rot, next_scale = next_m.decompose()
    alpha = (time_tick - prev_key.time) / (next_key.time - prev_key.time)

    loc = prev_loc.lerp(next_loc, alpha)
    rot = prev_rot.slerp(next_rot, alpha)
    scl = prev_scale.lerp(next_scale, alpha)
    local_m = _compose_trs_matrix(loc, rot, scl)
    return local_m


def _evaluate_cry_skeleton_pose(bind_pose, ctrl_by_bone, time_tick, evaluator_mode="DEFAULT"):
    pose = {}
    ordered = sorted(bind_pose.items(), key=lambda kv: kv[1]["bone_id"])
    for bone_name, item in ordered:
        default_local = item["bind_local"]
        ctrl_chunk = ctrl_by_bone.get(bone_name)
        local_m = default_local.copy()
        if ctrl_chunk is not None:
            local_m = _evaluate_crybone_controller_at_time(
                ctrl_chunk, time_tick, default_local=default_local, bone_name=bone_name, evaluator_mode=evaluator_mode
            )
        parent_name = item["parent_name"]
        if parent_name and parent_name in pose:
            world_m = pose[parent_name]["world"] @ local_m
        else:
            world_m = local_m.copy()
        pose[bone_name] = {
            "local": local_m,
            "world": world_m,
            "bind_local": item["bind_local"],
            "bind_world": item["bind_world"],
            "parent_name": parent_name,
            "bone_id": item["bone_id"],
        }
    return pose


def _apply_crybone_pose_at_time(arm_obj, ctrl_by_bone, time_tick, keyframe_frame=None, action=None, evaluator_mode="DEFAULT"):
    if arm_obj is None or arm_obj.pose is None:
        return

    bone_names = list(ctrl_by_bone.keys())
    _reset_pose_bones_to_rest(arm_obj, bone_names)
    bind_pose = None
    cry_pose = None
    geom_archive = getattr(arm_obj, "_cry_geom_archive_ref", None)
    if geom_archive is not None:
        try:
            bind_pose = _build_cry_bind_pose(geom_archive, arm_obj=arm_obj)
            if bind_pose:
                cry_pose = _evaluate_cry_skeleton_pose(bind_pose, ctrl_by_bone, time_tick, evaluator_mode=evaluator_mode)
        except Exception:
            bind_pose = None
            cry_pose = None
    ordered_bone_names = bone_names
    active_pose = cry_pose
    if active_pose:
        ordered_bone_names = [name for name, _ in sorted(
            active_pose.items(), key=lambda kv: kv[1]["bone_id"]
        ) if name in ctrl_by_bone]
        ordered_bone_names.extend(
            name for name in bone_names if name not in ordered_bone_names
        )
    for bone_name in ordered_bone_names:
        pbone = arm_obj.pose.bones.get(bone_name)
        if pbone is None:
            continue
        ctrl_chunk = ctrl_by_bone[bone_name]
        if cry_pose and bone_name in cry_pose:
            cry_local = cry_pose[bone_name]["local"].copy()
            cry_bind_local = cry_pose[bone_name]["bind_local"].copy()
            blender_rest_local = _bone_rest_local_matrix(pbone)
            try:
                local_delta = cry_bind_local.inverted_safe() @ cry_local
                local_m = blender_rest_local @ local_delta
            except Exception:
                local_m = cry_local
        else:
            local_m = _evaluate_crybone_controller_at_time(
                ctrl_chunk, time_tick, default_local=_bone_rest_local_matrix(pbone), bone_name=bone_name, evaluator_mode=evaluator_mode
            )
        _set_pose_from_anim_pose_matrix(pbone, local_m)

    try:
        bpy.context.view_layer.update()
    except Exception:
        pass

    if keyframe_frame is not None:
        for bone_name in bone_names:
            pbone = arm_obj.pose.bones.get(bone_name)
            if pbone is None:
                continue
            if action is not None:
                _insert_posebone_keys_into_action(action, pbone, keyframe_frame)
            else:
                pbone.rotation_mode = 'QUATERNION'
                pbone.keyframe_insert(data_path="location", frame=keyframe_frame)
                pbone.keyframe_insert(data_path="rotation_quaternion", frame=keyframe_frame)
                pbone.keyframe_insert(data_path="scale", frame=keyframe_frame)


def _debug_log_crybone_frame(arm_obj, ctrl_by_bone, time_tick, frame, evaluator_mode="DEFAULT"):
    focus = {
        "root", "root1", "weapon", "spitfire", "local_hs_weapon", "reload",
        "bone18", "bone19", "bone20",
    }
    print(f"[CAF-DEBUG] frame={frame:.3f} tick={time_tick}")
    for bone_name in sorted(ctrl_by_bone.keys()):
        if bone_name.lower() not in focus:
            continue
        pbone = arm_obj.pose.bones.get(bone_name)
        if pbone is None:
            continue
        ctrl_chunk = ctrl_by_bone[bone_name]
        local_m = _evaluate_crybone_controller_at_time(
            ctrl_chunk, time_tick, default_local=_bone_rest_local_matrix(pbone), bone_name=bone_name, evaluator_mode=evaluator_mode
        )
        l_loc, l_rot, _ = local_m.decompose()
        w_loc = pbone.matrix.translation.copy()
        w_rot = pbone.matrix.to_quaternion().copy()
        print(
            "[CAF-DEBUG] "
            f"{bone_name}: local_loc=({l_loc.x:.4f},{l_loc.y:.4f},{l_loc.z:.4f}) "
            f"local_rot=({l_rot.w:.4f},{l_rot.x:.4f},{l_rot.y:.4f},{l_rot.z:.4f}) "
            f"world_loc=({w_loc.x:.4f},{w_loc.y:.4f},{w_loc.z:.4f}) "
            f"world_rot=({w_rot.w:.4f},{w_rot.x:.4f},{w_rot.y:.4f},{w_rot.z:.4f})"
        )


def _debug_log_crybone_matrices(arm_obj, ctrl_by_bone, time_tick, frame, evaluator_mode="DEFAULT"):
    print(f"[CAF-MATRIX] frame={frame:.3f} tick={time_tick}")
    for bone_name in ("root1", "weapon", "reload"):
        pbone = arm_obj.pose.bones.get(bone_name)
        ctrl_chunk = ctrl_by_bone.get(bone_name)
        if pbone is None or ctrl_chunk is None:
            continue
        rest_local = _bone_rest_local_matrix(pbone)
        eval_local = _evaluate_crybone_controller_at_time(
            ctrl_chunk, time_tick, default_local=rest_local, bone_name=bone_name, evaluator_mode=evaluator_mode
        )
        basis_delta = rest_local.inverted() @ eval_local
        print(f"[CAF-MATRIX] bone={bone_name} rest_local={_fmt_matrix4(rest_local)}")
        print(f"[CAF-MATRIX] bone={bone_name} eval_local={_fmt_matrix4(eval_local)}")
        print(f"[CAF-MATRIX] bone={bone_name} basis_delta={_fmt_matrix4(basis_delta)}")
        print(f"[CAF-MATRIX] bone={bone_name} matrix_basis={_fmt_matrix4(pbone.matrix_basis)}")
        print(f"[CAF-MATRIX] bone={bone_name} pose_matrix={_fmt_matrix4(pbone.matrix)}")


def _debug_log_cry_pose(bind_pose, ctrl_by_bone, time_tick, frame):
    print(f"[CRYPOSE] frame={frame:.3f} tick={time_tick}")
    pose = _evaluate_cry_skeleton_pose(bind_pose, ctrl_by_bone, time_tick)
    for bone_name in ("root1", "weapon", "reload"):
        item = pose.get(bone_name)
        if item is None:
            continue
        print(f"[CRYPOSE] bone={bone_name} bind_local={_fmt_matrix4(item['bind_local'])}")
        print(f"[CRYPOSE] bone={bone_name} anim_local={_fmt_matrix4(item['local'])}")
        print(f"[CRYPOSE] bone={bone_name} anim_world={_fmt_matrix4(item['world'])}")


def _debug_log_v827_root_keys(ctrl_by_bone, time_tick, frame):
    print(f"[V827-ROOT] frame={frame:.3f} tick={time_tick}")
    for bone_name in ("root", "root1"):
        ctrl_chunk = ctrl_by_bone.get(bone_name)
        if ctrl_chunk is None:
            continue
        keys = _effective_ctrl_keys(ctrl_chunk)
        if not keys or hasattr(keys[0], 'rel_pos'):
            continue

        prev_key = keys[0]
        next_key = keys[-1]
        alpha = 0.0
        if len(keys) > 1:
            if time_tick <= keys[0].time:
                prev_key = next_key = keys[0]
            elif time_tick >= keys[-1].time:
                prev_key = next_key = keys[-1]
            else:
                for idx in range(1, len(keys)):
                    if time_tick <= keys[idx].time:
                        prev_key = keys[idx - 1]
                        next_key = keys[idx]
                        break
                if next_key.time != prev_key.time:
                    alpha = (time_tick - prev_key.time) / (next_key.time - prev_key.time)

        def _vec3(v):
            return f"({float(v[0]):.4f},{float(v[1]):.4f},{float(v[2]):.4f})"

        print(
            f"[V827-ROOT] bone={bone_name} "
            f"prev_time={int(prev_key.time)} next_time={int(next_key.time)} alpha={float(alpha):.4f}"
        )
        print(
            f"[V827-ROOT] bone={bone_name} "
            f"prev_pos_raw={_vec3(prev_key.pos)} next_pos_raw={_vec3(next_key.pos)}"
        )
        print(
            f"[V827-ROOT] bone={bone_name} "
            f"prev_rotlog={_vec3(prev_key.rot_log)} next_rotlog={_vec3(next_key.rot_log)}"
        )
        prev_local = _v827_local_from_key(prev_key, half_rot=False, bone_name=bone_name)
        next_local = _v827_local_from_key(next_key, half_rot=False, bone_name=bone_name)
        print(f"[V827-ROOT] bone={bone_name} prev_local={_fmt_matrix4(prev_local)}")
        print(f"[V827-ROOT] bone={bone_name} next_local={_fmt_matrix4(next_local)}")


def _debug_log_bone_space_deltas(arm_obj, bind_pose):
    if arm_obj is None or arm_obj.pose is None or not bind_pose:
        return
    print("[CRYBONEDELTA] begin")
    for bone_name in ("root1", "weapon", "reload"):
        pbone = arm_obj.pose.bones.get(bone_name)
        item = bind_pose.get(bone_name)
        if pbone is None or item is None:
            continue

        blender_rest_local = _bone_rest_local_matrix(pbone)
        blender_rest_world = pbone.bone.matrix_local.copy()
        cry_bind_local = item["bind_local"].copy()
        cry_bind_world = item["bind_world"].copy()

        try:
            local_delta = blender_rest_local.inverted_safe() @ cry_bind_local
        except Exception:
            local_delta = None
        try:
            world_delta = blender_rest_world.inverted_safe() @ cry_bind_world
        except Exception:
            world_delta = None

        print(f"[CRYBONEDELTA] bone={bone_name} blender_rest_local={_fmt_matrix4(blender_rest_local)}")
        print(f"[CRYBONEDELTA] bone={bone_name} cry_bind_local={_fmt_matrix4(cry_bind_local)}")
        print(f"[CRYBONEDELTA] bone={bone_name} local_delta={_fmt_matrix4(local_delta)}")
        print(f"[CRYBONEDELTA] bone={bone_name} blender_rest_world={_fmt_matrix4(blender_rest_world)}")
        print(f"[CRYBONEDELTA] bone={bone_name} cry_bind_world={_fmt_matrix4(cry_bind_world)}")
        print(f"[CRYBONEDELTA] bone={bone_name} world_delta={_fmt_matrix4(world_delta)}")
    print("[CRYBONEDELTA] end")


def _mesh_objects_for_armature(arm_obj):
    if arm_obj is None:
        return []
    result = []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object == arm_obj:
                result.append(obj)
                break
    return result


def _geom_mesh_chunk_by_object(obj, geom_archive):
    if obj is None or geom_archive is None:
        return None
    chunk_id = obj.get('cgf_chunk_id')
    if chunk_id is None:
        return None
    for mesh_chunk in geom_archive.mesh_chunks:
        if mesh_chunk.header and int(mesh_chunk.header.chunk_id) == int(chunk_id):
            return mesh_chunk
    return None


def _remove_cry_preview_shape_keys(obj):
    if obj is None or obj.data is None or obj.data.shape_keys is None:
        return
    key_blocks = obj.data.shape_keys.key_blocks
    removable = [kb.name for kb in key_blocks if kb.name.startswith("CRYPREVIEW_")]
    if not removable:
        return
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    for key_name in removable:
        kb = obj.data.shape_keys.key_blocks.get(key_name)
        if kb is None:
            continue
        obj.active_shape_key_index = list(obj.data.shape_keys.key_blocks).index(kb)
        bpy.ops.object.shape_key_remove(all=False)


def _ensure_basis_shape_key(obj):
    if obj.data.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)


def _source_bind_positions_from_object(obj, mesh_chunk=None, geom_archive=None):
    if mesh_chunk is not None and geom_archive is not None:
        node_chunk = geom_archive.get_node(mesh_chunk.header.chunk_id) if mesh_chunk.header else None
        rebuilt = _build_skinned_bind_positions(mesh_chunk, geom_archive, node_chunk)
        if rebuilt:
            return {int(k): v.copy() for k, v in rebuilt.items()}

    source_ids = obj.get("_cgf_source_vert_ids") or []
    if not source_ids:
        return {}
    result = {}
    for mesh_vid, src_vid in enumerate(source_ids):
        if src_vid in result:
            continue
        if mesh_vid < len(obj.data.vertices):
            result[int(src_vid)] = obj.data.vertices[mesh_vid].co.copy()
    return result


def _skin_vertex_from_cry_pose(bind_pos, bone_links, bone_name_by_id, bind_pose, cry_pose, mode,
                               obj_to_arm=None, arm_to_obj=None):
    out = mathutils.Vector((0.0, 0.0, 0.0))
    total = 0.0
    if obj_to_arm is None:
        obj_to_arm = mathutils.Matrix.Identity(4)
    if arm_to_obj is None:
        arm_to_obj = mathutils.Matrix.Identity(4)

    bind_h = mathutils.Vector((bind_pos.x, bind_pos.y, bind_pos.z, 1.0))
    bind_arm4 = obj_to_arm @ bind_h
    bind_arm = mathutils.Vector((bind_arm4.x, bind_arm4.y, bind_arm4.z))
    for link in bone_links.links:
        bone_name = bone_name_by_id.get(link.bone_id)
        if bone_name is None:
            continue
        pose_item = cry_pose.get(bone_name)
        bind_item = bind_pose.get(bone_name)
        if pose_item is None or bind_item is None:
            continue

        offset = mathutils.Vector((
            float(link.offset[0]) * INCHES_TO_METERS,
            float(link.offset[1]) * INCHES_TO_METERS,
            float(link.offset[2]) * INCHES_TO_METERS,
        ))

        if mode == "delta_col":
            skin_m = pose_item["world"] @ bind_item["bind_world"].inverted_safe()
            deformed4 = skin_m @ mathutils.Vector((bind_arm.x, bind_arm.y, bind_arm.z, 1.0))
            deformed_obj4 = arm_to_obj @ deformed4
            deformed = mathutils.Vector((deformed_obj4.x, deformed_obj4.y, deformed_obj4.z))
        elif mode == "delta_row":
            skin_m = pose_item["world"] @ bind_item["bind_world"].inverted_safe()
            deformed_arm = _mul_point_max_row(skin_m, mathutils.Vector((bind_arm.x, bind_arm.y, bind_arm.z)))
            deformed_obj4 = arm_to_obj @ mathutils.Vector((deformed_arm.x, deformed_arm.y, deformed_arm.z, 1.0))
            deformed = mathutils.Vector((deformed_obj4.x, deformed_obj4.y, deformed_obj4.z))
        elif mode == "offset_col":
            deformed_arm = _mul_point(pose_item["world"], offset)
            deformed_obj4 = arm_to_obj @ mathutils.Vector((deformed_arm.x, deformed_arm.y, deformed_arm.z, 1.0))
            deformed = mathutils.Vector((deformed_obj4.x, deformed_obj4.y, deformed_obj4.z))
        elif mode == "offset_row":
            deformed_arm = _mul_point_max_row(pose_item["world"], offset)
            deformed_obj4 = arm_to_obj @ mathutils.Vector((deformed_arm.x, deformed_arm.y, deformed_arm.z, 1.0))
            deformed = mathutils.Vector((deformed_obj4.x, deformed_obj4.y, deformed_obj4.z))
        else:
            continue

        weight = float(link.blending)
        out += deformed * weight
        total += weight

    if total > 1e-8:
        return out / total
    return bind_pos.copy()


def _choose_preview_skinning_mode(obj, mesh_chunk, geom_archive, bind_pose, obj_to_arm=None, arm_to_obj=None):
    source_bind = _source_bind_positions_from_object(obj, mesh_chunk=mesh_chunk, geom_archive=geom_archive)
    if not source_bind:
        return "delta_col"

    source_ids = obj.get("_cgf_source_vert_ids") or []
    if not source_ids:
        return "delta_col"

    bone_name_by_id = {item["bone_id"]: name for name, item in bind_pose.items()}
    modes = ("delta_col", "delta_row", "offset_col", "offset_row")
    bind_world_pose = {name: {"world": item["bind_world"]} for name, item in bind_pose.items()}
    errors = {mode: 0.0 for mode in modes}
    counts = {mode: 0 for mode in modes}

    for bone_links in mesh_chunk.physique:
        src_vid = int(bone_links.vertex_id)
        bind_pos = source_bind.get(src_vid)
        if bind_pos is None:
            continue
        for mode in modes:
            pred = _skin_vertex_from_cry_pose(
                bind_pos, bone_links, bone_name_by_id, bind_pose, bind_world_pose, mode,
                obj_to_arm=obj_to_arm, arm_to_obj=arm_to_obj,
            )
            delta = pred - bind_pos
            errors[mode] += float(delta.length_squared)
            counts[mode] += 1

    ranked = []
    for mode in modes:
        n = counts[mode]
        if n <= 0:
            continue
        ranked.append((errors[mode] / n, mode))
    if not ranked:
        return "delta_col"
    ranked.sort()
    best_error, best_mode = ranked[0]
    print(f"[CRYSKIN] best_mode={best_mode} bind_mse={best_error:.8f} all={[(m, round(errors[m] / counts[m], 8)) for m in modes if counts[m] > 0]}")
    return best_mode


def _skin_mesh_vertices_from_cry_pose(obj, mesh_chunk, geom_archive, bind_pose, cry_pose, skin_mode="delta_col"):
    if obj is None or mesh_chunk is None or not mesh_chunk.physique:
        return None
    source_bind = _source_bind_positions_from_object(obj, mesh_chunk=mesh_chunk, geom_archive=geom_archive)
    if not source_bind:
        return None
    bone_name_by_id = {item["bone_id"]: name for name, item in bind_pose.items()}
    arm_obj = obj.find_armature()
    arm_world = arm_obj.matrix_world.copy() if arm_obj is not None else mathutils.Matrix.Identity(4)
    node_chunk = geom_archive.get_node(mesh_chunk.header.chunk_id) if (geom_archive is not None and mesh_chunk.header) else None
    if node_chunk is not None and node_chunk.trans_matrix:
        chunk_world = cry_matrix_to_blender(node_chunk.trans_matrix)
    else:
        chunk_world = obj.matrix_world.copy()
    obj_to_arm = arm_world.inverted_safe() @ chunk_world
    arm_to_obj = chunk_world.inverted_safe() @ arm_world
    if obj.get("_crychunk_logged") != 1:
        obj["_crychunk_logged"] = 1
        print(f"[CRYCHUNK] obj={obj.name} chunk={int(mesh_chunk.header.chunk_id) if mesh_chunk.header else -1}")
        print(f"[CRYCHUNK] obj_world={_matrix_str(obj.matrix_world)}")
        print(f"[CRYCHUNK] chunk_world={_matrix_str(chunk_world)}")
        print(f"[CRYCHUNK] arm_world={_matrix_str(arm_world)}")

    source_out = {}
    for bone_links in mesh_chunk.physique:
        src_vid = int(bone_links.vertex_id)
        bind_pos = source_bind.get(src_vid)
        if bind_pos is None:
            continue
        if not bone_links.links:
            source_out[src_vid] = bind_pos.copy()
            continue
        source_out[src_vid] = _skin_vertex_from_cry_pose(
            bind_pos,
            bone_links,
            bone_name_by_id,
            bind_pose,
            cry_pose,
            skin_mode,
            obj_to_arm=obj_to_arm,
            arm_to_obj=arm_to_obj,
        )

    coords = [v.co.copy() for v in obj.data.vertices]
    source_ids = obj.get("_cgf_source_vert_ids") or []
    for mesh_vid, src_vid in enumerate(source_ids):
        if mesh_vid >= len(coords):
            continue
        if int(src_vid) in source_out:
            coords[mesh_vid] = source_out[int(src_vid)].copy()
    return coords


def _debug_compare_blender_vs_cry_skin(arm_obj, geom_archive, bind_pose, cry_pose, skin_mode="delta_col"):
    if arm_obj is None or geom_archive is None or not bind_pose or not cry_pose:
        return
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in _mesh_objects_for_armature(arm_obj):
        mesh_chunk = _geom_mesh_chunk_by_object(obj, geom_archive)
        if mesh_chunk is None or not mesh_chunk.physique:
            continue
        predicted = _skin_mesh_vertices_from_cry_pose(obj, mesh_chunk, geom_archive, bind_pose, cry_pose, skin_mode=skin_mode)
        if not predicted:
            continue
        try:
            eval_obj = obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
        except Exception:
            continue
        try:
            limit = min(12, len(eval_mesh.vertices), len(predicted))
            total = 0.0
            worst = 0.0
            worst_vid = -1
            for vid in range(limit):
                delta = eval_mesh.vertices[vid].co.copy() - predicted[vid]
                err = float(delta.length)
                total += err
                if err > worst:
                    worst = err
                    worst_vid = vid
            mean = (total / limit) if limit else 0.0
            print(f"[CRYDEFORM] obj={obj.name} sample_count={limit} mean_err={mean:.6f} worst_vid={worst_vid} worst_err={worst:.6f}")
            if worst_vid >= 0:
                eval_co = eval_mesh.vertices[worst_vid].co.copy()
                pred_co = predicted[worst_vid].copy()
                print(
                    "[CRYDEFORM] "
                    f"worst_vid={worst_vid} "
                    f"eval=({eval_co.x:.4f},{eval_co.y:.4f},{eval_co.z:.4f}) "
                    f"pred=({pred_co.x:.4f},{pred_co.y:.4f},{pred_co.z:.4f})"
                )
        finally:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass


def _convert_raw_pose_to_blender_pose(raw_pose, arm_obj):
    if not raw_pose:
        return {}
    arm_inv = arm_obj.matrix_world.inverted_safe() if arm_obj is not None else mathutils.Matrix.Identity(4)
    result = {}
    for bone_name, item in raw_pose.items():
        src_world = item.get("world", item.get("bind_world"))
        src_bind_world = item.get("bind_world", item.get("world"))
        if src_world is None or src_bind_world is None:
            continue
        world_m = arm_inv @ _raw_max_matrix_to_blender(src_world)
        bind_world = arm_inv @ _raw_max_matrix_to_blender(src_bind_world)
        parent_name = item.get("parent_name")
        if parent_name and parent_name in result:
            local_m = result[parent_name]["world"].inverted_safe() @ world_m
        else:
            local_m = world_m.copy()
        bind_parent_name = item.get("parent_name")
        if bind_parent_name and bind_parent_name in result:
            bind_local = result[bind_parent_name]["bind_world"].inverted_safe() @ bind_world
        else:
            bind_local = bind_world.copy()
        result[bone_name] = {
            "local": local_m,
            "world": world_m,
            "bind_local": bind_local,
            "bind_world": bind_world,
            "parent_name": parent_name,
            "bone_id": item["bone_id"],
        }
    return result


def _build_cry_proxy_hierarchy(arm_obj, bind_pose_raw):
    if arm_obj is None or not bind_pose_raw:
        return {}
    ordered = [name for name, _ in sorted(bind_pose_raw.items(), key=lambda kv: kv[1]["bone_id"])]
    proxies = _ensure_cry_proxy_targets(arm_obj, ordered)
    for bone_name in ordered:
        proxy = proxies.get(bone_name)
        item = bind_pose_raw.get(bone_name)
        if proxy is None or item is None:
            continue
        parent_name = item.get("parent_name")
        proxy.parent = proxies.get(parent_name) if parent_name else None
        proxy.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        local_bl = _raw_max_matrix_to_blender(item["bind_local"])
        _set_proxy_local_matrix(proxy, local_bl)
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    return proxies


def _insert_proxy_key(proxy, local_bl, frame):
    if proxy is None or local_bl is None:
        return
    _set_proxy_local_matrix(proxy, local_bl)
    proxy.keyframe_insert(data_path="location", frame=frame)
    proxy.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    proxy.keyframe_insert(data_path="scale", frame=frame)


def _animate_cry_proxies(arm_obj, proxies, ctrl_by_bone, ticks_per_frame, time_offset_ticks=0):
    key_times = set()
    for ctrl_chunk in ctrl_by_bone.values():
        for key in _effective_ctrl_keys(ctrl_chunk):
            key_times.add(int(key.time))
    if not key_times:
        return []

    for bone_name, ctrl_chunk in ctrl_by_bone.items():
        proxy = proxies.get(bone_name)
        if proxy is None:
            continue
        if proxy.animation_data is None:
            proxy.animation_data_create()
        for key in _effective_ctrl_keys(ctrl_chunk):
            frame = (int(key.time) + time_offset_ticks) / ticks_per_frame
            local_bl = _raw_max_matrix_to_blender(_raw_max_local_from_key(key))
            _insert_proxy_key(proxy, local_bl, frame)
    return sorted(key_times)


def _proxy_pose_to_blender_pose(proxies, bind_pose_raw, arm_obj):
    if not proxies or not bind_pose_raw:
        return {}
    arm_inv = arm_obj.matrix_world.inverted_safe() if arm_obj is not None else mathutils.Matrix.Identity(4)
    result = {}
    ordered = sorted(bind_pose_raw.items(), key=lambda kv: kv[1]["bone_id"])
    bind_bl = _convert_raw_pose_to_blender_pose(bind_pose_raw, arm_obj)
    for bone_name, item in ordered:
        proxy = proxies.get(bone_name)
        if proxy is None:
            continue
        world_m = arm_inv @ proxy.matrix_world.copy()
        parent_name = item.get("parent_name")
        if parent_name and parent_name in result:
            local_m = result[parent_name]["world"].inverted_safe() @ world_m
        else:
            local_m = world_m.copy()
        bind_item = bind_bl.get(bone_name)
        result[bone_name] = {
            "local": local_m,
            "world": world_m,
            "bind_local": bind_item["bind_local"] if bind_item else local_m.copy(),
            "bind_world": bind_item["bind_world"] if bind_item else world_m.copy(),
            "parent_name": parent_name,
            "bone_id": item["bone_id"],
        }
    return result


def _bake_cry_proxy_to_meshes(arm_obj, geom_archive, bind_pose_raw, proxies, sorted_ticks, ticks_per_frame, time_offset_ticks, action_name, debug_caf=False):
    if arm_obj is None or geom_archive is None or not bind_pose_raw or not proxies or not sorted_ticks:
        return False
    bind_bl = _convert_raw_pose_to_blender_pose(bind_pose_raw, arm_obj)
    mesh_objects = _mesh_objects_for_armature(arm_obj)
    if not mesh_objects:
        return False

    baked_any = False
    for obj in mesh_objects:
        mesh_chunk = _geom_mesh_chunk_by_object(obj, geom_archive)
        if mesh_chunk is None or not mesh_chunk.physique:
            continue

        _remove_cry_preview_shape_keys(obj)
        _ensure_basis_shape_key(obj)

        key_data = obj.data.shape_keys
        if key_data.animation_data is None:
            key_data.animation_data_create()
        preview_action = bpy.data.actions.new(name=f"{action_name}_{obj.name}_CRYPREVIEW")
        key_data.animation_data.action = preview_action

        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object == arm_obj:
                mod.show_viewport = False
                mod.show_render = False

        for idx, time_tick in enumerate(sorted_ticks):
            frame = (time_tick + time_offset_ticks) / ticks_per_frame
            try:
                frame_int = int(math.floor(frame))
                subframe = float(frame - frame_int)
                bpy.context.scene.frame_set(frame_int, subframe=subframe)
            except Exception:
                pass
            pose_bl = _proxy_pose_to_blender_pose(proxies, bind_pose_raw, arm_obj)
            coords = _skin_mesh_vertices_from_cry_pose(
                obj, mesh_chunk, geom_archive, bind_bl, pose_bl, skin_mode="delta_col"
            )
            if not coords:
                continue
            kb = obj.shape_key_add(name=f"CRYPREVIEW_{idx:04d}", from_mix=False)
            for vid, co in enumerate(coords):
                kb.data[vid].co = co
            kb.value = 0.0
            kb.keyframe_insert(data_path="value", frame=frame - 1.0)
            kb.value = 1.0
            kb.keyframe_insert(data_path="value", frame=frame)
            kb.value = 0.0
            kb.keyframe_insert(data_path="value", frame=frame + 1.0)

        fcurves = getattr(preview_action, "fcurves", None)
        if fcurves is not None:
            for fcurve in fcurves:
                for kp in fcurve.keyframe_points:
                    kp.interpolation = 'CONSTANT'
        baked_any = True

    if baked_any and debug_caf:
        try:
            pose_bl = _proxy_pose_to_blender_pose(proxies, bind_pose_raw, arm_obj)
            _debug_compare_blender_vs_cry_skin(arm_obj, geom_archive, bind_bl, pose_bl)
        except Exception:
            pass
    return baked_any


def _bake_cry_maxspace_to_meshes(arm_obj, geom_archive, ctrl_by_bone, ticks_per_frame, time_offset_ticks, action_name, debug_caf=False):
    if arm_obj is None or geom_archive is None:
        return False
    bind_pose_raw = _build_cry_bind_pose_raw(geom_archive)
    if not bind_pose_raw:
        return False

    mesh_objects = _mesh_objects_for_armature(arm_obj)
    if not mesh_objects:
        return False

    key_times = set()
    for ctrl_chunk in ctrl_by_bone.values():
        for key in getattr(ctrl_chunk, 'keys', None) or []:
            key_times.add(int(key.time))
    if not key_times:
        return False

    sorted_ticks = sorted(key_times)
    baked_any = False
    for obj in mesh_objects:
        mesh_chunk = _geom_mesh_chunk_by_object(obj, geom_archive)
        if mesh_chunk is None or not mesh_chunk.physique:
            continue
        node_chunk = geom_archive.get_node(mesh_chunk.header.chunk_id) if mesh_chunk.header else None
        if node_chunk is not None and node_chunk.trans_matrix:
            chunk_world = cry_matrix_to_blender(node_chunk.trans_matrix)
        else:
            chunk_world = obj.matrix_world.copy()
        arm_world = arm_obj.matrix_world.copy()
        obj_to_arm = arm_world.inverted_safe() @ chunk_world
        arm_to_obj = chunk_world.inverted_safe() @ arm_world
        bind_bl = _convert_raw_pose_to_blender_pose(bind_pose_raw, arm_obj)
        skin_mode = _choose_preview_skinning_mode(
            obj,
            mesh_chunk,
            geom_archive,
            bind_bl,
            obj_to_arm=obj_to_arm,
            arm_to_obj=arm_to_obj,
        )
        _remove_cry_preview_shape_keys(obj)
        _ensure_basis_shape_key(obj)

        key_data = obj.data.shape_keys
        if key_data.animation_data is None:
            key_data.animation_data_create()
        preview_action = bpy.data.actions.new(name=f"{action_name}_{obj.name}_CRYPREVIEW")
        key_data.animation_data.action = preview_action

        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object == arm_obj:
                mod.show_viewport = False
                mod.show_render = False

        for idx, time_tick in enumerate(sorted_ticks):
            frame = (time_tick + time_offset_ticks) / ticks_per_frame
            raw_pose = _evaluate_cry_skeleton_pose_raw(bind_pose_raw, ctrl_by_bone, time_tick)
            pose_bl = _convert_raw_pose_to_blender_pose(raw_pose, arm_obj)
            coords = _skin_mesh_vertices_from_cry_pose(
                obj, mesh_chunk, geom_archive, bind_bl, pose_bl, skin_mode=skin_mode
            )
            if not coords:
                continue
            kb = obj.shape_key_add(name=f"CRYPREVIEW_{idx:04d}", from_mix=False)
            for vid, co in enumerate(coords):
                kb.data[vid].co = co
            kb.value = 0.0
            kb.keyframe_insert(data_path="value", frame=frame - 1.0)
            kb.value = 1.0
            kb.keyframe_insert(data_path="value", frame=frame)
            kb.value = 0.0
            kb.keyframe_insert(data_path="value", frame=frame + 1.0)

        fcurves = getattr(preview_action, "fcurves", None)
        if fcurves is not None:
            for fcurve in fcurves:
                for kp in fcurve.keyframe_points:
                    kp.interpolation = 'CONSTANT'
        baked_any = True

    if baked_any and debug_caf:
        try:
            pose0 = _evaluate_cry_skeleton_pose_raw(bind_pose_raw, ctrl_by_bone, sorted_ticks[0])
            pose0_bl = _convert_raw_pose_to_blender_pose(pose0, arm_obj)
            bind_bl = _convert_raw_pose_to_blender_pose(bind_pose_raw, arm_obj)
            _debug_compare_blender_vs_cry_skin(arm_obj, geom_archive, bind_bl, pose0_bl, skin_mode=skin_mode)
        except Exception:
            pass
    return baked_any


def _bake_cry_preview_to_meshes(arm_obj, geom_archive, ctrl_by_bone, ticks_per_frame, time_offset_ticks, action_name):
    if arm_obj is None or geom_archive is None:
        return False
    bind_pose = _build_cry_bind_pose(geom_archive, arm_obj=arm_obj)
    if not bind_pose:
        return False

    mesh_objects = _mesh_objects_for_armature(arm_obj)
    if not mesh_objects:
        return False

    key_times = set()
    for ctrl_chunk in ctrl_by_bone.values():
        for key in getattr(ctrl_chunk, 'keys', None) or []:
            key_times.add(int(key.time))
    if not key_times:
        return False

    sorted_ticks = sorted(key_times)
    baked_any = False
    for obj in mesh_objects:
        mesh_chunk = _geom_mesh_chunk_by_object(obj, geom_archive)
        if mesh_chunk is None or not mesh_chunk.physique:
            continue
        arm_obj_local = obj.find_armature()
        arm_world = arm_obj_local.matrix_world.copy() if arm_obj_local is not None else mathutils.Matrix.Identity(4)
        node_chunk = geom_archive.get_node(mesh_chunk.header.chunk_id) if mesh_chunk.header else None
        if node_chunk is not None and node_chunk.trans_matrix:
            chunk_world = cry_matrix_to_blender(node_chunk.trans_matrix)
        else:
            chunk_world = obj.matrix_world.copy()
        obj_to_arm = arm_world.inverted_safe() @ chunk_world
        arm_to_obj = chunk_world.inverted_safe() @ arm_world
        skin_mode = _choose_preview_skinning_mode(obj, mesh_chunk, geom_archive, bind_pose, obj_to_arm=obj_to_arm, arm_to_obj=arm_to_obj)
        _remove_cry_preview_shape_keys(obj)
        _ensure_basis_shape_key(obj)

        key_data = obj.data.shape_keys
        if key_data.animation_data is None:
            key_data.animation_data_create()
        preview_action = bpy.data.actions.new(name=f"{action_name}_{obj.name}_CRYPREVIEW")
        key_data.animation_data.action = preview_action

        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object == arm_obj:
                mod.show_viewport = False
                mod.show_render = False

        for idx, time_tick in enumerate(sorted_ticks):
            frame = (time_tick + time_offset_ticks) / ticks_per_frame
            cry_pose = _evaluate_cry_skeleton_pose(bind_pose, ctrl_by_bone, time_tick)
            coords = _skin_mesh_vertices_from_cry_pose(obj, mesh_chunk, geom_archive, bind_pose, cry_pose, skin_mode=skin_mode)
            if not coords:
                continue
            kb = obj.shape_key_add(name=f"CRYPREVIEW_{idx:04d}", from_mix=False)
            for vid, co in enumerate(coords):
                kb.data[vid].co = co
            kb.value = 0.0
            kb.keyframe_insert(data_path="value", frame=frame - 1.0)
            kb.value = 1.0
            kb.keyframe_insert(data_path="value", frame=frame)
            kb.value = 0.0
            kb.keyframe_insert(data_path="value", frame=frame + 1.0)

        fcurves = getattr(preview_action, "fcurves", None)
        if fcurves is not None:
            for fcurve in fcurves:
                for kp in fcurve.keyframe_points:
                    kp.interpolation = 'CONSTANT'
        baked_any = True

    return baked_any


def _normalize_playback_mode(playback_mode):
    if isinstance(playback_mode, bool):
        return "PROXY" if playback_mode else "ARMATURE"
    mode = str(playback_mode or "ARMATURE").strip().upper()
    if mode not in {"ARMATURE", "PROXY", "MAXSPACE", "RAWMAX"}:
        mode = "ARMATURE"
    return mode


def _apply_crybone_controllers(
    arm_obj, geom_archive, ctrl_chunks, ctrl_to_bone, ticks_per_frame, time_offset_ticks=0,
    debug_caf=False, action=None, playback_mode="ARMATURE"
):
    _clear_cry_proxy_data(arm_obj)
    playback_mode = _normalize_playback_mode(playback_mode)
    evaluator_mode = "RAWMAX" if playback_mode == "RAWMAX" else "DEFAULT"

    ctrl_by_bone = {}
    key_times = set()
    for ctrl_chunk in ctrl_chunks:
        eff_keys = _effective_ctrl_keys(ctrl_chunk)
        if not eff_keys:
            continue
        bone_name = ctrl_to_bone.get(ctrl_chunk.ctrl_id)
        if not bone_name:
            continue
        if arm_obj.pose is None or bone_name not in arm_obj.pose.bones:
            continue
        ctrl_by_bone[bone_name] = ctrl_chunk
        key_times.update(key.time for key in eff_keys)

    if not ctrl_by_bone or not key_times:
        return False

    if playback_mode == "PROXY":
        bind_pose_raw = _build_cry_bind_pose_raw(geom_archive)
        if bind_pose_raw:
            proxies = _build_cry_proxy_hierarchy(arm_obj, bind_pose_raw)
            sorted_ticks = _animate_cry_proxies(
                arm_obj, proxies, ctrl_by_bone, ticks_per_frame, time_offset_ticks=time_offset_ticks
            )
            _drive_armature_from_proxies(arm_obj, proxies)
            baked_preview = _bake_cry_proxy_to_meshes(
                arm_obj,
                geom_archive,
                bind_pose_raw,
                proxies,
                sorted_ticks,
                ticks_per_frame,
                time_offset_ticks,
                getattr(action, "name", "CRYPOSE_PROXY"),
                debug_caf=debug_caf,
            )
            if debug_caf:
                print(f"[CRYPOSE] proxy_first baked_preview={baked_preview} proxy_count={len(proxies)}")
            _reset_pose_bones_to_rest(arm_obj, ctrl_by_bone.keys())
            return baked_preview or True

    if playback_mode == "MAXSPACE":
        baked_preview = _bake_cry_maxspace_to_meshes(
            arm_obj,
            geom_archive,
            ctrl_by_bone,
            ticks_per_frame,
            time_offset_ticks,
            getattr(action, "name", "CRYPOSE_MAXSPACE"),
            debug_caf=debug_caf,
        )
        if debug_caf:
            print(f"[CRYPOSE] maxspace baked_preview={baked_preview} ctrl_count={len(ctrl_by_bone)}")
        _reset_pose_bones_to_rest(arm_obj, ctrl_by_bone.keys())
        return baked_preview or True

    bind_pose = _build_cry_bind_pose(geom_archive, arm_obj=arm_obj)
    if debug_caf:
        sample_keys = list(bind_pose.keys())[:6]
        print(f"[CRYPOSE] bind_pose_count={len(bind_pose)} sample={sample_keys}")
        _debug_log_bone_space_deltas(arm_obj, bind_pose)
    sorted_ticks = sorted(key_times)
    last_index = len(sorted_ticks) - 1
    mid_index = last_index // 2 if last_index >= 0 else 0
    target_frame = 37.0
    target_tick = None
    if sorted_ticks:
        try:
            first_tick = sorted_ticks[0]
            raw_target_tick = int(round(first_tick + target_frame * ticks_per_frame))
            target_tick = min(sorted_ticks, key=lambda t: abs(int(t) - raw_target_tick))
        except Exception:
            target_tick = None
    for idx, time_tick in enumerate(sorted_ticks):
        shifted_tick = time_tick + time_offset_ticks
        frame = shifted_tick / ticks_per_frame
        try:
            frame_int = int(math.floor(frame))
            subframe = float(frame - frame_int)
            bpy.context.scene.frame_set(frame_int, subframe=subframe)
        except Exception:
            pass
        _apply_crybone_pose_at_time(
            arm_obj,
            ctrl_by_bone,
            time_tick,
            keyframe_frame=frame,
            action=action,
            evaluator_mode=evaluator_mode,
        )
        if debug_caf and (idx < 5 or idx == mid_index or idx == last_index or (target_tick is not None and int(time_tick) == int(target_tick))):
            _debug_log_crybone_frame(arm_obj, ctrl_by_bone, time_tick, frame, evaluator_mode=evaluator_mode)
        if debug_caf and (idx == 0 or idx == mid_index or idx == last_index or (target_tick is not None and int(time_tick) == int(target_tick))):
            _debug_log_crybone_matrices(arm_obj, ctrl_by_bone, time_tick, frame, evaluator_mode=evaluator_mode)
            _debug_log_v827_root_keys(ctrl_by_bone, time_tick, frame)
            if bind_pose:
                _debug_log_cry_pose(bind_pose, ctrl_by_bone, time_tick, frame)
                debug_cry_pose = _evaluate_cry_skeleton_pose(bind_pose, ctrl_by_bone, time_tick, evaluator_mode=evaluator_mode)
                _debug_compare_blender_vs_cry_skin(arm_obj, geom_archive, bind_pose, debug_cry_pose)

    _reset_pose_bones_to_rest(arm_obj, ctrl_by_bone.keys())
    return True


def build_material(mat_chunk, filepath, import_materials, game_root_path=""):
    if not import_materials:
        return None
    full_name = _build_cgf_mat_name(mat_chunk.name,
                                    mat_chunk.shader_name,
                                    mat_chunk.surface_name)

    mat = bpy.data.materials.new(name=full_name)
    # Store original CGF material info for round-trip export
    mat['cgf_chunk_id']     = int(mat_chunk.header.chunk_id)
    mat['cgf_shader_name']  = mat_chunk.shader_name
    mat['cgf_surface_name'] = mat_chunk.surface_name
    mat['cgf_full_name']    = full_name
    mat['cgf_source_name']  = mat_chunk.name
    # Populate CryEngine panel properties
    if hasattr(mat, 'cry'):
        # Set shader — check if it matches a preset
        shader = mat_chunk.shader_name or ''
        preset_values = [item[0] for item in [
            ('Phong',''),('TemplModelCommon',''),('TemplBumpDiffuse',''),
            ('TemplBumpSpec',''),('TemplBumpSpec_GlossAlpha',''),
            ('NoDraw',''),('Glass',''),('Vegetation',''),('Terrain',''),
        ]]
        if shader in preset_values:
            mat.cry.shader_preset = shader
        else:
            mat.cry.shader_preset = 'custom'
            mat.cry.shader_custom = shader
        # Set surface
        surface = mat_chunk.surface_name or 'mat_default'
        try:
            mat.cry.surface = surface
        except Exception:
            mat.cry.surface = 'mat_default'
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out  = nodes.new('ShaderNodeOutputMaterial'); out.location  = (400, 0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (0, 0)
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    if _is_nodraw_material(mat_chunk):
        _set_input(bsdf, 'Base Color', value=(0.0, 0.0, 0.0, 1.0))
        _set_input(bsdf, 'Alpha', value=0.0)
        if hasattr(mat, 'blend_method'):
            mat.blend_method = 'CLIP'
        if hasattr(mat, 'shadow_method'):
            try:
                mat.shadow_method = 'NONE'
            except Exception:
                pass
        return mat

    d = mat_chunk.diffuse
    _set_input(bsdf, 'Base Color', value=(d[0], d[1], d[2], 1.0))

    s = mat_chunk.specular
    spec = ((s[0]+s[1]+s[2])/3.0) * mat_chunk.specular_level
    _set_input(bsdf, 'Specular IOR Level', 'Specular', value=min(spec, 1.0))

    if mat_chunk.specular_shininess > 0:
        _set_input(bsdf, 'Roughness',
                   value=1.0 - min(mat_chunk.specular_shininess/100.0, 1.0))

    # opacity: 0.0 in CGF often means "unused", not "fully transparent"
    # Only apply if it's a meaningful value between 0 and 1 (exclusive)
    if 0.0 < mat_chunk.opacity < 1.0 and _uses_diffuse_alpha_as_opacity(mat_chunk):
        _set_input(bsdf, 'Alpha', value=mat_chunk.opacity)
        if hasattr(mat, 'blend_method'):
            mat.blend_method = 'BLEND'

    def add_tex(tex_data, x, y, color_space='sRGB'):
        if not tex_data or not tex_data.name:
            return None
        print(f"[CGF] Searching: '{tex_data.name}' | root='{game_root_path}'")
        path = _find_texture(tex_data.name, filepath, game_root_path)
        if not path:
            print(f"[CGF] NOT found: {tex_data.name}")
            return None
        print(f"[CGF] Found: {path}")
        node = nodes.new('ShaderNodeTexImage')
        node.location = (x, y)
        try:
            img = bpy.data.images.load(path, check_existing=True)
            img.colorspace_settings.name = color_space
            # Force absolute filepath so FBX export picks up the correct path
            img.filepath = os.path.abspath(path)
            img.filepath_raw = os.path.abspath(path)
            node.image = img
        except Exception as e:
            print(f"[CGF] Load error {path}: {e}")
        return node

    tex_diff = add_tex(mat_chunk.tex_diffuse, -400, 0)
    if tex_diff:
        mat['tex_diffuse'] = bpy.path.abspath(tex_diff.image.filepath) if tex_diff.image else ""
        _configure_diffuse_image_alpha(tex_diff.image, mat_chunk)
        links.new(tex_diff.outputs['Color'], bsdf.inputs['Base Color'])
        alpha_input = bsdf.inputs.get('Alpha')
        if alpha_input and _uses_diffuse_alpha_as_opacity(mat_chunk):
            links.new(tex_diff.outputs['Alpha'], alpha_input)
            if hasattr(mat, 'blend_method'):
                if mat_chunk.alpha_test > 0.0:
                    mat.blend_method = 'CLIP'
                elif mat_chunk.opacity < 1.0:
                    mat.blend_method = 'CLIP'
                elif tex_diff.image and getattr(tex_diff.image, 'depth', 0) in (32, 64):
                    mat.blend_method = 'CLIP'
        # Gloss packed in diffuse alpha → connect to Specular
        shader_name = mat_chunk.shader_name or ''
        if 'GlossAlpha' in shader_name or 'glossalpha' in shader_name.lower():
            spec_input = (bsdf.inputs.get('Specular IOR Level') or
                          bsdf.inputs.get('Specular'))
            if spec_input:
                links.new(tex_diff.outputs['Alpha'], spec_input)

    tex_bump = add_tex(mat_chunk.tex_bump, -600, -300, 'Non-Color')
    if tex_bump:
        mat['tex_bump'] = bpy.path.abspath(tex_bump.image.filepath) if tex_bump.image else ""
        tex_name = (mat_chunk.tex_bump.name or '').lower()
        if '_ddn' in tex_name:
            # DDN = normal map
            normal_map = nodes.new('ShaderNodeNormalMap')
            normal_map.location = (-200, -300)
            links.new(tex_bump.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
        else:
            # _bump or other = heightmap → Bump node
            bump = nodes.new('ShaderNodeBump')
            bump.location = (-200, -300)
            links.new(tex_bump.outputs['Color'], bump.inputs['Height'])
            links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    tex_detail = add_tex(mat_chunk.tex_detail, -600, -520, 'Non-Color')
    if tex_detail:
        mat['tex_detail'] = bpy.path.abspath(tex_detail.image.filepath) if tex_detail.image else ""
        detail_bump = nodes.new('ShaderNodeBump')
        detail_bump.location = (-200, -520)
        links.new(tex_detail.outputs['Color'], detail_bump.inputs['Height'])
        if not bsdf.inputs['Normal'].links:
            links.new(detail_bump.outputs['Normal'], bsdf.inputs['Normal'])

    return mat


def _find_texture(name, cgf_path, game_root_path=""):
    """
    Search for a texture. Mirrors getFullFilename from original Max script:
    1. game_root_path + name  (texture name is relative to game root)
    2. CGF folder + name
    3. CGF folder + basename only
    """
    if not name:
        return None

    name = name.replace('\\', os.sep).replace('/', os.sep)
    basename = os.path.basename(name)
    cgf_dir  = os.path.dirname(cgf_path)
    exts     = ['', '.dds', '.tga', '.png', '.jpg', '.bmp', '.tif', '.tiff']

    def try_path(p):
        if os.path.isfile(p): return p
        base_no_ext = os.path.splitext(p)[0]
        for ext in exts:
            if os.path.isfile(base_no_ext + ext):
                return base_no_ext + ext
        return None

    if game_root_path:
        r = try_path(os.path.join(game_root_path, name))
        if r: return r
        print(f"[CGF] Texture not found in game root: {os.path.join(game_root_path, name)}")

    r = try_path(os.path.join(cgf_dir, name))
    if r: return r

    r = try_path(os.path.join(cgf_dir, basename))
    if r: return r

    print(f"[CGF] Texture not found anywhere: '{name}'")
    return None


# ── Mesh ──────────────────────────────────────────────────────────────────────

def build_mesh(mesh_chunk, node_chunk, archive, collection,
               import_materials, import_normals, import_uvs,
               import_weights, blender_materials, filepath,
               skip_collision_geometry=False, apply_node_transform=True):

    mc = mesh_chunk
    if not mc.vertices or not mc.faces:
        return None

    name = node_chunk.name if node_chunk else f"Mesh_{mc.header.chunk_id}"
    mesh = bpy.data.meshes.new(name)
    obj  = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    # Build the mesh from face corners, not from the raw shared vertex table.
    # Some CryEngine meshes intentionally reuse the same geometric triangle more than once
    # with different UV/material data; bmesh collapses those faces, which is what caused the
    # visible UV breakage on coa_storage.cgf.
    verts = []
    faces = []
    vertex_source_ids = []
    face_texcoords = []
    face_normals = []
    face_material_ids = []
    face_smooth_flags = []
    face_smoothing_groups = []
    collision_material_ids = _global_collision_material_ids(archive) if skip_collision_geometry else set()
    bind_positions = _build_skinned_bind_positions(mc, archive, node_chunk) if import_weights else {}

    for fi, cf in enumerate(mc.faces):
        if collision_material_ids and cf.mat_id in collision_material_ids:
            continue
        src_vis = (cf.v0, cf.v1, cf.v2)
        if any(vi >= len(mc.vertices) for vi in src_vis):
            continue

        if mc.tex_faces and fi < len(mc.tex_faces):
            tf = mc.tex_faces[fi]
            src_tvis = (tf.t0, tf.t1, tf.t2)
        else:
            src_tvis = src_vis

        face_indices = []
        corner_uvs = []
        corner_normals = []

        for corner_idx, src_vi in enumerate(src_vis):
            verts.append(bind_positions.get(src_vi, cry_vec(mc.vertices[src_vi].pos)))
            vertex_source_ids.append(src_vi)
            face_indices.append(len(verts) - 1)

            tvi = src_tvis[corner_idx] if corner_idx < len(src_tvis) else src_vi
            if import_uvs and tvi is not None and tvi < len(mc.tex_vertices):
                corner_uvs.append(mc.tex_vertices[tvi])
            else:
                corner_uvs.append((0.0, 0.0))

            n = mc.vertices[src_vi].normal
            bn = mathutils.Vector(n)
            if bn.length > 1e-6:
                bn.normalize()
            else:
                bn = mathutils.Vector((0, 0, 1))
            corner_normals.append((bn.x, bn.y, bn.z))

        faces.append(face_indices)
        face_texcoords.append(corner_uvs)
        face_normals.append(corner_normals)
        face_material_ids.append(cf.mat_id)
        face_smooth_flags.append(cf.smooth_group != 0)
        face_smoothing_groups.append(_to_signed_i32(cf.smooth_group))

    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj["_cgf_source_vert_ids"] = vertex_source_ids
    obj["_cgf_face_smoothing_groups"] = face_smoothing_groups

    for poly_index, poly in enumerate(mesh.polygons):
        if poly_index < len(face_smooth_flags):
            poly.use_smooth = face_smooth_flags[poly_index]

    # Custom normals
    if import_normals and face_normals:
        normals = [n for face in face_normals for n in face]
        try:
            if hasattr(mesh, 'use_auto_smooth'):
                mesh.use_auto_smooth = True
            for poly in mesh.polygons:
                poly.use_smooth = True
            if len(normals) == len(mesh.loops):
                mesh.normals_split_custom_set(normals)
        except Exception:
            pass

    # UVs
    if import_uvs and face_texcoords:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        loop_index = 0
        for poly_index, poly in enumerate(mesh.polygons):
            corner_uvs = face_texcoords[poly_index]
            for corner_idx in range(poly.loop_total):
                if corner_idx < len(corner_uvs):
                    u, v = corner_uvs[corner_idx]
                    uv_layer.data[loop_index].uv = (u, v)
                loop_index += 1

    # Materials
    # face.mat_id in CGF = global index among ALL standard materials in file,
    # excluding Multi materials. This matches how Max buildMatMappings() works.
    # We build the same mapping: standard_mat_index → (blender_material, slot_index)
    if import_materials and blender_materials:
        global_material_map = _build_global_standard_material_map(archive, blender_materials)
        slot_map = {}  # face.mat_id → mesh material slot index
        for pi, poly in enumerate(mesh.polygons):
            if pi >= len(face_material_ids):
                continue
            face_mat_id = face_material_ids[pi]
            bmat = global_material_map.get(face_mat_id)
            if bmat is None:
                continue
            if bmat.name not in [m.name for m in mesh.materials]:
                mesh.materials.append(bmat)
            if face_mat_id not in slot_map:
                slot_map[face_mat_id] = list(mesh.materials).index(bmat)
            poly.material_index = slot_map[face_mat_id]

    # Transform
    if apply_node_transform and node_chunk and node_chunk.trans_matrix:
        obj.matrix_world = cry_matrix_to_blender(node_chunk.trans_matrix)
    elif apply_node_transform and node_chunk:
        obj.location = cry_vec(node_chunk.position)

    # Vertex weights
    if import_weights and mc.physique and archive.bone_anim_chunks:
        _assign_weights(obj, mc, archive, vertex_source_ids)

    return obj


def _assign_weights(obj, mc, archive, vertex_source_ids=None):
    print(f"[CGF] Assigning weights: {len(mc.physique)} source vertices...")
    names = {}
    if archive.bone_name_list_chunks:
        for i, n in enumerate(archive.bone_name_list_chunks[0].name_list):
            names[i] = n

    links_by_source_vid = {bl.vertex_id: bl.links for bl in mc.physique}

    if vertex_source_ids is None:
        vertex_source_ids = list(range(len(mc.vertices)))

    for vid, src_vid in enumerate(vertex_source_ids):
        for lnk in links_by_source_vid.get(src_vid, []):
            bname = names.get(lnk.bone_id, f"Bone_{lnk.bone_id}")
            if bname not in obj.vertex_groups:
                obj.vertex_groups.new(name=bname)
            obj.vertex_groups[bname].add([vid], lnk.blending, 'REPLACE')
    print(f"[CGF] Weights done")


def _collect_standard_chunks(mat_chunk, archive, result):
    """
    Recursively collect all STANDARD material chunks in order, skipping Multi.
    This mirrors Max's tempStandardMatArray — face.mat_id is an index into this list.

    materialType_Standard = 1
    materialType_Multi    = 2
    """
    if mat_chunk.type == 1:  # Standard
        result.append(mat_chunk)
    elif mat_chunk.type == 2:  # Multi — recurse into children
        for cid in mat_chunk.children:
            child = archive.get_material_chunk(cid)
            if child:
                _collect_standard_chunks(child, archive, result)
    else:
        # Unknown type — treat as standard
        result.append(mat_chunk)


# ── Armature ──────────────────────────────────────────────────────────────────

def _build_material_cache(archive, filepath, import_materials, game_root_path="", skip_collision_geometry=False):
    by_name = {}
    by_signature = {}

    if not import_materials:
        return by_name, by_signature

    for mc in archive.material_chunks:
        standard_chunks = []
        _collect_standard_chunks(mc, archive, standard_chunks)
        print(f"[CGF]   material chunk: {mc.name} type={mc.type} -> {len(standard_chunks)} standard")
        for std in standard_chunks:
            if skip_collision_geometry and _is_nodraw_material(std):
                continue
            std_key = _build_cgf_mat_name(std.name, std.shader_name, std.surface_name)
            signature = _material_signature(std, filepath, game_root_path)
            bmat = by_signature.get(signature)
            if bmat is None:
                bmat = build_material(std, filepath, import_materials, game_root_path)
                if bmat:
                    by_signature[signature] = bmat
            if bmat:
                by_name[std_key] = bmat

    return by_name, by_signature


def _build_global_standard_material_map(archive, blender_materials):
    slot_map = {}
    standard_index = 0

    for mc in _global_standard_material_chunks(archive):
        std_key = _build_cgf_mat_name(mc.name, mc.shader_name, mc.surface_name)
        bmat = blender_materials.get(std_key)
        if bmat is not None:
            slot_map[standard_index] = bmat
        standard_index += 1

    return slot_map


def _source_vert_map_from_object(obj):
    values = obj.get("_cgf_source_vert_ids")
    if not values:
        return None
    source_map = {}
    for mesh_vid, src_vid in enumerate(values):
        source_map.setdefault(int(src_vid), []).append(mesh_vid)
    return source_map


def _to_signed_i32(value):
    """Normalize integer to Blender custom-property signed 32-bit range."""
    v = int(value) & 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    return v


def _find_asset_root_node(archive):
    if archive.bone_initial_pos_chunks:
        target_object_id = archive.bone_initial_pos_chunks[0].mesh_chunk_id
        node = next((n for n in archive.node_chunks if n.object_id == target_object_id), None)
        if node:
            return node
    return next((n for n in archive.node_chunks if n.object_id >= 0), None)


def _build_asset_root(file_name, archive, collection):
    root_name = f"{file_name}_ROOT"
    root_obj = bpy.data.objects.new(root_name, None)
    root_obj.empty_display_type = 'PLAIN_AXES'
    root_obj.empty_display_size = 0.05
    collection.objects.link(root_obj)

    root_node = _find_asset_root_node(archive)
    if root_node and root_node.trans_matrix:
        try:
            root_obj.matrix_world = cry_matrix_to_blender(root_node.trans_matrix)
        except Exception as e:
            print(f"[CGF] Asset root transform error {root_node.name}: {e}")

    if root_node:
        root_obj["cry_node_name"] = root_node.name
        root_obj["cry_object_id"] = int(root_node.object_id)
        root_obj["cry_parent_id"] = int(root_node.parent_id)

    return root_obj, root_node


def _parent_object_under_root(obj, root_obj, world_matrix):
    if obj is None or root_obj is None:
        return
    obj.parent = root_obj
    obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    local_m = root_obj.matrix_world.inverted_safe() @ world_matrix
    obj.rotation_mode = 'QUATERNION'
    try:
        loc, rot, scl = local_m.decompose()
        obj.location = loc
        obj.rotation_quaternion = rot
        obj.scale = scl
    except Exception:
        obj.matrix_local = local_m


def _build_scene_node_objects(archive, collection, asset_root_obj=None, existing_objects=None):
    existing_objects = existing_objects or {}
    node_objects = {}

    for node in archive.node_chunks:
        if not node or not node.name:
            continue
        if node.name in existing_objects:
            continue

        obj = bpy.data.objects.new(node.name, None)
        obj.empty_display_type = 'PLAIN_AXES'
        obj.empty_display_size = 0.03
        collection.objects.link(obj)

        if node.trans_matrix:
            try:
                obj.matrix_world = cry_matrix_to_blender(node.trans_matrix)
            except Exception as e:
                print(f"[CGF] Scene node transform error {node.name}: {e}")

        obj["cry_node_name"] = node.name
        obj["cry_node_chunk_id"] = int(node.header.chunk_id) if node.header else -1
        obj["cry_object_id"] = int(node.object_id)
        obj["cry_parent_id"] = int(node.parent_id)
        if node.pos_ctrl_id:
            obj["cry_pos_ctrl_id"] = node.pos_ctrl_id
        if node.rot_ctrl_id:
            obj["cry_rot_ctrl_id"] = node.rot_ctrl_id
        if node.scale_ctrl_id:
            obj["cry_scale_ctrl_id"] = node.scale_ctrl_id

        node_objects[int(node.header.chunk_id) if node.header else len(node_objects)] = obj
        existing_objects[node.name] = obj

    for node in archive.node_chunks:
        if not node or not node.name:
            continue
        child = node_objects.get(int(node.header.chunk_id) if node.header else -1)
        if child is None:
            continue

        parent = node_objects.get(int(node.parent_id))
        if parent is not None:
            world_m = child.matrix_world.copy()
            _parent_object_under_root(child, parent, world_m)
        elif asset_root_obj is not None:
            world_m = child.matrix_world.copy()
            _parent_object_under_root(child, asset_root_obj, world_m)

    return node_objects


def _build_helper_objects(archive, collection, asset_root_obj=None, existing_objects=None):
    existing_objects = existing_objects or {}
    helper_objects = []

    for idx, helper in enumerate(getattr(archive, "helper_chunks", [])):
        name = f"Helper_{idx}"
        if name in existing_objects:
            continue
        obj = bpy.data.objects.new(name, None)
        obj.empty_display_type = 'CUBE'
        obj.empty_display_size = max(helper.size) * INCHES_TO_METERS if getattr(helper, "size", None) else 0.025
        collection.objects.link(obj)
        if asset_root_obj is not None:
            world_m = obj.matrix_world.copy()
            _parent_object_under_root(obj, asset_root_obj, world_m)
        obj["cry_helper_type"] = int(getattr(helper, "type", 0))
        obj["cry_helper_chunk_id"] = int(helper.header.chunk_id) if helper.header else idx
        helper_objects.append(obj)
        existing_objects[name] = obj

    return helper_objects


def _collection_bounds(objects):
    corners = []
    for obj in objects:
        if obj is None:
            continue
        try:
            bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
            corners.extend(bbox)
        except Exception:
            try:
                corners.append(obj.matrix_world.translation.copy())
            except Exception:
                pass
    if not corners:
        return mathutils.Vector((0.0, 0.0, 0.0)), 1.0
    min_v = mathutils.Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_v = mathutils.Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    center = (min_v + max_v) * 0.5
    radius = max((max_v - min_v).length * 0.75, 0.25)
    return center, radius


def _look_at_quaternion(location, target):
    direction = (target - location)
    if direction.length < 1e-8:
        return mathutils.Euler((0.0, 0.0, 0.0)).to_quaternion()
    return direction.to_track_quat('-Z', 'Y')


def _build_producer_cameras(file_name, collection, target_objects, asset_root_obj=None):
    center, radius = _collection_bounds(target_objects)
    camera_specs = [
        ("Producer Back",        mathutils.Vector((0.0, -radius * 2.5, radius * 0.8))),
        ("Producer Bottom",      mathutils.Vector((0.0, 0.0, -radius * 2.5))),
        ("Producer Front",       mathutils.Vector((0.0, radius * 2.5, radius * 0.8))),
        ("Producer Left",        mathutils.Vector((-radius * 2.5, 0.0, radius * 0.8))),
        ("Producer Perspective", mathutils.Vector((radius * 1.8, -radius * 1.8, radius * 1.4))),
        ("Producer Right",       mathutils.Vector((radius * 2.5, 0.0, radius * 0.8))),
        ("Producer Top",         mathutils.Vector((0.0, 0.0, radius * 2.5))),
    ]

    camera_objects = []
    for cam_name, offset in camera_specs:
        data = bpy.data.cameras.new(f"{file_name}_{cam_name}")
        obj = bpy.data.objects.new(cam_name, data)
        obj.location = center + offset
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = _look_at_quaternion(obj.location, center)
        collection.objects.link(obj)
        camera_objects.append(obj)

    return camera_objects


def build_armature(archive, collection, asset_root_obj=None, apply_asset_transform=True):
    if not archive.bone_anim_chunks or not archive.bone_anim_chunks[0].bones:
        return None, None

    names = archive.bone_name_list_chunks[0].name_list if archive.bone_name_list_chunks else []

    arm_data = bpy.data.armatures.new("Armature")
    arm_obj  = bpy.data.objects.new("Armature", arm_data)
    collection.objects.link(arm_obj)

    # Keep the armature object neutral. BoneInitialPos already contains the bind
    # placement; adding armature object-space here splits root/root1 away from the
    # scene pivots seen in Max.
    arm_obj.matrix_world = mathutils.Matrix.Identity(4)

    # Make active and enter edit mode
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    # Find window/area/region for temp_override
    win    = bpy.context.window_manager.windows[0]
    screen = win.screen
    area   = next((a for a in screen.areas if a.type == 'VIEW_3D'), None)

    if area is None:
        # No 3D viewport — try any area
        area = screen.areas[0]

    region = next((r for r in area.regions if r.type == 'WINDOW'), area.regions[0])

    ctx = {
        'window': win, 'screen': screen, 'area': area, 'region': region,
        'active_object': arm_obj, 'object': arm_obj,
        'selected_objects': [arm_obj], 'selected_editable_objects': [arm_obj],
    }

    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='EDIT')

    bones = archive.bone_anim_chunks[0].bones
    init_mats = {}
    child_map = {}
    for bone in bones:
        init = archive.get_bone_initial_pos(bone.bone_id)
        if init:
            init_mats[bone.bone_id] = cry_bone_matrix43_to_blender(init)
        child_map.setdefault(bone.parent_id, []).append(bone.bone_id)

    eb_map = {}
    for bone in bones:
        bid   = bone.bone_id
        bname = names[bid] if bid < len(names) else (bone.name or f"Bone_{bid}")
        eb = arm_data.edit_bones.new(bname)
        eb.head = (0, 0, 0)
        eb.tail = (0, 0.05 * INCHES_TO_METERS, 0)

        mx = init_mats.get(bid)
        if mx is not None:
            try:
                bone_len = 0.05 * INCHES_TO_METERS
                eb.head = mathutils.Vector((0.0, 0.0, 0.0))
                eb.tail = mathutils.Vector((0.0, bone_len, 0.0))
                try:
                    eb.matrix = mx.copy()
                    eb.length = bone_len
                except Exception:
                    head = mx.translation.copy()
                    # Fallback path for Blender builds that reject direct edit-bone
                    # matrix assignment: reconstruct the same basis manually.
                    tail_dir = mx.col[1].xyz.copy()
                    if tail_dir.length <= 1e-8:
                        tail_dir = mathutils.Vector((0.0, 1.0, 0.0))
                    tail = head + tail_dir.normalized() * bone_len
                    eb.head = head
                    eb.tail = tail
                    local_up = mx.col[2].xyz.copy()
                    if local_up.length > 1e-8:
                        eb.align_roll(local_up.normalized())
            except Exception as e:
                print(f"[CGF] Bone matrix error {bname}: {e}")
        eb_map[bid] = eb

    for bone in bones:
        if bone.parent_id >= 0 and bone.parent_id in eb_map:
            child  = eb_map[bone.bone_id]
            parent = eb_map[bone.parent_id]
            child.parent = parent
            # Match the Max 8 CryImporter more closely: bones are parented, but not
            # hard-connected. Connected Blender bones suppress/alter translation in
            # pose mode, which breaks CAF controllers that animate PRS directly.
            child.use_connect = False

    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')

    debug_bones = {"root1", "weapon", "reload"}
    if arm_data.bones:
        for bone in bones:
            bid = bone.bone_id
            bname = names[bid] if bid < len(names) else (bone.name or f"Bone_{bid}")
            if bname not in debug_bones:
                continue
            raw_init = init_mats.get(bid)
            data_bone = arm_data.bones.get(bname)
            if data_bone is None:
                continue
            if raw_init is not None:
                print(f"[RIG-DEBUG] bone={bname} bind_local={_fmt_matrix4(raw_init)}")
            print(f"[RIG-DEBUG] bone={bname} bone_local={_fmt_matrix4(data_bone.matrix_local.copy())}")

    # Preserve original bone metadata for round-trip export.
    for bone in archive.bone_anim_chunks[0].bones:
        bid   = bone.bone_id
        bname = names[bid] if bid < len(names) else (bone.name or f"Bone_{bid}")
        if not arm_obj.pose or bname not in arm_obj.pose.bones:
            continue
        pbone = arm_obj.pose.bones[bname]
        pbone['cry_ctrl_id'] = bone.ctrl_id
        pbone['cry_bone_id'] = int(bone.bone_id)
        pbone['cry_parent_id'] = int(bone.parent_id)
        pbone['cry_custom_property'] = bone.custom_property or ""
        if bone.bone_physics:
            pbone['cry_bone_mesh_id'] = int(bone.bone_physics.mesh_id)
            pbone['cry_bone_flags'] = f"{int(bone.bone_physics.flags) & 0xFFFFFFFF:08X}"

    # Store original CGF bone matrices on armature for round-trip export
    # Must be done AFTER exit from edit mode (data bones are accessible now)
    cgf_matrices = {}
    for bone in archive.bone_anim_chunks[0].bones:
        bid   = bone.bone_id
        bname = names[bid] if bid < len(names) else (bone.name or f"Bone_{bid}")
        init  = archive.get_bone_initial_pos(bid)
        if init:
            cgf_matrices[bname] = list(init)

    if cgf_matrices:
        import json
        arm_obj['cgf_bone_matrices'] = json.dumps(cgf_matrices)

    return arm_obj, arm_data


def apply_armature_to_meshes(arm_obj, mesh_objects, preserve_world=False):
    if not arm_obj:
        return
    for obj in mesh_objects:
        if obj and obj.vertex_groups:
            world_m = obj.matrix_world.copy()
            obj.parent = arm_obj
            if preserve_world:
                obj.matrix_parent_inverse = arm_obj.matrix_world.inverted_safe()
                try:
                    obj.matrix_world = world_m
                except Exception:
                    pass
            mod = obj.modifiers.new("Armature", 'ARMATURE')
            mod.object = arm_obj
            mod.use_vertex_groups = True


# ── Shape keys ────────────────────────────────────────────────────────────────

def build_shape_keys(obj, mesh_chunk, archive):
    morphs = archive.get_morphs_for_mesh(mesh_chunk.header.chunk_id)
    if not morphs:
        return
    source_map = _source_vert_map_from_object(obj)
    obj.shape_key_add(name="Basis", from_mix=False)
    for morph in morphs:
        sk = obj.shape_key_add(name=morph.name, from_mix=False)
        for mv in morph.target_vertices:
            target_ids = source_map.get(mv.vertex_id, []) if source_map else [mv.vertex_id]
            for target_id in target_ids:
                if target_id < len(sk.data):
                    sk.data[target_id].co = cry_vec(mv.target_point)


# ── Animation ─────────────────────────────────────────────────────────────────

def apply_animation(arm_obj, geom_archive, anim_archive, action_name="Action", debug_caf=False, playback_mode="ARMATURE"):
    """
    Apply controller chunks from anim_archive to the armature.
    Ported from CryImporter-scenebuilder.ms createController826/827 + addAnim.

    The controller chunk's ctrl_id matches the bone's ctrl_id.
    Keys are in ticks; divide by ticks_per_frame to get frame number.
    """
    if not arm_obj:
        return

    tpf = anim_archive.get_ticks_per_frame()
    fps = round(1.0 / (anim_archive.get_secs_per_tick() * tpf))
    if fps <= 0:
        fps = 25

    # Set scene FPS
    bpy.context.scene.render.fps = fps

    # Start from a neutral pose. Cry CAF files can animate only a subset of bones,
    # so unanimated bones should stay at identity pose deltas relative to rest.
    if arm_obj.pose:
        for pbone in arm_obj.pose.bones:
            pbone.rotation_mode = 'QUATERNION'
            pbone.location = (0.0, 0.0, 0.0)
            pbone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pbone.scale = (1.0, 1.0, 1.0)

    # Build ctrl_id → bone name map from geom archive
    # Bone ctrl_id is stored as 8-char hex string in CryBone
    ctrl_to_bone = {}
    if geom_archive.bone_anim_chunks:
        name_list = geom_archive.bone_name_list_chunks[0].name_list \
                    if geom_archive.bone_name_list_chunks else []
        for bone in geom_archive.bone_anim_chunks[0].bones:
            bid = bone.bone_id
            bname = name_list[bid] if bid < len(name_list) else f"Bone_{bid}"
            if bone.ctrl_id and bone.ctrl_id != "FFFFFFFF":
                ctrl_to_bone[bone.ctrl_id] = bname

    ctrl_to_objects = {}
    try:
        for obj in bpy.context.scene.objects:
            for prop_name in ("cry_pos_ctrl_id", "cry_rot_ctrl_id", "cry_scale_ctrl_id"):
                ctrl_id = obj.get(prop_name)
                if ctrl_id and ctrl_id != "FFFFFFFF":
                    ctrl_to_objects.setdefault(ctrl_id, []).append(obj)
    except Exception:
        pass

    if not anim_archive.controller_chunks:
        print("[CGF] No controller chunks found in animation file")
        return

    # Determine total frame range from timing chunk
    frame_start = 0
    frame_end   = 0
    time_offset_ticks = 0
    first_key_tick = None
    last_key_tick = None
    for ctrl_chunk in anim_archive.controller_chunks:
        for key in _effective_ctrl_keys(ctrl_chunk):
            tick = int(getattr(key, 'time', 0))
            first_key_tick = tick if first_key_tick is None else min(first_key_tick, tick)
            last_key_tick = tick if last_key_tick is None else max(last_key_tick, tick)

    if first_key_tick is not None and last_key_tick is not None:
        time_offset_ticks = -first_key_tick
        frame_start = 0
        frame_end = int(round((last_key_tick - first_key_tick) / tpf))
    elif anim_archive.timing_chunks:
        gr = anim_archive.timing_chunks[0].global_range
        if gr:
            time_offset_ticks = -int(gr[1])
            frame_start = 0
            frame_end = int(round((gr[2] - gr[1]) / tpf))

    # Create a fresh action for each imported animation.
    # Reusing an existing datablock by name can pick up an incompatible or stale
    # action object in newer Blender versions.
    action = bpy.data.actions.new(name=action_name)
    _assign_action_to_armature(arm_obj, action)
    try:
        arm_obj._cry_geom_archive_ref = geom_archive
    except Exception:
        pass

    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end   = max(frame_end, frame_start + 1)

    crybone_chunks = [
        ctrl_chunk for ctrl_chunk in anim_archive.controller_chunks
        if getattr(ctrl_chunk, 'ctrl_type', None) == CTRL_CRY_BONE and getattr(ctrl_chunk, 'keys', None)
    ]
    crybone_baked = False
    if crybone_chunks:
        _restore_mesh_armature_playback(arm_obj)
        crybone_baked = _apply_crybone_controllers(
            arm_obj,
            geom_archive,
            crybone_chunks,
            ctrl_to_bone,
            tpf,
            time_offset_ticks=time_offset_ticks,
            debug_caf=debug_caf,
            action=action,
            playback_mode=playback_mode,
        )

    for ctrl_chunk in anim_archive.controller_chunks:
        if not ctrl_chunk.keys:
            continue
        target_objects = ctrl_to_objects.get(ctrl_chunk.ctrl_id, [])
        for target_obj in target_objects:
            _apply_controller_to_object(
                target_obj, ctrl_chunk, action_name, tpf, time_offset_ticks=time_offset_ticks
            )

        if ctrl_chunk.ctrl_type != CTRL_CRY_BONE:
            bone_name = ctrl_to_bone.get(ctrl_chunk.ctrl_id)
            if not bone_name:
                continue

            if bone_name not in arm_obj.pose.bones:
                continue

            pbone = arm_obj.pose.bones[bone_name]
            _apply_controller_to_bone(
                pbone, ctrl_chunk, action, tpf, bone_name, time_offset_ticks=time_offset_ticks
            )

    # Keep CE1 imports on pure armature playback by default.
    # The CRYPREVIEW path bakes per-frame shape keys and disables the Armature
    # modifier, which is useful for diagnostics but can mask whether the bone
    # animation itself is correct.
    baked_preview = bool(crybone_baked)
    print(f"[CGF] CRYPREVIEW baked: {baked_preview}")

    # After key insertion Blender can leave pose bones in the last written state.
    # Reset back to rest deltas so action evaluation on frame_set() starts clean.
    if arm_obj.pose:
        for pbone in arm_obj.pose.bones:
            pbone.rotation_mode = 'QUATERNION'
            pbone.location = (0.0, 0.0, 0.0)
            pbone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pbone.scale = (1.0, 1.0, 1.0)

    # Re-assign after writing keys so Blender 5 action slots pick up the
    # populated datablock for playback/export reliably.
    _assign_action_to_armature(arm_obj, action)
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    try:
        bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    except Exception:
        pass

    if debug_caf:
        try:
            fcurves = _action_fcurves_view(action, arm_obj)
            fcount = len(fcurves) if fcurves is not None else 0
            frange = tuple(action.frame_range) if hasattr(action, 'frame_range') else None
            print(f"[CAF-DEBUG] action={action.name} fcurves={fcount} frame_range={frange}")
        except Exception:
            pass

    try:
        delattr(arm_obj, "_cry_geom_archive_ref")
    except Exception:
        pass

    print(f"[CGF] Animation '{action_name}' applied: {len(anim_archive.controller_chunks)} controllers, fps={fps}")


def _apply_controller_to_bone(pbone, ctrl_chunk, action, ticks_per_frame, bone_name, time_offset_ticks=0):
    """Apply a single controller chunk to a pose bone as F-Curves."""

    bone_path_loc  = f'pose.bones["{bone_name}"].location'
    bone_path_rot  = f'pose.bones["{bone_name}"].rotation_quaternion'
    bone_path_scl  = f'pose.bones["{bone_name}"].scale'
    pbone.rotation_mode = 'QUATERNION'

    from .cry_chunk_reader import (CTRL_CRY_BONE, CTRL_LINEAR3, CTRL_LINEAR_Q,
                              CTRL_BEZIER3, CTRL_BEZIER_Q,
                              CTRL_TCB3, CTRL_TCBQ)

    ct = ctrl_chunk.ctrl_type
    ctrl_ver = getattr(getattr(ctrl_chunk, 'header', None), 'version', 0)
    # v827 or v826 CryBone: pos + rotation (as quat or rotLog)
    if ct == CTRL_CRY_BONE:
        for key in ctrl_chunk.keys:
            frame = (key.time + time_offset_ticks) / ticks_per_frame

            # Position
            s = INCHES_TO_METERS
            if hasattr(key, 'rel_pos'):
                # CryBoneKey (v826)
                pos = key.rel_pos
                q   = cry_quat(key.rel_quat)
                pos_vec = _cry_anim_pos_to_blender((pos[0]*s, pos[1]*s, pos[2]*s))
                quat = _cry_anim_quat_to_blender(mathutils.Quaternion((q.w, q.x, q.y, q.z)))
            else:
                # CryKey (v827): rot_log is logarithm of quat
                pos = key.pos
                q   = quat_exp(key.rot_log)
                pos_vec = cry_vec(pos)
                quat = q.copy()

            if ctrl_ver == 0x0827 and not hasattr(key, 'rel_pos'):
                # Max 8 writes v827 as absolute local PRS on the bone object.
                # The closest Blender analogue is a direct local basis transform.
                quat = _quat_in_rest_basis(pbone, quat)
                _set_pose_from_anim_basis(
                    pbone,
                    pos=pos_vec,
                    quat=quat,
                )
            else:
                _set_pose_from_anim_local(
                    pbone,
                    pos=pos_vec,
                    quat=quat,
                )
            pbone.keyframe_insert(data_path="location", frame=frame)
            pbone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    # Linear position
    elif ct == CTRL_LINEAR3:
        s = INCHES_TO_METERS
        for key in ctrl_chunk.keys:
            frame = (key.time + time_offset_ticks) / ticks_per_frame
            pos_vec = _cry_anim_pos_to_blender((key.val[0]*s, key.val[1]*s, key.val[2]*s))
            _set_pose_from_anim_local(pbone, pos=pos_vec)
            pbone.keyframe_insert(data_path="location", frame=frame)

    # Linear rotation (quat)
    elif ct == CTRL_LINEAR_Q:
        for key in ctrl_chunk.keys:
            frame = (key.time + time_offset_ticks) / ticks_per_frame
            q = cry_quat(key.val)
            quat = _cry_anim_quat_to_blender(mathutils.Quaternion((q.w, q.x, q.y, q.z)))
            _set_pose_from_anim_local(pbone, quat=quat)
            pbone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    # Bezier position
    elif ct == CTRL_BEZIER3:
        s = INCHES_TO_METERS
        for key in ctrl_chunk.keys:
            frame = (key.time + time_offset_ticks) / ticks_per_frame
            pos_vec = _cry_anim_pos_to_blender((key.val[0]*s, key.val[1]*s, key.val[2]*s))
            _set_pose_from_anim_local(pbone, pos=pos_vec)
            pbone.keyframe_insert(data_path="location", frame=frame)

    # Bezier rotation (quat, no tangents for rotation)
    elif ct == CTRL_BEZIER_Q:
        for key in ctrl_chunk.keys:
            frame = (key.time + time_offset_ticks) / ticks_per_frame
            q = cry_quat(key.val)
            quat = _cry_anim_quat_to_blender(mathutils.Quaternion((q.w, q.x, q.y, q.z)))
            _set_pose_from_anim_local(pbone, quat=quat)
            pbone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    # TCB position
    elif ct == CTRL_TCB3:
        s = INCHES_TO_METERS
        for key in ctrl_chunk.keys:
            frame = (key.time + time_offset_ticks) / ticks_per_frame
            pos_vec = _cry_anim_pos_to_blender((key.val[0]*s, key.val[1]*s, key.val[2]*s))
            _set_pose_from_anim_local(pbone, pos=pos_vec)
            pbone.keyframe_insert(data_path="location", frame=frame)

    # TCB rotation
    elif ct == CTRL_TCBQ:
        for key in ctrl_chunk.keys:
            frame = (key.time + time_offset_ticks) / ticks_per_frame
            q = cry_quat(key.val)
            quat = _cry_anim_quat_to_blender(mathutils.Quaternion((q.w, q.x, q.y, q.z)))
            _set_pose_from_anim_local(pbone, quat=quat)
            pbone.keyframe_insert(data_path="rotation_quaternion", frame=frame)


def _apply_controller_to_object(obj, ctrl_chunk, action_name, ticks_per_frame, time_offset_ticks=0):
    if obj is None or not getattr(ctrl_chunk, 'keys', None):
        return

    action = bpy.data.actions.new(name=f"{action_name}_{obj.name}")
    _assign_action_to_object(obj, action)
    obj.rotation_mode = 'QUATERNION'

    ct = ctrl_chunk.ctrl_type
    ctrl_ver = getattr(getattr(ctrl_chunk, 'header', None), 'version', 0)

    if ct == CTRL_CRY_BONE:
        for key in ctrl_chunk.keys:
            frame = (key.time + time_offset_ticks) / ticks_per_frame
            if hasattr(key, 'rel_pos'):
                pos = key.rel_pos
                q = cry_quat(key.rel_quat)
                pos_vec = _cry_anim_pos_to_blender((pos[0] * INCHES_TO_METERS, pos[1] * INCHES_TO_METERS, pos[2] * INCHES_TO_METERS))
                quat = _cry_anim_quat_to_blender(mathutils.Quaternion((q.w, q.x, q.y, q.z)))
            else:
                if ctrl_ver == 0x0827:
                    pos_vec = cry_vec(key.pos)
                    quat = quat_exp(key.rot_log)
                else:
                    pos_vec = cry_vec(key.pos)
                    quat = quat_exp(key.rot_log)
            _set_object_from_anim_local(obj, pos=pos_vec, quat=quat, scale=(1.0, 1.0, 1.0))
            _keyframe_object_transform(obj, frame)

    elif ct in {CTRL_LINEAR3, CTRL_BEZIER3, CTRL_TCB3}:
        for key in ctrl_chunk.keys:
            frame = (key.time + time_offset_ticks) / ticks_per_frame
            pos_vec = _cry_anim_pos_to_blender((key.val[0] * INCHES_TO_METERS, key.val[1] * INCHES_TO_METERS, key.val[2] * INCHES_TO_METERS))
            _set_object_from_anim_local(obj, pos=pos_vec)
            obj.keyframe_insert(data_path="location", frame=frame)

    elif ct in {CTRL_LINEAR_Q, CTRL_BEZIER_Q, CTRL_TCBQ}:
        for key in ctrl_chunk.keys:
            frame = (key.time + time_offset_ticks) / ticks_per_frame
            q = cry_quat(key.val)
            quat = _cry_anim_quat_to_blender(mathutils.Quaternion((q.w, q.x, q.y, q.z)))
            _set_object_from_anim_local(obj, quat=quat)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    elif ct in {CTRL_LINEAR1, CTRL_BEZIER1, CTRL_TCB1}:
        for key in ctrl_chunk.keys:
            frame = (key.time + time_offset_ticks) / ticks_per_frame
            val = float(getattr(key, 'val', 1.0))
            _set_object_from_anim_local(obj, scale=(val, val, val))
            obj.keyframe_insert(data_path="scale", frame=frame)


# ── CAF file search (mirrors getCAFFilename from Max script) ──────────────────

def find_caf_file(caf_name, cal_filepath, geom_filepath):
    cal_dir  = os.path.dirname(cal_filepath)
    geom_dir = os.path.dirname(geom_filepath) if geom_filepath else ""
    candidates = [
        os.path.join(cal_dir,  caf_name),
        os.path.join(geom_dir, caf_name),
    ]
    for path in candidates:
        if os.path.isfile(path): return path
    return None


# ── Main load functions ───────────────────────────────────────────────────────

def load(operator, context, filepath,
         import_materials=True, import_normals=True, import_uvs=True,
         import_skeleton=True, import_weights=True, game_root_path="",
         skip_collision_geometry=False, create_asset_root_empty=True,
         apply_armature_node_transform=True, apply_mesh_node_transform=True,
         preserve_mesh_world_on_armature_parent=True, create_helper_nodes=True,
         create_controller_targets=True, create_producer_cameras=True):
    """Import a CGF/CGA geometry file."""

    print(f"[CGF] Loading: {filepath}")
    print(f"[CGF] Game root: '{game_root_path}'")
    reader = cry_chunk_reader.ChunkReader()
    try:
        print(f"[CGF] Reading file...")
        archive = reader.read_file(filepath)
    except ValueError as e:
        operator.report({'ERROR'}, str(e)); return {'CANCELLED'}

    print(f"[CGF] {archive.num_chunks} chunks — "
          f"meshes:{len(archive.mesh_chunks)} nodes:{len(archive.node_chunks)} "
          f"mats:{len(archive.material_chunks)} bones:{len(archive.bone_anim_chunks)}")

    file_name  = os.path.splitext(os.path.basename(filepath))[0]
    collection = bpy.data.collections.new(file_name)
    context.scene.collection.children.link(collection)
    asset_root_obj = None
    if create_asset_root_empty:
        asset_root_obj, _ = _build_asset_root(file_name, archive, collection)

    # Materials
    print(f"[CGF] Building materials...")
    blender_materials, _ = _build_material_cache(
        archive, filepath, import_materials, game_root_path, skip_collision_geometry
    )
    print(f"[CGF] Materials done: {len(blender_materials)}")

    # Armature
    arm_obj = None
    if import_skeleton and archive.bone_anim_chunks:
        print(f"[CGF] Building armature...")
        arm_obj, _ = build_armature(
            archive, collection, asset_root_obj=asset_root_obj,
            apply_asset_transform=apply_armature_node_transform
        )
        print(f"[CGF] Armature done: {arm_obj}")

    # Meshes
    print(f"[CGF] Building {len(archive.mesh_chunks)} mesh(es)...")
    mesh_objects = []
    for i, mc in enumerate(archive.mesh_chunks):
        print(f"[CGF]   mesh {i}: verts={len(mc.vertices)} faces={len(mc.faces)} bone_info={mc.has_bone_info} physique={len(mc.physique)}")
        if skip_collision_geometry and _mesh_is_collision_like(mc, archive):
            print(f"[CGF]   mesh {i} skipped as collision-like geometry")
            continue
        node = archive.get_node(mc.header.chunk_id)
        obj  = build_mesh(mc, node, archive, collection,
                          import_materials, import_normals, import_uvs,
                          import_weights, blender_materials, filepath,
                          skip_collision_geometry=skip_collision_geometry,
                          apply_node_transform=apply_mesh_node_transform)
        if obj:
            if asset_root_obj is not None:
                world_m = obj.matrix_world.copy()
                _parent_object_under_root(obj, asset_root_obj, world_m)
            obj['cgf_chunk_id'] = int(mc.header.chunk_id)
            obj['cgf_source_name'] = node.name if node and node.name else obj.name
            mesh_objects.append(obj)
            print(f"[CGF]   mesh {i} done: {obj.name}")
            if archive.mesh_morph_target_chunks:
                build_shape_keys(obj, mc, archive)

    print(f"[CGF] All meshes done")
    if arm_obj and import_skeleton and import_weights:
        apply_armature_to_meshes(arm_obj, mesh_objects, preserve_world=preserve_mesh_world_on_armature_parent)

    existing_objects = {obj.name: obj for obj in collection.objects}
    if create_controller_targets:
        _build_scene_node_objects(archive, collection, asset_root_obj=asset_root_obj, existing_objects=existing_objects)
    if create_helper_nodes:
        _build_helper_objects(archive, collection, asset_root_obj=asset_root_obj, existing_objects=existing_objects)
    if create_producer_cameras:
        _build_producer_cameras(file_name, collection, list(collection.objects), asset_root_obj=asset_root_obj)

    if arm_obj and import_skeleton and archive.controller_chunks:
        action_name = f"{file_name}_Embedded"
        apply_animation(arm_obj, archive, archive, action_name)
        try:
            context.scene.frame_set(context.scene.frame_start)
        except Exception:
            pass

    bpy.ops.object.select_all(action='DESELECT')
    for obj in collection.objects: obj.select_set(True)
    if mesh_objects: context.view_layer.objects.active = mesh_objects[0]

    operator.report({'INFO'},
        f"Imported {len(mesh_objects)} mesh(es) from {os.path.basename(filepath)}")
    return {'FINISHED'}


def _find_cgf_near(filepath):
    """
    Find a CGF or CGA file with the SAME base name as the given CAF/CAL file.
    Returns the path if found, or None.
    Does NOT fall back to random CGF files in the folder.
    """
    folder = os.path.dirname(filepath)
    base = os.path.splitext(os.path.basename(filepath))[0]
    for ext in ('.cgf', '.cga'):
        p = os.path.join(folder, base + ext)
        if os.path.isfile(p):
            return p
    return None


def _ensure_armature(operator, context, anim_filepath):
    # Check active object first
    arm_obj = context.active_object
    print(f"[CGF] active_object: {arm_obj} type: {arm_obj.type if arm_obj else None}")
    if arm_obj and arm_obj.type == 'ARMATURE':
        return arm_obj, None

    # Search the whole scene
    print(f"[CGF] Scene objects: {[o.name+':'+o.type for o in context.scene.objects]}")
    for obj in context.scene.objects:
        if obj.type == 'ARMATURE':
            return obj, None

    # No armature — try to auto-import CGF from same folder
    cgf_path = _find_cgf_near(anim_filepath)
    print(f"[CGF] CGF found near anim: {cgf_path}")
    if not cgf_path:
        base = os.path.splitext(os.path.basename(anim_filepath))[0]
        operator.report({'ERROR'},
            f"No CGF/CGA found with name '{base}' in the same folder. "
            f"Expected: {base}.cgf or {base}.cga")
        return None, None

    print(f"[CGF] Auto-importing geometry: {cgf_path}")
    reader = cry_chunk_reader.ChunkReader()
    try:
        archive = reader.read_file(cgf_path)
    except ValueError as e:
        operator.report({'ERROR'}, f"Failed to read CGF: {e}")
        return None, None

    print(f"[CGF] Archive: bone_anim_chunks={len(archive.bone_anim_chunks)} mesh={len(archive.mesh_chunks)}")

    file_name  = os.path.splitext(os.path.basename(cgf_path))[0]
    collection = bpy.data.collections.new(file_name)
    context.scene.collection.children.link(collection)
    prefs = bpy.context.preferences.addons.get('io_import_cgf')
    pref_obj = prefs.preferences if prefs else None
    full_scene_setup = bool(getattr(pref_obj, "enable_scene_setup", True))
    create_asset_root_empty = full_scene_setup
    apply_armature_node_transform = True
    apply_mesh_node_transform = True
    preserve_mesh_world_on_armature_parent = True
    create_helper_nodes = full_scene_setup
    create_controller_targets = full_scene_setup
    create_producer_cameras = full_scene_setup

    asset_root_obj = None
    if create_asset_root_empty:
        asset_root_obj, _ = _build_asset_root(file_name, archive, collection)

    arm_obj = None
    if archive.bone_anim_chunks:
        try:
            arm_obj, _ = build_armature(
                archive, collection, asset_root_obj=asset_root_obj,
                apply_asset_transform=apply_armature_node_transform
            )
            print(f"[CGF] build_armature result: {arm_obj}")
        except Exception as e:
            print(f"[CGF] build_armature FAILED: {e}")
            import traceback; traceback.print_exc()

        if arm_obj:
            arm_obj['cgf_source_path'] = cgf_path
            if archive.bone_name_list_chunks:
                name_list = archive.bone_name_list_chunks[0].name_list
                for bone in archive.bone_anim_chunks[0].bones:
                    bid   = bone.bone_id
                    bname = name_list[bid] if bid < len(name_list) else f"Bone_{bid}"
                    if arm_obj.pose and bname in arm_obj.pose.bones:
                        arm_obj.pose.bones[bname]['cry_ctrl_id'] = bone.ctrl_id

    # Get game root from addon preferences
    game_root_path = ""
    skip_collision_geometry = False
    try:
        prefs = bpy.context.preferences.addons.get('io_import_cgf')
        if prefs:
            game_root_path = prefs.preferences.game_root_path
            skip_collision_geometry = bool(getattr(prefs.preferences, "skip_collision_geometry", False))
    except Exception:
        pass

    blender_materials, _ = _build_material_cache(
        archive, cgf_path, True, game_root_path, skip_collision_geometry
    )

    mesh_objects = []
    for mc in archive.mesh_chunks:
        if skip_collision_geometry and _mesh_is_collision_like(mc, archive):
            continue
        node = archive.get_node(mc.header.chunk_id)
        obj = build_mesh(mc, node, archive, collection,
                         import_materials=True, import_normals=True,
                         import_uvs=True, import_weights=True,
                         blender_materials=blender_materials, filepath=cgf_path,
                         skip_collision_geometry=skip_collision_geometry,
                         apply_node_transform=apply_mesh_node_transform)
        if obj:
            if asset_root_obj is not None:
                world_m = obj.matrix_world.copy()
                _parent_object_under_root(obj, asset_root_obj, world_m)
            mesh_objects.append(obj)

    if arm_obj and mesh_objects:
        apply_armature_to_meshes(arm_obj, mesh_objects, preserve_world=preserve_mesh_world_on_armature_parent)

    existing_objects = {obj.name: obj for obj in collection.objects}
    if create_controller_targets:
        _build_scene_node_objects(archive, collection, asset_root_obj=asset_root_obj, existing_objects=existing_objects)
    if create_helper_nodes:
        _build_helper_objects(archive, collection, asset_root_obj=asset_root_obj, existing_objects=existing_objects)
    if create_producer_cameras:
        _build_producer_cameras(file_name, collection, list(collection.objects), asset_root_obj=asset_root_obj)

    if arm_obj is None:
        operator.report({'ERROR'},
            "CGF imported but no armature was created (file has no skeleton).")
        return None, None

    context.view_layer.objects.active = arm_obj
    return arm_obj, archive


def load_caf(operator, context, filepath, append=True, debug_caf=False, playback_mode="ARMATURE"):
    """Import a CAF animation file. Auto-imports CGF if no armature in scene."""

    arm_obj, auto_archive = _ensure_armature(operator, context, filepath)
    if arm_obj is None:
        return {'CANCELLED'}

    # Use the auto-imported archive directly if available (avoids re-reading CGF)
    if auto_archive is not None:
        geom_archive = auto_archive
    else:
        geom_archive = _build_geom_archive_from_armature(arm_obj)

    print(f"[CGF] Loading animation: {filepath}")
    reader = cry_chunk_reader.ChunkReader()
    try:
        anim_archive = reader.read_file(filepath)
    except ValueError as e:
        operator.report({'ERROR'}, str(e)); return {'CANCELLED'}

    print(f"[CGF] Controllers: {len(anim_archive.controller_chunks)}")

    action_name = os.path.splitext(os.path.basename(filepath))[0]
    apply_animation(arm_obj, geom_archive, anim_archive, action_name, debug_caf=debug_caf, playback_mode=playback_mode)
    try:
        context.scene.frame_set(context.scene.frame_start)
    except Exception:
        pass

    operator.report({'INFO'}, f"Animation '{action_name}' imported")
    return {'FINISHED'}


def load_cal(operator, context, filepath, debug_caf=False, playback_mode="ARMATURE"):
    """Import all animations from a CAL file. Auto-imports CGF if needed."""

    arm_obj, auto_archive = _ensure_armature(operator, context, filepath)
    if arm_obj is None:
        return {'CANCELLED'}

    if auto_archive is not None:
        geom_archive = auto_archive
    else:
        geom_archive = _build_geom_archive_from_armature(arm_obj)

    records = cry_chunk_reader.read_cal_file(filepath)
    if not records:
        operator.report({'WARNING'}, "CAL file is empty or could not be parsed")
        return {'CANCELLED'}

    imported = 0
    for rec in records:
        caf_path = find_caf_file(rec.path, filepath,
                                  arm_obj.get('cgf_source_path', ''))
        if not caf_path:
            print(f"[CGF] CAF not found: {rec.path}"); continue
        reader = cry_chunk_reader.ChunkReader()
        try:
            anim_archive = reader.read_file(caf_path)
        except Exception as e:
            print(f"[CGF] Failed {caf_path}: {e}"); continue

        apply_animation(arm_obj, geom_archive, anim_archive, rec.name, debug_caf=debug_caf, playback_mode=playback_mode)
        try:
            context.scene.frame_set(context.scene.frame_start)
        except Exception:
            pass
        imported += 1

    operator.report({'INFO'}, f"Imported {imported}/{len(records)} animations from CAL")
    return {'FINISHED'}


def _build_geom_archive_from_armature(arm_obj):
    """
    Reconstruct a minimal CryChunkArchive from an imported armature
    so we can match controller IDs to bone names during CAF import.
    Bone ctrl_ids are stored as custom properties on pose bones.
    Falls back to re-reading the source CGF if pose data is unavailable.
    """
    archive = cry_chunk_reader.CryChunkArchive()
    archive.geom_file_name = arm_obj.get('cgf_source_path', '')

    # Try to reload from source CGF first — most reliable
    source_path = arm_obj.get('cgf_source_path', '')
    if source_path and os.path.isfile(source_path):
        try:
            reader = cry_chunk_reader.ChunkReader()
            src = reader.read_file(source_path)
            archive.bone_anim_chunks      = src.bone_anim_chunks
            archive.bone_name_list_chunks = src.bone_name_list_chunks
            archive.bone_initial_pos_chunks = src.bone_initial_pos_chunks
            archive.node_chunks = src.node_chunks
            archive.mesh_chunks = src.mesh_chunks
            return archive
        except Exception as e:
            print(f"[CGF] Could not reload source CGF: {e}")

    # Fallback: build from pose bones + stored ctrl_ids
    bac  = cry_chunk_reader.CryBoneAnimChunk()
    bac.header  = cry_chunk_reader.ChunkHeader()
    bnlc = cry_chunk_reader.CryBoneNameListChunk()
    bnlc.header = cry_chunk_reader.ChunkHeader()

    pose_bones = arm_obj.pose.bones if arm_obj.pose else []
    for i, pbone in enumerate(pose_bones):
        bone = cry_chunk_reader.CryBone()
        bone.bone_id = i
        bone.name    = pbone.name
        bone.ctrl_id = pbone.get('cry_ctrl_id', 'FFFFFFFF')
        bac.bones.append(bone)
        bnlc.name_list.append(pbone.name)

    archive.bone_anim_chunks.append(bac)
    archive.bone_name_list_chunks.append(bnlc)
    return archive
