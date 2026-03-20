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

from . import cgf_reader
from .cgf_reader import (CTRL_CRY_BONE, CTRL_LINEAR1, CTRL_LINEAR3, CTRL_LINEAR_Q,
                         CTRL_BEZIER1, CTRL_BEZIER3, CTRL_BEZIER_Q,
                         CTRL_TCB1, CTRL_TCB3, CTRL_TCBQ)

# ── Scale ─────────────────────────────────────────────────────────────────────
# 3ds Max default units = inches. 1 inch = 0.0254 meters.
INCHES_TO_METERS = 0.0254


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


# ── Material ──────────────────────────────────────────────────────────────────

def _build_cgf_mat_name(name, shader_name, surface_name):
    """Reconstruct full CGF material name: 'name(shader)/surface'"""
    result = name
    if shader_name:
        result += f"({shader_name})"
    if surface_name:
        result += f"/{surface_name}"
    return result


def _set_input(node, *names, value):
    for name in names:
        if name in node.inputs:
            try: node.inputs[name].default_value = value
            except Exception: pass
            return


def build_material(mat_chunk, filepath, import_materials, game_root_path=""):
    if not import_materials:
        return None
    mat = bpy.data.materials.get(mat_chunk.name)
    if mat:
        return mat

    mat = bpy.data.materials.new(name=mat_chunk.name)
    # Store original CGF material info for round-trip export
    mat['cgf_shader_name']  = mat_chunk.shader_name
    mat['cgf_surface_name'] = mat_chunk.surface_name
    mat['cgf_full_name']    = _build_cgf_mat_name(mat_chunk.name,
                                                    mat_chunk.shader_name,
                                                    mat_chunk.surface_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out  = nodes.new('ShaderNodeOutputMaterial'); out.location  = (400, 0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (0, 0)
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    d = mat_chunk.diffuse
    _set_input(bsdf, 'Base Color', value=(d[0], d[1], d[2], 1.0))

    s = mat_chunk.specular
    spec = ((s[0]+s[1]+s[2])/3.0) * mat_chunk.specular_level
    _set_input(bsdf, 'Specular IOR Level', 'Specular', value=min(spec, 1.0))

    if mat_chunk.specular_shininess > 0:
        _set_input(bsdf, 'Roughness',
                   value=1.0 - min(mat_chunk.specular_shininess/100.0, 1.0))

    if mat_chunk.opacity < 1.0:
        _set_input(bsdf, 'Alpha', value=mat_chunk.opacity)
        if hasattr(mat, 'blend_method'):
            mat.blend_method = 'BLEND'

    def add_tex(tex_data, x, y, color_space='sRGB'):
        if not tex_data or not tex_data.name:
            print(f"[CGF] add_tex: tex_data={tex_data}")
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
            node.image = img
        except Exception as e:
            print(f"[CGF] Load error {path}: {e}")
        return node

    tex_diff = add_tex(mat_chunk.tex_diffuse, -400, 0)
    if tex_diff:
        links.new(tex_diff.outputs['Color'], bsdf.inputs['Base Color'])

    tex_bump = add_tex(mat_chunk.tex_bump, -600, -300, 'Non-Color')
    if tex_bump:
        bump = nodes.new('ShaderNodeBump'); bump.location = (-200, -300)
        links.new(tex_bump.outputs['Color'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

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
    exts     = ['', '.dds', '.tga', '.png', '.jpg', '.bmp']

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

    r = try_path(os.path.join(cgf_dir, name))
    if r: return r

    r = try_path(os.path.join(cgf_dir, basename))
    if r: return r

    return None


# ── Mesh ──────────────────────────────────────────────────────────────────────

def build_mesh(mesh_chunk, node_chunk, archive, collection,
               import_materials, import_normals, import_uvs,
               import_weights, blender_materials, filepath):

    mc = mesh_chunk
    if not mc.vertices or not mc.faces:
        return None

    name = node_chunk.name if node_chunk else f"Mesh_{mc.header.chunk_id}"
    mesh = bpy.data.meshes.new(name)
    obj  = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    bm = bmesh.new()

    # Vertices
    # For skinned meshes, Max recalculates vertex positions using bone transforms:
    #   p = sum(link.offset * bone.transform * link.blending) for each link
    # link.offset is the vertex position in the bone's local space.
    # We precompute this if bone initial positions are available.
    for vi, cv in enumerate(mc.vertices):
        bm.verts.new(cry_vec(cv.pos))
    bm.verts.ensure_lookup_table()

    # Faces
    skipped = 0
    for cf in mc.faces:
        if cf.v0>=len(bm.verts) or cf.v1>=len(bm.verts) or cf.v2>=len(bm.verts):
            skipped += 1; continue
        try:
            f = bm.faces.new((bm.verts[cf.v0], bm.verts[cf.v1], bm.verts[cf.v2]))
            f.smooth = True
        except ValueError:
            skipped += 1
    bm.faces.ensure_lookup_table()

    # UVs
    if import_uvs and mc.tex_vertices:
        uv = bm.loops.layers.uv.new("UVMap")
        real_fi = 0
        for fi, cf in enumerate(mc.faces):
            if real_fi >= len(bm.faces): break
            face = bm.faces[real_fi]
            real_fi += 1

            # If texFaces exist use them, otherwise use geometry face indices (Max fallback)
            if mc.tex_faces and fi < len(mc.tex_faces):
                tf = mc.tex_faces[fi]
                tv = [tf.t0, tf.t1, tf.t2]
            else:
                tv = [cf.v0, cf.v1, cf.v2]

            for li, loop in enumerate(face.loops):
                tvi = tv[li]
                if tvi < len(mc.tex_vertices):
                    u, v = mc.tex_vertices[tvi]
                    loop[uv].uv = (u, v)

    bm.to_mesh(mesh)
    bm.free()

    # Custom normals
    if import_normals and mc.vertices:
        normals = []
        for cf in mc.faces:
            for vi in (cf.v0, cf.v1, cf.v2):
                if vi < len(mc.vertices):
                    n = mc.vertices[vi].normal
                    bn = mathutils.Vector(n)
                    if bn.length > 1e-6: bn.normalize()
                    normals.append(bn)
                else:
                    normals.append(mathutils.Vector((0, 0, 1)))
        try:
            if hasattr(mesh, 'use_auto_smooth'):
                mesh.use_auto_smooth = True
            if len(normals) == len(mesh.loops):
                mesh.normals_split_custom_set([(n.x,n.y,n.z) for n in normals])
        except Exception:
            pass

    # Materials
    # face.mat_id in CGF = global index among ALL standard materials in file,
    # excluding Multi materials. This matches how Max buildMatMappings() works.
    # We build the same mapping: standard_mat_index → (blender_material, slot_index)
    if import_materials and blender_materials:
        mat_chunk = archive.get_material_chunk(node_chunk.material_id) if node_chunk else None
        if mat_chunk:
            # Collect all standard material chunks in order (same as Max tempStandardMatArray)
            standard_chunks = []
            _collect_standard_chunks(mat_chunk, archive, standard_chunks)

            # Add all standard materials as slots and build matID → slot map
            slot_map = {}  # face.mat_id → mesh material slot index
            for i, std_chunk in enumerate(standard_chunks):
                if std_chunk.name in blender_materials:
                    bmat = blender_materials[std_chunk.name]
                    if bmat.name not in [m.name for m in mesh.materials]:
                        mesh.materials.append(bmat)
                    slot_map[i] = list(mesh.materials).index(bmat)

            # Assign material slots to polygons
            for pi, poly in enumerate(mesh.polygons):
                if pi < len(mc.faces):
                    mid = mc.faces[pi].mat_id
                    poly.material_index = slot_map.get(mid, 0)

    # Transform
    if node_chunk and node_chunk.trans_matrix:
        obj.matrix_world = cry_matrix_to_blender(node_chunk.trans_matrix)
    elif node_chunk:
        obj.location = cry_vec(node_chunk.position)

    # Vertex weights
    if import_weights and mc.physique and archive.bone_anim_chunks:
        _assign_weights(obj, mc, archive)

    return obj


def _assign_weights(obj, mc, archive):
    print(f"[CGF] Assigning weights: {len(mc.physique)} vertices...")
    names = {}
    if archive.bone_name_list_chunks:
        for i, n in enumerate(archive.bone_name_list_chunks[0].name_list):
            names[i] = n
    for bl in mc.physique:
        vid = bl.vertex_id
        for lnk in bl.links:
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

def build_armature(archive, collection):
    if not archive.bone_anim_chunks or not archive.bone_anim_chunks[0].bones:
        return None, None

    names = archive.bone_name_list_chunks[0].name_list if archive.bone_name_list_chunks else []

    arm_data = bpy.data.armatures.new("Armature")
    arm_obj  = bpy.data.objects.new("Armature", arm_data)
    collection.objects.link(arm_obj)

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

    eb_map = {}
    for bone in archive.bone_anim_chunks[0].bones:
        bid   = bone.bone_id
        bname = names[bid] if bid < len(names) else (bone.name or f"Bone_{bid}")
        eb = arm_data.edit_bones.new(bname)
        eb.head = (0, 0, 0)
        eb.tail = (0, 0.05 * INCHES_TO_METERS, 0)

        init = archive.get_bone_initial_pos(bid)
        if init:
            try:
                mx = cry_matrix43_to_blender(init)
                head = mx.translation
                local_x = mx.col[0].xyz.normalized() * (0.05 * INCHES_TO_METERS)
                eb.head = head
                eb.tail = head + local_x
            except Exception as e:
                print(f"[CGF] Bone matrix error {bname}: {e}")
        eb_map[bid] = eb

    for bone in archive.bone_anim_chunks[0].bones:
        if bone.parent_id >= 0 and bone.parent_id in eb_map:
            child  = eb_map[bone.bone_id]
            parent = eb_map[bone.parent_id]
            child.parent = parent
            if (child.head - parent.tail).length < 0.0001:
                child.use_connect = True

    with bpy.context.temp_override(**ctx):
        bpy.ops.object.mode_set(mode='OBJECT')

    return arm_obj, arm_data


def apply_armature_to_meshes(arm_obj, mesh_objects):
    if not arm_obj:
        return
    for obj in mesh_objects:
        if obj and obj.vertex_groups:
            obj.parent = arm_obj
            mod = obj.modifiers.new("Armature", 'ARMATURE')
            mod.object = arm_obj
            mod.use_vertex_groups = True


# ── Shape keys ────────────────────────────────────────────────────────────────

def build_shape_keys(obj, mesh_chunk, archive):
    morphs = archive.get_morphs_for_mesh(mesh_chunk.header.chunk_id)
    if not morphs:
        return
    obj.shape_key_add(name="Basis", from_mix=False)
    for morph in morphs:
        sk = obj.shape_key_add(name=morph.name, from_mix=False)
        for mv in morph.target_vertices:
            if mv.vertex_id < len(sk.data):
                sk.data[mv.vertex_id].co = cry_vec(mv.target_point)


# ── Animation ─────────────────────────────────────────────────────────────────

def apply_animation(arm_obj, geom_archive, anim_archive, action_name="Action"):
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

    if not anim_archive.controller_chunks:
        print("[CGF] No controller chunks found in animation file")
        return

    # Determine total frame range from timing chunk
    frame_start = 0
    frame_end   = 0
    if anim_archive.timing_chunks:
        gr = anim_archive.timing_chunks[0].global_range
        if gr:
            frame_start = gr[1]
            frame_end   = gr[2]

    # Create or get action
    action = bpy.data.actions.get(action_name)
    if action is None:
        action = bpy.data.actions.new(name=action_name)

    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    arm_obj.animation_data.action = action

    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end   = max(frame_end, frame_start + 1)

    for ctrl_chunk in anim_archive.controller_chunks:
        if not ctrl_chunk.keys:
            continue

        bone_name = ctrl_to_bone.get(ctrl_chunk.ctrl_id)
        if not bone_name:
            continue

        if bone_name not in arm_obj.pose.bones:
            continue

        pbone = arm_obj.pose.bones[bone_name]
        _apply_controller_to_bone(pbone, ctrl_chunk, action, tpf, bone_name)

    print(f"[CGF] Animation '{action_name}' applied: {len(anim_archive.controller_chunks)} controllers, fps={fps}")


def _apply_controller_to_bone(pbone, ctrl_chunk, action, ticks_per_frame, bone_name):
    """Apply a single controller chunk to a pose bone as F-Curves."""

    bone_path_loc  = f'pose.bones["{bone_name}"].location'
    bone_path_rot  = f'pose.bones["{bone_name}"].rotation_quaternion'
    bone_path_scl  = f'pose.bones["{bone_name}"].scale'
    pbone.rotation_mode = 'QUATERNION'

    def get_or_make_fcurve(data_path, index):
        fc = action.fcurves.find(data_path, index=index)
        if fc is None:
            fc = action.fcurves.new(data_path, index=index)
        return fc

    from .cgf_reader import (CTRL_CRY_BONE, CTRL_LINEAR3, CTRL_LINEAR_Q,
                              CTRL_BEZIER3, CTRL_BEZIER_Q,
                              CTRL_TCB3, CTRL_TCBQ)

    ct = ctrl_chunk.ctrl_type

    # v827 or v826 CryBone: pos + rotation (as quat or rotLog)
    if ct == CTRL_CRY_BONE:
        fc_loc = [get_or_make_fcurve(bone_path_loc, i) for i in range(3)]
        fc_rot = [get_or_make_fcurve(bone_path_rot, i) for i in range(4)]

        for key in ctrl_chunk.keys:
            frame = key.time / ticks_per_frame

            # Position
            s = INCHES_TO_METERS
            if hasattr(key, 'rel_pos'):
                # CryBoneKey (v826)
                pos = key.rel_pos
                q   = cry_quat(key.rel_quat)
            else:
                # CryKey (v827): rot_log is logarithm of quat
                pos = key.pos
                q   = quat_exp(key.rot_log)

            for i, v in enumerate((pos[0]*s, pos[1]*s, pos[2]*s)):
                fc_loc[i].keyframe_points.insert(frame, v, options={'FAST'})

            # Rotation (w, x, y, z)
            for i, v in enumerate((q.w, q.x, q.y, q.z)):
                fc_rot[i].keyframe_points.insert(frame, v, options={'FAST'})

    # Linear position
    elif ct == CTRL_LINEAR3:
        fc = [get_or_make_fcurve(bone_path_loc, i) for i in range(3)]
        s = INCHES_TO_METERS
        for key in ctrl_chunk.keys:
            frame = key.time / ticks_per_frame
            for i, v in enumerate((key.val[0]*s, key.val[1]*s, key.val[2]*s)):
                fc[i].keyframe_points.insert(frame, v, options={'FAST'})

    # Linear rotation (quat)
    elif ct == CTRL_LINEAR_Q:
        fc = [get_or_make_fcurve(bone_path_rot, i) for i in range(4)]
        for key in ctrl_chunk.keys:
            frame = key.time / ticks_per_frame
            q = cry_quat(key.val)
            for i, v in enumerate((q.w, q.x, q.y, q.z)):
                fc[i].keyframe_points.insert(frame, v, options={'FAST'})

    # Bezier position
    elif ct == CTRL_BEZIER3:
        fc = [get_or_make_fcurve(bone_path_loc, i) for i in range(3)]
        s = INCHES_TO_METERS
        for key in ctrl_chunk.keys:
            frame = key.time / ticks_per_frame
            for i, v in enumerate((key.val[0]*s, key.val[1]*s, key.val[2]*s)):
                fc[i].keyframe_points.insert(frame, v, options={'FAST'})

    # Bezier rotation (quat, no tangents for rotation)
    elif ct == CTRL_BEZIER_Q:
        fc = [get_or_make_fcurve(bone_path_rot, i) for i in range(4)]
        for key in ctrl_chunk.keys:
            frame = key.time / ticks_per_frame
            q = cry_quat(key.val)
            for i, v in enumerate((q.w, q.x, q.y, q.z)):
                fc[i].keyframe_points.insert(frame, v, options={'FAST'})

    # TCB position
    elif ct == CTRL_TCB3:
        fc = [get_or_make_fcurve(bone_path_loc, i) for i in range(3)]
        s = INCHES_TO_METERS
        for key in ctrl_chunk.keys:
            frame = key.time / ticks_per_frame
            for i, v in enumerate((key.val[0]*s, key.val[1]*s, key.val[2]*s)):
                fc[i].keyframe_points.insert(frame, v, options={'FAST'})

    # TCB rotation
    elif ct == CTRL_TCBQ:
        fc = [get_or_make_fcurve(bone_path_rot, i) for i in range(4)]
        for key in ctrl_chunk.keys:
            frame = key.time / ticks_per_frame
            q = cry_quat(key.val)
            for i, v in enumerate((q.w, q.x, q.y, q.z)):
                fc[i].keyframe_points.insert(frame, v, options={'FAST'})

    # Update F-Curve handles
    for fc in action.fcurves:
        fc.update()


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
         import_skeleton=True, import_weights=True, game_root_path=""):
    """Import a CGF/CGA geometry file."""

    print(f"[CGF] Loading: {filepath}")
    print(f"[CGF] Game root: '{game_root_path}'")
    reader = cgf_reader.ChunkReader()
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

    # Materials
    print(f"[CGF] Building materials...")
    blender_materials = {}
    if import_materials:
        for mc in archive.material_chunks:
            standard_chunks = []
            _collect_standard_chunks(mc, archive, standard_chunks)
            print(f"[CGF]   material chunk: {mc.name} type={mc.type} → {len(standard_chunks)} standard")
            for std in standard_chunks:
                if std.name not in blender_materials:
                    bmat = build_material(std, filepath, import_materials, game_root_path)
                    if bmat: blender_materials[std.name] = bmat
    print(f"[CGF] Materials done: {len(blender_materials)}")

    # Armature
    arm_obj = None
    if import_skeleton and archive.bone_anim_chunks:
        print(f"[CGF] Building armature...")
        arm_obj, _ = build_armature(archive, collection)
        print(f"[CGF] Armature done: {arm_obj}")

    # Meshes
    print(f"[CGF] Building {len(archive.mesh_chunks)} mesh(es)...")
    mesh_objects = []
    for i, mc in enumerate(archive.mesh_chunks):
        print(f"[CGF]   mesh {i}: verts={len(mc.vertices)} faces={len(mc.faces)} bone_info={mc.has_bone_info} physique={len(mc.physique)}")
        node = archive.get_node(mc.header.chunk_id)
        obj  = build_mesh(mc, node, archive, collection,
                          import_materials, import_normals, import_uvs,
                          import_weights, blender_materials, filepath)
        if obj:
            mesh_objects.append(obj)
            print(f"[CGF]   mesh {i} done: {obj.name}")
            if archive.mesh_morph_target_chunks:
                build_shape_keys(obj, mc, archive)

    print(f"[CGF] All meshes done")
    if arm_obj and import_skeleton and import_weights:
        apply_armature_to_meshes(arm_obj, mesh_objects)

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
    reader = cgf_reader.ChunkReader()
    try:
        archive = reader.read_file(cgf_path)
    except ValueError as e:
        operator.report({'ERROR'}, f"Failed to read CGF: {e}")
        return None, None

    print(f"[CGF] Archive: bone_anim_chunks={len(archive.bone_anim_chunks)} mesh={len(archive.mesh_chunks)}")

    file_name  = os.path.splitext(os.path.basename(cgf_path))[0]
    collection = bpy.data.collections.new(file_name)
    context.scene.collection.children.link(collection)

    arm_obj = None
    if archive.bone_anim_chunks:
        try:
            arm_obj, _ = build_armature(archive, collection)
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

    for mc in archive.mesh_chunks:
        node = archive.get_node(mc.header.chunk_id)
        build_mesh(mc, node, archive, collection,
                   import_materials=False, import_normals=False,
                   import_uvs=False, import_weights=True,
                   blender_materials={}, filepath=cgf_path)

    if arm_obj is None:
        operator.report({'ERROR'},
            "CGF imported but no armature was created (file has no skeleton).")
        return None, None

    context.view_layer.objects.active = arm_obj
    return arm_obj, archive


def load_caf(operator, context, filepath, append=True):
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
    reader = cgf_reader.ChunkReader()
    try:
        anim_archive = reader.read_file(filepath)
    except ValueError as e:
        operator.report({'ERROR'}, str(e)); return {'CANCELLED'}

    print(f"[CGF] Controllers: {len(anim_archive.controller_chunks)}")

    action_name = os.path.splitext(os.path.basename(filepath))[0]
    apply_animation(arm_obj, geom_archive, anim_archive, action_name)

    operator.report({'INFO'}, f"Animation '{action_name}' imported")
    return {'FINISHED'}


def load_cal(operator, context, filepath):
    """Import all animations from a CAL file. Auto-imports CGF if needed."""

    arm_obj, auto_archive = _ensure_armature(operator, context, filepath)
    if arm_obj is None:
        return {'CANCELLED'}

    if auto_archive is not None:
        geom_archive = auto_archive
    else:
        geom_archive = _build_geom_archive_from_armature(arm_obj)

    records = cgf_reader.read_cal_file(filepath)
    if not records:
        operator.report({'WARNING'}, "CAL file is empty or could not be parsed")
        return {'CANCELLED'}

    imported = 0
    for rec in records:
        caf_path = find_caf_file(rec.path, filepath,
                                  arm_obj.get('cgf_source_path', ''))
        if not caf_path:
            print(f"[CGF] CAF not found: {rec.path}"); continue
        reader = cgf_reader.ChunkReader()
        try:
            anim_archive = reader.read_file(caf_path)
        except Exception as e:
            print(f"[CGF] Failed {caf_path}: {e}"); continue

        apply_animation(arm_obj, geom_archive, anim_archive, rec.name)
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
    archive = cgf_reader.CryChunkArchive()
    archive.geom_file_name = arm_obj.get('cgf_source_path', '')

    # Try to reload from source CGF first — most reliable
    source_path = arm_obj.get('cgf_source_path', '')
    if source_path and os.path.isfile(source_path):
        try:
            reader = cgf_reader.ChunkReader()
            src = reader.read_file(source_path)
            archive.bone_anim_chunks      = src.bone_anim_chunks
            archive.bone_name_list_chunks = src.bone_name_list_chunks
            return archive
        except Exception as e:
            print(f"[CGF] Could not reload source CGF: {e}")

    # Fallback: build from pose bones + stored ctrl_ids
    bac  = cgf_reader.CryBoneAnimChunk()
    bac.header  = cgf_reader.ChunkHeader()
    bnlc = cgf_reader.CryBoneNameListChunk()
    bnlc.header = cgf_reader.ChunkHeader()

    pose_bones = arm_obj.pose.bones if arm_obj.pose else []
    for i, pbone in enumerate(pose_bones):
        bone = cgf_reader.CryBone()
        bone.bone_id = i
        bone.name    = pbone.name
        bone.ctrl_id = pbone.get('cry_ctrl_id', 'FFFFFFFF')
        bac.bones.append(bone)
        bnlc.name_list.append(pbone.name)

    archive.bone_anim_chunks.append(bac)
    archive.bone_name_list_chunks.append(bnlc)
    return archive
