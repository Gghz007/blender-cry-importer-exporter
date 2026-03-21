"""
cgf_writer.py — Binary chunk writer for CryEngine 1 CGF/CGA/CAF files (Far Cry)

Mirror of cgf_reader.py — same format, writing instead of reading.
"""

import struct
import os
import zlib
import math

from .cgf_reader import (
    FILE_SIGNATURE,
    FILE_TYPE_GEOM_HIGH, FILE_TYPE_GEOM_LOW,
    FILE_TYPE_ANIM_HIGH, FILE_TYPE_ANIM_LOW,
    CHUNK_TYPE_MESH, CHUNK_TYPE_NODE, CHUNK_TYPE_MATERIAL,
    CHUNK_TYPE_BONE_ANIM, CHUNK_TYPE_BONE_NAME_LIST,
    CHUNK_TYPE_CONTROLLER, CHUNK_TYPE_TIMING,
    CHUNK_TYPE_BONE_INITIAL_POS, CHUNK_TYPE_SOURCE_INFO,
    SIZE_CHUNK_HEADER,
    CTRL_CRY_BONE,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def ctrl_id_from_name(name):
    """Generate a stable 32-bit controller ID from bone name using CRC32."""
    return zlib.crc32(name.encode('utf-8')) & 0xFFFFFFFF


def pack_u8(v):   return struct.pack('<B', v & 0xFF)
def pack_u16(v):  return struct.pack('<H', v & 0xFFFF)
def pack_i32(v):  return struct.pack('<i', v)
def pack_u32(v):  return struct.pack('<I', v & 0xFFFFFFFF)
def pack_f32(v):  return struct.pack('<f', float(v))

def pack_point3(v):
    return pack_f32(v[0]) + pack_f32(v[1]) + pack_f32(v[2])

def pack_quat(q):
    # q = (x, y, z, w)
    return pack_f32(q[0]) + pack_f32(q[1]) + pack_f32(q[2]) + pack_f32(q[3])

def pack_fixed_string(s, length):
    """Write a null-terminated string padded to fixed length."""
    b = s.encode('latin-1')[:length-1]
    return b + b'\x00' * (length - len(b))

def pack_c_string(s):
    """Write a null-terminated C string."""
    return s.encode('latin-1') + b'\x00'

def pack_color_byte(r, g, b):
    """Write RGB as 3 bytes (0-255)."""
    return bytes([int(r*255) & 0xFF, int(g*255) & 0xFF, int(b*255) & 0xFF])

def pack_matrix44(m44):
    """Write 4x4 matrix as 16 floats (row-major, rows = basis vectors)."""
    return b''.join(pack_f32(v) for v in m44)

def pack_matrix43(m43):
    """Write 4x3 matrix as 12 floats."""
    return b''.join(pack_f32(v) for v in m43)

def pack_chunk_header(chunk_type, version, file_offset, chunk_id):
    """Write 16-byte chunk header."""
    return (pack_u16(chunk_type) + pack_u16(0xCCCC) +
            pack_u32(version) + pack_u32(file_offset) + pack_u32(chunk_id))


# ── Chunk builders ────────────────────────────────────────────────────────────

def build_source_info_chunk(chunk_id, source_file="", date="", user=""):
    """0x0013 SourceInfo chunk."""
    data = pack_c_string(source_file) + pack_c_string(date) + pack_c_string(user)
    return data, 0x0000, chunk_id  # type, version, id


def build_timing_chunk(chunk_id, ticks_per_frame=160, secs_per_tick=1.0/4800.0,
                        start_frame=0, end_frame=0):
    """0x000E Timing chunk."""
    data = (pack_f32(secs_per_tick) +
            pack_u32(ticks_per_frame) +
            # global range: name(32) + start(4) + end(4)
            pack_fixed_string("", 32) + pack_i32(start_frame) + pack_i32(end_frame) +
            pack_u32(0))  # num sub ranges
    return data, 0x0918, chunk_id


def build_bone_name_list_chunk(chunk_id, names):
    """
    0x0005 BoneNameList chunk.
    We write v0745 format: no chunk header skip, null-terminated strings.
    """
    data = pack_u32(len(names))
    for name in names:
        data += pack_c_string(name)
    return data, 0x0745, chunk_id


def build_bone_anim_chunk(chunk_id, bones):
    """
    0x0003 BoneAnim chunk v0290.
    Bone struct = 152 bytes:
      bone_id(4) + parent_id(4) + num_children(4) + ctrl_id(8) + custom_prop(32) + physics(100)
    ctrl_id is stored as uint32 + 4 zero padding bytes = 8 bytes total.
    """
    data = pack_u32(len(bones))
    for b in bones:
        data += pack_i32(b['bone_id'])
        data += pack_i32(b['parent_id'])
        data += pack_i32(b['num_children'])
        data += pack_u32(b['ctrl_id']) + b'\x00\x00\x00\x00'  # ctrl_id = 8 bytes (uint32 + 4 padding)
        data += pack_fixed_string(b.get('custom_property', ''), 32)
        data += _pack_bone_physics(b.get('bone_physics'))
    return data, 0x0290, chunk_id


def _pack_bone_physics(phys):
    """
    Pack CryBonePhysics struct = 100 bytes:
      mesh_id(4) + minimum(12) + maximum(12) + spring_angle(12)
      + spring_tension(12) + damping(12) + frame_3x3_matrix(36)
    Note: no flags field — confirmed from hex analysis of original files.
    """
    if phys is None:
        return b'\x00\x00\x00\xff' + b'\x00' * 96  # mesh_id=-1 (0xFFFFFFFF) + 96 zeros
    return (pack_i32(phys.get('mesh_id', -1)) +
            pack_point3(phys.get('minimum', (0,0,0))) +
            pack_point3(phys.get('maximum', (0,0,0))) +
            pack_point3(phys.get('spring_angle', (0,0,0))) +
            pack_point3(phys.get('spring_tension', (0,0,0))) +
            pack_point3(phys.get('damping', (0,0,0))) +
            pack_point3(phys.get('frame_matrix_row0', (1,0,0))) +
            pack_point3(phys.get('frame_matrix_row1', (0,1,0))) +
            pack_point3(phys.get('frame_matrix_row2', (0,0,1))))


def build_bone_initial_pos_chunk(chunk_id, mesh_chunk_id, matrices):
    """
    0x0012 BoneInitialPos chunk.
    matrices: list of 12-float lists (4x3 row-major)
    """
    data = pack_u32(mesh_chunk_id)
    data += pack_u32(len(matrices))
    for m in matrices:
        data += pack_matrix43(m)
    return data, 0x0001, chunk_id


def build_mesh_chunk(chunk_id, vertices, faces, tex_vertices, tex_faces,
                      physique=None, has_bone_info=False,
                      bone_matrices=None, bone_initial_pos_id=None):
    """
    0x0000 Mesh chunk (v0744).
    vertices: list of (pos, normal) where pos/normal are (x,y,z)
    faces: list of (v0, v1, v2, mat_id, smooth_group)
    tex_vertices: list of (u, v)
    tex_faces: list of (t0, t1, t2)
    physique: list of bone links per vertex
    bone_matrices: list of 12-float matrices for BoneInitialPos (embedded at end)
    bone_initial_pos_id: chunk_id for BoneInitialPos (for chunk table entry)

    BoneInitialPos is embedded at the END of mesh chunk data (not a separate chunk).
    The chunk table has a separate entry pointing to this embedded data.
    """
    num_verts  = len(vertices)
    num_tverts = len(tex_vertices)
    num_faces  = len(faces)
    has_vcol   = 1  # always write vertex colors (white) — engine expects them

    data  = pack_u8(1 if has_bone_info else 0)
    data += pack_u8(has_vcol)
    data += b'\x00\x00'  # padding
    data += pack_u32(num_verts)
    data += pack_u32(num_tverts)
    data += pack_u32(num_faces)
    data += pack_i32(-1)  # vert_anim_id

    # Vertices
    for pos, normal in vertices:
        data += pack_point3(pos)
        data += pack_point3(normal)

    # Faces
    for v0, v1, v2, mat_id, smooth_group in faces:
        data += pack_u32(v0) + pack_u32(v1) + pack_u32(v2)
        data += pack_u32(mat_id) + pack_u32(smooth_group)

    # Tex vertices
    for u, v in tex_vertices:
        data += pack_f32(u) + pack_f32(v)

    # Tex faces
    for t0, t1, t2 in tex_faces:
        data += pack_u32(t0) + pack_u32(t1) + pack_u32(t2)

    # Physique (bone links)
    if has_bone_info and physique:
        for bl in physique:
            data += pack_u32(len(bl))  # num links
            for bone_id, offset, blending in bl:
                data += pack_i32(bone_id)
                data += pack_point3(offset)
                data += pack_f32(blending)

    # Vertex colors — write white (255,255,255) for all verts
    for _ in range(num_verts):
        data += b'\xFF\xFF\xFF'

    # BoneInitialPos embedded at end of mesh chunk (original format)
    # mesh_chunk_id(4) + num_matrices(4) + matrices(n×48)
    bone_initial_pos_offset = None
    if has_bone_info and bone_matrices:
        bone_initial_pos_offset = len(data)  # offset within mesh data
        data += pack_u32(chunk_id)           # mesh_chunk_id = this chunk's id
        data += pack_u32(len(bone_matrices))
        for m in bone_matrices:
            data += pack_matrix43(m)

    return data, 0x0744, chunk_id, bone_initial_pos_offset


def build_node_chunk(chunk_id, name, object_id, parent_id, material_id,
                      trans_matrix, position, rotation, scale,
                      pos_ctrl_id=0xFFFFFFFF, rot_ctrl_id=0xFFFFFFFF,
                      scale_ctrl_id=0xFFFFFFFF, prop=""):
    """
    0x000B Node chunk (v0823).
    trans_matrix: flat list of 16 floats (row-major, rows=basis vectors)
    position: (x,y,z)
    rotation: (x,y,z,w)
    scale: (x,y,z)
    """
    data  = pack_fixed_string(name, 64)
    data += pack_i32(object_id)
    data += pack_i32(parent_id)
    data += pack_u32(0)  # num_children
    data += pack_i32(material_id)
    data += pack_u8(0)   # is_group_head
    data += pack_u8(0)   # is_group_member
    data += b'\x00\x00' # padding
    data += pack_matrix44(trans_matrix)
    data += pack_point3(position)
    data += pack_quat(rotation)
    data += pack_point3(scale)
    # Controller IDs: low u16 + high u16
    data += _pack_ctrl_id(pos_ctrl_id)
    data += _pack_ctrl_id(rot_ctrl_id)
    data += _pack_ctrl_id(scale_ctrl_id)
    # Property string
    if prop:
        data += pack_u32(len(prop) + 1)
        data += pack_c_string(prop)
    else:
        data += pack_u32(0)
    return data, 0x0823, chunk_id


def _pack_ctrl_id(ctrl_id):
    """Pack ctrl_id as low_u16 + high_u16."""
    low  = ctrl_id & 0xFFFF
    high = (ctrl_id >> 16) & 0xFFFF
    return pack_u16(low) + pack_u16(high)


def build_material_chunk(chunk_id, name, mat_type=1, children=None,
                          diffuse=(0.8,0.8,0.8), specular=(0,0,0), ambient=(0,0,0),
                          specular_level=0.0, specular_shininess=0.0,
                          self_illumination=0.0, opacity=1.0,
                          tex_diffuse="", tex_bump="", tex_detail="",
                          flags=0, alpha_test=0.0):
    """
    0x000C Material chunk (v0746).
    mat_type: 1=Standard, 2=Multi
    """
    # Name field: 124 bytes + 4 bytes alphaTest
    data  = pack_fixed_string(name, 124)
    data += pack_f32(alpha_test)
    data += pack_i32(mat_type)

    if mat_type == 2:  # Multi
        data += pack_i32(len(children or []))
        # Children IDs must be at offset (chunk_file_offset + 2552) from file
        # = 16 (embedded header) + 2536 bytes of data before children
        # current data written so far: name(124) + alphaTest(4) + type(4) + numChildren(4) = 136
        current = 124 + 4 + 4 + 4
        pad = 2536 - current  # pad to 2536 bytes total data before children
        data += b'\x00' * pad
        for cid in (children or []):
            data += pack_i32(cid)
    else:  # Standard
        # Colors as RGB bytes (3 bytes each), then 3 bytes padding to 4-byte boundary
        data += pack_color_byte(*diffuse)
        data += pack_color_byte(*specular)
        data += pack_color_byte(*ambient)
        data += b'\x00\x00\x00'  # 3 padding bytes (not 1!)
        data += pack_f32(specular_level)
        data += pack_f32(specular_shininess)
        data += pack_f32(self_illumination)
        data += pack_f32(opacity)
        # 10 texture slots (each 236 bytes with 152-byte name)
        for tex_name in [
            "",          # 0 ambient
            tex_diffuse, # 1 diffuse ← main texture
            "",          # 2 specular
            "",          # 3 opacity
            tex_bump,    # 4 bump ← DDN normal map (_ddn)
            "",          # 5 gloss
            "",          # 6 filter
            "",          # 7 reflection
            "",          # 8 subsurface
            tex_detail,  # 9 detail ← heightmap/bump (_bump)
        ]:
            data += _pack_texture(tex_name)
        # Flags + physics params
        data += pack_u32(flags)
        data += pack_f32(0.0)  # dynamicBounce
        data += pack_f32(0.0)  # dynamicStaticFriction
        data += pack_f32(0.0)  # dynamicSlidingFriction

    return data, 0x0746, chunk_id


def _pack_texture(name):
    """Pack a CryTexture struct. v746 uses 152-byte name field (verified from original files)."""
    data  = pack_fixed_string(name, 152)  # 152 bytes confirmed from hex analysis
    data += pack_u32(0)    # type = Normal
    data += pack_u32(0)    # flags
    data += pack_i32(100)  # amount
    data += pack_u8(1)     # u_tile
    data += pack_u8(0)     # u_mirror
    data += pack_u8(1)     # v_tile
    data += pack_u8(0)     # v_mirror
    data += pack_i32(1)    # nth_frame
    data += pack_i32(256)  # ref_size
    data += pack_f32(0.0)  # ref_blur
    data += pack_f32(0.0)  # u_offset
    data += pack_f32(1.0)  # u_scale
    data += pack_f32(0.0)  # u_rotation
    data += pack_f32(0.0)  # v_offset
    data += pack_f32(1.0)  # v_scale
    data += pack_f32(0.0)  # v_rotation
    data += pack_f32(0.0)  # w_rotation
    data += pack_u32(0xFFFFFFFF) + pack_u32(0xFFFFFFFF) + pack_u32(0xFFFFFFFF) + pack_u32(0xFFFFFFFF) + pack_u32(0xFFFFFFFF) + pack_u32(0xFFFFFFFF) + pack_u32(0xFFFFFFFF)
    return data


# ── CAF controller chunk ───────────────────────────────────────────────────────

def build_controller_chunk_v827(chunk_id, ctrl_id, keys):
    """
    0x000D Controller chunk v0827.
    keys: list of (time_ticks, pos_xyz, rot_log_xyz)
    """
    low  = ctrl_id & 0xFFFF
    high = (ctrl_id >> 16) & 0xFFFF
    data  = pack_u32(len(keys))
    data += pack_u16(low) + pack_u16(high)
    for time, pos, rot_log in keys:
        data += pack_i32(time)
        data += pack_point3(pos)
        data += pack_point3(rot_log)
    return data, 0x0827, chunk_id


# ── CGF file assembler ────────────────────────────────────────────────────────

class CGFWriter:
    """
    Assembles chunks into a valid CGF/CAF binary file.

    Usage:
        w = CGFWriter(is_anim=False)
        w.add_chunk(type, version, chunk_id, data)
        w.write(filepath)
    """

    def __init__(self, is_anim=False):
        self.is_anim = is_anim
        self.chunks = []  # list of (type, version, chunk_id, data_bytes)
        self.extra_table_entries = []  # list of (type, version, absolute_offset, chunk_id)
                                       # for embedded chunks like BoneInitialPos

    def add_chunk(self, chunk_type, version, chunk_id, data):
        self.chunks.append((chunk_type, version, chunk_id, data))

    def add_embedded_chunk_entry(self, chunk_type, version, chunk_id, parent_chunk_idx, data_offset):
        """
        Register an embedded chunk that lives inside another chunk's data.
        parent_chunk_idx: index in self.chunks of the parent chunk
        data_offset: byte offset within parent chunk's data (after the 16-byte header)
        """
        self.extra_table_entries.append((chunk_type, version, chunk_id, parent_chunk_idx, data_offset))

    def write(self, filepath):
        FILE_HEADER_SIZE = 20

        offsets = []
        pos = FILE_HEADER_SIZE
        for chunk_type, version, chunk_id, data in self.chunks:
            offsets.append(pos)
            pos += SIZE_CHUNK_HEADER + len(data)

        chunk_table_offset = pos

        out = bytearray()

        out += FILE_SIGNATURE
        out += b'\x00\x00'
        if self.is_anim:
            out += pack_u16(FILE_TYPE_ANIM_LOW)
            out += pack_u16(FILE_TYPE_ANIM_HIGH)
            out += pack_u32(0x0744)
        else:
            out += pack_u16(FILE_TYPE_GEOM_LOW)
            out += pack_u16(FILE_TYPE_GEOM_HIGH)
            out += pack_u32(0x0744)
        out += pack_u32(chunk_table_offset)

        # Chunks
        for i, (chunk_type, version, chunk_id, data) in enumerate(self.chunks):
            out += pack_chunk_header(chunk_type, version, offsets[i], chunk_id)
            out += data

        # Chunk table — all normal chunks + embedded chunks
        total_entries = len(self.chunks) + len(self.extra_table_entries)
        out += pack_u32(total_entries)

        for i, (chunk_type, version, chunk_id, data) in enumerate(self.chunks):
            out += pack_chunk_header(chunk_type, version, offsets[i], chunk_id)

        # Extra embedded chunk entries
        for chunk_type, version, chunk_id, parent_idx, data_offset in self.extra_table_entries:
            # Absolute offset = parent chunk offset + 16 (header) + data_offset
            abs_offset = offsets[parent_idx] + SIZE_CHUNK_HEADER + data_offset
            out += pack_chunk_header(chunk_type, version, abs_offset, chunk_id)

        with open(filepath, 'wb') as f:
            f.write(out)

        with open(filepath, 'wb') as f:
            f.write(out)
