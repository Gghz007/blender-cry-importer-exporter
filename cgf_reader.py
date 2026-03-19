"""
cgf_reader.py - Binary chunk reader for CryEngine 1 CGF/CGA files (Far Cry)

Ported from CryImporter for 3ds Max by Takaro Pty. Ltd.
Original format reverse-engineered by the community.

CGF File Structure:
  - 20-byte file header: "CryTek" signature, file type, version, chunk table offset
  - Chunk table: count + array of 16-byte chunk headers (type, version, offset, ID)
  - Chunk data: each chunk at its offset

Supported chunk types (from CryEngine 1 / Far Cry):
  0x0000  Mesh
  0x0001  Helper (point/dummy)
  0x0003  BoneAnim (skeleton)
  0x0005  BoneNameList
  0x000B  Node (scene hierarchy + transform)
  0x000C  Material
  0x000E  Timing
  0x000F  BoneMesh (bone physics mesh)
  0x0011  MeshMorphTarget (shape keys)
  0x0012  BoneInitialPos
"""

import struct
import os


# ---- File / chunk type constants ----

FILE_SIGNATURE       = b"CryTek"

FILE_TYPE_GEOM_HIGH  = 0xFFFF
FILE_TYPE_GEOM_LOW   = 0x0000
FILE_TYPE_ANIM_HIGH  = 0xFFFF
FILE_TYPE_ANIM_LOW   = 0x0001

FILE_VERSION_GEOM    = 0x0744
FILE_VERSION_ANIM    = 0x0744

CHUNK_TYPE_MESH              = 0x0000
CHUNK_TYPE_HELPER            = 0x0001
CHUNK_TYPE_VERT_ANIM         = 0x0002
CHUNK_TYPE_BONE_ANIM         = 0x0003
CHUNK_TYPE_BONE_NAME_LIST    = 0x0005
CHUNK_TYPE_SCENE_PROP        = 0x0008
CHUNK_TYPE_LIGHT             = 0x0009
CHUNK_TYPE_NODE              = 0x000B
CHUNK_TYPE_MATERIAL          = 0x000C
CHUNK_TYPE_CONTROLLER        = 0x000D
CHUNK_TYPE_TIMING            = 0x000E
CHUNK_TYPE_BONE_MESH         = 0x000F
CHUNK_TYPE_BONE_LIGHT_BIND   = 0x0010
CHUNK_TYPE_MESH_MORPH_TARGET = 0x0011
CHUNK_TYPE_BONE_INITIAL_POS  = 0x0012
CHUNK_TYPE_SOURCE_INFO       = 0x0013

SIZE_CHUNK_HEADER   = 16
SIZE_BONE_LINK      = 20  # boneID(4) + offset(12) + blending(4)


# ---- Data classes (plain Python, no Blender dependency) ----

class ChunkHeader:
    __slots__ = ('type', 'version', 'file_offset', 'chunk_id')

    def __init__(self):
        self.type = 0
        self.version = 0
        self.file_offset = 0
        self.chunk_id = 0

    def __repr__(self):
        return (f"<ChunkHeader type=0x{self.type:04X} ver=0x{self.version:04X} "
                f"offset={self.file_offset} id={self.chunk_id}>")


class CryVertex:
    __slots__ = ('pos', 'normal')

    def __init__(self, pos, normal):
        self.pos = pos        # (x, y, z) floats
        self.normal = normal  # (x, y, z) floats


class CryFace:
    __slots__ = ('v0', 'v1', 'v2', 'mat_id', 'smooth_group')

    def __init__(self, v0, v1, v2, mat_id, smooth_group):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.mat_id = mat_id
        self.smooth_group = smooth_group


class CryTexFace:
    __slots__ = ('t0', 't1', 't2')

    def __init__(self, t0, t1, t2):
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2


class CryLink:
    """Single bone influence on a vertex."""
    __slots__ = ('bone_id', 'offset', 'blending')

    def __init__(self, bone_id, offset, blending):
        self.bone_id = bone_id  # int
        self.offset = offset    # (x, y, z)
        self.blending = blending  # float weight


class CryBoneLinks:
    """All bone influences for a single vertex."""
    __slots__ = ('vertex_id', 'links')

    def __init__(self, vertex_id):
        self.vertex_id = vertex_id
        self.links = []  # list of CryLink


class CryMeshChunk:
    __slots__ = ('header', 'has_bone_info', 'has_vertex_colors', 'vert_anim_id',
                 'vertices', 'faces', 'tex_vertices', 'tex_faces',
                 'physique', 'vertex_colors', 'is_bone_mesh')

    def __init__(self):
        self.header = None
        self.has_bone_info = False
        self.has_vertex_colors = False
        self.vert_anim_id = -1
        self.vertices = []       # list of CryVertex
        self.faces = []          # list of CryFace
        self.tex_vertices = []   # list of (u, v) floats
        self.tex_faces = []      # list of CryTexFace
        self.physique = []       # list of CryBoneLinks
        self.vertex_colors = []  # list of (r, g, b) bytes
        self.is_bone_mesh = False


class CryHelperChunk:
    __slots__ = ('header', 'type', 'size')

    def __init__(self):
        self.header = None
        self.type = 0
        self.size = (0.0, 0.0, 0.0)


class CryBonePhysics:
    __slots__ = ('mesh_id', 'flags', 'minimum', 'maximum',
                 'spring_angle', 'spring_tension', 'damping', 'frame_matrix')

    def __init__(self):
        self.mesh_id = -1
        self.flags = 0
        self.minimum = (0, 0, 0)
        self.maximum = (0, 0, 0)
        self.spring_angle = (0, 0, 0)
        self.spring_tension = (0, 0, 0)
        self.damping = (0, 0, 0)
        self.frame_matrix = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


class CryBone:
    __slots__ = ('bone_id', 'name', 'parent_id', 'num_children',
                 'ctrl_id', 'custom_property', 'bone_physics')

    def __init__(self):
        self.bone_id = 0
        self.name = ""
        self.parent_id = -1
        self.num_children = 0
        self.ctrl_id = ""
        self.custom_property = ""
        self.bone_physics = None


class CryBoneAnimChunk:
    __slots__ = ('header', 'bones')

    def __init__(self):
        self.header = None
        self.bones = []  # list of CryBone


class CryBoneNameListChunk:
    __slots__ = ('header', 'name_list')

    def __init__(self):
        self.header = None
        self.name_list = []  # list of str


class CryNodeChunk:
    __slots__ = ('header', 'name', 'object_id', 'parent_id', 'material_id',
                 'is_group_head', 'is_group_member',
                 'trans_matrix',  # 4x4 as flat list of 16 floats (row-major)
                 'position', 'rotation', 'scale',
                 'pos_ctrl_id', 'rot_ctrl_id', 'scale_ctrl_id',
                 'property', 'child_nodes')

    def __init__(self):
        self.header = None
        self.name = ""
        self.object_id = -1
        self.parent_id = -1
        self.material_id = -1
        self.is_group_head = False
        self.is_group_member = False
        self.trans_matrix = None
        self.position = (0, 0, 0)
        self.rotation = (0, 0, 0, 1)  # x, y, z, w
        self.scale = (1, 1, 1)
        self.pos_ctrl_id = ""
        self.rot_ctrl_id = ""
        self.scale_ctrl_id = ""
        self.property = ""
        self.child_nodes = []


class CryTexture:
    __slots__ = ('name', 'type', 'flags', 'amount',
                 'u_tile', 'u_mirror', 'v_tile', 'v_mirror',
                 'u_offset', 'u_scale', 'v_offset', 'v_scale')

    def __init__(self):
        self.name = ""
        self.type = 0
        self.flags = 0
        self.amount = 100
        self.u_tile = True
        self.u_mirror = False
        self.v_tile = True
        self.v_mirror = False
        self.u_offset = 0.0
        self.u_scale = 1.0
        self.v_offset = 0.0
        self.v_scale = 1.0


class CryMaterialChunk:
    __slots__ = ('header', 'name', 'shader_name', 'surface_name', 'type',
                 'alpha_test', 'children',
                 'diffuse', 'specular', 'ambient',
                 'specular_level', 'specular_shininess', 'self_illumination',
                 'opacity',
                 'tex_diffuse', 'tex_specular', 'tex_bump',
                 'tex_reflection', 'tex_detail', 'flags')

    def __init__(self):
        self.header = None
        self.name = ""
        self.shader_name = ""
        self.surface_name = ""
        self.type = 0
        self.alpha_test = 0.0
        self.children = []
        self.diffuse = (0.8, 0.8, 0.8)
        self.specular = (0.0, 0.0, 0.0)
        self.ambient = (0.0, 0.0, 0.0)
        self.specular_level = 0.0
        self.specular_shininess = 0.0
        self.self_illumination = 0.0
        self.opacity = 1.0
        self.tex_diffuse = None
        self.tex_specular = None
        self.tex_bump = None
        self.tex_reflection = None
        self.tex_detail = None
        self.flags = 0


class CryTimingChunk:
    __slots__ = ('header', 'secs_per_tick', 'ticks_per_frame', 'global_range', 'sub_ranges')

    def __init__(self):
        self.header = None
        self.secs_per_tick = 1.0 / 4800.0
        self.ticks_per_frame = 160
        self.global_range = None
        self.sub_ranges = []


class CryMeshMorphTargetVertex:
    __slots__ = ('vertex_id', 'target_point')

    def __init__(self, vertex_id, target_point):
        self.vertex_id = vertex_id
        self.target_point = target_point  # (x, y, z)


class CryMeshMorphTargetChunk:
    __slots__ = ('header', 'mesh_chunk_id', 'name', 'target_vertices')

    def __init__(self):
        self.header = None
        self.mesh_chunk_id = -1
        self.name = ""
        self.target_vertices = []  # list of CryMeshMorphTargetVertex


class CryBoneInitialPosChunk:
    __slots__ = ('header', 'mesh_chunk_id', 'initial_positions')

    def __init__(self):
        self.header = None
        self.mesh_chunk_id = -1
        self.initial_positions = []  # list of 4x3 matrices (as list of 12 floats)


class CryChunkArchive:
    """Container for all parsed chunks from a CGF file."""

    def __init__(self):
        self.mesh_chunks = []
        self.helper_chunks = []
        self.vert_anim_chunks = []
        self.bone_anim_chunks = []
        self.bone_name_list_chunks = []
        self.scene_prop_chunks = []
        self.light_chunks = []
        self.node_chunks = []
        self.material_chunks = []
        self.controller_chunks = []
        self.timing_chunks = []
        self.bone_mesh_chunks = []
        self.bone_light_binding_chunks = []
        self.mesh_morph_target_chunks = []
        self.bone_initial_pos_chunks = []
        self.source_info_chunks = []
        self.geom_file_name = ""
        self.num_chunks = 0

    def add(self, chunk):
        t = chunk.header.type
        if   t == CHUNK_TYPE_MESH:              self.mesh_chunks.append(chunk)
        elif t == CHUNK_TYPE_HELPER:            self.helper_chunks.append(chunk)
        elif t == CHUNK_TYPE_VERT_ANIM:         self.vert_anim_chunks.append(chunk)
        elif t == CHUNK_TYPE_BONE_ANIM:         self.bone_anim_chunks.append(chunk)
        elif t == CHUNK_TYPE_BONE_NAME_LIST:    self.bone_name_list_chunks.append(chunk)
        elif t == CHUNK_TYPE_SCENE_PROP:        self.scene_prop_chunks.append(chunk)
        elif t == CHUNK_TYPE_LIGHT:             self.light_chunks.append(chunk)
        elif t == CHUNK_TYPE_NODE:              self.node_chunks.append(chunk)
        elif t == CHUNK_TYPE_MATERIAL:          self.material_chunks.append(chunk)
        elif t == CHUNK_TYPE_CONTROLLER:        self.controller_chunks.append(chunk)
        elif t == CHUNK_TYPE_TIMING:            self.timing_chunks.append(chunk)
        elif t == CHUNK_TYPE_BONE_MESH:         self.bone_mesh_chunks.append(chunk)
        elif t == CHUNK_TYPE_BONE_LIGHT_BIND:   self.bone_light_binding_chunks.append(chunk)
        elif t == CHUNK_TYPE_MESH_MORPH_TARGET: self.mesh_morph_target_chunks.append(chunk)
        elif t == CHUNK_TYPE_BONE_INITIAL_POS:  self.bone_initial_pos_chunks.append(chunk)
        elif t == CHUNK_TYPE_SOURCE_INFO:       self.source_info_chunks.append(chunk)
        self.num_chunks += 1

    def get_bone_name(self, bone_id):
        if self.bone_name_list_chunks:
            lst = self.bone_name_list_chunks[0].name_list
            i = bone_id  # 0-based index
            if 0 <= i < len(lst):
                return lst[i]
        return None

    def get_ticks_per_frame(self):
        if self.timing_chunks:
            return self.timing_chunks[0].ticks_per_frame
        return 160

    def get_node(self, object_id):
        for n in self.node_chunks:
            if n.object_id == object_id:
                return n
        return None

    def get_material_chunk(self, chunk_id):
        for m in self.material_chunks:
            if m.header.chunk_id == chunk_id:
                return m
        return None

    def get_bone_mesh_from_chunk_id(self, chunk_id):
        for bm in self.bone_mesh_chunks:
            if bm.header.chunk_id == chunk_id:
                return bm
        return None

    def get_bone_initial_pos(self, bone_id):
        if self.bone_initial_pos_chunks:
            positions = self.bone_initial_pos_chunks[0].initial_positions
            if 0 <= bone_id < len(positions):
                return positions[bone_id]
        return None

    def get_morphs_for_mesh(self, mesh_chunk_id):
        return [m for m in self.mesh_morph_target_chunks
                if m.mesh_chunk_id == mesh_chunk_id]


# ---- Binary Reader ----

class ChunkReader:
    """
    Reads a CGF/CGA binary file and populates a CryChunkArchive.

    The CGF file format (CryEngine 1, Far Cry):
      Offset 0:   6 bytes  "CryTek" signature
      Offset 6:   2 bytes  (padding/reserved)
      Offset 8:   2 bytes  file type low word
      Offset 10:  2 bytes  file type high word
      Offset 12:  4 bytes  file version
      Offset 16:  4 bytes  chunk table offset
      Offset 20:  (padding to chunk table)

      Chunk table at chunkTableOffset:
        4 bytes  number of chunks
        Then numChunks * 16 bytes of chunk headers:
          2 bytes  chunk type (low word of full 32-bit type)
          2 bytes  chunk type (high word, always 0xCCCC for valid chunks)
          4 bytes  chunk version
          4 bytes  chunk file offset
          4 bytes  chunk ID

      Each chunk at its file offset starts with the 16-byte header repeated,
      then the chunk-specific data.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.pos = 0

    # -- Low-level readers --

    def _read(self, n):
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def _seek(self, pos):
        self.pos = pos

    def _tell(self):
        return self.pos

    def _skip(self, n):
        self.pos += n

    def _read_u8(self):
        v, = struct.unpack_from('<B', self.data, self.pos)
        self.pos += 1
        return v

    def _read_i16(self):
        v, = struct.unpack_from('<h', self.data, self.pos)
        self.pos += 2
        return v

    def _read_u16(self):
        v, = struct.unpack_from('<H', self.data, self.pos)
        self.pos += 2
        return v

    def _read_i32(self):
        v, = struct.unpack_from('<i', self.data, self.pos)
        self.pos += 4
        return v

    def _read_u32(self):
        v, = struct.unpack_from('<I', self.data, self.pos)
        self.pos += 4
        return v

    def _read_f32(self):
        v, = struct.unpack_from('<f', self.data, self.pos)
        self.pos += 4
        return v

    def _read_point3(self):
        x = self._read_f32()
        y = self._read_f32()
        z = self._read_f32()
        return (x, y, z)

    def _read_quat(self):
        """Reads x, y, z, w quaternion."""
        x = self._read_f32()
        y = self._read_f32()
        z = self._read_f32()
        w = self._read_f32()
        return (x, y, z, w)

    def _read_color_rgb_byte(self):
        r = self._read_u8()
        g = self._read_u8()
        b = self._read_u8()
        return (r, g, b)

    def _read_color_rgb_float(self):
        r = self._read_f32()
        g = self._read_f32()
        b = self._read_f32()
        return (r, g, b)

    def _read_fixed_string(self, length):
        """Read a null-terminated string from a fixed-size buffer."""
        raw = self._read(length)
        end = raw.find(b'\x00')
        if end == -1:
            end = length
        try:
            return raw[:end].decode('latin-1')
        except Exception:
            return raw[:end].decode('ascii', errors='replace')

    def _read_c_string(self):
        """Read a null-terminated string of variable length."""
        start = self.pos
        while self.pos < len(self.data) and self.data[self.pos] != 0:
            self.pos += 1
        s = self.data[start:self.pos].decode('latin-1')
        self.pos += 1  # skip null terminator
        return s

    def _read_matrix44(self):
        """Read a 4x4 matrix as 16 floats (row-major)."""
        return [self._read_f32() for _ in range(16)]

    def _read_matrix43(self):
        """Read a 4x3 matrix as 12 floats (used for bone initial positions)."""
        return [self._read_f32() for _ in range(12)]

    # -- Chunk header --

    def _read_chunk_header(self):
        h = ChunkHeader()
        h.type = self._read_u16()
        self._skip(2)           # high word of type (0xCCCC), skip
        h.version = self._read_u32()
        h.file_offset = self._read_u32()
        h.chunk_id = self._read_u32()
        return h

    # -- Primitive data readers --

    def _read_vertex(self):
        pos = self._read_point3()
        normal = self._read_point3()
        return CryVertex(pos, normal)

    def _read_face(self):
        v0 = self._read_u32()
        v1 = self._read_u32()
        v2 = self._read_u32()
        mat_id = self._read_u32()
        smooth_group = self._read_u32()
        return CryFace(v0, v1, v2, mat_id, smooth_group)

    def _read_tex_face(self):
        t0 = self._read_u32()
        t1 = self._read_u32()
        t2 = self._read_u32()
        return CryTexFace(t0, t1, t2)

    def _read_link(self):
        bone_id = self._read_i32()
        offset = self._read_point3()
        blending = self._read_f32()
        return CryLink(bone_id, offset, blending)

    def _read_bone_links(self, vertex_id):
        bl = CryBoneLinks(vertex_id)
        num_links = self._read_u32()
        for _ in range(num_links):
            bl.links.append(self._read_link())
        return bl

    def _read_bone_physics(self):
        bp = CryBonePhysics()
        bp.mesh_id = self._read_i32()
        bp.flags = self._read_u32()
        bp.minimum = self._read_point3()
        bp.maximum = self._read_point3()
        bp.spring_angle = self._read_point3()
        bp.spring_tension = self._read_point3()
        bp.damping = self._read_point3()
        bp.frame_matrix = [self._read_point3() for _ in range(3)]
        return bp

    def _read_bone(self):
        b = CryBone()
        b.bone_id = self._read_i32()
        b.parent_id = self._read_i32()
        b.num_children = self._read_i32()
        ctrl_id_raw = self._read_u32()
        b.ctrl_id = str(ctrl_id_raw)
        b.custom_property = self._read_fixed_string(32)
        b.bone_physics = self._read_bone_physics()
        return b

    def _read_texture(self, chunk_version):
        tex = CryTexture()

        if chunk_version >= 0x0746:
            tex.name = self._read_fixed_string(128)
        else:
            tex.name = self._read_fixed_string(32)

        tex.type = self._read_u32()
        tex.flags = self._read_u32()
        tex.amount = self._read_i32()

        tex.u_tile   = bool(self._read_u8())
        tex.u_mirror = bool(self._read_u8())
        tex.v_tile   = bool(self._read_u8())
        tex.v_mirror = bool(self._read_u8())

        nth_frame = self._read_i32()
        ref_size  = self._read_i32()
        ref_blur  = self._read_f32()

        tex.u_offset   = self._read_f32()
        tex.u_scale    = self._read_f32()
        u_rotation     = self._read_f32()
        tex.v_offset   = self._read_f32()
        tex.v_scale    = self._read_f32()
        v_rotation     = self._read_f32()
        w_rotation     = self._read_f32()

        # Skip controller IDs (7 * 4 bytes each for v746+, or 7 * 4 bytes for older)
        if chunk_version >= 0x0746:
            self._skip(7 * 4)  # 7 controller IDs as 32-bit integers
        else:
            self._skip(7 * 4)

        return tex

    def _read_range(self):
        name = self._read_fixed_string(32)
        start = self._read_i32()
        end = self._read_i32()
        return (name, start, end)

    # -- Chunk readers --

    def _read_mesh_chunk(self, header, next_chunk_pos, is_bone_mesh=False):
        self._seek(header.file_offset)
        self._skip(SIZE_CHUNK_HEADER)  # skip the embedded header copy

        chunk = CryMeshChunk()
        chunk.header = header
        chunk.is_bone_mesh = is_bone_mesh

        has_bone_info_byte    = self._read_u8()
        has_vertex_colors_byte = self._read_u8()
        self._skip(2)  # padding

        num_verts   = self._read_u32()
        num_tverts  = self._read_u32()
        num_faces   = self._read_u32()
        chunk.vert_anim_id = self._read_i32()

        chunk.has_bone_info     = has_bone_info_byte > 0
        chunk.has_vertex_colors = has_vertex_colors_byte > 0

        num_tfaces = num_faces if num_tverts > 0 else 0

        # Vertices
        chunk.vertices = [self._read_vertex() for _ in range(num_verts)]

        # Faces
        chunk.faces = [self._read_face() for _ in range(num_faces)]

        # Texture vertices (UV)
        for _ in range(num_tverts):
            u = self._read_f32()
            v = self._read_f32()
            chunk.tex_vertices.append((u, v))

        # Texture faces
        chunk.tex_faces = [self._read_tex_face() for _ in range(num_tfaces)]

        # Physique / bone links
        if chunk.has_bone_info and num_verts > 0:
            cur_pos = self._tell()
            if next_chunk_pos is not None and (cur_pos + SIZE_BONE_LINK > next_chunk_pos):
                chunk.has_bone_info = False
            else:
                for i in range(num_verts):
                    chunk.physique.append(self._read_bone_links(i))

        # Vertex colors
        if chunk.has_vertex_colors and num_verts > 0:
            for _ in range(num_verts):
                chunk.vertex_colors.append(self._read_color_rgb_byte())

        return chunk

    def _read_helper_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset)
        self._skip(SIZE_CHUNK_HEADER)

        chunk = CryHelperChunk()
        chunk.header = header
        chunk.type = self._read_u32()
        chunk.size = self._read_point3()
        return chunk

    def _read_bone_anim_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset)
        self._skip(SIZE_CHUNK_HEADER)

        chunk = CryBoneAnimChunk()
        chunk.header = header

        num_bones = self._read_u32()
        for _ in range(num_bones):
            chunk.bones.append(self._read_bone())
        return chunk

    def _read_bone_name_list_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset)
        self._skip(SIZE_CHUNK_HEADER)

        chunk = CryBoneNameListChunk()
        chunk.header = header

        # The name list is stored as a count then a block of fixed strings
        # Each name is 32 bytes
        num_names = self._read_u32()
        for _ in range(num_names):
            chunk.name_list.append(self._read_fixed_string(32))
        return chunk

    def _read_node_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset)
        self._skip(SIZE_CHUNK_HEADER)

        chunk = CryNodeChunk()
        chunk.header = header

        # Exact field order from CryImporter-chunkreader.ms readNodeChunk:
        #   name(64) → objectID(4) → parentID(4) → numChildren(4) → materialID(4)
        #   → isGroupHead(1) → isGroupMember(1) → padding(2)
        #   → transMatrix(64) → position(12) → rotation(16) → scale(12)
        #   → posCtrlID(4) → rotCtrlID(4) → scaleCtrlID(4) → propStrLen(4) → ...

        chunk.name        = self._read_fixed_string(64)
        chunk.object_id   = self._read_i32()
        chunk.parent_id   = self._read_i32()
        num_children      = self._read_u32()          # ← was missing before!
        chunk.material_id = self._read_i32()

        igh = self._read_u8()                         # isGroupHead byte
        igm = self._read_u8()                         # isGroupMember byte
        self._skip(2)                                 # 2 padding bytes to 4-byte boundary
        chunk.is_group_head   = igh > 0
        chunk.is_group_member = igm > 0

        # 4×4 transform matrix — 16 floats, row-major
        # Stored as: row0=Xaxis(+w), row1=Yaxis(+w), row2=Zaxis(+w), row3=Translation(+w)
        # Max builds Matrix3 directly from these rows: Matrix3 [r1] [r2] [r3] [r4]
        chunk.trans_matrix = self._read_matrix44()    # flat list of 16 floats

        # Decomposed TRS (pre-computed by exporter, same data as matrix)
        chunk.position = self._read_point3()          # x, y, z
        chunk.rotation = self._read_quat()            # x, y, z, w
        chunk.scale    = self._read_point3()          # x, y, z

        # Controller IDs: each stored as low_u16 + high_u16 → hex string in Max
        # We store as raw u32 for now (used for animation lookup)
        low  = self._read_u16(); high = self._read_u16()
        chunk.pos_ctrl_id   = f"{high:04X}{low:04X}"
        low  = self._read_u16(); high = self._read_u16()
        chunk.rot_ctrl_id   = f"{high:04X}{low:04X}"
        low  = self._read_u16(); high = self._read_u16()
        chunk.scale_ctrl_id = f"{high:04X}{low:04X}"

        # User-defined property string
        prop_len = self._read_u32()
        if prop_len > 0:
            chunk.property = self._read_c_string()

        # Child node chunk IDs
        for _ in range(num_children):
            chunk.child_nodes.append(self._read_u32())

        return chunk

    def _read_material_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset)
        self._skip(SIZE_CHUNK_HEADER)

        chunk = CryMaterialChunk()
        chunk.header = header

        chunk.name         = self._read_fixed_string(64)
        chunk.shader_name  = self._read_fixed_string(64)
        chunk.surface_name = self._read_fixed_string(64)
        chunk.flags        = self._read_u32()
        chunk.type         = self._read_u32()

        if header.version >= 0x0746:
            chunk.alpha_test = self._read_f32()

        # Number of sub-materials if multi-material
        num_children = self._read_u32()
        for _ in range(num_children):
            chunk.children.append(self._read_u32())

        # Color components (as floats 0..1 for ambient/diffuse/specular)
        chunk.ambient   = self._read_color_rgb_float()
        chunk.diffuse   = self._read_color_rgb_float()
        chunk.specular  = self._read_color_rgb_float()

        chunk.specular_level      = self._read_f32()
        chunk.specular_shininess  = self._read_f32()
        chunk.self_illumination   = self._read_f32()
        chunk.opacity             = self._read_f32()

        # Textures - order differs between versions
        if header.version >= 0x0746:
            # v746 order: ambient, diffuse, specular, opacity, bump, gloss, filter/detail, reflection, subsurface, detail(normalmap)
            tex_ambient    = self._read_texture(header.version)
            tex_diffuse    = self._read_texture(header.version)
            tex_specular   = self._read_texture(header.version)
            tex_opacity    = self._read_texture(header.version)
            tex_bump       = self._read_texture(header.version)
            tex_gloss      = self._read_texture(header.version)
            tex_filter     = self._read_texture(header.version)
            tex_reflection = self._read_texture(header.version)
            tex_subsurface = self._read_texture(header.version)
            tex_detail     = self._read_texture(header.version)
        else:
            # v745 order
            tex_ambient    = self._read_texture(header.version)
            tex_diffuse    = self._read_texture(header.version)
            tex_specular   = self._read_texture(header.version)
            tex_opacity    = self._read_texture(header.version)
            tex_bump       = self._read_texture(header.version)
            tex_gloss      = self._read_texture(header.version)
            tex_filter     = self._read_texture(header.version)
            tex_reflection = self._read_texture(header.version)
            tex_subsurface = self._read_texture(header.version)
            tex_detail     = self._read_texture(header.version)

        if tex_diffuse and tex_diffuse.name:
            chunk.tex_diffuse = tex_diffuse
        if tex_specular and tex_specular.name:
            chunk.tex_specular = tex_specular
        if tex_bump and tex_bump.name:
            chunk.tex_bump = tex_bump
        if tex_reflection and tex_reflection.name:
            chunk.tex_reflection = tex_reflection
        if tex_detail and tex_detail.name:
            chunk.tex_detail = tex_detail

        return chunk

    def _read_timing_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset)
        self._skip(SIZE_CHUNK_HEADER)

        chunk = CryTimingChunk()
        chunk.header = header

        chunk.secs_per_tick   = self._read_f32()
        chunk.ticks_per_frame = self._read_u32()
        chunk.global_range    = self._read_range()

        num_sub_ranges = self._read_u32()
        for _ in range(num_sub_ranges):
            chunk.sub_ranges.append(self._read_range())
        return chunk

    def _read_mesh_morph_target_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset)
        self._skip(SIZE_CHUNK_HEADER)

        chunk = CryMeshMorphTargetChunk()
        chunk.header = header
        chunk.mesh_chunk_id = self._read_u32()
        chunk.name = self._read_fixed_string(64)

        num_verts = self._read_u32()
        for _ in range(num_verts):
            vid = self._read_u32()
            pt  = self._read_point3()
            chunk.target_vertices.append(CryMeshMorphTargetVertex(vid, pt))
        return chunk

    def _read_bone_initial_pos_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset)

        chunk = CryBoneInitialPosChunk()
        chunk.header = header
        chunk.mesh_chunk_id = self._read_u32()

        num_bones = self._read_u32()
        for _ in range(num_bones):
            chunk.initial_positions.append(self._read_matrix43())
        return chunk

    def _read_chunk(self, header, next_chunk_pos):
        """Dispatch to the correct chunk reader based on type."""
        # Validate offset
        if header.file_offset >= len(self.data):
            return None

        t = header.type
        try:
            if   t == CHUNK_TYPE_MESH:
                return self._read_mesh_chunk(header, next_chunk_pos, is_bone_mesh=False)
            elif t == CHUNK_TYPE_HELPER:
                return self._read_helper_chunk(header, next_chunk_pos)
            elif t == CHUNK_TYPE_BONE_ANIM:
                return self._read_bone_anim_chunk(header, next_chunk_pos)
            elif t == CHUNK_TYPE_BONE_NAME_LIST:
                return self._read_bone_name_list_chunk(header, next_chunk_pos)
            elif t == CHUNK_TYPE_NODE:
                return self._read_node_chunk(header, next_chunk_pos)
            elif t == CHUNK_TYPE_MATERIAL:
                return self._read_material_chunk(header, next_chunk_pos)
            elif t == CHUNK_TYPE_TIMING:
                return self._read_timing_chunk(header, next_chunk_pos)
            elif t == CHUNK_TYPE_BONE_MESH:
                return self._read_mesh_chunk(header, next_chunk_pos, is_bone_mesh=True)
            elif t == CHUNK_TYPE_MESH_MORPH_TARGET:
                return self._read_mesh_morph_target_chunk(header, next_chunk_pos)
            elif t == CHUNK_TYPE_BONE_INITIAL_POS:
                return self._read_bone_initial_pos_chunk(header, next_chunk_pos)
            else:
                return None  # unsupported/obsolete chunk type
        except Exception as e:
            print(f"[CGF] Warning: failed to read chunk type=0x{t:04X} id={header.chunk_id}: {e}")
            return None

    # -- Main entry point --

    def read_file(self, filepath):
        """
        Parse a CGF/CGA file and return a CryChunkArchive.
        Raises ValueError with a description on failure.
        """
        if not os.path.isfile(filepath):
            raise ValueError(f"File not found: {filepath}")

        with open(filepath, 'rb') as f:
            self.data = f.read()
        self.pos = 0

        file_size = len(self.data)

        # -- Validate signature --
        sig = self._read(6)
        if sig != FILE_SIGNATURE:
            raise ValueError(f"Not a CryTek file (bad signature: {sig!r})")

        # -- File type (at offset 8) --
        self._seek(8)
        file_type_low  = self._read_u16()
        file_type_high = self._read_u16()

        is_geom = (file_type_high == FILE_TYPE_GEOM_HIGH and file_type_low == FILE_TYPE_GEOM_LOW)
        is_anim = (file_type_high == FILE_TYPE_ANIM_HIGH and file_type_low == FILE_TYPE_ANIM_LOW)

        if not (is_geom or is_anim):
            raise ValueError(f"Unknown file type: high=0x{file_type_high:04X} low=0x{file_type_low:04X}")

        # -- Chunk table --
        self._seek(16)
        chunk_table_pos = self._read_u32()

        self._seek(chunk_table_pos)
        num_chunks = self._read_u32()
        chunk_headers_start = self._tell()

        # Read all chunk headers first
        headers = []
        header_positions = []
        for i in range(num_chunks):
            self._seek(chunk_headers_start + i * SIZE_CHUNK_HEADER)
            header_positions.append(self._tell())
            h = self._read_chunk_header()
            headers.append(h)

        # Build archive
        archive = CryChunkArchive()
        archive.geom_file_name = filepath if is_geom else ""

        # Read each chunk's data
        for i, h in enumerate(headers):
            if i + 1 < num_chunks:
                next_pos = header_positions[i + 1]
            else:
                next_pos = chunk_table_pos

            chunk = self._read_chunk(h, next_pos)
            if chunk is not None:
                archive.add(chunk)

        return archive
