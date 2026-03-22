"""
cgf_reader.py — Binary chunk reader for CryEngine 1 CGF/CGA/CAF files (Far Cry)

Ported from CryImporter for 3ds Max by Takaro Pty. Ltd.
"""

import struct
import os

# ── File / chunk type constants ──────────────────────────────────────────────

FILE_SIGNATURE      = b"CryTek"

FILE_TYPE_GEOM_HIGH = 0xFFFF
FILE_TYPE_GEOM_LOW  = 0x0000
FILE_TYPE_ANIM_HIGH = 0xFFFF
FILE_TYPE_ANIM_LOW  = 0x0001

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

# Controller types (v826)
CTRL_NONE           = 0
CTRL_CRY_BONE       = 1
CTRL_LINEAR1        = 2
CTRL_LINEAR3        = 3
CTRL_LINEAR_Q       = 4
CTRL_BEZIER1        = 5
CTRL_BEZIER3        = 6
CTRL_BEZIER_Q       = 7
CTRL_TCB1           = 8
CTRL_TCB3           = 9
CTRL_TCBQ           = 10
CTRL_BSPLINE2_OPEN  = 11
CTRL_BSPLINE1_OPEN  = 12
CTRL_BSPLINE2_CLOSE = 13
CTRL_BSPLINE1_CLOSE = 14
CTRL_CONSTANT       = 15

SIZE_CHUNK_HEADER = 16
SIZE_BONE_LINK    = 20


# ── Data classes ─────────────────────────────────────────────────────────────

class ChunkHeader:
    __slots__ = ('type', 'version', 'file_offset', 'chunk_id')
    def __init__(self):
        self.type=0; self.version=0; self.file_offset=0; self.chunk_id=0
    def __repr__(self):
        return f"<Chunk 0x{self.type:04X} v{self.version:04X} @{self.file_offset} id={self.chunk_id}>"


class CryVertex:
    __slots__ = ('pos', 'normal')
    def __init__(self, pos, normal): self.pos=pos; self.normal=normal


class CryFace:
    __slots__ = ('v0','v1','v2','mat_id','smooth_group')
    def __init__(self, v0, v1, v2, mat_id, sg):
        self.v0=v0; self.v1=v1; self.v2=v2; self.mat_id=mat_id; self.smooth_group=sg


class CryTexFace:
    __slots__ = ('t0','t1','t2')
    def __init__(self, t0, t1, t2): self.t0=t0; self.t1=t1; self.t2=t2


class CryLink:
    __slots__ = ('bone_id','offset','blending')
    def __init__(self, bone_id, offset, blending):
        self.bone_id=bone_id; self.offset=offset; self.blending=blending


class CryBoneLinks:
    __slots__ = ('vertex_id','links')
    def __init__(self, vertex_id): self.vertex_id=vertex_id; self.links=[]


class CryMeshChunk:
    __slots__ = ('header','has_bone_info','has_vertex_colors','vert_anim_id',
                 'vertices','faces','tex_vertices','tex_faces',
                 'physique','vertex_colors','is_bone_mesh')
    def __init__(self):
        self.header=None; self.has_bone_info=False; self.has_vertex_colors=False
        self.vert_anim_id=-1; self.vertices=[]; self.faces=[]
        self.tex_vertices=[]; self.tex_faces=[]; self.physique=[]
        self.vertex_colors=[]; self.is_bone_mesh=False


class CryHelperChunk:
    __slots__ = ('header','type','size')
    def __init__(self): self.header=None; self.type=0; self.size=(0,0,0)


class CryBonePhysics:
    __slots__ = ('mesh_id','flags','minimum','maximum',
                 'spring_angle','spring_tension','damping','frame_matrix')
    def __init__(self):
        self.mesh_id=-1; self.flags=0
        self.minimum=(0,0,0); self.maximum=(0,0,0)
        self.spring_angle=(0,0,0); self.spring_tension=(0,0,0); self.damping=(0,0,0)
        self.frame_matrix=[(1,0,0),(0,1,0),(0,0,1)]


class CryBone:
    __slots__ = ('bone_id','name','parent_id','num_children',
                 'ctrl_id','custom_property','bone_physics')
    def __init__(self):
        self.bone_id=0; self.name=""; self.parent_id=-1; self.num_children=0
        self.ctrl_id=""; self.custom_property=""; self.bone_physics=None


class CryBoneAnimChunk:
    __slots__ = ('header','bones')
    def __init__(self): self.header=None; self.bones=[]


class CryBoneNameListChunk:
    __slots__ = ('header','name_list')
    def __init__(self): self.header=None; self.name_list=[]


class CryNodeChunk:
    __slots__ = ('header','name','object_id','parent_id','material_id',
                 'is_group_head','is_group_member','trans_matrix',
                 'position','rotation','scale',
                 'pos_ctrl_id','rot_ctrl_id','scale_ctrl_id',
                 'property','child_nodes')
    def __init__(self):
        self.header=None; self.name=""; self.object_id=-1
        self.parent_id=-1; self.material_id=-1
        self.is_group_head=False; self.is_group_member=False
        self.trans_matrix=None; self.position=(0,0,0)
        self.rotation=(0,0,0,1); self.scale=(1,1,1)
        self.pos_ctrl_id=""; self.rot_ctrl_id=""; self.scale_ctrl_id=""
        self.property=""; self.child_nodes=[]


class CryTexture:
    __slots__ = ('name','type','flags','amount',
                 'u_tile','u_mirror','v_tile','v_mirror',
                 'u_offset','u_scale','v_offset','v_scale')
    def __init__(self):
        self.name=""; self.type=0; self.flags=0; self.amount=100
        self.u_tile=True; self.u_mirror=False; self.v_tile=True; self.v_mirror=False
        self.u_offset=0.0; self.u_scale=1.0; self.v_offset=0.0; self.v_scale=1.0


class CryMaterialChunk:
    __slots__ = ('header','name','shader_name','surface_name','type',
                 'alpha_test','children','diffuse','specular','ambient',
                 'specular_level','specular_shininess','self_illumination','opacity',
                 'tex_diffuse','tex_specular','tex_bump','tex_reflection','tex_detail','flags')
    def __init__(self):
        self.header=None; self.name=""; self.shader_name=""; self.surface_name=""
        self.type=0; self.alpha_test=0.0; self.children=[]
        self.diffuse=(0.8,0.8,0.8); self.specular=(0,0,0); self.ambient=(0,0,0)
        self.specular_level=0.0; self.specular_shininess=0.0
        self.self_illumination=0.0; self.opacity=1.0
        self.tex_diffuse=None; self.tex_specular=None; self.tex_bump=None
        self.tex_reflection=None; self.tex_detail=None; self.flags=0


class CryTimingChunk:
    __slots__ = ('header','secs_per_tick','ticks_per_frame','global_range','sub_ranges')
    def __init__(self):
        self.header=None; self.secs_per_tick=1.0/4800.0
        self.ticks_per_frame=160; self.global_range=None; self.sub_ranges=[]


class CryMeshMorphTargetVertex:
    __slots__ = ('vertex_id','target_point')
    def __init__(self, vid, pt): self.vertex_id=vid; self.target_point=pt


class CryMeshMorphTargetChunk:
    __slots__ = ('header','mesh_chunk_id','name','target_vertices')
    def __init__(self):
        self.header=None; self.mesh_chunk_id=-1; self.name=""; self.target_vertices=[]


class CryBoneInitialPosChunk:
    __slots__ = ('header','mesh_chunk_id','initial_positions')
    def __init__(self): self.header=None; self.mesh_chunk_id=-1; self.initial_positions=[]


# ── Animation key structs ─────────────────────────────────────────────────────

class CryKey:
    """v827: time + position + rotation logarithm"""
    __slots__ = ('time','pos','rot_log')
    def __init__(self, t, p, r): self.time=t; self.pos=p; self.rot_log=r

class CryBoneKey:
    __slots__ = ('time','abs_pos','rel_pos','rel_quat')
    def __init__(self, t, ap, rp, rq):
        self.time=t; self.abs_pos=ap; self.rel_pos=rp; self.rel_quat=rq

class CryLin1Key:
    __slots__ = ('time','val')
    def __init__(self, t, v): self.time=t; self.val=v

class CryLin3Key:
    __slots__ = ('time','val')
    def __init__(self, t, v): self.time=t; self.val=v

class CryLinQKey:
    __slots__ = ('time','val')
    def __init__(self, t, v): self.time=t; self.val=v

class CryTCB1Key:
    __slots__ = ('time','val','t','c','b','ease_to','ease_from')
    def __init__(self, time, val, t, c, b, et, ef):
        self.time=time; self.val=val; self.t=t; self.c=c; self.b=b
        self.ease_to=et; self.ease_from=ef

class CryTCB3Key:
    __slots__ = ('time','val','t','c','b','ease_to','ease_from')
    def __init__(self, time, val, t, c, b, et, ef):
        self.time=time; self.val=val; self.t=t; self.c=c; self.b=b
        self.ease_to=et; self.ease_from=ef

class CryTCBQKey:
    __slots__ = ('time','val','t','c','b','ease_to','ease_from')
    def __init__(self, time, val, t, c, b, et, ef):
        self.time=time; self.val=val; self.t=t; self.c=c; self.b=b
        self.ease_to=et; self.ease_from=ef

class CryBez1Key:
    __slots__ = ('time','val','in_tan','out_tan')
    def __init__(self, t, v, i, o): self.time=t; self.val=v; self.in_tan=i; self.out_tan=o

class CryBez3Key:
    __slots__ = ('time','val','in_tan','out_tan')
    def __init__(self, t, v, i, o): self.time=t; self.val=v; self.in_tan=i; self.out_tan=o

class CryBezQKey:
    __slots__ = ('time','val')
    def __init__(self, t, v): self.time=t; self.val=v

class CryControllerChunk:
    __slots__ = ('header','ctrl_id','ctrl_type','flags','keys')
    def __init__(self):
        self.header=None; self.ctrl_id=""
        self.ctrl_type=CTRL_NONE; self.flags=0; self.keys=[]


# ── Chunk Archive ─────────────────────────────────────────────────────────────

class CryChunkArchive:
    def __init__(self):
        self.mesh_chunks=[]; self.helper_chunks=[]; self.vert_anim_chunks=[]
        self.bone_anim_chunks=[]; self.bone_name_list_chunks=[]
        self.scene_prop_chunks=[]; self.light_chunks=[]; self.node_chunks=[]
        self.material_chunks=[]; self.controller_chunks=[]; self.timing_chunks=[]
        self.bone_mesh_chunks=[]; self.bone_light_binding_chunks=[]
        self.mesh_morph_target_chunks=[]; self.bone_initial_pos_chunks=[]
        self.source_info_chunks=[]; self.geom_file_name=""; self.num_chunks=0

    def add(self, chunk):
        t = chunk.header.type
        m = {
            CHUNK_TYPE_MESH:              self.mesh_chunks,
            CHUNK_TYPE_HELPER:            self.helper_chunks,
            CHUNK_TYPE_VERT_ANIM:         self.vert_anim_chunks,
            CHUNK_TYPE_BONE_ANIM:         self.bone_anim_chunks,
            CHUNK_TYPE_BONE_NAME_LIST:    self.bone_name_list_chunks,
            CHUNK_TYPE_SCENE_PROP:        self.scene_prop_chunks,
            CHUNK_TYPE_LIGHT:             self.light_chunks,
            CHUNK_TYPE_NODE:              self.node_chunks,
            CHUNK_TYPE_MATERIAL:          self.material_chunks,
            CHUNK_TYPE_CONTROLLER:        self.controller_chunks,
            CHUNK_TYPE_TIMING:            self.timing_chunks,
            CHUNK_TYPE_BONE_MESH:         self.bone_mesh_chunks,
            CHUNK_TYPE_BONE_LIGHT_BIND:   self.bone_light_binding_chunks,
            CHUNK_TYPE_MESH_MORPH_TARGET: self.mesh_morph_target_chunks,
            CHUNK_TYPE_BONE_INITIAL_POS:  self.bone_initial_pos_chunks,
            CHUNK_TYPE_SOURCE_INFO:       self.source_info_chunks,
        }
        if t in m: m[t].append(chunk)
        self.num_chunks += 1

    def merge(self, other):
        for attr in ('mesh_chunks','helper_chunks','vert_anim_chunks',
                     'bone_anim_chunks','bone_name_list_chunks','scene_prop_chunks',
                     'light_chunks','node_chunks','material_chunks','controller_chunks',
                     'timing_chunks','bone_mesh_chunks','bone_light_binding_chunks',
                     'mesh_morph_target_chunks','bone_initial_pos_chunks','source_info_chunks'):
            getattr(self, attr).extend(getattr(other, attr))
        self.num_chunks += other.num_chunks

    def get_ticks_per_frame(self):
        return self.timing_chunks[0].ticks_per_frame if self.timing_chunks else 160

    def get_secs_per_tick(self):
        return self.timing_chunks[0].secs_per_tick if self.timing_chunks else 1.0/4800.0

    def get_bone_name(self, bone_id):
        if self.bone_name_list_chunks:
            lst = self.bone_name_list_chunks[0].name_list
            if 0 <= bone_id < len(lst): return lst[bone_id]
        return None

    def get_node(self, object_id):
        for n in self.node_chunks:
            if n.object_id == object_id: return n
        return None

    def get_material_chunk(self, chunk_id):
        for m in self.material_chunks:
            if m.header.chunk_id == chunk_id: return m
        return None

    def get_bone_initial_pos(self, bone_id):
        if self.bone_initial_pos_chunks:
            pos = self.bone_initial_pos_chunks[0].initial_positions
            if 0 <= bone_id < len(pos): return pos[bone_id]
        return None

    def get_morphs_for_mesh(self, mesh_chunk_id):
        return [m for m in self.mesh_morph_target_chunks if m.mesh_chunk_id == mesh_chunk_id]

    def get_controller(self, ctrl_id):
        for c in self.controller_chunks:
            if c.ctrl_id == ctrl_id: return c
        return None


# ── Binary Reader ─────────────────────────────────────────────────────────────

class ChunkReader:
    def __init__(self, filepath=None):
        self.filepath=filepath; self.data=None; self.pos=0

    def _read(self, n):
        b=self.data[self.pos:self.pos+n]; self.pos+=n; return b
    def _seek(self, p): self.pos=p
    def _tell(self): return self.pos
    def _skip(self, n): self.pos+=n

    def _read_u8(self):
        v,=struct.unpack_from('<B',self.data,self.pos); self.pos+=1; return v
    def _read_u16(self):
        v,=struct.unpack_from('<H',self.data,self.pos); self.pos+=2; return v
    def _read_i32(self):
        v,=struct.unpack_from('<i',self.data,self.pos); self.pos+=4; return v
    def _read_u32(self):
        v,=struct.unpack_from('<I',self.data,self.pos); self.pos+=4; return v
    def _read_f32(self):
        v,=struct.unpack_from('<f',self.data,self.pos); self.pos+=4; return v
    def _read_point3(self):
        return (self._read_f32(), self._read_f32(), self._read_f32())
    def _read_quat(self):
        return (self._read_f32(), self._read_f32(), self._read_f32(), self._read_f32())
    def _read_color_rgb_byte(self):
        return (self._read_u8(), self._read_u8(), self._read_u8())
    def _read_color_rgb_float(self):
        return (self._read_f32(), self._read_f32(), self._read_f32())

    def _read_fixed_string(self, length):
        raw=self._read(length); end=raw.find(b'\x00')
        if end==-1: end=length
        return raw[:end].decode('latin-1')

    def _read_c_string(self):
        start = self.pos
        end = len(self.data)
        while self.pos < end and self.data[self.pos] != 0:
            self.pos += 1
        s = self.data[start:self.pos].decode('latin-1')
        if self.pos < end:
            self.pos += 1  # skip null terminator
        return s

    def _read_matrix44(self): return [self._read_f32() for _ in range(16)]
    def _read_matrix43(self): return [self._read_f32() for _ in range(12)]

    def _read_chunk_header(self):
        h=ChunkHeader()
        h.type=self._read_u16(); self._skip(2)
        h.version=self._read_u32(); h.file_offset=self._read_u32(); h.chunk_id=self._read_u32()
        return h

    def _read_vertex(self): return CryVertex(self._read_point3(), self._read_point3())
    def _read_face(self): return CryFace(self._read_u32(),self._read_u32(),self._read_u32(),self._read_u32(),self._read_u32())
    def _read_tex_face(self): return CryTexFace(self._read_u32(),self._read_u32(),self._read_u32())

    def _read_link(self): return CryLink(self._read_i32(), self._read_point3(), self._read_f32())

    def _read_bone_links(self, vid):
        bl=CryBoneLinks(vid)
        for _ in range(self._read_u32()): bl.links.append(self._read_link())
        return bl

    def _read_bone_physics(self):
        bp=CryBonePhysics()
        bp.mesh_id=self._read_i32(); bp.flags=self._read_u32()
        bp.minimum=self._read_point3(); bp.maximum=self._read_point3()
        bp.spring_angle=self._read_point3(); bp.spring_tension=self._read_point3()
        bp.damping=self._read_point3()
        bp.frame_matrix=[self._read_point3() for _ in range(3)]
        return bp

    def _read_bone(self):
        b=CryBone()
        b.bone_id=self._read_i32(); b.parent_id=self._read_i32()
        b.num_children=self._read_i32()
        b.ctrl_id=f"{self._read_u32():08X}"
        b.custom_property=self._read_fixed_string(32)
        b.bone_physics=self._read_bone_physics()
        return b

    def _read_range(self):
        return (self._read_fixed_string(32), self._read_i32(), self._read_i32())

    def _read_texture(self, chunk_version):
        tex = CryTexture()
        p = self._tell()
        # v746: name field = 152 bytes; v745: 32 bytes
        name_len = 152 if chunk_version >= 0x0746 else 32
        tex.name   = self._read_fixed_string(name_len)
        self._seek(p + name_len)
        tex.type   = self._read_u32()
        tex.flags  = self._read_u32()
        tex.amount = self._read_i32()
        tex.u_tile   = bool(self._read_u8())
        tex.u_mirror = bool(self._read_u8())
        tex.v_tile   = bool(self._read_u8())
        tex.v_mirror = bool(self._read_u8())
        self._skip(12)  # nth_frame(4) + ref_size(4) + ref_blur(4)
        tex.u_offset = self._read_f32()
        tex.u_scale  = self._read_f32()
        self._skip(4)   # u_rotation
        tex.v_offset = self._read_f32()
        tex.v_scale  = self._read_f32()
        self._skip(4)   # v_rotation
        self._skip(4)   # w_rotation
        self._skip(7*4) # 7 controller IDs
        return tex

    # ── Animation key readers ─────────────────────────────────────────────────

    def _read_cry_key(self):   return CryKey(self._read_i32(), self._read_point3(), self._read_point3())
    def _read_bone_key(self):  return CryBoneKey(self._read_i32(), self._read_point3(), self._read_point3(), self._read_quat())
    def _read_lin1_key(self):  return CryLin1Key(self._read_i32(), self._read_f32())
    def _read_lin3_key(self):  return CryLin3Key(self._read_i32(), self._read_point3())
    def _read_linq_key(self):  return CryLinQKey(self._read_i32(), self._read_quat())
    def _read_bezq_key(self):  return CryBezQKey(self._read_i32(), self._read_quat())

    def _read_tcb1_key(self):
        t=self._read_i32(); v=self._read_f32()
        return CryTCB1Key(t,v,self._read_f32(),self._read_f32(),self._read_f32(),self._read_f32(),self._read_f32())

    def _read_tcb3_key(self):
        t=self._read_i32(); v=self._read_point3()
        return CryTCB3Key(t,v,self._read_f32(),self._read_f32(),self._read_f32(),self._read_f32(),self._read_f32())

    def _read_tcbq_key(self):
        t=self._read_i32(); v=self._read_quat()
        return CryTCBQKey(t,v,self._read_f32(),self._read_f32(),self._read_f32(),self._read_f32(),self._read_f32())

    def _read_bez1_key(self):
        return CryBez1Key(self._read_i32(),self._read_f32(),self._read_f32(),self._read_f32())

    def _read_bez3_key(self):
        return CryBez3Key(self._read_i32(),self._read_point3(),self._read_point3(),self._read_point3())

    # ── Chunk readers ─────────────────────────────────────────────────────────

    def _read_mesh_chunk(self, header, next_chunk_pos, is_bone_mesh=False):
        self._seek(header.file_offset); self._skip(SIZE_CHUNK_HEADER)
        chunk=CryMeshChunk(); chunk.header=header; chunk.is_bone_mesh=is_bone_mesh
        has_bone=self._read_u8(); has_vcol=self._read_u8(); self._skip(2)
        num_verts=self._read_u32(); num_tverts=self._read_u32()
        num_faces=self._read_u32(); chunk.vert_anim_id=self._read_i32()
        chunk.has_bone_info=has_bone>0; chunk.has_vertex_colors=has_vcol>0
        num_tfaces=num_faces if num_tverts>0 else 0
        chunk.vertices=[self._read_vertex() for _ in range(num_verts)]
        chunk.faces=[self._read_face() for _ in range(num_faces)]
        for _ in range(num_tverts): chunk.tex_vertices.append((self._read_f32(),self._read_f32()))
        chunk.tex_faces=[self._read_tex_face() for _ in range(num_tfaces)]
        if chunk.has_bone_info and num_verts>0:
            if next_chunk_pos and (self._tell()+SIZE_BONE_LINK>next_chunk_pos):
                chunk.has_bone_info=False
            else:
                for i in range(num_verts): chunk.physique.append(self._read_bone_links(i))
        if chunk.has_vertex_colors and num_verts>0:
            for _ in range(num_verts): chunk.vertex_colors.append(self._read_color_rgb_byte())
        return chunk

    def _read_helper_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset); self._skip(SIZE_CHUNK_HEADER)
        chunk=CryHelperChunk(); chunk.header=header
        chunk.type=self._read_u32(); chunk.size=self._read_point3()
        return chunk

    def _read_bone_anim_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset); self._skip(SIZE_CHUNK_HEADER)
        chunk=CryBoneAnimChunk(); chunk.header=header
        for _ in range(self._read_u32()): chunk.bones.append(self._read_bone())
        return chunk

    def _read_bone_name_list_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset)
        chunk = CryBoneNameListChunk()
        chunk.header = header

        if header.version == 0x0745:
            # v745: NO chunk header skip, names are null-terminated C strings
            num = self._read_u32()
            for _ in range(num):
                chunk.name_list.append(self._read_c_string())
        else:
            # Other versions: skip chunk header, names are fixed 64-byte buffers
            self._skip(SIZE_CHUNK_HEADER)
            num = self._read_u32()
            for _ in range(num):
                p = self._tell()
                chunk.name_list.append(self._read_c_string())
                self._seek(p + 64)  # skip to end of 64-byte buffer

        return chunk

    def _read_node_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset); self._skip(SIZE_CHUNK_HEADER)
        chunk=CryNodeChunk(); chunk.header=header
        chunk.name=self._read_fixed_string(64)
        chunk.object_id=self._read_i32(); chunk.parent_id=self._read_i32()
        num_children=self._read_u32(); chunk.material_id=self._read_i32()
        igh=self._read_u8(); igm=self._read_u8(); self._skip(2)
        chunk.is_group_head=igh>0; chunk.is_group_member=igm>0
        chunk.trans_matrix=self._read_matrix44()
        chunk.position=self._read_point3(); chunk.rotation=self._read_quat(); chunk.scale=self._read_point3()
        low=self._read_u16(); high=self._read_u16(); chunk.pos_ctrl_id=f"{high:04X}{low:04X}"
        low=self._read_u16(); high=self._read_u16(); chunk.rot_ctrl_id=f"{high:04X}{low:04X}"
        low=self._read_u16(); high=self._read_u16(); chunk.scale_ctrl_id=f"{high:04X}{low:04X}"
        prop_len=self._read_u32()
        if prop_len>0: chunk.property=self._read_c_string()
        for _ in range(num_children): chunk.child_nodes.append(self._read_u32())
        return chunk

    def _read_material_chunk(self, header, next_chunk_pos):
        """
        Ported 1:1 from CryImporter-chunkreader.ms readMaterialChunk.

        v745: name=64 bytes, no alphaTest, colors as RGB bytes
        v746: name=124 bytes + 4 bytes alphaTest before type, colors as RGB bytes

        Size constants from header.ms:
          size_MTL_CHUNK_DESC_0746 = 2552
          size_MTL_CHUNK_DESC_0745 = 1208
        """
        self._seek(header.file_offset)
        self._skip(SIZE_CHUNK_HEADER)

        chunk = CryMaterialChunk()
        chunk.header = header

        p = self._tell()

        # Read name string (null-terminated) then skip to end of allocated space
        raw_name = self._read_c_string()

        if header.version == 0x0745:
            self._seek(p + 64)          # 64-byte name field
        else:
            # v746: 124-byte name field + 4-byte reserved = 128 bytes total, then alphaTest float
            self._seek(p + 124)
            chunk.alpha_test = self._read_f32()

        # Parse shader name and surface name out of the raw name string
        # Format can be: "matname(shaderName)/surfaceName"
        s_start = raw_name.find('(')
        s_end   = raw_name.find(')')
        m_start = raw_name.find('/')

        if s_start != -1 and s_end != -1:
            chunk.shader_name = raw_name[s_start+1:s_end]
        if m_start != -1:
            chunk.surface_name = raw_name[m_start+1:]

        if m_start != -1 and s_start == -1:
            chunk.name = raw_name[:m_start]
        elif s_start != -1:
            chunk.name = raw_name[:s_start]
        else:
            chunk.name = raw_name

        chunk.name = chunk.name.strip()
        if not chunk.name:
            chunk.name = raw_name.strip()

        # Material type
        chunk.type = self._read_i32()

        if chunk.type == 2:  # materialType_Multi
            num_children = self._read_i32()

            # Skip to end of material chunk descriptor, then read child IDs
            if header.version == 0x0746:
                self._seek(header.file_offset + 2552)
            else:
                self._seek(header.file_offset + 1208)

            for _ in range(num_children):
                child_id = self._read_i32()
                if child_id > 0:
                    chunk.children.append(child_id)

        else:  # materialType_Standard (1) or other
            # Colors stored as RGB bytes (not floats!)
            # Order: diffuse, specular, ambient + 3 padding bytes
            def read_color_byte():
                r = self._read_u8() / 255.0
                g = self._read_u8() / 255.0
                b = self._read_u8() / 255.0
                return (r, g, b)

            chunk.diffuse  = read_color_byte()
            chunk.specular = read_color_byte()
            chunk.ambient  = read_color_byte()
            self._skip(3)  # padding to 4-byte boundary

            chunk.specular_level     = self._read_f32()
            chunk.specular_shininess = self._read_f32()
            chunk.self_illumination  = self._read_f32()
            chunk.opacity            = self._read_f32()

            # 10 textures in order: ambient, diffuse, specular, opacity,
            #                       bump, gloss, filter(detail), reflection, subsurface, detail(normalmap)
            textures = [self._read_texture(header.version) for _ in range(10)]
            if textures[1] and textures[1].name: chunk.tex_diffuse    = textures[1]
            if textures[2] and textures[2].name: chunk.tex_specular   = textures[2]
            if textures[4] and textures[4].name: chunk.tex_bump       = textures[4]
            if textures[7] and textures[7].name: chunk.tex_reflection = textures[7]
            if textures[9] and textures[9].name: chunk.tex_detail     = textures[9]

            # Flags come AFTER textures
            chunk.flags = self._read_u32()
            # dynamicBounce, staticFriction, slidingFriction (not used in Blender)
            self._skip(12)

        return chunk

    def _read_controller_chunk(self, header, next_chunk_pos):
        """
        Ported from CryImporter-chunkreader.ms readControllerChunk.
        v826: typed keys (CryBone, Linear, Bezier, TCB).
        v827: always pos + rotation logarithm (CryKey).
        """
        self._seek(header.file_offset)
        chunk=CryControllerChunk(); chunk.header=header

        if header.version==0x0826:
            self._skip(SIZE_CHUNK_HEADER)
            chunk.ctrl_type=self._read_u32()
            num_keys=self._read_u32()
            chunk.flags=self._read_u32()
            low=self._read_u16(); high=self._read_u16()
            chunk.ctrl_id=f"{high:04X}{low:04X}"
            readers={
                CTRL_CRY_BONE: self._read_bone_key,
                CTRL_LINEAR1:  self._read_lin1_key,
                CTRL_LINEAR3:  self._read_lin3_key,
                CTRL_LINEAR_Q: self._read_linq_key,
                CTRL_BEZIER1:  self._read_bez1_key,
                CTRL_BEZIER3:  self._read_bez3_key,
                CTRL_BEZIER_Q: self._read_bezq_key,
                CTRL_TCB1:     self._read_tcb1_key,
                CTRL_TCB3:     self._read_tcb3_key,
                CTRL_TCBQ:     self._read_tcbq_key,
            }
            fn=readers.get(chunk.ctrl_type)
            if fn and num_keys>0:
                for _ in range(num_keys): chunk.keys.append(fn())

        elif header.version==0x0827:
            num_keys=self._read_u32()
            low=self._read_u16(); high=self._read_u16()
            chunk.ctrl_id=f"{high:04X}{low:04X}"
            chunk.ctrl_type=CTRL_CRY_BONE
            for _ in range(num_keys): chunk.keys.append(self._read_cry_key())
        else:
            return None

        return chunk

    def _read_timing_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset); self._skip(SIZE_CHUNK_HEADER)
        chunk=CryTimingChunk(); chunk.header=header
        chunk.secs_per_tick=self._read_f32(); chunk.ticks_per_frame=self._read_u32()
        chunk.global_range=self._read_range()
        for _ in range(self._read_u32()): chunk.sub_ranges.append(self._read_range())
        return chunk

    def _read_mesh_morph_target_chunk(self, header, next_chunk_pos):
        # Original Max script order: meshChunkID → numVerts → targetVertices → name (ReadString at end)
        self._seek(header.file_offset); self._skip(SIZE_CHUNK_HEADER)
        chunk=CryMeshMorphTargetChunk(); chunk.header=header
        chunk.mesh_chunk_id=self._read_u32()
        for _ in range(self._read_u32()):
            chunk.target_vertices.append(CryMeshMorphTargetVertex(self._read_u32(), self._read_point3()))
        chunk.name=self._read_c_string()  # name is at the END, variable length
        return chunk

    def _read_bone_initial_pos_chunk(self, header, next_chunk_pos):
        self._seek(header.file_offset)
        chunk=CryBoneInitialPosChunk(); chunk.header=header
        chunk.mesh_chunk_id=self._read_u32()
        for _ in range(self._read_u32()): chunk.initial_positions.append(self._read_matrix43())
        return chunk

    def _read_chunk(self, header, next_chunk_pos):
        if header.file_offset>=len(self.data): return None
        t=header.type
        try:
            if   t==CHUNK_TYPE_MESH:              return self._read_mesh_chunk(header, next_chunk_pos)
            elif t==CHUNK_TYPE_HELPER:            return self._read_helper_chunk(header, next_chunk_pos)
            elif t==CHUNK_TYPE_BONE_ANIM:         return self._read_bone_anim_chunk(header, next_chunk_pos)
            elif t==CHUNK_TYPE_BONE_NAME_LIST:    return self._read_bone_name_list_chunk(header, next_chunk_pos)
            elif t==CHUNK_TYPE_NODE:              return self._read_node_chunk(header, next_chunk_pos)
            elif t==CHUNK_TYPE_MATERIAL:          return self._read_material_chunk(header, next_chunk_pos)
            elif t==CHUNK_TYPE_CONTROLLER:        return self._read_controller_chunk(header, next_chunk_pos)
            elif t==CHUNK_TYPE_TIMING:            return self._read_timing_chunk(header, next_chunk_pos)
            elif t==CHUNK_TYPE_BONE_MESH:         return self._read_mesh_chunk(header, next_chunk_pos, is_bone_mesh=True)
            elif t==CHUNK_TYPE_MESH_MORPH_TARGET: return self._read_mesh_morph_target_chunk(header, next_chunk_pos)
            elif t==CHUNK_TYPE_BONE_INITIAL_POS:  return self._read_bone_initial_pos_chunk(header, next_chunk_pos)
            else: return None
        except Exception as e:
            print(f"[CGF] Warning: chunk 0x{t:04X} id={header.chunk_id}: {e}")
            return None

    def read_file(self, filepath):
        if not os.path.isfile(filepath):
            raise ValueError(f"File not found: {filepath}")
        with open(filepath,'rb') as f: self.data=f.read()
        self.pos=0
        if self._read(6)!=FILE_SIGNATURE:
            raise ValueError("Not a CryTek file")
        self._seek(8)
        file_type_low=self._read_u16(); file_type_high=self._read_u16()
        is_geom=(file_type_high==FILE_TYPE_GEOM_HIGH and file_type_low==FILE_TYPE_GEOM_LOW)
        is_anim=(file_type_high==FILE_TYPE_ANIM_HIGH and file_type_low==FILE_TYPE_ANIM_LOW)
        if not (is_geom or is_anim):
            raise ValueError(f"Unknown file type 0x{file_type_high:04X}:0x{file_type_low:04X}")
        self._seek(16); chunk_table_pos=self._read_u32()
        self._seek(chunk_table_pos); num_chunks=self._read_u32(); hstart=self._tell()
        headers=[]
        for i in range(num_chunks):
            self._seek(hstart+i*SIZE_CHUNK_HEADER)
            headers.append(self._read_chunk_header())
        headers.sort(key=lambda h: (h.file_offset, h.type, h.chunk_id))
        archive=CryChunkArchive()
        archive.geom_file_name=filepath if is_geom else ""
        for i,h in enumerate(headers):
            next_pos=headers[i+1].file_offset if i+1<num_chunks else chunk_table_pos
            print(f"[CGF] chunk {i}/{num_chunks} type=0x{h.type:04X} ver=0x{h.version:04X} offset={h.file_offset}")
            chunk=self._read_chunk(h, next_pos)
            if chunk is not None: archive.add(chunk)
        return archive


# ── CAL reader ────────────────────────────────────────────────────────────────

class CALRecord:
    __slots__ = ('name','path','start_frame','end_frame')
    def __init__(self, name, path):
        self.name=name; self.path=path; self.start_frame=0; self.end_frame=0


def read_cal_file(filepath):
    records=[]
    if not os.path.isfile(filepath): return records
    with open(filepath,'r',errors='replace') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#') or line.startswith(';'): continue
            parts=line.split(None,1)
            if len(parts)==2: records.append(CALRecord(parts[0], parts[1]))
            elif len(parts)==1: records.append(CALRecord(parts[0], parts[0]))
    return records
