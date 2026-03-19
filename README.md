# CryEngine 1 CGF Importer for Blender

A Blender addon for importing **CryEngine 1 / Far Cry (2004)** geometry and animation files into Blender 4.0+.

Ported from the original [CryImporter for 3ds Max 8](https://www.takaro.net) by Takaro Pty. Ltd.

---

## Features

- Import `.cgf` and `.cga` geometry files
- Import `.caf` animation files onto armatures
- Import `.cal` animation list files (multiple animations at once)
- Mesh with correct vertex positions, normals, and UV coordinates
- Multi-material support (single and multi-sub materials)
- Texture auto-detection from file path (`.dds`, `.tga`, `.png`)
- Skeleton / armature import from bone chunks
- Vertex weights (skinning) for character models
- Shape keys (morph targets / facial expressions)
- Full animation support: CryBone, Linear, Bezier, TCB controller types (v826 and v827)
- **Auto-import:** when opening CAF or CAL, the CGF is found and imported automatically
- Correct scale conversion: Max inches → Blender meters (`× 0.0254`) applied to geometry AND animation
- Supports Blender 4.0, 4.1, 4.2, 5.0+

---

## Installation

1. Download `io_import_cgf.zip` from [Releases](../../releases)
2. In Blender: **Edit → Preferences → Add-ons → Install...**
3. Select the downloaded `.zip` file
4. Enable **"CryEngine 1 CGF Importer (Far Cry)"**

---

## Usage

### Importing geometry — CGF / CGA

**File → Import → CryEngine Geometry (.cgf, .cga)**

Just select the file and import. No preparation needed.

**Required file layout:**
```
any_folder/
├── model.cgf       ← select this
├── texture.dds     ← textures should be in the same folder (or subfolders)
└── texture2.dds
```

| Option | Description |
|---|---|
| Import UVs | Import texture coordinates |
| Import Normals | Use normals from file |
| Import Materials | Create Principled BSDF materials from chunk data |
| Import Skeleton | Build armature from bone chunks |
| Import Vertex Weights | Assign bone weights for skinned meshes |

---

### Importing a single animation — CAF

**File → Import → CryEngine Animation (.caf)**

The addon **automatically finds and imports the CGF** from the same folder before loading the animation. You do not need to import the CGF manually first.

**Required file layout:**
```
any_folder/
├── model.cgf       ← found and imported automatically
└── model.caf       ← select this
```

If there are multiple CGF files in the folder, the addon first tries the one with the same base name as the CAF (`model.caf` → `model.cgf`), then falls back to any CGF found in the folder.

If you already have the CGF imported and the armature is in the scene — just select it and import the CAF directly, the auto-import step is skipped.

---

### Importing multiple animations — CAL

**File → Import → CryEngine Animation List (.cal)**

A CAL file is a plain text list of animation names and their CAF file paths. The addon reads it, **automatically imports the CGF** from the same folder, then imports all listed CAF files as separate Actions.

**Required file layout:**
```
any_folder/
├── model.cgf       ← found and imported automatically
├── model.cal       ← select this
├── idle.caf        ← imported automatically from the list
├── walk.caf
└── run.caf
```

After import, switch to the **Action Editor** (or **NLA Editor**) to see and switch between all imported animations.

---

### Switching between animations after import

Open the **Action Editor** (bottom of the viewport → change editor type to Action Editor). All imported animations appear in the dropdown next to the action name field.

---

## Supported Chunk Types

| Chunk | Description |
|---|---|
| `0x0000` Mesh | Geometry, UVs, normals, vertex colors |
| `0x000B` Node | Scene hierarchy and world transform |
| `0x000C` Material | Colors and texture references |
| `0x0003` BoneAnim | Skeleton definition |
| `0x0005` BoneNameList | Bone names |
| `0x000D` Controller | Animation keys (v826 and v827) |
| `0x000F` BoneMesh | Bone physics meshes |
| `0x0011` MeshMorphTarget | Shape keys / facial expressions |
| `0x0012` BoneInitialPos | Bone rest pose matrices |
| `0x000E` Timing | Animation timing / FPS info |

### Supported controller types (v826)

| Type | Description |
|---|---|
| CryBone | Position + quaternion per bone |
| Linear3 / LinearQ | Linear interpolation (position / rotation) |
| Bezier3 / BezierQ | Bezier interpolation (position / rotation) |
| TCB3 / TCBQ | Tension-Continuity-Bias (position / rotation) |

---

## Coordinate System & Scale

CryEngine 1 / Far Cry was authored in **3ds Max with inches** as the unit system.

- **Scale:** `1 Max inch = 0.0254 Blender meters` — applied automatically to all geometry and animation data
- **Axes:** Max Z-up → Blender Z-up (compatible, node transforms handle orientation)
- **Object scale:** always `(1, 1, 1)` — scale is baked into coordinates, not the object transform

---

## Known Limitations

- BSpline controller types not supported (rare in Far Cry assets)
- VertAnim (vertex animation) chunks not yet applied
- Physics-only meshes are skipped

---

## File Format

The CGF format is a chunk-based binary format developed by Crytek.
File versions supported: `0x0744`, `0x0745`, `0x0746`, `0x0826`, `0x0827`

---

## Credits

- Original **CryImporter for 3ds Max** by [Takaro Pty. Ltd.](https://www.takaro.net) — binary format parsing and animation logic ported directly from their MaxScript
- Blender addon port and Python rewrite — this project

---

## License

This software is provided **as-is**, without any express or implied warranty.
Based on CryImporter by Takaro Pty. Ltd. — original license terms apply to the format knowledge derived from that work.

Free for non-commercial use. See `LICENSE` for details.
