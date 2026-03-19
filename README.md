# CryEngine 1 CGF Importer for Blender

A Blender addon for importing **CryEngine 1 / Far Cry (2004)** geometry files (`.cgf`, `.cga`) into Blender 4.0+.

Ported from the original [CryImporter for 3ds Max 8](https://www.takaro.net) by Takaro Pty. Ltd.

---

## Features

- Import `.cgf` and `.cga` geometry files
- Mesh with correct vertex positions, normals, and UV coordinates
- Multi-material support (single and multi-sub materials)
- Texture auto-detection from file path (`.dds`, `.tga`, `.png`)
- Skeleton / armature import from bone chunks
- Vertex weights (skinning) for character models
- Shape keys (morph targets / facial expressions)
- Correct coordinate system conversion (Max inches → Blender meters)
- Supports Blender 4.0, 4.1, 4.2, 5.0+

---

## Installation

1. Download `io_import_cgf.zip` from [Releases](../../releases)
2. In Blender: **Edit → Preferences → Add-ons → Install...**
3. Select the downloaded `.zip` file
4. Enable **"CryEngine 1 CGF Importer (Far Cry)"**

---

## Usage

**File → Import → CryEngine CGF (.cgf, .cga)**

### Import options

| Option | Description |
|---|---|
| Import UVs | Import texture coordinates |
| Import Normals | Use normals from file (otherwise Blender recalculates) |
| Import Materials | Create Principled BSDF materials from chunk data |
| Import Skeleton | Build armature from bone chunks |
| Import Vertex Weights | Assign bone weights for skinned meshes |

### Tips

- Place textures in the same folder as the `.cgf` file for auto-detection
- Far Cry textures are `.dds` — Blender can load them natively
- For character models, import the `.cgf` first, then you can import `.caf` animations (planned)

---

## Supported Chunk Types

| Chunk | Description |
|---|---|
| `0x0000` Mesh | Geometry, UVs, normals, vertex colors |
| `0x000B` Node | Scene hierarchy and world transform |
| `0x000C` Material | Colors and texture references |
| `0x0003` BoneAnim | Skeleton definition |
| `0x0005` BoneNameList | Bone names |
| `0x000F` BoneMesh | Bone physics meshes |
| `0x0011` MeshMorphTarget | Shape keys / facial expressions |
| `0x0012` BoneInitialPos | Bone rest pose matrices |
| `0x000E` Timing | Animation timing info |

---

## Coordinate System

CryEngine 1 / Far Cry was authored in **3ds Max with inches** as the unit system.

- Scale: `1 Max inch = 0.0254 Blender meters` — applied automatically
- Axes: Max Z-up → Blender Z-up (no rotation needed, node transforms handle orientation)

---

## Known Limitations

- `.caf` / `.cal` animation import not yet implemented
- BSpline controller types not supported (rare in Far Cry assets)
- Physics-only meshes are skipped
- Some very early CGF versions may not parse correctly

---

## File Format

The CGF format is a chunk-based binary format developed by Crytek.  
File versions supported: `0x0744`, `0x0745`, `0x0746`, `0x0826`, `0x0827`

Format reference based on reverse engineering by the modding community and the original CryImporter source by Takaro Pty. Ltd.

---

## Credits

- Original **CryImporter for 3ds Max** by [Takaro Pty. Ltd.](https://www.takaro.net) — the binary format parsing is ported directly from their MaxScript
- Blender addon port and Python rewrite — this project

---

## License

This software is provided **as-is**, without any express or implied warranty.  
Based on CryImporter by Takaro Pty. Ltd. — original license terms apply to the format knowledge derived from that work.

Free for non-commercial use. See `LICENSE` for details.
