# CryEngine 1 Import/Export Addon for Blender

Blender addon for working with CryEngine 1 / Far Cry era geometry and animation files.

The addon focuses on practical round-trip work:

- import geometry, materials, skeletons, weights, morph targets, and animation data
- edit assets in Blender
- export geometry and animation back to CryEngine 1 formats

The codebase is based on the original CryImporter / CryExport toolchain used for 3ds Max and on additional legacy CryEngine exporter references.

## Supported Blender Versions

- Blender 4.x
- Blender 5.x

## Supported Formats

### Import

- `.cgf` - geometry
- `.cga` - animated / skinned geometry
- `.bld` - building geometry
- `.caf` - animation
- `.anm` - animation container handled through the same importer path as CAF
- `.cal` - animation list

### Export

- `.cgf` - geometry
- `.cga` - animated / skinned geometry
- `.bld` - building geometry
- `.caf` - animation
- `.anm` - animation container handled through the same exporter path as CAF
- `.cal` - animation list
- auto export mode that decides between geometry and animation outputs from scene contents

## Current Feature Set

### Geometry Import

- imports one or many mesh chunks from a single file as separate Blender mesh objects
- keeps mesh objects under their source node names when available
- imports UVs and custom split normals
- imports material assignments per mesh and per polygon
- imports morph targets as shape keys
- can globally skip collision-like helper geometry based on reliable CryEngine material markers
- creates a dedicated collection per imported asset

### Material Import

- creates Blender node materials from CryEngine material chunks
- loads diffuse, bump / normal, and detail textures when found
- supports automatic texture lookup through the configured Game Textures Path
- supports common Far Cry texture formats: `.dds`, `.tga`, `.tif`, `.tiff`, `.png`, `.jpg`, `.bmp`
- maps DDN textures to a normal-map workflow
- preserves Cry shader and surface metadata on the Blender material
- reuses equivalent materials inside a single imported file instead of creating duplicates for every mesh

### Skeleton / Skin Import

- imports bone hierarchy
- imports vertex weights
- preserves original Cry bone metadata needed for later export
- preserves bone initial matrices for round-trip export
- applies embedded controller data from geometry files when present so rigged assets are not left only in rest pose

### Animation Import

- imports CAF and ANM animation data onto an armature
- imports CAL files and creates separate Blender actions
- if no armature is present, the importer can auto-load the matching CGF / CGA from the same folder
- uses controller ids from imported Cry data to match animation tracks back to bones

### Geometry Export

- exports visible meshes or selected meshes only
- exports materials
- exports skeletons and skin weights
- exports CGF, CGA, and BLD through the geometry exporter path
- preserves stored Cry bone matrices for better round-trip behavior
- writes vertex colors, including default white when needed for engine compatibility

### Animation Export

- exports the active action to CAF
- exports the active action to ANM through the same backend path
- exports all action tracks as CAF files plus a CAL list
- supports automatic asset export from a single command

### Auto Export

Auto export inspects the current scene and writes the most useful output automatically:

- exports `.cga` for skinned meshes when that option is enabled
- exports `.cgf` for non-skinned geometry
- can also write `.cal` plus multiple `.caf` files when actions are present on the armature

## Installation

1. Download or package the addon as a zip that contains the `io_import_cgf` folder.
2. In Blender open `Edit -> Preferences -> Add-ons`.
3. Click `Install...`.
4. Select the addon zip.
5. Enable `CryEngine 1 CGF Importer/Exporter (Far Cry)`.
6. Open the addon preferences and set `Game Textures Path` to your Far Cry / CryEngine 1 game folder.
7. Optionally enable `Skip Collision-Like Geometry` if you do not want helper / collision-like meshes to be imported by default.

## Addon Preferences

### Game Textures Path

Set the textures path to the folder that contains directories such as:

- `Objects`
- `Textures`
- `Levels`

Example:

```text
C:\FarCry
|-- Objects
|-- Textures
`-- Levels
```

This path is used to resolve relative texture paths stored inside CryEngine files.

### Skip Collision-Like Geometry

When enabled, the geometry importer skips collision-like helper content based on reliable CryEngine material markers such as:

- shader `NoDraw`
- shader `no_draw`
- surface `mat_obstruct`
- surface `mat_nodraw`
- diffuse texture `nodraw.dds`

This is useful when CryEngine assets contain extra proxy or collision-only geometry that you do not want cluttering the Blender scene.

The filter can remove:

- full helper meshes that are entirely collision-like
- collision-like polygons inside mixed meshes
- corresponding NoDraw helper materials

## Import Workflows

### Import Geometry

Menu:

- `File -> Import -> CryEngine Geometry (.cgf, .cga, .bld)`

Options:

- `Import UVs`
- `Import Normals`
- `Import Materials`
- `Import Skeleton`
- `Import Vertex Weights`
- `Override Textures Path`

Typical results:

- static assets import as one or more mesh objects
- multi-mesh assets stay split into separate Blender objects
- rigged assets can import with armature and weights
- embedded controller data is applied when present in the source file
- collision-like helper meshes can be skipped globally from addon preferences

### Import CAF / ANM

Menu:

- `File -> Import -> CryEngine Animation (.caf)`
- `File -> Import -> CryEngine Animation (.anm)`

Workflow:

1. Import the source geometry first, or select an existing armature.
2. Import the animation file.
3. The addon creates or updates a Blender action on the target armature.

If no armature exists, the importer tries to find a matching `.cgf` or `.cga` in the same folder and imports it automatically.

### Import CAL

Menu:

- `File -> Import -> CryEngine Animation List (.cal)`

Workflow:

1. Import or select the target armature.
2. Import the CAL file.
3. The addon resolves the listed CAF files and creates Blender actions for them.

## Export Workflows

### Export CGF

Menu:

- `File -> Export -> CryEngine Geometry (.cgf)`

Use for:

- static props
- geometry without character-style animation playback

Options:

- `Selected Only`
- `Export Materials`
- `Export Skeleton`
- `Export Vertex Weights`

### Export CGA

Menu:

- `File -> Export -> CryEngine Animated Geometry (.cga)`

Use for:

- skinned meshes
- animated or character-style geometry

Options:

- `Selected Only`
- `Export Materials`
- `Export Skeleton`
- `Export Vertex Weights`

### Export BLD

Menu:

- `File -> Export -> CryEngine Building (.bld)`

Use for:

- building / level geometry

Defaults:

- skeleton export disabled
- weight export disabled

### Export CAF / ANM

Menu:

- `File -> Export -> CryEngine Animation (.caf)`
- `File -> Export -> CryEngine Animation (.anm)`

Workflow:

1. Select the armature.
2. Make sure the desired action is active.
3. Export the animation.

### Export CAL

Menu:

- `File -> Export -> CryEngine Animation List (.cal)`

Workflow:

1. Keep all needed actions in the Blender file.
2. Export CAL.
3. The addon writes the CAL file and companion CAF files for eligible actions.

### Auto Export

Menu:

- `File -> Export -> CryEngine Auto Export`

Useful when:

- you want the addon to decide whether geometry should be written as CGF or CGA
- you want geometry plus animation set export from a single command

Options:

- `Selected Only`
- `Export Materials`
- `Export CAL/CAF When Present`
- `Use CGA For Skinned Meshes`

## CryEngine Material Panel

Open:

- `View3D -> N Panel -> CryEngine`

Per-material properties:

- shader preset
- custom shader name
- surface / physics material

The panel also rebuilds the full Cry material name in the expected format:

```text
material_name(ShaderName)/surface_name
```

Example:

```text
s_mut_abrr(TemplBumpSpec_GlossAlpha)/mat_default
```

## Texture Slots Used by the Addon

- slot 1: diffuse
- slot 4: bump / normal
- slot 9: detail / height

## Round-Trip Notes

The addon stores extra CryEngine metadata on imported data where possible:

- shader name
- surface name
- full Cry material name
- controller ids
- bone ids and parent ids
- bone initial matrices

This improves export fidelity when you import a CryEngine asset, edit it, and export it again.

## Known Limitations

- some legacy controller types are still uncommon and may not be fully covered in every asset
- CryEngine format edge cases can still exist on unusual assets from old pipelines
- engine-side crashes are usually caused by malformed chunk structure, invalid hierarchy data, or mismatched animation / skeleton data; the addon is being aligned against legacy exporter references to reduce those cases
- very custom material graphs in Blender will not map back to CryEngine 1 one-to-one

## Recommended Usage

For geometry only:

1. Import CGF or BLD.
2. Edit mesh or materials.
3. Export CGF or BLD.

For rigged assets:

1. Import CGA or skinned CGF with skeleton and weights enabled.
2. Verify the imported armature and material assignments.
3. Import CAF / ANM / CAL if needed.
4. Edit the mesh, rig, or actions.
5. Export CGA, CAF, ANM, or CAL as needed.

## Credits

- original CryImporter / CryExport pipeline by Takaro Pty. Ltd. and legacy CryEngine tool authors
- Blender addon port and ongoing compatibility work in this project

## License

See [LICENSE](LICENSE).
