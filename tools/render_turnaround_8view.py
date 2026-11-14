import sys, os, math, time, tempfile
import bpy
import mathutils
from PIL import Image, ImageDraw, ImageFont

def render_8view(glb_path, out_png_path):
    print(f"[render_turnaround_8view] Loading: {glb_path}")
    
    # Reset scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    
    # Use Workbench for lightning-fast high quality flat/texture preview or EEVEE
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'TEXTURE'
    scene.display.render_aa = 'FXAA'
    
    # 512x512 per view
    tile_w, tile_h = 512, 512
    scene.render.resolution_x = tile_w
    scene.render.resolution_y = tile_h
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    
    # Import GLB
    bpy.ops.import_scene.gltf(filepath=glb_path)
    
    # Find mesh objects and calculate collective bounding box
    mesh_objs = [o for o in scene.objects if o.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError("No mesh objects found in GLB")
    
    total_verts = sum(len(o.data.vertices) for o in mesh_objs)
    total_faces = sum(len(o.data.polygons) for o in mesh_objs)
    
    min_c = [float('inf')] * 3
    max_c = [float('-inf')] * 3
    for o in mesh_objs:
        for v in o.bound_box:
            world_v = o.matrix_world @ mathutils.Vector(v)
            for i in range(3):
                min_c[i] = min(min_c[i], world_v[i])
                max_c[i] = max(max_c[i], world_v[i])
                
    center = mathutils.Vector([(min_c[i] + max_c[i]) / 2.0 for i in range(3)])
    dims = mathutils.Vector([max_c[i] - min_c[i] for i in range(3)])
    max_dim = max(dims.x, dims.y, dims.z)
    
    # Center all objects
    for o in mesh_objs:
        o.location -= center
        
    # Camera setup
    cam_data = bpy.data.cameras.new("ReviewCamera")
    cam_data.lens = 50  # 50mm normal lens
    cam_obj = bpy.data.objects.new("ReviewCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    
    # Distance to fit object with margin
    dist = (max_dim / (2.0 * math.tan(math.radians(cam_data.angle / 2.0)))) * 1.35
    if dist < 0.1: dist = 2.0
    
    pitch = math.radians(14.0)  # slight downward tilt
    
    # 8 canonical azimuth angles (0 to 315 in 45 deg steps)
    angles = [
        (0, "FRONT"),
        (45, "FRONT-RIGHT 45°"),
        (90, "RIGHT 90°"),
        (135, "BACK-RIGHT 135°"),
        (180, "BACK 180°"),
        (225, "BACK-LEFT 225°"),
        (270, "LEFT 270°"),
        (315, "FRONT-LEFT 315°"),
    ]
    
    tmp_renders = []
    with tempfile.TemporaryDirectory(prefix="trellis_8view_", dir=os.environ.get("TMPDIR")) as tmp_dir:
        for idx, (deg, label) in enumerate(angles):
            rad = math.radians(deg)
            # Position camera on sphere around center
            cx = dist * math.sin(rad) * math.cos(pitch)
            cy = -dist * math.cos(rad) * math.cos(pitch)
            cz = dist * math.sin(pitch)

            cam_obj.location = mathutils.Vector((cx, cy, cz))

            # Look at origin
            direction = mathutils.Vector((0, 0, 0)) - cam_obj.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            cam_obj.rotation_euler = rot_quat.to_euler()

            frame_path = os.path.join(tmp_dir, f"view_{idx}_{deg}.png")
            scene.render.filepath = frame_path
            bpy.ops.render.render(write_still=True)
            tmp_renders.append((frame_path, label))

        # Composite into 4x2 contact sheet
        sheet_w = tile_w * 4
        header_h = 70
        footer_h = 30
        sheet_h = tile_h * 2 + header_h + footer_h

        sheet = Image.new("RGBA", (sheet_w, sheet_h), (24, 24, 27, 255)) # Dark slate #18181b
        draw = ImageDraw.Draw(sheet)

        # Header
        asset_name = os.path.basename(glb_path)
        header_title = "TRELLIS 3D / ART3D PIPELINE ASSET ACCEPTANCE REVIEW"
        stats_text = f"Asset: {asset_name}  |  Verts: {total_verts:,}  |  Polys: {total_faces:,}  |  Bounds: {dims.x:.2f} × {dims.y:.2f} × {dims.z:.2f}m"

        draw.text((20, 15), header_title, fill=(244, 244, 245, 255))
        draw.text((20, 42), stats_text, fill=(161, 161, 170, 255))

        # Paste 8 tiles (2 rows x 4 cols)
        for idx, (fpath, label) in enumerate(tmp_renders):
            row = idx // 4
            col = idx % 4
            x = col * tile_w
            y = header_h + row * tile_h

            tile_img = Image.open(fpath)
            # Paste with alpha
            sheet.paste(tile_img, (x, y), tile_img)

            # Tile border
            draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=(39, 39, 42, 255), width=1)

            # Tile Label tag
            tag_bg = (39, 39, 42, 220)
            draw.rectangle([x + 10, y + 10, x + 10 + len(label)*9 + 16, y + 32], fill=tag_bg)
            draw.text((x + 18, y + 13), label, fill=(228, 228, 231, 255))
            
    # Footer
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    draw.text((20, sheet_h - 22), f"Generated by Trellis 3D Studio  •  {ts}", fill=(113, 113, 122, 255))
    
    sheet.save(out_png_path, "PNG")
    print(f"[render_turnaround_8view] Saved 8-view turnaround panel to: {out_png_path}")

if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        args = argv[argv.index("--") + 1:]
    else:
        args = argv[1:]
    if len(args) < 2:
        print("Usage: blender -b -P render_turnaround_8view.py -- <glb_path> <out_png_path>")
        sys.exit(1)
    render_8view(args[0], args[1])
