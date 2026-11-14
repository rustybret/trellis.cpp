import argparse
import sys, os
from PIL import Image, ImageOps
import numpy as np

def auto_preprocess(input_path, output_path, target_size=1024, padding_ratio=0.15, alpha_threshold=15):
    """
    Standardizes input images for Trellis 3D Studio / DiT flow matching:
    1. Inspects alpha channel & background.
    2. Centers and proportionally fits the subject into a square canvas with safety margin.
    3. Cleans up edge spill and semi-transparent alpha halos.
    4. Outputs clean 32-bit RGBA PNG.
    """
    img = Image.open(input_path)
    
    # Check if image already has alpha
    has_alpha = False
    if img.mode == 'RGBA':
        alpha_arr = np.array(img.split()[-1])
        # If more than 5% of pixels are transparent/semi-transparent, treat as pre-matted
        if np.mean(alpha_arr < 250) > 0.05:
            has_alpha = True
            
    print(f"[Preprocess] Input: {img.size}, mode={img.mode}, has_alpha={has_alpha}")
    
    if not has_alpha:
        # Step A: Run background removal or corner-color threshold
        img = img.convert('RGB')
        arr = np.array(img)
        # Check if corners are uniform color (e.g. solid white or solid gray studio background)
        corners = np.concatenate([
            arr[:15, :15].reshape(-1, 3),
            arr[:15, -15:].reshape(-1, 3),
            arr[-15:, :15].reshape(-1, 3),
            arr[-15:, -15:].reshape(-1, 3)
        ], axis=0)
        corner_mean = corners.mean(axis=0)
        corner_std = corners.std(axis=0)
        
        # If corners are very uniform (std < 10)
        if corner_std.max() < 10:
            print(f"[Preprocess] Detected uniform studio background: RGB={corner_mean.round(1)} (std={corner_std.round(1)})")
            # Calculate color distance to corner background
            dist = np.linalg.norm(arr.astype(float) - corner_mean, axis=2)
            # Create alpha mask with soft transition
            alpha = np.clip((dist - 15.0) / 20.0 * 255.0, 0, 255).astype(np.uint8)
            # Zero out background RGB
            r, g, b = img.split()
            img = Image.merge('RGBA', (r, g, b, Image.fromarray(alpha)))
            has_alpha = True
        else:
            print("[Preprocess] Complex non-uniform background detected. BiRefNet/Rembg recommended.")
            img = img.convert('RGBA')

    # Step B: Tight bounding box crop based on alpha
    if has_alpha:
        alpha_channel = np.array(img.split()[-1])
        # Mask where alpha is significant (> alpha_threshold)
        fg_mask = alpha_channel > alpha_threshold
        if np.any(fg_mask):
            y_indices, x_indices = np.where(fg_mask)
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            
            # Crop to foreground
            img = img.crop((x_min, y_min, x_max + 1, y_max + 1))
            print(f"[Preprocess] Cropped to subject bbox: ({x_max - x_min + 1} x {y_max - y_min + 1})")
            
            # Clean alpha thresholding: harden weak alpha < 20 to 0 (eliminates fog/blobs)
            clean_alpha = np.array(img.split()[-1])
            clean_alpha[clean_alpha < alpha_threshold] = 0
            img.putalpha(Image.fromarray(clean_alpha))

    # Step C: Proportional letterboxing into square canvas with safety margin
    # Subject should occupy (1.0 - 2 * padding_ratio) of the square canvas (e.g. 80-85%)
    w, h = img.size
    max_dim = max(w, h)
    
    # Calculate scale factor to fit within target inner area
    usable_size = int(target_size * (1.0 - 2.0 * padding_ratio))
    scale = usable_size / float(max_dim)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Step D: Center into target square canvas with 100% transparent background
    canvas = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y), img_resized)
    
    # Step E: Zero out RGB for fully transparent pixels (prevent bilinear color bleeding)
    canvas_arr = np.array(canvas)
    zero_alpha = canvas_arr[:, :, 3] == 0
    canvas_arr[zero_alpha, :3] = 0
    clean_output = Image.fromarray(canvas_arr)
    
    clean_output.save(output_path, 'PNG')
    print(f"[Preprocess] Saved clean 3D-ready input: {output_path} ({clean_output.size}, mode={clean_output.mode})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Normalize an explicitly selected input image.")
    parser.add_argument("input", help="source image path")
    parser.add_argument("output", help="processed PNG path")
    args = parser.parse_args()
    auto_preprocess(args.input, args.output)
