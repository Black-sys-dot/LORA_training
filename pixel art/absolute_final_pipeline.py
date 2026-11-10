import cv2
import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def execute_flawless_pipeline(input_path, output_path, colors=32, offset=2):
    print(f"Executing Flawless Pipeline on {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img).astype(np.float32)
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # ---------------------------------------------------------
    # PHASE 1: PERFECT BACKGROUND REMOVAL (No jagged holes)
    # ---------------------------------------------------------
    # We now know the AI generated a (112, 254, 255) background, not pure cyan.
    # We strictly target ONLY that exact milky-cyan color to delete the solid background,
    # completely protecting her skin, hair, and the dark anti-aliased edge!
    cyan_mask = (g > 240) & (b > 240) & (r > 90) & (r < 130)
    data[cyan_mask, 3] = 0
    
    # Isolate the exact 1-pixel boundary (the soft anti-aliased edge)
    alpha = data[:, :, 3]
    kernel = np.ones((3,3), np.uint8)
    transparent_mask = (alpha == 0).astype(np.uint8)
    dilated_transparent = cv2.dilate(transparent_mask, kernel, iterations=1)
    boundary_mask = (alpha > 0) & (dilated_transparent == 1)
    
    # Hollywood Matting: Fade the boundary smoothly instead of deleting it
    # We despill the cyan by forcing green and blue to not exceed red
    g_flat, b_flat, r_flat = g[boundary_mask], b[boundary_mask], r[boundary_mask]
    
    g_flat = np.minimum(g_flat, r_flat)
    b_flat = np.minimum(b_flat, r_flat)
    
    g[boundary_mask] = g_flat
    b[boundary_mask] = b_flat
    
    data[:, :, 1] = g
    data[:, :, 2] = b
    
    # Apply a tiny Alpha Blur ONLY to the boundary to ensure it is buttery smooth
    blurred_alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    data[boundary_mask, 3] = blurred_alpha[boundary_mask]
    
    # ---------------------------------------------------------
    # PHASE 2: MIXEL SNAP & PALETTE (No eye warping)
    # ---------------------------------------------------------
    processed_img = Image.fromarray(data.astype(np.uint8))
    
    # Shift grid to align with AI's internal grid
    shifted_img = ImageChops.offset(processed_img, offset, offset)
    
    # Mixel Snap
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    
    # Revert Shift
    final_img = ImageChops.offset(snapped_img, -offset, -offset)
    
    # Palette Quantization
    final_data = np.array(final_img)
    final_alpha = final_data[:, :, 3]
    mask = final_alpha > 0
    visible_pixels = final_data[mask][:, :3] 
    
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(visible_pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    final_data[mask, :3] = quantized_pixels
    
    # Save the ultimate masterpiece
    final_pil = Image.fromarray(final_data)
    final_pil.save(output_path, "PNG")
    print(f"Saved Flawless Asset to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    execute_flawless_pipeline(
        "../results/batch_test/hands_test/hands_seed_6.png", 
        "../results/final_assets/cozy_girl_FLAWLESS_6.png"
    )
    execute_flawless_pipeline(
        "../results/batch_test/hands_test/hands_seed_7.png", 
        "../results/final_assets/cozy_girl_FLAWLESS_7.png"
    )
    print("\nFlawless Pipeline complete!")

if __name__ == "__main__":
    main()
