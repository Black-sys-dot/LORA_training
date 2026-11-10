import cv2
import numpy as np
from PIL import Image
import os

def apply_ultimate_key(input_path, output_path):
    print(f"Applying Ultimate Key to {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img).astype(np.float32)
    
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # 1. DELETE THE SOLID BACKGROUND
    # This safely deletes the massive blocks of solid cyan.
    cyan_mask = (g > r + 30) & (b > r + 30)
    data[cyan_mask, 3] = 0
    
    # 2. ISOLATE THE BOUNDARY
    # We find the exact 1-pixel jagged edge where the background was deleted.
    alpha = data[:, :, 3]
    kernel = np.ones((3,3), np.uint8)
    transparent_mask = (alpha == 0).astype(np.uint8)
    dilated_transparent = cv2.dilate(transparent_mask, kernel, iterations=1)
    boundary_mask = (alpha > 0) & (dilated_transparent == 1)
    
    # 3. HOLLYWOOD MATTING (Only on the boundary!)
    # Instead of deleting these edge pixels (which causes the cursed jagged holes),
    # we calculate EXACTLY how much cyan is mixed into them.
    # Cyan-ness = min(Green, Blue) - Red
    cyan_intensity = np.minimum(g, b) - r
    cyan_intensity = np.clip(cyan_intensity, 0, 255)
    
    # We turn that cyan intensity directly into transparency! 
    # (If a pixel is 50% cyan, it becomes 50% transparent, perfectly preserving the anti-aliasing)
    new_alpha = 255.0 - cyan_intensity
    
    # Apply the new calculated alpha ONLY to the boundary pixels
    alpha_flat = alpha[boundary_mask]
    new_alpha_flat = new_alpha[boundary_mask]
    
    # We only lower the alpha, never raise it
    alpha_flat = np.minimum(alpha_flat, new_alpha_flat)
    alpha[boundary_mask] = alpha_flat
    data[:, :, 3] = alpha
    
    # 4. DESPILL THE EDGE (Remove the cyan color)
    # Now that the edge has the perfect transparency, we suck the actual cyan color out of it
    # so it matches the black/gray outline.
    g_flat = g[boundary_mask]
    b_flat = b[boundary_mask]
    r_flat = r[boundary_mask]
    
    g_flat = np.minimum(g_flat, r_flat)
    b_flat = np.minimum(b_flat, r_flat)
    
    g[boundary_mask] = g_flat
    b[boundary_mask] = b_flat
    
    data[:, :, 1] = g
    data[:, :, 2] = b
    
    # Save the result
    final_pil = Image.fromarray(data.astype(np.uint8))
    final_pil.save(output_path, "PNG")
    print(f"Saved to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    apply_ultimate_key(
        "../results/batch_test/hands_test/hands_seed_6.png", 
        "../results/final_assets/cozy_girl_ULTIMATE_6.png"
    )
    apply_ultimate_key(
        "../results/batch_test/hands_test/hands_seed_7.png", 
        "../results/final_assets/cozy_girl_ULTIMATE_7.png"
    )
    print("\nUltimate Matting complete!")

if __name__ == "__main__":
    main()
