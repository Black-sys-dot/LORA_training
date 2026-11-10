import cv2
import numpy as np
from PIL import Image
import os

def apply_ultimate_key_v2(input_path, output_path):
    print(f"Applying Ultimate Key V2 to {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img).astype(np.float32)
    
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # 1. DELETE ONLY THE PUREST BACKGROUND
    # The original script deleted ANY pixel that had more green/blue than red. 
    # This was a massive mistake, because it aggressively ate the dark anti-aliasing pixels 
    # (which are a mix of black outline and cyan background), causing the remaining sharp edges!
    # NOW, we ONLY delete 100% pure cyan pixels. 
    cyan_mask = (g > 240) & (b > 240) & (r < 20)
    data[cyan_mask, 3] = 0
    
    # 2. ISOLATE THE BOUNDARY (Which now safely includes ALL the soft anti-aliasing!)
    alpha = data[:, :, 3]
    kernel = np.ones((5,5), np.uint8) # Slightly wider boundary detection
    transparent_mask = (alpha == 0).astype(np.uint8)
    dilated_transparent = cv2.dilate(transparent_mask, kernel, iterations=1)
    boundary_mask = (alpha > 0) & (dilated_transparent == 1)
    
    # 3. HOLLYWOOD MATTING 
    cyan_intensity = np.minimum(g, b) - r
    cyan_intensity = np.clip(cyan_intensity, 0, 255)
    
    # Calculate transparency
    new_alpha = 255.0 - cyan_intensity
    
    # Apply to boundary
    alpha_flat = alpha[boundary_mask]
    new_alpha_flat = new_alpha[boundary_mask]
    alpha_flat = np.minimum(alpha_flat, new_alpha_flat)
    alpha[boundary_mask] = alpha_flat
    data[:, :, 3] = alpha
    
    # 4. DESPILL
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
    apply_ultimate_key_v2(
        "../results/batch_test/hands_test/hands_seed_6.png", 
        "../results/final_assets/cozy_girl_ULTIMATE_V2_6.png"
    )
    apply_ultimate_key_v2(
        "../results/batch_test/hands_test/hands_seed_7.png", 
        "../results/final_assets/cozy_girl_ULTIMATE_V2_7.png"
    )
    print("\nUltimate Matting V2 complete!")

if __name__ == "__main__":
    main()
