import cv2
import numpy as np
from PIL import Image
import os

def apply_defringe(input_path, output_path):
    print(f"Applying Defringe to {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)
    
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # 1. PRIMARY CHROMA KEY (Only remove the 100% pure background)
    # We only delete pixels that are overwhelmingly cyan.
    cyan_mask = (g.astype(int) > r.astype(int) + 30) & (b.astype(int) > r.astype(int) + 30)
    data[cyan_mask, 3] = 0
    
    # 2. BOUNDARY MASKING
    # We only want to despill the outer boundary, not the whole image (otherwise her blue eyes turn gray!)
    alpha = data[:, :, 3]
    kernel = np.ones((3,3), np.uint8)
    transparent_mask = (alpha == 0).astype(np.uint8)
    dilated_transparent = cv2.dilate(transparent_mask, kernel, iterations=1)
    boundary_mask = (alpha > 0) & (dilated_transparent == 1)
    
    # 3. VFX DESPILL / DEFRINGE (The Magic Step)
    # Instead of deleting the semi-transparent anti-aliased edge pixels (which causes the cursed spiky look),
    # we just mathematically suck the cyan color out of them!
    
    # Cap Green to not exceed Red on the boundary
    excess_green = (data[:,:,1] > data[:,:,0]) & boundary_mask
    data[excess_green, 1] = data[excess_green, 0]
    
    # Cap Blue to not exceed Red on the boundary
    excess_blue = (data[:,:,2] > data[:,:,0]) & boundary_mask
    data[excess_blue, 2] = data[excess_blue, 0]
    
    # Save the result
    final_pil = Image.fromarray(data)
    final_pil.save(output_path, "PNG")
    print(f"Saved to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    
    apply_defringe(
        "../results/batch_test/hands_test/hands_seed_6.png", 
        "../results/final_assets/cozy_girl_defringed_6.png"
    )
    
    apply_defringe(
        "../results/batch_test/hands_test/hands_seed_7.png", 
        "../results/final_assets/cozy_girl_defringed_7.png"
    )
    print("\nDefringe complete!")

if __name__ == "__main__":
    main()
