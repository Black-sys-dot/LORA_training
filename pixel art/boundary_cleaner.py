import cv2
import numpy as np
from PIL import Image
import os

def clean_boundary(input_path, output_path):
    print(f"Processing {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)
    
    # 1. PRIMARY CHROMA KEY (Remove the obvious cyan)
    r = data[:,:,0].astype(int)
    g = data[:,:,1].astype(int)
    b = data[:,:,2].astype(int)
    
    cyan_mask = (g > r + 30) & (b > r + 30)
    data[cyan_mask, 3] = 0
    
    # 2. BOUNDARY CLEANING
    # Find the boundaries (pixels that are non-transparent but touch a transparent pixel)
    alpha = data[:, :, 3]
    kernel = np.ones((3,3), np.uint8)
    
    # Dilate the transparent area to find the border
    transparent_mask = (alpha == 0).astype(np.uint8)
    dilated_transparent = cv2.dilate(transparent_mask, kernel, iterations=1)
    
    # The boundary is where it is currently NON-transparent, but touches the dilated transparent area
    boundary_mask = (alpha > 0) & (dilated_transparent == 1)
    
    # For any pixel on this boundary, if it has a cyan/blue tint (b > r and g > r), delete it!
    # Because she has zero blue/cyan on her, any blue/cyan on the boundary is leftover background residue.
    residual_cyan = (b[boundary_mask] > r[boundary_mask]) & (g[boundary_mask] > r[boundary_mask])
    
    # Apply the deletion
    alpha_flat = alpha[boundary_mask]
    alpha_flat[residual_cyan] = 0
    alpha[boundary_mask] = alpha_flat
    
    data[:, :, 3] = alpha
    
    # SAVE (No mixel snap, no palette quantization)
    final_pil = Image.fromarray(data)
    final_pil.save(output_path, "PNG")
    print(f"Saved to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    
    clean_boundary(
        "../results/batch_test/hands_test/hands_seed_6.png", 
        "../results/final_assets/cozy_girl_boundary_cleaned_6.png"
    )
    
    clean_boundary(
        "../results/batch_test/hands_test/hands_seed_7.png", 
        "../results/final_assets/cozy_girl_boundary_cleaned_7.png"
    )
    print("\nBoundary cleaning complete!")

if __name__ == "__main__":
    main()
