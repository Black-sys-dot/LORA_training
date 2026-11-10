import cv2
import numpy as np
from PIL import Image
import os

def apply_mixel_snap(input_path, output_path):
    print(f"Applying Mixel Snap to {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    
    # MIXEL SNAP
    # 1024 / 4 = 256. 
    # Downscaling to 256 using Nearest Neighbor forces the raw, high-res jagged edges 
    # into a strict 4x4 mathematical grid, making them look like authentic pixel blocks.
    img = img.resize((256, 256), Image.Resampling.NEAREST)
    img = img.resize((1024, 1024), Image.Resampling.NEAREST)
    
    # Save the result
    img.save(output_path, "PNG")
    print(f"Saved to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    
    apply_mixel_snap(
        "../results/final_assets/cozy_girl_boundary_cleaned_6.png", 
        "../results/final_assets/cozy_girl_mixel_snapped_6.png"
    )
    
    apply_mixel_snap(
        "../results/final_assets/cozy_girl_boundary_cleaned_7.png", 
        "../results/final_assets/cozy_girl_mixel_snapped_7.png"
    )
    print("\nMixel Snap complete!")

if __name__ == "__main__":
    main()
