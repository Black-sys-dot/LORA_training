import cv2
import numpy as np
from PIL import Image
import os

def apply_edge_smoothing(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)
    
    alpha = data[:, :, 3]
    
    # Isolate the exact outer edge of the image
    kernel = np.ones((3,3), np.uint8)
    transparent_mask = (alpha == 0).astype(np.uint8)
    dilated_transparent = cv2.dilate(transparent_mask, kernel, iterations=1)
    boundary_mask = (alpha > 0) & (dilated_transparent == 1)
    
    # Create a slightly blurred version of the alpha channel
    blurred_alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    
    # We only apply the smooth blurred alpha to the sharp jagged boundary pixels
    data[boundary_mask, 3] = blurred_alpha[boundary_mask]
    
    final_pil = Image.fromarray(data)
    final_pil.save(output_path, "PNG")
    print(f"Saved smooth edge asset to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    apply_edge_smoothing(
        "../results/final_assets/cozy_girl_ULTIMATE_6.png", 
        "../results/final_assets/cozy_girl_SMOOTH_6.png"
    )
    apply_edge_smoothing(
        "../results/final_assets/cozy_girl_ULTIMATE_7.png", 
        "../results/final_assets/cozy_girl_SMOOTH_7.png"
    )
    print("\nEdge Smoothing complete!")

if __name__ == "__main__":
    main()
