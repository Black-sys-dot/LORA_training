import cv2
import numpy as np
from PIL import Image
import os

def safe_remove_background(input_path, output_path):
    print(f"Processing manual mask from {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img).astype(np.float32)
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # 1. DETECT THE YELLOW LINE
    yellow_mask = (r > 200) & (g > 200) & (b < 50)
    
    # 2. DETECT THE PURE CYAN BACKGROUND
    cyan_mask = (g > 240) & (b > 240) & (r < 150)
    
    # 3. CREATE A SMART FLOOD FILL MASK
    # The water is ONLY allowed to flow through pure cyan.
    # The "walls" are the yellow line AND the character (anything not pure cyan).
    h, w = yellow_mask.shape
    ff_mask = np.ones((h+2, w+2), np.uint8) 
    
    # Open up paths where it is pure cyan, EXCEPT where the yellow line is.
    path_mask = cyan_mask & (~yellow_mask)
    ff_mask[1:h+1, 1:w+1] = (~path_mask).astype(np.uint8)
    
    flood_canvas = np.zeros((h, w), np.uint8)
    
    # Find a cyan pixel on the far left to start the left flood
    left_start = None
    for x in range(w):
        cyan_y_indices = np.where(cyan_mask[:, x])[0]
        if len(cyan_y_indices) > 0:
            left_start = (x, cyan_y_indices[0])
            break
            
    # Find a cyan pixel on the far right to start the right flood
    right_start = None
    for x in range(w-1, -1, -1):
        cyan_y_indices = np.where(cyan_mask[:, x])[0]
        if len(cyan_y_indices) > 0:
            right_start = (x, cyan_y_indices[0])
            break
            
    print(f"Flooding from Left: {left_start}, Right: {right_start}")
    
    if left_start:
        cv2.floodFill(flood_canvas, ff_mask, left_start, 255)
    if right_start:
        cv2.floodFill(flood_canvas, ff_mask, right_start, 255)
        
    # 4. DELETE THE FLOODED BACKGROUND AND THE YELLOW LINE
    data[flood_canvas == 255, 3] = 0
    data[yellow_mask, 3] = 0
    
    # Delete pure white/transparent dead space on top left from the manual shift
    dead_space = (r > 250) & (g > 250) & (b > 250) & (a == 0)
    data[dead_space, 3] = 0
    
    # 5. DESPILL THE EDGE (Neutralize bluish-pink halos remaining inside the mask)
    alpha = data[:, :, 3]
    kernel = np.ones((3,3), np.uint8)
    transparent_mask = (alpha == 0).astype(np.uint8)
    dilated_transparent = cv2.dilate(transparent_mask, kernel, iterations=1)
    boundary_mask = (alpha > 0) & (dilated_transparent == 1)
    
    g_flat, b_flat, r_flat = g[boundary_mask], b[boundary_mask], r[boundary_mask]
    g_flat = np.minimum(g_flat, r_flat)
    b_flat = np.minimum(b_flat, r_flat)
    g[boundary_mask] = g_flat
    b[boundary_mask] = b_flat
    data[:, :, 1] = g
    data[:, :, 2] = b
    
    # Save the result
    final_pil = Image.fromarray(data.astype(np.uint8))
    final_pil.save(output_path, "PNG")
    print(f"Saved background-removed asset to {output_path}")

def main():
    os.makedirs("/home/black/Projects/LORA_training/results/final_assets", exist_ok=True)
    safe_remove_background(
        "/home/black/Downloads/cozy_girl_seed_6(2).png", 
        "/home/black/Projects/LORA_training/results/final_assets/cozy_girl_hybrid_safe.png"
    )

if __name__ == "__main__":
    main()
