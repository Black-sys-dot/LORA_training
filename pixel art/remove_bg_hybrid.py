import cv2
import numpy as np
from PIL import Image
import os

def remove_background(input_path, output_path):
    print(f"Processing manual mask from {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img).astype(np.float32)
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # 1. DETECT THE YELLOW LINE
    yellow_mask = (r > 200) & (g > 200) & (b < 50)
    
    # 2. DETECT THE CYAN BACKGROUND TO FIND SAFE STARTING POINTS
    cyan_mask = (g > 240) & (b > 240) & (r < 150)
    
    # We need a mask for OpenCV floodFill (must be 2 pixels larger)
    h, w = yellow_mask.shape
    ff_mask = np.zeros((h+2, w+2), np.uint8)
    ff_mask[1:h+1, 1:w+1] = yellow_mask.astype(np.uint8)
    
    flood_canvas = np.zeros((h, w), np.uint8)
    
    # Find a cyan pixel on the far left to start the left flood
    # We scan columns from left to right until we hit cyan
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
    
    # Flood from left
    if left_start:
        cv2.floodFill(flood_canvas, ff_mask, left_start, 255)
        
    # Flood from right
    if right_start:
        cv2.floodFill(flood_canvas, ff_mask, right_start, 255)
        
    # 3. DELETE THE FLOODED BACKGROUND AND THE YELLOW LINE
    data[flood_canvas == 255, 3] = 0
    data[yellow_mask, 3] = 0
    
    # 4. DELETE ANY PURE WHITE/TRANSPARENT DEAD SPACE AT THE TOP (from shifting)
    # The shift created empty white space (255,255,255).
    # Since the yellow line protects her sweater, we can safely delete pure white
    # that is OUTSIDE the yellow line. Wait, the floodFill didn't hit it if it wasn't cyan!
    # Let's just delete anything that is completely outside the character's bounding box.
    # Actually, the user's shift might have left raw transparency or white space.
    # We will just flood fill from (0,0) as well just in case!
    if not yellow_mask[0,0]:
        cv2.floodFill(flood_canvas, ff_mask, (0,0), 255)
        data[flood_canvas == 255, 3] = 0
        
    # 5. DESPILL THE NEW BOUNDARY (Convert bluish-pink to shadow)
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
    remove_background(
        "/home/black/Downloads/cozy_girl_seed_6(2).png", 
        "/home/black/Projects/LORA_training/results/final_assets/cozy_girl_hybrid_cutout.png"
    )

if __name__ == "__main__":
    main()
