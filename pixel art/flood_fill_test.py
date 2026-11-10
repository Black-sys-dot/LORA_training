import cv2
import numpy as np
from PIL import Image
import os

def flood_fill_cleaner(input_path, output_path):
    print(f"Applying Flood Fill to {input_path}...")
    # Load image using cv2
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
    h, w = img.shape[:2]
    
    # Create a mask for floodFill (OpenCV requires it to be 2 pixels larger)
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Start the "water" slightly inward at (10,10) because the AI drew a 1-pixel dark border on the absolute edge!
    seed_pt = (10, 10)
    
    # Tolerance: How much the color is allowed to change before the water stops.
    # A low tolerance (like 15) means it stops the SECOND it hits anything that isn't perfectly cyan.
    tolerance = 15
    lo_diff = (tolerance, tolerance, tolerance) 
    up_diff = (tolerance, tolerance, tolerance) 
    
    # We only care about checking the RGB color channels to find the wall
    img_rgb = img[:,:,:3].copy()
    
    # Run the flood fill. We tell it to ONLY generate the mask, not actually color the image.
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
    cv2.floodFill(img_rgb, mask, seed_pt, (0,0,0), lo_diff, up_diff, flags)
    
    # The mask now contains exactly the area of the water. Crop it back to image size.
    bg_mask = mask[1:h+1, 1:w+1]
    
    # Delete the background based on the water mask
    img[bg_mask == 255, 3] = 0
    
    cv2.imwrite(output_path, img)
    print(f"Saved to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    flood_fill_cleaner(
        "../results/batch_test/hands_test/hands_seed_6.png", 
        "../results/final_assets/cozy_girl_floodfill_6.png"
    )

if __name__ == "__main__":
    main()
