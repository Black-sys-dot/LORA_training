import cv2
import numpy as np
from PIL import Image
import os

def clean_residue(input_path, output_path):
    print(f"Cleaning disconnected residue from {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)
    
    alpha = data[:, :, 3]
    
    # We find all connected groups of visible pixels
    visible_mask = (alpha > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(visible_mask, connectivity=8)
    
    # We want to keep the background (label 0) and the LARGEST component (the character).
    # All other smaller components are the floating residue!
    
    # Find the largest component (excluding the background, which is label 0)
    largest_label = 1
    max_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            largest_label = i
            
    print(f"Main character component found with {max_area} pixels.")
    
    # Create a mask that ONLY includes the main character
    main_char_mask = (labels == largest_label)
    
    # Delete everything that is NOT the main character (all the floating residue)
    data[~main_char_mask, 3] = 0
    
    # Save the cleaned result
    final_pil = Image.fromarray(data)
    final_pil.save(output_path, "PNG")
    print(f"Cleaned asset saved to {output_path}")

def main():
    os.makedirs("/home/black/Projects/LORA_training/results/final_assets", exist_ok=True)
    clean_residue(
        "/home/black/Projects/LORA_training/results/final_assets/cozy_girl_hybrid_safe.png", 
        "/home/black/Projects/LORA_training/results/final_assets/cozy_girl_hybrid_clean.png"
    )

if __name__ == "__main__":
    main()
