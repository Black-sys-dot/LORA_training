import cv2
import numpy as np
from PIL import Image
import os

def simple_bg_remove(input_path, output_path):
    print(f"Removing background from {input_path}...")
    
    # 1. LOAD IMAGE
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)
    
    # 2. CHROMA KEY (Exact same logic used for the Pink Hair Girl)
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    cyan_mask = (g.astype(int) > r.astype(int) + 30) & (b.astype(int) > r.astype(int) + 30)
    data[cyan_mask, 3] = 0
    
    # Save EXACTLY in the same folder, doing nothing else.
    final_pil = Image.fromarray(data)
    final_pil.save(output_path, "PNG")
    print(f"Saved to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    
    # Process God Seed 6
    simple_bg_remove(
        "../results/batch_test/hands_test/hands_seed_6.png", 
        "../results/final_assets/cozy_girl_final_seed_6.png"
    )
    
    # Process God Seed 7
    simple_bg_remove(
        "../results/batch_test/hands_test/hands_seed_7.png", 
        "../results/final_assets/cozy_girl_final_seed_7.png"
    )

if __name__ == "__main__":
    main()
