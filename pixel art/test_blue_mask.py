import numpy as np
from PIL import Image
import os

def test_mask(input_path, output_path):
    img = Image.open(input_path).convert("RGB")
    data = np.array(img)
    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
    
    # Create the strict cyan mask
    bg_mask = (g > 240) & (b > 240) & (r < 150)
    
    # Visualize it: Background is white, character is black
    mask_img = np.zeros_like(data)
    mask_img[bg_mask] = [255, 255, 255]
    
    Image.fromarray(mask_img).save(output_path)
    print("Mask saved!")

if __name__ == "__main__":
    os.makedirs("/home/black/Projects/LORA_training/results/final_assets", exist_ok=True)
    test_mask(
        "/home/black/Downloads/blue_girl_seed_7.png", 
        "/home/black/Projects/LORA_training/results/final_assets/blue_girl_mask_test.png"
    )
