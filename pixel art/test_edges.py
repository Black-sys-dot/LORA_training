import cv2
import numpy as np
from PIL import Image
import os

def apply_feathering(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)
    
    # 1. Extract Alpha Channel
    alpha = data[:, :, 3]
    
    # 2. Apply a slight Gaussian Blur to the Alpha to recreate anti-aliasing
    blurred_alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    
    # We only want to soften the jagged edges, not make the whole image transparent,
    # so we'll combine the original solid core with the blurred edge.
    data[:, :, 3] = np.where(alpha == 255, 255, blurred_alpha)
    
    Image.fromarray(data).save(output_path, "PNG")
    print(f"Saved Feathered version to {output_path}")

def apply_sticker_outline(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)
    
    # 1. Extract Alpha
    alpha = data[:, :, 3]
    
    # 2. Dilate the alpha channel to create a 2-pixel thick border outside the character
    kernel = np.ones((5, 5), np.uint8)
    dilated_alpha = cv2.dilate(alpha, kernel, iterations=1)
    
    # 3. The outline area is where the dilated alpha is solid, but the original alpha is empty
    outline_mask = (dilated_alpha > 0) & (alpha == 0)
    
    # 4. Color the outline area solid dark gray/black (retro outline color)
    data[outline_mask, 0] = 30  # R
    data[outline_mask, 1] = 30  # G
    data[outline_mask, 2] = 30  # B
    data[outline_mask, 3] = 255 # A
    
    # 5. We also want to smooth the outermost edge of the new sticker outline so IT isn't jagged
    new_alpha = data[:, :, 3]
    smooth_new_alpha = cv2.GaussianBlur(new_alpha, (3, 3), 0)
    data[:, :, 3] = smooth_new_alpha
    
    Image.fromarray(data).save(output_path, "PNG")
    print(f"Saved Sticker Outline version to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    
    input_img = "../results/final_assets/cozy_girl_boundary_cleaned_6.png"
    
    # Test 1: Feathering (Smooth Edge)
    apply_feathering(input_img, "../results/final_assets/cozy_girl_feathered_6.png")
    
    # Test 2: Sticker Outline (Retro Border)
    apply_sticker_outline(input_img, "../results/final_assets/cozy_girl_sticker_6.png")

if __name__ == "__main__":
    main()
