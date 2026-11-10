import numpy as np
from PIL import Image
import os

def apply_perfect_key(input_path, output_path):
    print(f"Applying Advanced Chroma Key to {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img).astype(np.float32)
    
    R, G, B, A = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # 1. Calculate how "Cyan" each pixel is. 
    # Cyan is high Green and Blue, low Red. 
    # So Cyan-ness = min(Green, Blue) - Red
    cyan_weight = np.minimum(G, B) - R
    
    # Clamp the weight between 0 and 255
    cyan_weight = np.clip(cyan_weight, 0, 255)
    
    # 2. Extract the new Alpha channel!
    # If a pixel is 100% cyan, it becomes 0 alpha (transparent).
    # If a pixel is 0% cyan, it stays 255 alpha (opaque).
    # If a pixel is 50% cyan (like the blended chin edge!), it becomes 127 alpha!
    new_alpha = 255.0 - cyan_weight
    data[:, :, 3] = new_alpha
    
    # 3. Despill the Cyan tint from the remaining semi-transparent edge pixels
    # We cap the Green and Blue channels to not exceed the Red channel.
    data[:, :, 1] = np.minimum(G, R)
    data[:, :, 2] = np.minimum(B, R)
    
    # Save the result
    final_pil = Image.fromarray(data.astype(np.uint8))
    final_pil.save(output_path, "PNG")
    print(f"Saved perfectly keyed asset to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    apply_perfect_key(
        "../results/batch_test/hands_test/hands_seed_6.png", 
        "../results/final_assets/cozy_girl_perfect_key_6.png"
    )
    apply_perfect_key(
        "../results/batch_test/hands_test/hands_seed_7.png", 
        "../results/final_assets/cozy_girl_perfect_key_7.png"
    )
    print("\nAdvanced Matting complete!")

if __name__ == "__main__":
    main()
