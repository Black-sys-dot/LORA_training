import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def true_solid_pipeline(input_path, output_path, offset=2, colors=32):
    print(f"Applying True Solid Pipeline to {input_path}...")
    
    img = Image.open(input_path).convert("RGB")
    
    # 1. GRID PHASE OFFSET & MIXEL SNAP
    shifted_img = ImageChops.offset(img, offset, offset)
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    final_img = ImageChops.offset(snapped_img, -offset, -offset)
    
    data = np.array(final_img)
    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
    
    # 2. SEPARATE BACKGROUND FROM CHARACTER (AND EDGES)
    # The AI's true background is milky cyan (112, 254, 255).
    # We target ONLY the pure background, leaving the anti-aliased edges with the character.
    bg_mask = (g > 240) & (b > 240) & (r < 150)
    char_mask = ~bg_mask
    
    # 3. K-MEANS PALETTE (TRAINED ONLY ON THE CHARACTER)
    # We give all 32 colors exclusively to the character so we don't lose quality!
    char_pixels = data[char_mask]
    
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(char_pixels)
    quantized_char_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    # 4. APPLY THE PALETTE & DESPILL
    # Because the background is excluded from the palette, the anti-aliased edges
    # are forced to snap to the closest character color (pink/shadow), naturally despilling them!
    data[char_mask] = quantized_char_pixels
    
    # 5. FLATTEN THE BACKGROUND
    # We turn the noisy AI background into a perfectly flat, single retro color
    data[bg_mask] = [112, 254, 255]
    
    # Save the absolute final asset
    final_pil = Image.fromarray(data.astype(np.uint8))
    final_pil.save(output_path, "PNG")
    print(f"Flawless solid background asset saved to {output_path}")

def main():
    os.makedirs("/home/black/Projects/LORA_training/results/final_assets", exist_ok=True)
    true_solid_pipeline(
        "/home/black/Projects/LORA_training/results/batch_test/hands_test/hands_seed_6.png", 
        "/home/black/Projects/LORA_training/results/final_assets/cozy_girl_seed_6_PERFECT.png"
    )

if __name__ == "__main__":
    main()
