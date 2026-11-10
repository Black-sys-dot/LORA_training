import cv2
import numpy as np
from PIL import Image
import os
from sklearn.cluster import MiniBatchKMeans

def apply_post_processing(input_path, output_path, colors=32):
    print(f"Applying Post-Processing to {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    
    # 1. MIXEL SNAP
    # Downscale to 256x256 using Nearest Neighbor, then back up to 1024x1024.
    # This mathematically forces all jagged edges and messy pixels into a strict 4x4 chunky retro grid.
    img = img.resize((256, 256), Image.Resampling.NEAREST)
    img = img.resize((1024, 1024), Image.Resampling.NEAREST)
    
    data = np.array(img)
    alpha = data[:, :, 3]
    
    # 2. PALETTE QUANTIZATION (K-Means)
    # We only want to cluster the visible pixels, not the transparent background
    mask = alpha > 0
    visible_pixels = data[mask][:, :3]  # Get RGB of visible pixels
    
    # Use k=32 to preserve eye color and highlights while still flattening digital gradients
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(visible_pixels)
    
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    # Put the quantized colors back into the image
    data[mask, :3] = quantized_pixels
    
    # Save the final masterpiece
    final_pil = Image.fromarray(data)
    final_pil.save(output_path, "PNG")
    print(f"Saved FINAL asset to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    apply_post_processing(
        "../results/final_assets/cozy_girl_ULTIMATE_6.png", 
        "../results/final_assets/cozy_girl_FINAL_6.png",
        colors=32
    )
    apply_post_processing(
        "../results/final_assets/cozy_girl_ULTIMATE_7.png", 
        "../results/final_assets/cozy_girl_FINAL_7.png",
        colors=32
    )
    print("\nFull pipeline complete!")

if __name__ == "__main__":
    main()
