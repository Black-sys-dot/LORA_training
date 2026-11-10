import cv2
import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def flawless_no_despill(input_path, output_path, offset=2, colors=32):
    print(f"Executing raw color pipeline on {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img).astype(np.float32)
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    yellow_mask = (r > 200) & (g > 200) & (b < 50)
    cyan_mask = (g > 240) & (b > 240) & (r < 150)
    
    h, w = yellow_mask.shape
    ff_mask = np.ones((h+2, w+2), np.uint8) 
    path_mask = cyan_mask & (~yellow_mask)
    ff_mask[1:h+1, 1:w+1] = (~path_mask).astype(np.uint8)
    
    flood_canvas = np.zeros((h, w), np.uint8)
    
    left_start, right_start = None, None
    for x in range(w):
        if len(np.where(cyan_mask[:, x])[0]) > 0:
            left_start = (x, np.where(cyan_mask[:, x])[0][0])
            break
    for x in range(w-1, -1, -1):
        if len(np.where(cyan_mask[:, x])[0]) > 0:
            right_start = (x, np.where(cyan_mask[:, x])[0][0])
            break
            
    if left_start: cv2.floodFill(flood_canvas, ff_mask, left_start, 255)
    if right_start: cv2.floodFill(flood_canvas, ff_mask, right_start, 255)
        
    data[flood_canvas == 255, 3] = 0
    data[yellow_mask, 3] = 0
    
    dead_space = (r > 250) & (g > 250) & (b > 250) & (a == 0)
    data[dead_space, 3] = 0
    
    # --- NO DESPILL! We let K-Means handle the raw raw colors ---
    
    # Clean residue using Connected Components
    visible_mask = (data[:, :, 3] > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(visible_mask, connectivity=8)
    
    largest_label = 1
    max_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            largest_label = i
            
    main_char_mask = (labels == largest_label)
    data[~main_char_mask, 3] = 0
    
    # Final Pixel Pass (Offset + Mixel + KMeans)
    processed_img = Image.fromarray(data.astype(np.uint8))
    shifted_img = ImageChops.offset(processed_img, offset, offset)
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    final_img = ImageChops.offset(snapped_img, -offset, -offset)
    
    final_data = np.array(final_img)
    final_alpha = final_data[:, :, 3]
    mask = final_alpha > 0
    visible_pixels = final_data[mask][:, :3] 
    
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(visible_pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    final_data[mask, :3] = quantized_pixels
    
    final_pil = Image.fromarray(final_data)
    final_pil.save(output_path, "PNG")
    print(f"Vibrant asset saved to {output_path}")

def main():
    os.makedirs("/home/black/Projects/LORA_training/results/final_assets", exist_ok=True)
    flawless_no_despill(
        "/home/black/Downloads/cozy_girl_seed_6(2).png", 
        "/home/black/Projects/LORA_training/results/final_assets/cozy_girl_seed_6_VIBRANT.png"
    )

if __name__ == "__main__":
    main()
