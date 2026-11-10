import cv2
import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def process_hybrid_asset(input_path, output_path, offset=2, colors=32):
    print(f"Processing human-guided asset: {input_path}")
    
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img).astype(np.float32)
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # 1. FIND THE YELLOW TRACE
    # Detect pure yellow pixels (with a tiny tolerance for anti-aliasing if they messed up)
    yellow_mask = (r > 240) & (g > 240) & (b < 20)
    
    # We create an OpenCV flood-fill mask. OpenCV requires it to be 2 pixels larger.
    h, w = yellow_mask.shape
    ff_mask = np.zeros((h+2, w+2), np.uint8)
    
    # The yellow pixels act as the "walls" for the flood fill
    ff_mask[1:h+1, 1:w+1] = yellow_mask.astype(np.uint8)
    
    # 2. FLOOD FILL THE OUTSIDE
    # We create a blank canvas to flood
    flood_canvas = np.zeros((h, w), np.uint8)
    
    # We flood fill from the top-left corner (0,0)
    # The 'water' will spread everywhere until it hits the yellow walls!
    cv2.floodFill(flood_canvas, ff_mask, (0,0), 255)
    
    # Now, flood_canvas is 255 everywhere OUTSIDE the character.
    # Delete the flooded background
    data[flood_canvas == 255, 3] = 0
    
    # 3. DELETE INNER HOLES
    # If the user drew yellow loops for inner holes, the flood fill from (0,0) couldn't reach them.
    # We find all closed yellow contours and fill them!
    yellow_uint8 = yellow_mask.astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(yellow_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            # We look for internal holes (contours that have a parent)
            if hierarchy[0][i][3] != -1:
                cv2.drawContours(data[:, :, 3], [cnt], -1, 0, thickness=cv2.FILLED)
                
    # 4. DELETE THE YELLOW LINE ITSELF
    data[yellow_mask, 3] = 0
    
    # 5. DESPILL THE EDGE (Neutralize remaining bluish-pink halos)
    # We find the new boundary pixels and cap G/B to R to turn them into dark pink/gray shadows
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
    
    # 6. MIXEL SNAP WITH GRID OFFSET
    processed_img = Image.fromarray(data.astype(np.uint8))
    shifted_img = ImageChops.offset(processed_img, offset, offset)
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    final_img = ImageChops.offset(snapped_img, -offset, -offset)
    
    # 7. K-MEANS PALETTE
    final_data = np.array(final_img)
    final_alpha = final_data[:, :, 3]
    mask = final_alpha > 0
    visible_pixels = final_data[mask][:, :3] 
    
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(visible_pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    final_data[mask, :3] = quantized_pixels
    
    # Save the masterpiece
    final_pil = Image.fromarray(final_data)
    final_pil.save(output_path, "PNG")
    print(f"Hybrid pipeline complete! Saved to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    # We will run this once the user provides the file!
    pass

if __name__ == "__main__":
    main()
