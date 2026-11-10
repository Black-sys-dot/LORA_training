import numpy as np
from PIL import Image

def analyze_pixel_size(img_path):
    img = Image.open(img_path).convert("RGB")
    data = np.array(img)
    
    print(f"Image shape: {data.shape}")
    
    # Check the top-left of the jacket to measure block size
    # Just print a small 20x20 patch
    print("Sample patch of green channel (20x20):")
    for r in range(400, 420):
        row_str = ""
        for c in range(400, 420):
            # Format to 3 chars for alignment
            row_str += f"{data[r, c, 1]:3d} "
        print(row_str)

if __name__ == "__main__":
    analyze_pixel_size("/home/black/Downloads/blue_girl_seed_7.png")
