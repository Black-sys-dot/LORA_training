from PIL import Image

# 1. Load the final eroded composite
img_path = "../results/smart_greenscreen/4_final_eroded_composite.png"
img = Image.open(img_path)

# 2. Downscale to 256x256 using strict Nearest Neighbor
# Since the original is 1024x1024, dividing by 4 gives 256. 
# This forces every chaotic, random-sized AI block to snap perfectly into a true 1-pixel grid.
downscaled = img.resize((256, 256), Image.Resampling.NEAREST)

# 3. Upscale back to 1024x1024 using Nearest Neighbor
# This blows those true 1-pixels back up into mathematically perfect 4x4 pixel blocks.
# Mixels are completely destroyed.
upscaled = downscaled.resize((1024, 1024), Image.Resampling.NEAREST)

# 4. Save
save_path = "../results/smart_greenscreen/5_final_grid_snapped.png"
upscaled.save(save_path, "PNG")
print(f"Mixels destroyed! Saved {save_path}")
