import cv2
import numpy as np
from PIL import Image
import os

# Load the raw character image generated on Cyan
char_img = Image.open("../results/smart_greenscreen/1_character_cyan.png").convert("RGBA")
# Load the raw background (from the first greenscreen test, same seed!)
bg_img = Image.open("../results/greenscreen/2_background_raw.png").convert("RGBA")

# Convert character to numpy for processing
data = np.array(char_img)
r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]

# 1. Smart Cyan Chroma-Key
cyan_mask = (g.astype(int) > r.astype(int) + 30) & (b.astype(int) > r.astype(int) + 30)
data[cyan_mask, 3] = 0

# 2. EROSION (Shave off the edges)
alpha = data[:, :, 3]
# The user wants to remove the outer layer where the black outline overlapped.
# An erosion of 1 pixel trims the 0.001% fringe.
# An erosion of 2 pixels ensures we cut deeply enough to remove ANY edge blending.
kernel = np.ones((3,3), np.uint8)
eroded_alpha = cv2.erode(alpha, kernel, iterations=2) 
data[:, :, 3] = eroded_alpha

# Convert back to PIL
char_transparent = Image.fromarray(data)

# 3. Crop and Snap to Bottom
non_empty_columns = np.where(eroded_alpha > 0)[0]
non_empty_rows = np.where(eroded_alpha > 0)[1]

if len(non_empty_columns) > 0 and len(non_empty_rows) > 0:
    y_indices, x_indices = np.where(eroded_alpha > 0)
    crop_box = (min(x_indices), min(y_indices), max(x_indices), max(y_indices))
    cropped_char = char_transparent.crop(crop_box)
    char_w, char_h = cropped_char.size
    paste_x = (1024 - char_w) // 2
    paste_y = 1024 - char_h
else:
    cropped_char = char_transparent
    paste_x, paste_y = 0, 0

# 4. Composite
bg_img.paste(cropped_char, (paste_x, paste_y), cropped_char)
final_composite = bg_img.convert("RGB")

# Save
save_path = "../results/smart_greenscreen/4_final_eroded_composite.png"
final_composite.save(save_path, "PNG")
print(f"Saved {save_path}!")
