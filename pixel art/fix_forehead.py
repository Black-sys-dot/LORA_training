import cv2
import numpy as np
from PIL import Image

# 1. Load the original Seed 7 masterpiece
img_path = "../results/batch_test/blue_girl_seed_7.png"
img = cv2.imread(img_path)

# 2. Precisely target the rogue black line
# Based on the grid, the black line is located between X=465 and X=480, and Y=150 and Y=200.
# We stop at Y=200 to ensure we don't accidentally erase her beautiful eyelashes!
target_region = img[150:200, 465:480]

# 3. Find the black pixels in this specific region (BGR format in cv2)
black_mask = (target_region[:, :, 0] < 50) & (target_region[:, :, 1] < 50) & (target_region[:, :, 2] < 50)

# 4. Sample the exact skin color from her forehead (just to the left of the line)
skin_color = img[180, 460]

# 5. Overwrite only the rogue black pixels with the skin color
target_region[black_mask] = skin_color
img[150:200, 465:480] = target_region

# Save the fixed version
fixed_path = "../results/batch_test/blue_girl_seed_7_fixed.png"
cv2.imwrite(fixed_path, img)
print(f"Surgical precision complete! Fixed image saved to {fixed_path}")
