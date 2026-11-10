from PIL import Image

# Load the mixel-fixed image (it's in RGB mode)
img_path = "../results/smart_greenscreen/5_final_grid_snapped.png"
img = Image.open(img_path).convert("RGB")

# ---------------------------------------------------------
# Test 1: 32 Colors (No Dithering)
# ---------------------------------------------------------
# colors=32: strictly limit the entire image to 32 dominant colors
# dither=0: forces the math to use flat color blocks instead of mixing pixels
quantized_32 = img.quantize(colors=32, method=2, kmeans=3, dither=0)
quantized_32.save("../results/smart_greenscreen/6_final_quantized_32.png")

# ---------------------------------------------------------
# Test 2: 16 Colors (No Dithering)
# ---------------------------------------------------------
# Extremely strict retro limitation
quantized_16 = img.quantize(colors=16, method=2, kmeans=3, dither=0)
quantized_16.save("../results/smart_greenscreen/6_final_quantized_16.png")

print("Color quantization complete! Saved 32-color and 16-color versions.")
