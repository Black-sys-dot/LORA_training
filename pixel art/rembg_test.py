from rembg import remove
from PIL import Image
import os

def apply_ai_rembg(input_path, output_path):
    print(f"Applying AI Background Removal to {input_path}...")
    
    input_image = Image.open(input_path)
    
    # Run the image through the U-2-Net AI model to cleanly slice out the foreground subject
    output_image = remove(input_image)
    
    output_image.save(output_path)
    print(f"Saved to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    apply_ai_rembg(
        "../results/batch_test/hands_test/hands_seed_6.png", 
        "../results/final_assets/cozy_girl_rembg_6.png"
    )

if __name__ == "__main__":
    main()
