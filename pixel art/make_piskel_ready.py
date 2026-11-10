from PIL import Image

def make_piskel_ready():
    img_path = "/home/black/Projects/LORA_training/results/final_assets/blue_girl_seed_7_INCLUDED.png"
    out_path = "/home/black/Projects/LORA_training/results/final_assets/blue_girl_seed_7_INCLUDED_256.png"
    
    img = Image.open(img_path)
    
    # We downscale it back to its true native 256x256 resolution for easy editing
    img_256 = img.resize((256, 256), Image.Resampling.NEAREST)
    img_256.save(out_path)
    print("Created 256x256 version for Piskel!")

if __name__ == "__main__":
    make_piskel_ready()
