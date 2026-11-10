import modal
import io
import os

app = modal.App("flux-lora-greenscreen")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "peft",
        "Pillow",
        "numpy"
    )
)

volume = modal.Volume.from_name("flux-lora-volume")

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1200, 
)
def generate_and_greenscreen() -> dict:
    from diffusers import FluxPipeline
    import torch
    import numpy as np
    from PIL import Image
    
    print("Loading FLUX.1-dev base model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("Loading Pixel Art LoRA (Step 3000)...")
    pipe.load_lora_weights("/data/output/pixel_art_lora/pixel_art_lora_000003000.safetensors", adapter_name="pixel_art")
    pipe.fuse_lora(lora_scale=0.85)
    
    # 1. Generate the isolated character on NEON GREEN
    char_prompt = "beautiful pixel art illustration of a cute anime girl with pink hair, wearing a simple casual dress, full body, isolated on a pure bright neon green solid flat background, highly detailed retro-modern cel-shaded pixel art, crisp 1-pixel aliased lines, masterpiece"
    print("Generating isolated character on green...")
    char_img = pipe(
        prompt=char_prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=30,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(4096)
    ).images[0]
    
    # 2. Generate the empty background scenery
    bg_prompt = "beautiful pixel art scenery of a glowing neon cyberpunk city rooftop, night sky, empty scene, no characters, no people, highly detailed retro-modern cel-shaded pixel art, crisp 1-pixel aliased lines, masterpiece"
    print("Generating empty background...")
    bg_img = pipe(
        prompt=bg_prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=30,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(2048)
    ).images[0]
    
    # 3. Greenscreen Removal using Numpy
    print("Running Chroma-Key (Greenscreen) removal...")
    char_rgba = char_img.convert("RGBA")
    data = np.array(char_rgba)
    
    # Extract color channels
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # The AI might not output PERFECT #00FF00 due to VAE compression, so we use a range
    # A pixel is "green background" if Green is high, and Red/Blue are relatively low
    green_mask = (g > 150) & (r < 100) & (b < 100)
    
    # Set alpha to 0 for all background pixels
    data[green_mask, 3] = 0
    
    # Create the transparent image
    char_transparent = Image.fromarray(data)
    
    # 4. Smart Positioning!
    # Instead of leaving her floating, we will find where she is and snap her to the bottom of the screen.
    # Find bounding box of non-transparent pixels
    non_empty_columns = np.where(green_mask.min(axis=0) == False)[0]
    non_empty_rows = np.where(green_mask.min(axis=1) == False)[0]
    
    if len(non_empty_columns) > 0 and len(non_empty_rows) > 0:
        crop_box = (min(non_empty_columns), min(non_empty_rows), max(non_empty_columns), max(non_empty_rows))
        cropped_char = char_transparent.crop(crop_box)
        
        char_w, char_h = cropped_char.size
        # Center horizontally, stick to the absolute bottom vertically
        paste_x = (1024 - char_w) // 2
        paste_y = 1024 - char_h
    else:
        # Fallback if masking fails entirely
        cropped_char = char_transparent
        paste_x, paste_y = 0, 0
    
    print("Compositing grounded character onto background...")
    bg_img_rgba = bg_img.convert("RGBA")
    bg_img_rgba.paste(cropped_char, (paste_x, paste_y), cropped_char)
    final_composite = bg_img_rgba.convert("RGB")
    
    # Save results
    results = {}
    
    char_bytes = io.BytesIO()
    char_img.save(char_bytes, format="PNG")
    results["1_character_green.png"] = char_bytes.getvalue()
    
    bg_bytes = io.BytesIO()
    bg_img.save(bg_bytes, format="PNG")
    results["2_background_raw.png"] = bg_bytes.getvalue()
    
    final_bytes = io.BytesIO()
    final_composite.save(final_bytes, format="PNG")
    results["3_final_greenscreen_composite.png"] = final_bytes.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    os.makedirs("../results/greenscreen", exist_ok=True)
    
    print("Starting generation and Greenscreen compositing on Modal...")
    
    images_dict = generate_and_greenscreen.remote()
    
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/greenscreen/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"Saved {filepath}")
        
    print("Done! Check the 'results/greenscreen' folder.")
