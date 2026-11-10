import modal
import io
import os

app = modal.App("flux-lora-smart-greenscreen-outline")

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
        "numpy",
        "opencv-python-headless"
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
def generate_and_outline() -> dict:
    from diffusers import FluxPipeline
    import torch
    import numpy as np
    import cv2
    from PIL import Image
    
    print("Loading FLUX.1-dev base model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("Loading Pixel Art LoRA (Step 3000)...")
    pipe.load_lora_weights("/data/output/pixel_art_lora/pixel_art_lora_000003000.safetensors", adapter_name="pixel_art")
    pipe.fuse_lora(lora_scale=0.85)
    
    char_prompt = "beautiful pixel art illustration of a cute anime girl with pink hair, wearing a simple casual white dress, full body, isolated on a pure bright neon cyan solid flat background, highly detailed retro-modern cel-shaded pixel art, crisp 1-pixel aliased lines, masterpiece"
    char_img = pipe(
        prompt=char_prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=30,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(5000)
    ).images[0]
    
    bg_prompt = "beautiful pixel art scenery of a glowing neon cyberpunk city rooftop, night sky, empty scene, no characters, no people, highly detailed retro-modern cel-shaded pixel art, crisp 1-pixel aliased lines, masterpiece"
    bg_img = pipe(
        prompt=bg_prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=30,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(2048)
    ).images[0]
    
    print("Running Smart Chroma-Key with Outline Logic...")
    char_rgba = char_img.convert("RGBA")
    data = np.array(char_rgba)
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # 1. Base Cyan Removal
    cyan_mask = (g.astype(int) > r.astype(int) + 30) & (b.astype(int) > r.astype(int) + 30)
    data[cyan_mask, 3] = 0
    
    # 2. Edge EROSION (Shave off the 1-pixel 0.001% Cyan fringe!)
    alpha = data[:, :, 3]
    kernel = np.ones((3,3), np.uint8)
    # Erode the alpha channel so the outermost edge of the character is deleted
    eroded_alpha = cv2.erode(alpha, kernel, iterations=1)
    data[:, :, 3] = eroded_alpha
    
    # 3. Create a thick Black Stroke Outline
    # Expand (dilate) the shaved alpha to create an outline footprint
    dilated_alpha = cv2.dilate(eroded_alpha, kernel, iterations=1)
    # The outline is the expanded area MINUS the eroded character body
    outline_mask = (dilated_alpha > 0) & (eroded_alpha == 0)
    
    # Color the outline pure black!
    data[outline_mask, 0] = 0   # R
    data[outline_mask, 1] = 0   # G
    data[outline_mask, 2] = 0   # B
    data[outline_mask, 3] = 255 # Alpha
    
    char_transparent = Image.fromarray(data)
    
    # Auto-crop and snap to bottom
    non_empty_columns = np.where(dilated_alpha > 0)[0]
    non_empty_rows = np.where(dilated_alpha > 0)[1] # Wait, axis is (y, x). rows is y, cols is x.
    # Safe fallback crop logic:
    if len(non_empty_columns) > 0 and len(non_empty_rows) > 0:
        # np.where returns (y_indices, x_indices)
        y_indices, x_indices = np.where(dilated_alpha > 0)
        crop_box = (min(x_indices), min(y_indices), max(x_indices), max(y_indices))
        cropped_char = char_transparent.crop(crop_box)
        char_w, char_h = cropped_char.size
        paste_x = (1024 - char_w) // 2
        paste_y = 1024 - char_h 
    else:
        cropped_char = char_transparent
        paste_x, paste_y = 0, 0
    
    print("Compositing grounded outlined character onto background...")
    bg_img_rgba = bg_img.convert("RGBA")
    bg_img_rgba.paste(cropped_char, (paste_x, paste_y), cropped_char)
    final_composite = bg_img_rgba.convert("RGB")
    
    results = {}
    
    final_bytes = io.BytesIO()
    final_composite.save(final_bytes, format="PNG")
    results["3_final_outline_composite.png"] = final_bytes.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    os.makedirs("../results/smart_greenscreen", exist_ok=True)
    images_dict = generate_and_outline.remote()
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/smart_greenscreen/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
    print("Done!")
