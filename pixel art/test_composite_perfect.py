import modal
import io
import os

app = modal.App("flux-lora-perfect-composite")

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
        "rembg",
        "Pillow",
        "onnxruntime",
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
def generate_and_composite() -> dict:
    from diffusers import FluxPipeline
    import torch
    from rembg import remove
    from PIL import Image
    import numpy as np
    
    print("Loading FLUX.1-dev base model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("Loading Pixel Art LoRA (Step 3000)...")
    pipe.load_lora_weights("/data/output/pixel_art_lora/pixel_art_lora_000003000.safetensors", adapter_name="pixel_art")
    pipe.fuse_lora(lora_scale=0.85)
    
    # Generate on solid black to eliminate white halos
    char_prompt = "beautiful pixel art illustration of a cute anime girl with pink hair, wearing a simple casual dress, full body, isolated on a pure solid black flat background, highly detailed retro-modern cel-shaded pixel art, crisp 1-pixel aliased lines, masterpiece"
    char_img = pipe(
        prompt=char_prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=30,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(1024)
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
    
    print("Removing black background...")
    # Use rembg
    char_transparent = remove(char_img)
    
    # Auto-crop bounding box to anchor her to the bottom
    char_rgba_data = np.array(char_transparent)
    alpha = char_rgba_data[:, :, 3]
    non_empty_columns = np.where(alpha.max(axis=0) > 0)[0]
    non_empty_rows = np.where(alpha.max(axis=1) > 0)[0]
    
    if len(non_empty_columns) > 0 and len(non_empty_rows) > 0:
        crop_box = (min(non_empty_columns), min(non_empty_rows), max(non_empty_columns), max(non_empty_rows))
        cropped_char = char_transparent.crop(crop_box)
        
        char_w, char_h = cropped_char.size
        paste_x = (1024 - char_w) // 2
        paste_y = 1024 - char_h # Snap to bottom
    else:
        cropped_char = char_transparent
        paste_x, paste_y = 0, 0
    
    print("Pasting...")
    bg_img_rgba = bg_img.convert("RGBA")
    bg_img_rgba.paste(cropped_char, (paste_x, paste_y), cropped_char)
    final_composite = bg_img_rgba.convert("RGB")
    
    results = {}
    
    final_bytes = io.BytesIO()
    final_composite.save(final_bytes, format="PNG")
    results["perfect_composite.png"] = final_bytes.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    os.makedirs("../results/composite_perfect", exist_ok=True)
    images_dict = generate_and_composite.remote()
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/composite_perfect/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
    print("Done!")
