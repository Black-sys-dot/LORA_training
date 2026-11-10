import modal
import io
import os

app = modal.App("flux-lora-smart-greenscreen")

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
    
    # Generate on CYAN because she has pink hair! Pink/Magenta would blend with her hair. Cyan is the exact opposite.
    char_prompt = "beautiful pixel art illustration of a cute anime girl with pink hair, wearing a simple casual white dress, full body, isolated on a pure bright neon cyan solid flat background, highly detailed retro-modern cel-shaded pixel art, crisp 1-pixel aliased lines, masterpiece"
    print("Generating isolated character on Cyan...")
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
    
    print("Running Smart Chroma-Key (Cyan) removal...")
    char_rgba = char_img.convert("RGBA")
    data = np.array(char_rgba)
    
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # SMART MASK: Cyan is high Green and high Blue, low Red.
    # Instead of strict numbers, we just say: "Is Green and Blue significantly brighter than Red?"
    # This mathematically catches ANY shade of Cyan, even pastel or dark cyan!
    cyan_mask = (g.astype(int) > r.astype(int) + 30) & (b.astype(int) > r.astype(int) + 30)
    
    # Make all cyan background pixels transparent
    data[cyan_mask, 3] = 0
    char_transparent = Image.fromarray(data)
    
    # Auto-crop and snap to bottom
    non_empty_columns = np.where(~cyan_mask.min(axis=0))[0]
    non_empty_rows = np.where(~cyan_mask.min(axis=1))[0]
    
    if len(non_empty_columns) > 0 and len(non_empty_rows) > 0:
        crop_box = (min(non_empty_columns), min(non_empty_rows), max(non_empty_columns), max(non_empty_rows))
        cropped_char = char_transparent.crop(crop_box)
        char_w, char_h = cropped_char.size
        paste_x = (1024 - char_w) // 2
        paste_y = 1024 - char_h # Snap to bottom border
    else:
        cropped_char = char_transparent
        paste_x, paste_y = 0, 0
    
    print("Compositing grounded character onto background...")
    bg_img_rgba = bg_img.convert("RGBA")
    bg_img_rgba.paste(cropped_char, (paste_x, paste_y), cropped_char)
    final_composite = bg_img_rgba.convert("RGB")
    
    results = {}
    
    char_bytes = io.BytesIO()
    char_img.save(char_bytes, format="PNG")
    results["1_character_cyan.png"] = char_bytes.getvalue()
    
    final_bytes = io.BytesIO()
    final_composite.save(final_bytes, format="PNG")
    results["2_final_smart_composite.png"] = final_bytes.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    os.makedirs("../results/smart_greenscreen", exist_ok=True)
    images_dict = generate_and_greenscreen.remote()
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/smart_greenscreen/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
    print("Done!")
