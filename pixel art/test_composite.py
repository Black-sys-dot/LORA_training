import modal
import io
import os

app = modal.App("flux-lora-composite")

# We added 'rembg' and 'Pillow' for the automatic background removal and pasting
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
        "onnxruntime"
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
    
    print("Loading FLUX.1-dev base model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("Loading Pixel Art LoRA (Step 3000)...")
    pipe.load_lora_weights("/data/output/pixel_art_lora/pixel_art_lora_000003000.safetensors", adapter_name="pixel_art")
    # Setting LoRA scale to 0.85 to get beautiful textures but keep anatomy flawless
    pipe.fuse_lora(lora_scale=0.85)
    
    # 1. Generate the isolated character
    char_prompt = "beautiful pixel art illustration of a cute anime girl with pink hair, wearing a simple casual dress, full body, isolated on a pure flat solid white background, highly detailed retro-modern cel-shaded pixel art, crisp 1-pixel aliased lines, masterpiece"
    print("Generating isolated character...")
    char_img = pipe(
        prompt=char_prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=30,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(1024)
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
    
    # 3. Automatic Background Removal and Compositing
    print("Removing white background from character using AI...")
    char_transparent = remove(char_img)
    
    print("Compositing character onto background...")
    bg_img_rgba = bg_img.convert("RGBA")
    # Paste the transparent character directly over the background
    bg_img_rgba.paste(char_transparent, (0, 0), char_transparent)
    final_composite = bg_img_rgba.convert("RGB")
    
    # Save all 3 steps to send back
    results = {}
    
    char_bytes = io.BytesIO()
    char_img.save(char_bytes, format="PNG")
    results["1_character_raw.png"] = char_bytes.getvalue()
    
    bg_bytes = io.BytesIO()
    bg_img.save(bg_bytes, format="PNG")
    results["2_background_raw.png"] = bg_bytes.getvalue()
    
    final_bytes = io.BytesIO()
    final_composite.save(final_bytes, format="PNG")
    results["3_final_composite.png"] = final_bytes.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    os.makedirs("../results/composite", exist_ok=True)
    
    print("Starting generation and AI compositing on Modal...")
    
    images_dict = generate_and_composite.remote()
    
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/composite/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"Saved {filepath}")
        
    print("Done! Check the 'results/composite' folder.")
