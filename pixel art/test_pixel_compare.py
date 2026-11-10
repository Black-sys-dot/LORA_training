import modal
import io
import os

app = modal.App("flux-lora-pixel-compare")

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
        "peft"
    )
)

volume = modal.Volume.from_name("flux-lora-volume")

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1200, # 20 mins just in case
)
def generate_images() -> dict:
    from diffusers import FluxPipeline
    import torch
    
    print("Loading FLUX.1-dev base model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # A single detailed but safe prompt to let the LoRA shine
    prompt = "beautiful pixel art illustration of a cute anime girl with pink hair wearing casual streetwear and headphones, standing in a sunny cozy bedroom, highly detailed retro-modern cel-shaded pixel art, crisp 1-pixel aliased lines, smooth stepped tonal bands, no dithering, masterpiece"
    
    results = {}
    
    # ---------------------------------------------------------
    # Generate 5 images with Step 2000
    # ---------------------------------------------------------
    print("Loading Pixel Art LoRA (Step 2000)...")
    pipe.load_lora_weights("/data/output/pixel_art_lora/pixel_art_lora_000002000.safetensors", adapter_name="pixel_art")
    pipe.fuse_lora(lora_scale=1.0)
    
    print("Generating 5 images for Step 2000...")
    for i in range(5):
        print(f"Step 2000 - Image {i+1}/5...")
        generator = torch.Generator("cuda").manual_seed(500 + i)
        img = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=30,
            max_sequence_length=512,
            joint_attention_kwargs={"scale": 1.0},
            generator=generator
        ).images[0]
        
        byte_stream = io.BytesIO()
        img.save(byte_stream, format="PNG")
        results[f"step2000_img{i+1}.png"] = byte_stream.getvalue()
        
    pipe.unfuse_lora()
    pipe.unload_lora_weights()

    # ---------------------------------------------------------
    # Generate 5 images with Step 3000 (EXACT SAME SEEDS)
    # ---------------------------------------------------------
    print("Loading Pixel Art LoRA (Step 3000)...")
    pipe.load_lora_weights("/data/output/pixel_art_lora/pixel_art_lora_000003000.safetensors", adapter_name="pixel_art_3000")
    pipe.fuse_lora(lora_scale=1.0)
    
    print("Generating 5 images for Step 3000...")
    for i in range(5):
        print(f"Step 3000 - Image {i+1}/5...")
        # Exact same seeds to ensure direct 1-to-1 comparison
        generator = torch.Generator("cuda").manual_seed(500 + i)
        img = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=30,
            max_sequence_length=512,
            joint_attention_kwargs={"scale": 1.0},
            generator=generator
        ).images[0]
        
        byte_stream = io.BytesIO()
        img.save(byte_stream, format="PNG")
        results[f"step3000_img{i+1}.png"] = byte_stream.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    os.makedirs("../results/pixel_compare", exist_ok=True)
    
    print("Starting side-by-side comparison on Modal...")
    
    images_dict = generate_images.remote()
    
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/pixel_compare/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"Saved {filepath}")
        
    print("Done! Check the 'results/pixel_compare' folder.")
