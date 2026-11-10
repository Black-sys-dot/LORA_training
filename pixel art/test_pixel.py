import modal
import io
import os

app = modal.App("flux-lora-pixel-test")

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
    
    # Pixel art style prompts based directly on the dataset captions
    prompts = [
        "beautiful pixel art portrait of 2B from Nier Automata, white hair, black blindfold, serene mood, surrounded by pristine white lilies floating, extreme low-angle perspective, high-contrast triadic color scheme, intricate lace patterns, crisp aliased 1-pixel borders, no dithering, masterpiece",
        "beautiful pixel art illustration of Fern from Frieren, long purple hair, wearing a dark heavy cloak, holding a magic staff with purple wrapping, floating in the clouds, serene night sky with faint stars, elegant soft stepped gradients, no dithering, high-end stylized pixel art, clean graphic silhouette, masterpiece",
        "beautiful pixel art illustration of a cyberpunk city hacker girl, sitting on a glowing neon rooftop, highly detailed retro-modern cel-shaded pixel art, smooth stepped tonal bands, crisp 1-pixel aliased lines, no dithering, masterpiece"
    ]
    
    results = {}
    
    # Load and Fuse the 2000-Step LoRA
    print("Loading and Fusing Pixel Art LoRA (Step 2000)...")
    pipe.load_lora_weights("/data/output/pixel_art_lora/pixel_art_lora_000002000.safetensors", adapter_name="pixel_art")
    pipe.fuse_lora(lora_scale=1.0)
    
    print("Generating LORA model images...")
    for i, prompt in enumerate(prompts):
        print(f"Generating Image {i+1}/3...")
        generator = torch.Generator("cuda").manual_seed(999 + i)
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
        results[f"pixel_test_{i+1}.png"] = byte_stream.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    os.makedirs("../results/pixel_test", exist_ok=True)
    
    print("Starting generation on Modal...")
    
    images_dict = generate_images.remote()
    
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/pixel_test/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"Saved {filepath}")
        
    print("Done! Check the 'results/pixel_test' folder.")
