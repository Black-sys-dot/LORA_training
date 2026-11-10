import modal
import io
import os

app = modal.App("flux-lora-comparison")

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
def generate_comparisons() -> dict:
    from diffusers import FluxPipeline
    import torch
    
    print("Loading FLUX.1-dev base model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    prompts = [
        "Satoru Gojo drinking tea in a garden.",
        "Satoru Gojo, adult male, pure-white hair, icy-blue eyes, wearing a casual hoodie, standing in a rainy city street.",
        "Satoru Gojo, adult male, pure-white hair, icy-blue eyes, glowing purple energy around his hands, levitating above a crumbling temple, dramatic cinematic lighting, masterpiece illustration, 8k resolution, volumetric fog."
    ]
    
    results = {}
    
    # 1. Generate Base Model Images
    print("Generating BASE model images...")
    for i, prompt in enumerate(prompts):
        print(f"Base Image {i+1}/3...")
        generator = torch.Generator("cuda").manual_seed(100 + i) # Changed seed to get fresh compositions
        img = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=30,
            max_sequence_length=512,
            generator=generator
        ).images[0]
        
        byte_stream = io.BytesIO()
        img.save(byte_stream, format="PNG")
        results[f"comparison2_{i+1}_base.png"] = byte_stream.getvalue()
        
    # 2. Load and Fuse LoRA
    print("Loading and Fusing Gojo LoRA...")
    pipe.load_lora_weights("/data/output/gojo_lora/gojo_lora.safetensors", adapter_name="gojo")
    pipe.fuse_lora(lora_scale=1.0)
    
    # 3. Generate LoRA Images with EXACT SAME SEEDS
    print("Generating LORA model images...")
    for i, prompt in enumerate(prompts):
        print(f"LoRA Image {i+1}/3...")
        generator = torch.Generator("cuda").manual_seed(100 + i)
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
        results[f"comparison2_{i+1}_lora.png"] = byte_stream.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    os.makedirs("../results/comparison2", exist_ok=True)
    
    print("Starting side-by-side comparison on Modal...")
    print("This will generate 6 images total (3 Base + 3 LoRA)...")
    
    images_dict = generate_comparisons.remote()
    
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/comparison2/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"Saved {filepath}")
        
    print("Comparison complete! Check the 'results/comparison2' folder.")
