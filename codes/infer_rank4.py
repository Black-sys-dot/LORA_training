import modal
import io
import os

app = modal.App("flux-lora-comparison-rank4")

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
        "Satoru Gojo, adult male, pure-white hair, icy-blue eyes, glowing purple energy around his hands, levitating above a crumbling temple, dramatic cinematic lighting, masterpiece illustration, 8k resolution, volumetric fog.",
        "Satoru Gojo wearing heavy medieval knight armor holding a broadsword."
    ]
    
    results = {}
    
    # We will just generate the LoRA (Rank 4) images here to compare against older LoRA.
    # No need to generate base images again if they already exist, but for completeness, we'll generate the new Rank 4 ones.
    
    print("Loading and Fusing Gojo LoRA Rank 4...")
    pipe.load_lora_weights("/data/output/gojo_lora_rank4/gojo_lora_rank4.safetensors", adapter_name="gojo_rank4")
    pipe.fuse_lora(lora_scale=1.0)
    
    print("Generating LORA (Rank 4) model images...")
    for i, prompt in enumerate(prompts):
        print(f"LoRA Image {i+1}/{len(prompts)}...")
        generator = torch.Generator("cuda").manual_seed(100 + i) # Exact same seed as before
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
        results[f"comparison_rank4_{i+1}_lora.png"] = byte_stream.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    os.makedirs("../results/comparison_rank4", exist_ok=True)
    
    print("Starting generation on Modal for Rank 4 LoRA...")
    
    images_dict = generate_comparisons.remote()
    
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/comparison_rank4/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"Saved {filepath}")
        
    print("Generation complete! Check the 'results/comparison_rank4' folder.")

