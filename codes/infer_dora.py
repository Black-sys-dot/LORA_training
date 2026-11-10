import modal
import io
import os

app = modal.App("flux-dora-comparison")

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
    
    print("Loading DoRA Rank 4 Weights...")
    # Diffusers natively supports DoRA checkpoints loading via peft backend
    pipe.load_lora_weights("/data/output/gojo_dora_rank4/gojo_dora_rank4.safetensors", adapter_name="gojo_dora")
    
    # We omit fuse_lora() for DoRA to prevent potential merge errors in older diffuser versions,
    # and instead rely purely on joint_attention_kwargs scaling during generation.
    
    print("Generating DoRA model images...")
    for i, prompt in enumerate(prompts):
        print(f"DoRA Image {i+1}/{len(prompts)}...")
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
        results[f"comparison_dora_{i+1}.png"] = byte_stream.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    os.makedirs("../results/comparison_dora", exist_ok=True)
    
    print("Starting generation on Modal for DoRA...")
    
    images_dict = generate_comparisons.remote()
    
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/comparison_dora/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"Saved {filepath}")
        
    print("Generation complete! Check the 'results/comparison_dora' folder.")

