import modal
import io
import os

app = modal.App("flux-lycoris-comparison")

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
        "lycoris_lora"
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
    
    print("Loading LyCORIS (LoKr) Rank 4 Weights...")
    pipe.load_lora_weights("/data/output/gojo_lycoris_rank4/gojo_lycoris_rank4.safetensors", adapter_name="gojo_lycoris")
    
    print("Generating LyCORIS model images...")
    for i, prompt in enumerate(prompts):
        print(f"LyCORIS Image {i+1}/{len(prompts)}...")
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
        results[f"comparison_lycoris_{i+1}.png"] = byte_stream.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    os.makedirs("../results/comparison_lycoris", exist_ok=True)
    
    print("Starting generation on Modal for LyCORIS...")
    
    images_dict = generate_comparisons.remote()
    
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/comparison_lycoris/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"Saved {filepath}")
        
    print("Generation complete! Check the 'results/comparison_lycoris' folder.")
