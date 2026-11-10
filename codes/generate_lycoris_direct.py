import modal
import os
import io

app = modal.App("flux-lycoris-inference-direct")

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
def generate_direct() -> dict:
    from diffusers import FluxPipeline
    import torch
    import lycoris
    
    print("Loading FLUX.1-dev base model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("Directly wrapping Diffusers pipeline with LyCORIS library...")
    # This bypasses Diffusers' internal check and loads LoKr directly!
    lyco_net, _ = lycoris.create_lycoris_from_weights(
        1.0, 
        "/data/output/gojo_lycoris_rank4/gojo_lycoris_rank4.safetensors",
        pipe.transformer,
        dtype=torch.bfloat16
    )
    lyco_net.merge_to() # Fuses it permanently into the transformer layers!
    
    prompts = [
        "Satoru Gojo drinking tea in a garden.",
        "Satoru Gojo, adult male, pure-white hair, icy-blue eyes, wearing a casual hoodie, standing in a rainy city street.",
        "Satoru Gojo, adult male, pure-white hair, icy-blue eyes, glowing purple energy around his hands, levitating above a crumbling temple, dramatic cinematic lighting, masterpiece illustration, 8k resolution, volumetric fog.",
        "Satoru Gojo wearing heavy medieval knight armor holding a broadsword."
    ]
    
    results = {}
    
    print("Generating LyCORIS model images (direct merge)...")
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
            generator=generator
        ).images[0]
        
        byte_stream = io.BytesIO()
        img.save(byte_stream, format="PNG")
        results[f"comparison_lycoris_{i+1}.png"] = byte_stream.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    os.makedirs("../results/comparison_lycoris_final", exist_ok=True)
    
    print("Starting pure LyCORIS pipeline wrapper...")
    
    images_dict = generate_direct.remote()
    
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/comparison_lycoris_final/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"Saved {filepath}")
        
    print("Generation complete! Check the 'results/comparison_lycoris_final' folder.")
