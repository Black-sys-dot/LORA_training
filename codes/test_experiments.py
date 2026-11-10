import modal
import io
import os

app = modal.App("flux-lora-test-experiments")

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
    timeout=1200,
)
def test_experiments() -> dict:
    from diffusers import FluxPipeline
    import torch
    
    print("Loading FLUX.1-dev base model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    prompts = [
        "Satoru Gojo, adult male, pure-white hair, icy-blue eyes, wearing a casual hoodie, standing in a rainy city street, photorealistic.",
        "Satoru Gojo, adult male, pure-white hair, icy-blue eyes, in an intense battle stance, ruined city background, dynamic action pose, highly detailed anime art style.",
        "Satoru Gojo, adult male, pure-white hair, icy-blue eyes, cinematic shot sitting on a luxury modern sofa drinking coffee, soft sunlight coming from the window, photorealistic."
    ]
    
    results = {}
    
    # --- TEST 1: RANK 32 LORA ---
    print("Loading Rank 32 LoRA...")
    pipe.load_lora_weights("/data/output/gojo_lora_rank32/gojo_lora_rank32.safetensors", adapter_name="rank32")
    pipe.set_adapters("rank32", adapter_weights=[1.0])
    
    print("Generating Rank 32 images...")
    for i, prompt in enumerate(prompts):
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
        results[f"experiment_rank32_prompt{i+1}.png"] = byte_stream.getvalue()
        
    pipe.unload_lora_weights() # Unload rank 32 before loading the next one
    
    # --- TEST 2: SLOW BAKE LORA (STEP 3000) ---
    print("Loading Slow Bake LoRA (Step 3000)...")
    pipe.load_lora_weights("/data/output/gojo_lora_slowbake/gojo_lora_slowbake_000003000.safetensors", adapter_name="slowbake")
    pipe.set_adapters("slowbake", adapter_weights=[1.0])
    
    print("Generating Slow Bake images...")
    for i, prompt in enumerate(prompts):
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
        results[f"experiment_slowbake_prompt{i+1}.png"] = byte_stream.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    import os
    os.makedirs("../results/experiments", exist_ok=True)
    
    print("Running experiments test on Modal...")
    images_dict = test_experiments.remote()
    
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/experiments/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"Saved {filepath}")
        
    print("Done! Check the 'results/experiments' folder.")
