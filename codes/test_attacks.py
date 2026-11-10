import modal
import io
import os

app = modal.App("flux-lora-test-attacks")

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
def test_attacks() -> dict:
    from diffusers import FluxPipeline
    import torch
    
    print("Loading FLUX.1-dev base model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # We use the exact trigger phrases from gojo_06.txt and gojo_09.txt in the dataset
    prompts = [
        "Satoru Gojo, adult male, pure-white hair, icy-blue eyes, dark navy-black high-collared Jujutsu Kaisen-style uniform. A concentrated bright purple energy sphere is formed directly above the tip of his raised index finger. Large swirling ribbons and arcs of purple cursed energy spiral around his body, accompanied by glowing fragments and sharp energy trails. Cinematic lighting, highly detailed anime art style, 8k.",
        "Satoru Gojo, adult male, pure-white hair, icy-blue eyes, dark navy-black high-collared Jujutsu Kaisen-style uniform. Two massive opposing cursed-energy spheres are being generated simultaneously. A brilliant red sphere surrounds his left hand, while a brilliant blue sphere surrounds his right hand. The spheres surge toward each other and collide at the center, producing an intense white-violet core surrounded by explosive red, blue, and purple energy fragments. Masterpiece illustration, dynamic action pose, 8k."
    ]
    
    results = {}
    
    print("Loading ORIGINAL Masterpiece LoRA...")
    pipe.load_lora_weights("/data/output/gojo_lora/gojo_lora.safetensors", adapter_name="gojo")
    pipe.fuse_lora(lora_scale=1.0)
    
    print("Generating Attack images...")
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/2...")
        # Fixed seed to ensure a great composition
        generator = torch.Generator("cuda").manual_seed(1234 + i)
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
        results[f"attack_test_{i+1}.png"] = byte_stream.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    os.makedirs("../results/attacks", exist_ok=True)
    
    print("Running attack tests on Modal...")
    images_dict = test_attacks.remote()
    
    for filename, image_bytes in images_dict.items():
        filepath = f"../results/attacks/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"Saved {filepath}")
        
    print("Done! Check the 'results/attacks' folder.")
