import modal
import io
import os

app = modal.App("flux-lora-batch-generate")

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
        "Pillow"
    )
)

volume = modal.Volume.from_name("flux-lora-volume")

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800, 
)
def batch_generate() -> dict:
    from diffusers import FluxPipeline
    import torch
    
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    pipe.load_lora_weights("/data/output/pixel_art_lora/pixel_art_lora_000003000.safetensors", adapter_name="pixel_art")
    pipe.fuse_lora(lora_scale=0.85)
    
    # NEW PROMPT: Heavy emphasis on ONE cup, perfect hands, other hand in pocket
    char_prompt = "beautiful pixel art illustration of a cute anime girl with short pastel pink hair, wearing a fluffy oversized white sweater and a black pleated skirt, ONE hand holding a single steaming coffee cup with perfectly drawn wrapping fingers, her other hand resting hidden in her pocket, exactly one coffee cup, anatomically correct hands, exactly five fingers, flawless detailed fingers, full body, isolated on a pure bright neon cyan solid flat background, highly detailed retro-modern cel-shaded pixel art, masterpiece"
    
    results = {}
    
    print("Generating 10 variations for the hands test...")
    for seed in range(1, 11):
        print(f"Generating Seed {seed}...")
        char_img = pipe(
            prompt=char_prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=30,
            max_sequence_length=512,
            generator=torch.Generator("cuda").manual_seed(seed)
        ).images[0]
        
        char_bytes = io.BytesIO()
        char_img.save(char_bytes, format="PNG")
        results[f"hands_seed_{seed}.png"] = char_bytes.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    out_dir = "../results/batch_test/hands_test"
    os.makedirs(out_dir, exist_ok=True)
    images_dict = batch_generate.remote()
    for filename, image_bytes in images_dict.items():
        filepath = f"{out_dir}/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
    print("Done! Saved all remaining images.")
