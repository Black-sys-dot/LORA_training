import modal
import io
import os

app = modal.App("flux-lora-emo-girl")

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
    
    # We use step 3000 as it yielded perfect pixel art before!
    pipe.load_lora_weights("/data/output/pixel_art_lora/pixel_art_lora_000003000.safetensors", adapter_name="pixel_art")
    pipe.fuse_lora(lora_scale=0.85)
    
    char_prompt = "beautiful detailed pixel art illustration of an edgy gothic emo girl, medium-length messy black wolf cut hair with vibrant purple under-lights, heavy dark eyeliner, pale skin, wearing a spiked collar, completely plain blank black baggy distressed tee shirt falling off one shoulder with absolutely no text and no logos and no graphics, deep cleavage, fishnet arm warmers, dark red plaid pleated skirt. smoking a cigarette with purple smoke. slightly explicit, revealing, sexy. isolated on a pure bright neon green solid flat background, retro-modern cel-shaded pixel art, crisp 1-pixel aliased lines, masterpiece, high quality"
    
    results = {}
    
    print("Generating 10 new vibe variations (v4)...")
    for seed in range(31, 41):
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
        results[f"emo_girl_v4_seed_{seed}.png"] = char_bytes.getvalue()
        
    return results

@app.local_entrypoint()
def main():
    out_dir = "../results/emo_girl_pixel"
    os.makedirs(out_dir, exist_ok=True)
    images_dict = batch_generate.remote()
    for filename, image_bytes in images_dict.items():
        filepath = f"{out_dir}/{filename}"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
    print(f"Done! Saved 10 images to {out_dir}")

