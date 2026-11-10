import modal
import io
import os

app = modal.App("flux-lora-single-test")

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
    timeout=600, 
)
def generate_single() -> bytes:
    from diffusers import FluxPipeline
    import torch
    
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    pipe.load_lora_weights("/data/output/pixel_art_lora/pixel_art_lora_000003000.safetensors", adapter_name="pixel_art")
    pipe.fuse_lora(lora_scale=0.85)
    
    # NEW PROMPT: Heavy emphasis on perfect hands and fingers
    char_prompt = "beautiful pixel art illustration of a cute anime girl with short pastel pink hair, wearing a fluffy oversized white sweater and a black pleated skirt, holding a steaming coffee cup, perfectly drawn hands, anatomically correct hands, exactly five fingers, flawless detailed fingers, full body, isolated on a pure bright neon cyan solid flat background, highly detailed retro-modern cel-shaded pixel art, crisp 1-pixel aliased lines, masterpiece"
    
    print("Generating perfect hands test...")
    char_img = pipe(
        prompt=char_prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=30,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(9999)
    ).images[0]
    
    char_bytes = io.BytesIO()
    char_img.save(char_bytes, format="PNG")
    return char_bytes.getvalue()

@app.local_entrypoint()
def main():
    os.makedirs("../results/batch_test", exist_ok=True)
    image_bytes = generate_single.remote()
    
    filepath = "../results/batch_test/cozy_girl_perfect_hands_2.png"
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    print(f"Done! Saved test image to {filepath}")
