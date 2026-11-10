import modal
import os
import io

app = modal.App("test-gojo-lora-scales")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "peft"
    )
)

@app.function(
    image=image,
    gpu="T4",
    timeout=300,
)
def test_scales(lora_bytes: bytes) -> list[dict]:
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image

    print("Loading SD 1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    temp_lora_path = "/tmp/lora.safetensors"
    with open(temp_lora_path, "wb") as f:
        f.write(lora_bytes)
    pipe.load_lora_weights(temp_lora_path)

    prompt = "A portrait of Satoru Gojo from Jujutsu Kaisen, white hair, black blindfold, anime illustration, masterpiece"
    negative_prompt = "ugly, deformed, disfigured, poor anatomy, bad hands, blurry, low quality, oversaturated"
    
    scales = [0.3, 0.5, 0.7, 1.0]
    results = []

    for scale in scales:
        print(f"Generating with LoRA scale: {scale}...")
        generator = torch.Generator(device="cuda").manual_seed(42)
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.0,
            cross_attention_kwargs={"scale": scale},
            generator=generator
        ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        results.append({
            "scale": scale,
            "filename": f"gojo_scale_{int(scale*100)}.png",
            "bytes": buf.getvalue()
        })

    return results

@app.local_entrypoint()
def main():
    lora_path = "models/sd15_lora/pytorch_lora_weights.safetensors"
    with open(lora_path, "rb") as f:
        lora_bytes = f.read()

    output_dir = "sd15_results/scale_test"
    os.makedirs(output_dir, exist_ok=True)

    print("Running scale comparison...")
    data = test_scales.remote(lora_bytes)

    for item in data:
        p = os.path.join(output_dir, item["filename"])
        with open(p, "wb") as f:
            f.write(item["bytes"])
        print(f"Saved scale {item['scale']} to {p}")
