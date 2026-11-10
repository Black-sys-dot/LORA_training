import modal
import os
import io

app = modal.App("sd15-gojo-lora-infer")

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
    timeout=600,
)
def run_gojo_inference(lora_bytes: bytes) -> dict:
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image

    print("Loading base Stable Diffusion 1.5 on Modal GPU...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    prompts = [
        "A portrait of Satoru Gojo from Jujutsu Kaisen, white spiky hair, black blindfold, dark high-collar uniform, anime illustration, 8k resolution, masterpiece",
        "Satoru Gojo with glowing vibrant blue six eyes, white hair, domain expansion unlimited void, dramatic cinematic lighting, anime aesthetic",
        "Satoru Gojo smiling casually in Tokyo street, white hair, stylish sunglasses, modern clothes, sharp focus, vibrant colors"
    ]

    results = {"base": [], "lora": []}

    # 1. Base Model
    print("\n--- Generating 3 BASE Model Images (Zero LoRA) ---")
    for idx, prompt in enumerate(prompts):
        print(f"Generating Base Image {idx + 1}/3...")
        generator = torch.Generator(device="cuda").manual_seed(42 + idx)
        image = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        results["base"].append({
            "filename": f"base_gojo_{idx + 1}.png",
            "prompt": prompt,
            "bytes": buf.getvalue()
        })

    # 2. LoRA Model
    print("\n--- Loading trained LoRA adapter weights ---")
    temp_lora_path = "/tmp/gojo_lora.safetensors"
    with open(temp_lora_path, "wb") as f:
        f.write(lora_bytes)
    pipe.load_lora_weights(temp_lora_path)

    print("\n--- Generating 3 LoRA Adapted Images ---")
    for idx, prompt in enumerate(prompts):
        print(f"Generating LoRA Image {idx + 1}/3...")
        generator = torch.Generator(device="cuda").manual_seed(42 + idx)
        image = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        results["lora"].append({
            "filename": f"lora_gojo_{idx + 1}.png",
            "prompt": prompt,
            "bytes": buf.getvalue()
        })

    print("All generations complete! Container will terminate now.")
    return results

@app.local_entrypoint()
def main():
    lora_weights_path = "models/sd15_lora/pytorch_lora_weights.safetensors"
    with open(lora_weights_path, "rb") as f:
        lora_bytes = f.read()

    output_dir = "sd15_results/gojo_comparison"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loaded {len(lora_bytes)} bytes of LoRA weights. Dispatching to Modal...")
    data = run_gojo_inference.remote(lora_bytes)

    for item in data["base"]:
        out_p = os.path.join(output_dir, item["filename"])
        with open(out_p, "wb") as f:
            f.write(item["bytes"])
        print(f"Saved: {out_p}")

    for item in data["lora"]:
        out_p = os.path.join(output_dir, item["filename"])
        with open(out_p, "wb") as f:
            f.write(item["bytes"])
        print(f"Saved: {out_p}")

    print("\nInference complete and container terminated!")
