import modal
import os
import io

app = modal.App("sd15-will-smith-infer")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate"
    )
)

@app.function(
    image=image,
    gpu="T4",
    timeout=300,
)
def generate_will_smith() -> list[dict]:
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image

    print("Loading base Stable Diffusion 1.5 model on Modal GPU...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    prompts = [
        "A realistic portrait photo of Will Smith in a sharp black suit, studio lighting, highly detailed, 8k resolution",
        "Will Smith smiling warmly outdoors in natural sunlight, candid cinematic portrait photography",
        "Will Smith as a futuristic sci-fi detective in a cyber city, neon rim lighting, cinematic masterpiece"
    ]

    images_data = []
    print("Generating 3 images of Will Smith on base SD 1.5...")
    for idx, prompt in enumerate(prompts):
        print(f"Generating image {idx + 1}/3: '{prompt}'")
        generator = torch.Generator(device="cuda").manual_seed(100 + idx)
        image = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        images_data.append({
            "prompt": prompt,
            "filename": f"will_smith_base_{idx + 1}.png",
            "bytes": buf.getvalue()
        })

    print("Generation complete! Container will now terminate.")
    return images_data

@app.local_entrypoint()
def main():
    output_dir = "sd15_results/will_smith"
    os.makedirs(output_dir, exist_ok=True)
    print("Dispatching job to Modal...")
    results = generate_will_smith.remote()

    print(f"\nSaving {len(results)} images locally to {output_dir}/...")
    for item in results:
        save_path = os.path.join(output_dir, item["filename"])
        with open(save_path, "wb") as f:
            f.write(item["bytes"])
        print(f"Saved: {save_path} (Prompt: {item['prompt'][:50]}...)")

    print("\nAll done! Container on Modal has terminated.")
