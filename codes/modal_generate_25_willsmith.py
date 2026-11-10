import modal
import os
import io

app = modal.App("sd15-will-smith-25-dataset")

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

PROMPTS = [
    "Portrait photo of Will Smith as a 1990s hip hop artist in Philadelphia, colorful retro windbreaker, backwards cap, authentic film grain",
    "Close-up cinematic headshot of Will Smith as a stern military general in uniform, harsh studio lighting, sharp focus on eyes",
    "Will Smith smiling warmly at an outdoor coffee shop, wearing casual linen shirt, golden hour sunlight, bokeh background",
    "Will Smith in a formal black tuxedo on a red carpet premiere, flash photography, sharp focus, photorealistic",
    "Will Smith as an intense cyberpunk detective in a dark neon-lit alleyway, rainy atmosphere, cyan and magenta reflections",
    "Will Smith wearing pilot sunglasses and a leather bomber jacket standing in front of an aircraft hangar, afternoon sky",
    "Will Smith laughing genuinely in a comedy club setting, warm stage lighting, microphone in hand, natural expression",
    "Will Smith as a weathered post-apocalyptic survivor in rugged wasteland gear, dusty environment, dramatic portrait lighting",
    "Will Smith looking thoughtful in an art museum, black turtleneck, gallery lighting, elegant candid portrait",
    "Will Smith running in athletic sports gear on an urban track, dynamic pose, morning mist, sharp motion",
    "Will Smith as a wise college professor in a library surrounded by books, wearing tweed vest and glasses, soft warm lighting",
    "High-fashion magazine editorial portrait of Will Smith, monochrome black and white photography, strong shadows and contrast",
    "Will Smith as a charismatic jazz musician, smoky jazz bar atmosphere, moody blue and amber spotlight",
    "Will Smith standing on a rainy New York rooftop overlooking the skyline, dark trench coat, overcast moody sky",
    "Will Smith as a futuristic astronaut looking through a spaceship window, reflection of Earth, sci-fi interior lighting",
    "Will Smith dressed in distinguished historical Victorian formal attire, vintage portrait lighting, dignified expression",
    "Will Smith smiling at the beach during sunset, tropical shirt, ocean waves in background, warm orange glow",
    "Intense emotional close-up portrait of Will Smith in dramatic cinema lighting, focused eyes, intense drama acting expression",
    "Will Smith as a rugged wilderness mountain explorer, warm puffer jacket, snowy peaks in background, crisp winter light",
    "Will Smith in modern casual streetwear, hoodie and denim jacket, standing in front of vibrant urban graffiti wall",
    "Will Smith in a business meeting boardroom, crisp navy blue suit and tie, sleek modern office background",
    "Will Smith as a medieval knight in ornate dark iron armor, atmospheric torchlight, cinematic fantasy portrait",
    "Will Smith looking up at falling autumn leaves in Central Park, cozy wool sweater, soft diffused autumn sunlight",
    "Will Smith as an undercover detective in a dimly lit vintage sedan at night, streetlights reflecting on windshield",
    "Studio glamour beauty headshot of Will Smith, neutral grey background, clean three-point commercial portrait lighting"
]

@app.function(
    image=image,
    gpu="T4",
    timeout=900,
)
def generate_batch() -> list[dict]:
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image

    print("Loading base Stable Diffusion 1.5 model on Modal GPU...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    results = []
    print(f"Generating {len(PROMPTS)} diverse Will Smith images on base SD 1.5...")
    for idx, prompt in enumerate(PROMPTS):
        print(f"Generating image {idx + 1}/{len(PROMPTS)}: {prompt[:60]}...")
        generator = torch.Generator(device="cuda").manual_seed(200 + idx)
        image = pipe(
            prompt,
            num_inference_steps=28,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        results.append({
            "index": idx + 1,
            "filename": f"will_smith_{idx + 1:02d}.png",
            "prompt": prompt,
            "bytes": buf.getvalue()
        })

    print("All 25 images generated! Container will now terminate.")
    return results

@app.local_entrypoint()
def main():
    output_dir = "sd15_results/will_smith_dataset"
    os.makedirs(output_dir, exist_ok=True)
    print("Dispatching 25-image generation job to Modal GPU...")
    results = generate_batch.remote()

    print(f"\nSaving {len(results)} images locally to {output_dir}/...")
    metadata_lines = []
    for item in results:
        save_path = os.path.join(output_dir, item["filename"])
        with open(save_path, "wb") as f:
            f.write(item["bytes"])
        metadata_lines.append(f"{item['filename']}: {item['prompt']}\n")
        print(f"Saved: {save_path}")

    # Write prompts index
    with open(os.path.join(output_dir, "prompts.txt"), "w") as f:
        f.writelines(metadata_lines)

    print(f"\nSuccessfully generated and saved all 25 images to {output_dir}/!")
    print("Modal container has terminated.")
