import torch
from diffusers import StableDiffusionPipeline
import os
import argparse

def run_inference(lora_weights_path=None, output_dir="sd15_results"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading Stable Diffusion 1.5 base model on {device} ({dtype})...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)

    prompts = [
        "A portrait of Satoru Gojo from Jujutsu Kaisen, white hair, black blindfold, high collar jacket, anime illustration, 8k resolution, masterpiece",
        "Satoru Gojo with glowing vibrant blue eyes, white spiky hair, domain expansion unlimited void, dramatic lighting, anime aesthetic",
        "Satoru Gojo smiling casually in a street cafe, white hair, stylish sunglasses, modern clothes, sharp focus, vibrant colors"
    ]

    # 1. Base Model (No LoRA)
    print("\n--- Generating BASE Model Images (Without LoRA) ---")
    for i, prompt in enumerate(prompts):
        generator = torch.Generator(device=device).manual_seed(42 + i)
        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, generator=generator).images[0]
        save_path = os.path.join(output_dir, f"base_sample_{i+1}.png")
        image.save(save_path)
        print(f"Saved: {save_path}")

    # 2. With LoRA
    if lora_weights_path and os.path.exists(lora_weights_path):
        print(f"\n--- Loading LoRA weights from {lora_weights_path} ---")
        pipe.load_lora_weights(lora_weights_path)

        print("\n--- Generating LoRA Adapted Images ---")
        for i, prompt in enumerate(prompts):
            generator = torch.Generator(device=device).manual_seed(42 + i)
            image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, generator=generator).images[0]
            save_path = os.path.join(output_dir, f"lora_sample_{i+1}.png")
            image.save(save_path)
            print(f"Saved: {save_path}")
    else:
        print(f"\nNo LoRA weights provided or path does not exist: {lora_weights_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default=None, help="Path to lora weights dir or safetensors")
    parser.add_argument("--output_dir", type=str, default="sd15_results", help="Output directory")
    args = parser.parse_args()
    run_inference(args.lora_path, args.output_dir)
