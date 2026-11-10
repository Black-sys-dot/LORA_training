import os
import torch
from diffusers import StableDiffusionPipeline

def main():
    output_dir = "/opt/ml/model"
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on AWS SageMaker with device: {device}...")

    # Load SD 1.5 base
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)

    # Load Gojo LoRA weights
    lora_path = "/opt/ml/code/pytorch_lora_weights.safetensors"
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}...")
        pipe.load_lora_weights(lora_path)
    else:
        print(f"Warning: {lora_path} not found, generating base...")

    prompts = [
        "A portrait of Satoru Gojo from Jujutsu Kaisen, white spiky hair, black blindfold, dark high-collar uniform, anime illustration, 8k resolution, masterpiece",
        "Satoru Gojo with glowing vibrant blue six eyes, white hair, domain expansion unlimited void, dramatic cinematic lighting, anime aesthetic",
        "Satoru Gojo smiling casually in Tokyo street, white hair, stylish sunglasses, modern clothes, sharp focus, vibrant colors"
    ]

    for idx, prompt in enumerate(prompts):
        print(f"Generating image {idx + 1}/3: '{prompt}'")
        generator = torch.Generator(device=device).manual_seed(42 + idx)
        image = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        save_path = os.path.join(output_dir, f"gojo_aws_{idx + 1}.png")
        image.save(save_path)
        print(f"Saved to {save_path}")

    print("AWS Inference job completed successfully! Container will now terminate.")

if __name__ == "__main__":
    main()
