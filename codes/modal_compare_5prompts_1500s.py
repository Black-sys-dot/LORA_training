import modal
import os
import io
import json

app = modal.App("flux-gojo-5prompts-1500s-comparison")

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
        "lycoris_lora"
    )
)

volume = modal.Volume.from_name("flux-lora-volume")

PROMPTS = [
    # 1. Iconic Combat / Hollow Purple
    "A dynamic action portrait of Satoru Gojo preparing his Hollow Purple technique, intense glowing violet energy swirling between his palms, floating above a shattered city crater, cosmic distortion and debris in the air, dramatic anime masterpiece.",
    # 2. Casual / Rain & Tokyo Cafe
    "Satoru Gojo sitting relaxed at an outdoor Tokyo cafe during a gentle rain, wearing an oversized black wool sweater, holding a warm cup of coffee, raindrops on the window, soft cinematic ambient lighting.",
    # 3. Cyberpunk Neo-Tokyo
    "Satoru Gojo in a futuristic cyberpunk Neo-Tokyo alleyway, glowing cyan and magenta holographic advertisements, high-tech tactical windbreaker, neon light reflections on his black blindfold, ultra-detailed.",
    # 4. Traditional Japanese Shrine
    "Satoru Gojo dressed in an elegant white and navy traditional Japanese yukata, walking through a tranquil Kyoto shrine garden surrounded by blooming pink cherry blossom petals falling gently.",
    # 5. Wild West Gunslinger
    "Satoru Gojo reimagined as a rugged Wild West gunslinger, wearing a brown leather duster coat, holster on his hip, wide-brim cowboy hat slightly tilted showing his icy-blue eyes, dusty frontier town at golden hour."
]

BASE_SEED = 2024

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def generate_lora() -> dict:
    from diffusers import FluxPipeline
    import torch

    print("=== [1/2] Generating LoRA (Rank 4, 1500 steps) ===")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    print("Loading and fusing LoRA Rank 4 (1500s)...")
    pipe.load_lora_weights("/data/output/gojo_lora_rank4_1500s/gojo_lora_rank4_1500s.safetensors", adapter_name="gojo_rank4_1500s")
    pipe.fuse_lora(lora_scale=1.0)

    results = {}
    for i, prompt in enumerate(PROMPTS):
        seed = BASE_SEED + i * 17
        print(f"LoRA 1500s Prompt {i+1}/5 (seed={seed})...")
        generator = torch.Generator("cuda").manual_seed(seed)
        img = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=30,
            max_sequence_length=512,
            generator=generator
        ).images[0]

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        results[f"prompt_{i+1:02d}_lora_1500s.png"] = buf.getvalue()

    return results

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def generate_lokr() -> dict:
    from diffusers import FluxPipeline
    import torch
    import lycoris

    print("=== [2/2] Generating LoKr / LyCORIS (Rank 4, 1500 steps) ===")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    print("Directly wrapping and merging LoKr (1500s) weights...")
    lyco_net, _ = lycoris.create_lycoris_from_weights(
        1.0,
        "/data/output/gojo_lycoris_rank4_1500s/gojo_lycoris_rank4_1500s.safetensors",
        pipe.transformer,
        dtype=torch.bfloat16
    )
    lyco_net.merge_to()

    results = {}
    for i, prompt in enumerate(PROMPTS):
        seed = BASE_SEED + i * 17
        print(f"LoKr 1500s Prompt {i+1}/5 (seed={seed})...")
        generator = torch.Generator("cuda").manual_seed(seed)
        img = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=30,
            max_sequence_length=512,
            generator=generator
        ).images[0]

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        results[f"prompt_{i+1:02d}_lokr_1500s.png"] = buf.getvalue()

    return results

@app.local_entrypoint()
def main():
    out_dir = "results/comparison_1500s"
    os.makedirs(out_dir, exist_ok=True)

    # Save prompt reference
    with open(os.path.join(out_dir, "prompts.json"), "w") as f:
        json.dump([{"id": i+1, "prompt": p, "seed": BASE_SEED + i * 17} for i, p in enumerate(PROMPTS)], f, indent=2)

    print(f"Loaded 5 test prompts. Output target: {out_dir}")

    print("\n>>> Launching LoRA and LoKr 1500s inference concurrently on Modal...")
    # Spawn both concurrently
    lora_handle = generate_lora.spawn()
    lokr_handle = generate_lokr.spawn()

    lora_results = lora_handle.get()
    for filename, raw_bytes in lora_results.items():
        with open(os.path.join(out_dir, filename), "wb") as f:
            f.write(raw_bytes)
        print(f"Saved: {filename}")

    lokr_results = lokr_handle.get()
    for filename, raw_bytes in lokr_results.items():
        with open(os.path.join(out_dir, filename), "wb") as f:
            f.write(raw_bytes)
        print(f"Saved: {filename}")

    print("\nAll 10 images generated and saved successfully!")

if __name__ == "__main__":
    main()
