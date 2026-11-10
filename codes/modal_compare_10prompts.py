import modal
import os
import io
import json

app = modal.App("flux-gojo-10prompts-comparison")

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
    # 1. Iconic Action / Ability
    "A dynamic action portrait of Satoru Gojo preparing his Hollow Purple technique, intense glowing violet energy swirling between his palms, floating above a shattered city crater, cosmic distortion and debris in the air, dramatic anime masterpiece.",
    # 2. Cozy Rain / Cafe
    "Satoru Gojo sitting relaxed at an outdoor Tokyo cafe during a gentle rain, wearing an oversized black wool sweater, holding a warm cup of coffee, raindrops on the window, soft cinematic ambient lighting.",
    # 3. Cyberpunk / Sci-Fi
    "Satoru Gojo in a futuristic cyberpunk Neo-Tokyo alleyway, glowing cyan and magenta holographic advertisements, high-tech tactical windbreaker, neon light reflections on his black blindfold, ultra-detailed.",
    # 4. Traditional Japanese
    "Satoru Gojo dressed in an elegant white and navy traditional Japanese yukata, walking through a tranquil Kyoto shrine garden surrounded by blooming pink cherry blossom petals falling gently.",
    # 5. Wild West
    "Satoru Gojo reimagined as a rugged Wild West gunslinger, wearing a brown leather duster coat, holster on his hip, wide-brim cowboy hat slightly tilted showing his icy-blue eyes, dusty frontier town at golden hour.",
    # 6. Winter Blizzard
    "Satoru Gojo standing alone atop a jagged snowy mountain peak in a violent blizzard, wind fiercely whipping his spiky white hair, icy frost on his lashes, epic glacial landscape backdrop.",
    # 7. High-Fashion Streetwear
    "High-fashion editorial street photograph of Satoru Gojo, wearing designer dark streetwear, loose utility cargo pants and luxury sneakers, modern minimalist brutalist architecture background, natural midday daylight.",
    # 8. Gothic Cathedral
    "Satoru Gojo walking through an ancient abandoned gothic cathedral, intricate stained glass windows casting vivid crimson and cobalt rays onto the stone floor, floating dust particles, moody atmospheric chiaroscuro lighting.",
    # 9. Tropical Beach
    "Satoru Gojo enjoying a sunny summer vacation at a tropical beach, wearing stylish tinted sunglasses, an open Hawaiian floral shirt, and shorts, turquoise ocean waves and palm trees in the background.",
    # 10. Fantasy Knight Armor (Overfitting / Generalization Stress Test)
    "Satoru Gojo as a fantasy warrior clad in ornate heavy silver knight plate armor with gold engravings, resting his hands on the hilt of a glowing enchanted broadsword, ruined fantasy fortress in the background."
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

    print("=== [1/3] Generating LoRA (Rank 4, 1000 steps) ===")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    print("Loading and fusing LoRA Rank 4...")
    pipe.load_lora_weights("/data/output/gojo_lora_rank4/gojo_lora_rank4.safetensors", adapter_name="gojo_rank4")
    pipe.fuse_lora(lora_scale=1.0)

    results = {}
    for i, prompt in enumerate(PROMPTS):
        seed = BASE_SEED + i * 17
        print(f"LoRA Prompt {i+1}/10 (seed={seed})...")
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
        results[f"prompt_{i+1:02d}_lora.png"] = buf.getvalue()

    return results

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def generate_dora() -> dict:
    from diffusers import FluxPipeline
    import torch

    print("=== [2/3] Generating DoRA (Rank 4, 600 steps) ===")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    print("Loading DoRA Rank 4 weights...")
    pipe.load_lora_weights("/data/output/gojo_dora_rank4/gojo_dora_rank4.safetensors", adapter_name="gojo_dora")

    results = {}
    for i, prompt in enumerate(PROMPTS):
        seed = BASE_SEED + i * 17
        print(f"DoRA Prompt {i+1}/10 (seed={seed})...")
        generator = torch.Generator("cuda").manual_seed(seed)
        img = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=30,
            max_sequence_length=512,
            joint_attention_kwargs={"scale": 1.0},
            generator=generator
        ).images[0]

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        results[f"prompt_{i+1:02d}_dora.png"] = buf.getvalue()

    return results

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def generate_lycoris() -> dict:
    from diffusers import FluxPipeline
    import torch
    import lycoris

    print("=== [3/3] Generating LyCORIS / LoKr (Rank 4, 600 steps) ===")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    print("Directly wrapping and merging LyCORIS weights...")
    lyco_net, _ = lycoris.create_lycoris_from_weights(
        1.0,
        "/data/output/gojo_lycoris_rank4/gojo_lycoris_rank4.safetensors",
        pipe.transformer,
        dtype=torch.bfloat16
    )
    lyco_net.merge_to()

    results = {}
    for i, prompt in enumerate(PROMPTS):
        seed = BASE_SEED + i * 17
        print(f"LyCORIS Prompt {i+1}/10 (seed={seed})...")
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
        results[f"prompt_{i+1:02d}_lycoris.png"] = buf.getvalue()

    return results

@app.local_entrypoint()
def main():
    out_dir = "results/comparison_10prompts"
    os.makedirs(out_dir, exist_ok=True)

    # Save prompt reference
    with open(os.path.join(out_dir, "prompts.json"), "w") as f:
        json.dump([{"id": i+1, "prompt": p, "seed": BASE_SEED + i * 17} for i, p in enumerate(PROMPTS)], f, indent=2)

    with open(os.path.join(out_dir, "prompts.txt"), "w") as f:
        for i, p in enumerate(PROMPTS):
            f.write(f"Prompt {i+1} (seed={BASE_SEED + i * 17}):\n{p}\n\n")

    print(f"Loaded 10 test prompts. Output target: {out_dir}")

    print("\n>>> Launching LoRA inference on Modal...")
    lora_results = generate_lora.remote()
    for filename, raw_bytes in lora_results.items():
        with open(os.path.join(out_dir, filename), "wb") as f:
            f.write(raw_bytes)
        print(f"Saved: {filename}")

    print("\n>>> Launching DoRA inference on Modal...")
    dora_results = generate_dora.remote()
    for filename, raw_bytes in dora_results.items():
        with open(os.path.join(out_dir, filename), "wb") as f:
            f.write(raw_bytes)
        print(f"Saved: {filename}")

    print("\n>>> Launching LyCORIS inference on Modal...")
    lyco_results = generate_lycoris.remote()
    for filename, raw_bytes in lyco_results.items():
        with open(os.path.join(out_dir, filename), "wb") as f:
            f.write(raw_bytes)
        print(f"Saved: {filename}")

    print("\nAll 30 images generated and saved successfully!")
