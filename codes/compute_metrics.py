#!/usr/bin/env python3
"""Compute real mathematical evaluation metrics for generated Gojo images:
1. CLIP-T (Text-to-Image Prompt Alignment): Cosine similarity between prompt text and generated image.
2. CLIP-I (Subject Identity Preservation): Cosine similarity between generated image and ground-truth dataset images.

Evaluates: LoRA vs. DoRA vs. LyCORIS across 10 prompts from results/comparison_10prompts/
"""

import os
import sys
import json
import csv
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPARISON_DIR = Path("results/comparison_10prompts")
DATASET_DIR = Path("dataset/gojo")
OUTPUT_CSV = Path("results/metrics_comparison.csv")
OUTPUT_JSON = Path("results/metrics_summary.json")

def extract_embed(output) -> torch.Tensor:
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        t = output.pooler_output
    elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        t = output.last_hidden_state[:, 0, :]
    elif isinstance(output, torch.Tensor):
        t = output
    else:
        raise ValueError(f"Unknown output type: {type(output)}")
    return t / t.norm(p=2, dim=-1, keepdim=True)

def main():
    print(f"=== Computing Real Generative AI Metrics on Device: {DEVICE} ===", flush=True)

    # 1. Load CLIP Model
    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading {model_name}...", flush=True)
    model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # 2. Compute Reference Subject Embedding from Dataset (25 Gojo images)
    dataset_files = sorted(list(DATASET_DIR.glob("*.png")))
    print(f"Computing Reference Subject Embedding from {len(dataset_files)} dataset images...", flush=True)

    ref_embeds = []
    with torch.no_grad():
        for img_path in dataset_files:
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(DEVICE)
                raw_out = model.get_image_features(**inputs)
                feats = extract_embed(raw_out)
                ref_embeds.append(feats)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    ref_embed_matrix = torch.cat(ref_embeds, dim=0) # [25, D]
    mean_ref_embed = ref_embed_matrix.mean(dim=0, keepdim=True)
    mean_ref_embed = mean_ref_embed / mean_ref_embed.norm(p=2, dim=-1, keepdim=True)
    print(f"Subject Reference Embedding computed successfully across {len(ref_embeds)} dataset images!\n", flush=True)

    # 3. Load 10 Prompts
    prompts_file = COMPARISON_DIR / "prompts.json"
    with open(prompts_file, "r") as f:
        prompts_data = json.load(f)

    models = ["lora", "dora", "lycoris"]
    results_rows = []
    model_scores = {m: {"clip_t": [], "clip_i": []} for m in models}

    print("Evaluating Generated Images across Models & Prompts:\n" + "-"*75)

    for item in prompts_data:
        p_id = item["id"]
        prompt_text = item["prompt"]
        print(f"\n[Prompt {p_id:02d}]: {prompt_text[:60]}...")

        # Encode text prompt
        with torch.no_grad():
            text_inputs = processor(text=[prompt_text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            raw_text = model.get_text_features(**text_inputs)
            text_feats = extract_embed(raw_text)

        for m in models:
            filename = f"prompt_{p_id:02d}_{m}.png"
            img_path = COMPARISON_DIR / filename
            if not img_path.exists():
                print(f"  Missing {filename}, skipping...")
                continue

            with torch.no_grad():
                img = Image.open(img_path).convert("RGB")
                img_inputs = processor(images=img, return_tensors="pt").to(DEVICE)
                raw_img = model.get_image_features(**img_inputs)
                img_feats = extract_embed(raw_img)

                # Metric 1: CLIP-T (Text-Image Cosine Similarity * 100)
                clip_t = (img_feats * text_feats).sum().item() * 100

                # Metric 2: CLIP-I (Subject Identity Cosine Similarity, 0 to 1)
                clip_i = (img_feats * mean_ref_embed).sum().item()

            model_scores[m]["clip_t"].append(clip_t)
            model_scores[m]["clip_i"].append(clip_i)

            results_rows.append({
                "prompt_id": p_id,
                "model": m,
                "clip_t_prompt_alignment": round(clip_t, 3),
                "clip_i_identity_fidelity": round(clip_i, 4),
                "image_filename": filename,
            })

            print(f"  [{m.upper():7s}] CLIP-T (Alignment): {clip_t:.2f} | CLIP-I (Identity): {clip_i:.4f}")

    # 4. Save CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "model", "clip_t_prompt_alignment", "clip_i_identity_fidelity", "image_filename"])
        writer.writeheader()
        writer.writerows(results_rows)

    # 5. Compute Aggregates
    summary = {}
    print("\n" + "="*75)
    print("FINAL AGGREGATE EVALUATION RESULTS:")
    print("="*75)
    for m in models:
        t_scores = model_scores[m]["clip_t"]
        i_scores = model_scores[m]["clip_i"]
        t_mean = sum(t_scores) / len(t_scores) if t_scores else 0
        i_mean = sum(i_scores) / len(i_scores) if i_scores else 0

        summary[m] = {
            "mean_clip_t_alignment": round(t_mean, 2),
            "mean_clip_i_identity": round(i_mean, 4),
            "num_evaluated": len(t_scores)
        }
        print(f"Model: {m.upper():8s} | Mean CLIP-T (Alignment): {t_mean:.2f} | Mean CLIP-I (Identity): {i_mean:.4f}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetailed CSV written to: {OUTPUT_CSV}")
    print(f"Summary JSON written to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
