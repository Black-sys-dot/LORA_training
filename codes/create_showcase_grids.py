#!/usr/bin/env python3
"""
create_showcase_grids.py
Generates two polished, presentation-ready image grids:
1. results/dataset_gojo_grid.png: 15 samples from the actual Gojo training dataset.
2. results/lora_showcase_grid.png: 12 favorite early high-fidelity LoRA generations.
"""

import os
from PIL import Image, ImageDraw, ImageFont

def get_font(font_path, size, fallback="DejaVuSans-Bold.ttf"):
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        try:
            return ImageFont.truetype(f"/usr/share/fonts/TTF/{fallback}", size)
        except Exception:
            return ImageFont.load_default()

def create_dataset_grid():
    root_dir = "/home/black/Projects/LORA_training"
    out_path = os.path.join(root_dir, "results/dataset_gojo_grid.png")
    dataset_dir = os.path.join(root_dir, "dataset/gojo")

    # Select 15 clean, diverse images from dataset
    selected_indices = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 18, 22, 25]
    img_files = [os.path.join(dataset_dir, f"gojo_{i:02d}.png") for i in selected_indices]

    cols = 5
    rows = 3
    cell_w = 320
    cell_h = 480  # 2:3 aspect ratio matching dataset 1024x1536
    gap = 16
    margin = 32
    header_h = 90

    total_w = margin * 2 + cols * cell_w + (cols - 1) * gap
    total_h = margin * 2 + header_h + 16 + rows * cell_h + (rows - 1) * gap

    bg_color = (15, 18, 26)
    panel_bg = (24, 28, 40)
    card_border = (45, 52, 72)
    text_white = (245, 247, 250)
    accent_purple = (168, 85, 247)

    f_title = get_font("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 24)
    f_subtitle = get_font("/usr/share/fonts/TTF/DejaVuSans.ttf", 14)

    canvas = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Header Card
    banner_box = [margin, margin, total_w - margin, margin + header_h]
    draw.rounded_rectangle(banner_box, radius=12, fill=panel_bg, outline=card_border, width=2)
    draw.text((margin + 24, margin + 20), "Ground-Truth Training Dataset Samples (Satoru Gojo)", fill=text_white, font=f_title)
    draw.text((margin + 24, margin + 54), "15 Reference Portraits from dataset/gojo/ (Curated 1024x1536 Source Artwork)", fill=accent_purple, font=f_subtitle)

    # Place Images
    grid_y = margin + header_h + 16
    for idx, path in enumerate(img_files):
        r = idx // cols
        c = idx % cols
        x = margin + c * (cell_w + gap)
        y = grid_y + r * (cell_h + gap)

        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            img = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            canvas.paste(img, (x, y))
        else:
            draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(35, 35, 45))

        # Border
        draw.rectangle([x, y, x + cell_w, y + cell_h], outline=card_border, width=2)

    print(f"Saving dataset grid to {out_path}...")
    canvas.save(out_path, "PNG", optimize=True)
    print("Dataset grid created!")

def create_lora_showcase_grid():
    root_dir = "/home/black/Projects/LORA_training"
    out_path = os.path.join(root_dir, "results/lora_showcase_grid.png")

    items = [
        {"file": "results/comparison2/comparison2_4_lora.png", "title": "Hollow Purple Technique"},
        {"file": "results/comparison2/comparison2_3_lora.png", "title": "Crumbling Temple Levitation"},
        {"file": "results/comparison2/comparison2_2_lora.png", "title": "Rainy Tokyo Street (Hoodie)"},
        {"file": "results/comparison2/comparison2_1_lora.png", "title": "Zen Garden Tea Drinking"},
        {"file": "results/comparison/comparison_1_lora.png", "title": "Classic Close-Up Portrait"},
        {"file": "results/comparison/comparison_2_lora.png", "title": "Dynamic Combat Pose"},
        {"file": "results/comparison/test_crepe_s1000_lora.png", "title": "Eating Crepe in Harajuku"},
        {"file": "results/comparison/test_rubble_s1000_lora.png", "title": "Sitting on Destroyed Pillar"},
        {"file": "results/comparison/test_teaching_s1000_lora.png", "title": "Teaching at Jujutsu High"},
        {"file": "results/comparison/test_floating_s1000_lora.png", "title": "Floating Meditation in Sky"},
        {"file": "results/comparison/test_subway_s1000_lora.png", "title": "Subway Station Battle"},
        {"file": "results/comparison/comparison_3_lora.png", "title": "Cinematic Night Lighting"}
    ]

    cols = 4
    rows = 3
    cell_w = 420
    cell_h = 420  # Square 1024x1024 source
    gap = 16
    margin = 32
    header_h = 90

    total_w = margin * 2 + cols * cell_w + (cols - 1) * gap
    total_h = margin * 2 + header_h + 16 + rows * cell_h + (rows - 1) * gap

    bg_color = (15, 18, 26)
    panel_bg = (24, 28, 40)
    card_border = (45, 52, 72)
    text_white = (245, 247, 250)
    accent_purple = (168, 85, 247)
    pill_bg = (10, 12, 18, 225)

    f_title = get_font("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 24)
    f_subtitle = get_font("/usr/share/fonts/TTF/DejaVuSans.ttf", 14)
    f_badge = get_font("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 13)

    canvas = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Header Card
    banner_box = [margin, margin, total_w - margin, margin + header_h]
    draw.rounded_rectangle(banner_box, radius=12, fill=panel_bg, outline=card_border, width=2)
    draw.text((margin + 24, margin + 20), "Best Results We Got: FLUX.1-dev LoRA Showcase", fill=text_white, font=f_title)
    draw.text((margin + 24, margin + 54), "Standard LoRA (Rank 16 • 1,000 Steps)  |  Subject: Satoru Gojo (Jujutsu Kaisen)", fill=accent_purple, font=f_subtitle)

    # Place Images
    grid_y = margin + header_h + 16
    for idx, item in enumerate(items):
        r = idx // cols
        c = idx % cols
        x = margin + c * (cell_w + gap)
        y = grid_y + r * (cell_h + gap)

        full_path = os.path.join(root_dir, item["file"])
        if os.path.exists(full_path):
            img = Image.open(full_path).convert("RGB")
            img = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            canvas.paste(img, (x, y))
        else:
            draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(35, 35, 45))

        # Border
        draw.rectangle([x, y, x + cell_w, y + cell_h], outline=card_border, width=2)

        # Label Pill Overlay at Bottom
        pill_h = 30
        pill_box = [x + 8, y + cell_h - pill_h - 8, x + cell_w - 8, y + cell_h - 8]
        draw.rounded_rectangle(pill_box, radius=6, fill=pill_bg, outline=(60, 70, 95, 240), width=1)
        draw.text((x + 18, y + cell_h - pill_h), item["title"], fill=text_white, font=f_badge)

    print(f"Saving LoRA showcase grid to {out_path}...")
    canvas.save(out_path, "PNG", optimize=True)
    print("LoRA showcase grid created!")

if __name__ == "__main__":
    create_dataset_grid()
    create_lora_showcase_grid()
