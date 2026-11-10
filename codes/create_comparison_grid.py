#!/usr/bin/env python3
"""
create_comparison_grid.py
Generates a clean, polished comparison grid without overlapping text.
"""

import os
import csv
from PIL import Image, ImageDraw, ImageFont

def get_font(font_path, size, fallback="DejaVuSans-Bold.ttf"):
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        try:
            return ImageFont.truetype(f"/usr/share/fonts/TTF/{fallback}", size)
        except Exception:
            return ImageFont.load_default()

def main():
    root_dir = "/home/black/Projects/LORA_training"
    img_dir = os.path.join(root_dir, "results/comparison_10prompts")
    csv_path = os.path.join(root_dir, "results/metrics_comparison.csv")
    output_path = os.path.join(root_dir, "results/comparison_10prompts_grid.png")

    # Load metrics from CSV
    metrics = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p_id = int(row["prompt_id"])
                model = row["model"].strip().lower()
                metrics[(p_id, model)] = {
                    "clip_t": float(row["clip_t_prompt_alignment"]),
                    "clip_i": float(row["clip_i_identity_fidelity"])
                }

    models = [
        {
            "key": "lora",
            "name": "Standard LoRA",
            "spec": "Rank 4  •  1,000 Steps  •  43.1 MB",
            "badge": "Avg Face: 0.667 | Prompt: 36.3",
            "color": (59, 130, 246)   # Blue
        },
        {
            "key": "dora",
            "name": "DoRA (Winner)",
            "spec": "Rank 4  •  600 Steps  •  49.2 MB",
            "badge": "Avg Face: 0.742 (Best) | Prompt: 36.0",
            "color": (16, 185, 129)  # Green
        },
        {
            "key": "lycoris",
            "name": "LyCORIS (LoKr)",
            "spec": "Rank 4  •  600 Steps  •  9.5 MB",
            "badge": "Avg Face: 0.615 | Prompt: 34.8",
            "color": (245, 158, 11)  # Amber
        }
    ]

    prompts = [
        {"id": 1, "title": "01. Hollow Purple", "desc": "Combat Pose & Violet Energy"},
        {"id": 2, "title": "02. Rainy Cafe", "desc": "Sweater, Coffee & Ambient Rain"},
        {"id": 3, "title": "03. Cyberpunk", "desc": "Neo-Tokyo, Neon Lights & Blindfold"},
        {"id": 4, "title": "04. Kyoto Shrine", "desc": "Traditional Yukata & Sakura"},
        {"id": 5, "title": "05. Wild West", "desc": "Cowboy Hat, Duster & Blue Eyes"},
        {"id": 6, "title": "06. Snowy Peak", "desc": "Blizzard, Frost & Spiky White Hair"},
        {"id": 7, "title": "07. Streetwear", "desc": "Modern Fashion & Cargo Pants"},
        {"id": 8, "title": "08. Cathedral", "desc": "Gothic Windows & Moody Light"},
        {"id": 9, "title": "09. Beach Vacation", "desc": "Hawaiian Shirt & Sunglasses"},
        {"id": 10, "title": "10. Knight Armor", "desc": "Heavy Silver Plate & Sword"}
    ]

    # Clean Spacing Layout
    margin = 32
    row_label_w = 260
    gap = 16
    cell_w = 420
    cell_h = 420

    master_banner_h = 110
    col_header_h = 80
    banner_to_col_gap = 16
    col_to_grid_gap = 16

    total_w = margin + row_label_w + gap + (3 * cell_w) + (2 * gap) + margin
    # Height calculation:
    grid_y_start = margin + master_banner_h + banner_to_col_gap + col_header_h + col_to_grid_gap
    total_h = grid_y_start + (len(prompts) * cell_h) + ((len(prompts) - 1) * gap) + margin

    # Colors
    bg_color = (15, 18, 26)
    panel_bg = (24, 28, 40)
    card_border = (45, 52, 72)
    text_white = (245, 247, 250)
    text_muted = (156, 163, 175)
    accent_purple = (168, 85, 247)

    # Fonts
    font_bold = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
    font_reg = "/usr/share/fonts/TTF/DejaVuSans.ttf"
    font_mono = "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf"

    f_title = get_font(font_bold, 26)
    f_subtitle = get_font(font_reg, 15)
    f_col_title = get_font(font_bold, 18)
    f_col_spec = get_font(font_reg, 12)
    f_col_badge = get_font(font_mono, 11)
    f_row_title = get_font(font_bold, 16)
    f_row_desc = get_font(font_reg, 13)
    f_score_badge = get_font(font_mono, 13)

    canvas = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(canvas, "RGBA")

    # ==================== 1. MASTER TOP BANNER ====================
    banner_box = [margin, margin, total_w - margin, margin + master_banner_h]
    draw.rounded_rectangle(banner_box, radius=12, fill=panel_bg, outline=card_border, width=2)

    # Title & Subtitle cleanly separated
    draw.text((margin + 28, margin + 22), "FLUX.1-dev Architecture Comparison: LoRA vs DoRA vs LyCORIS", fill=text_white, font=f_title)
    draw.text(
        (margin + 28, margin + 62),
        "Stress Test at Rank 4  |  Subject: Satoru Gojo  |  Evaluated on 10 Prompts with Computer Vision Metrics",
        fill=accent_purple,
        font=f_subtitle
    )

    # ==================== 2. COLUMN HEADERS ====================
    col_header_y = margin + master_banner_h + banner_to_col_gap

    # Left Column Header (Prompts)
    left_col_box = [margin, col_header_y, margin + row_label_w, col_header_y + col_header_h]
    draw.rounded_rectangle(left_col_box, radius=8, fill=(30, 35, 50), outline=card_border, width=1)
    draw.text((margin + 16, col_header_y + 16), "Test Prompts", fill=text_white, font=f_col_title)
    draw.text((margin + 16, col_header_y + 44), "10 Standardized Scenes", fill=text_muted, font=f_col_spec)

    # Model Column Headers
    col_x_start = margin + row_label_w + gap
    for c_idx, model in enumerate(models):
        cx = col_x_start + c_idx * (cell_w + gap)
        col_box = [cx, col_header_y, cx + cell_w, col_header_y + col_header_h]
        draw.rounded_rectangle(col_box, radius=8, fill=(28, 33, 48), outline=model["color"], width=2)

        # 3 lines of clean, non-overlapping text inside each card
        draw.text((cx + 16, col_header_y + 12), model["name"], fill=text_white, font=f_col_title)
        draw.text((cx + 16, col_header_y + 36), model["spec"], fill=text_muted, font=f_col_spec)
        draw.text((cx + 16, col_header_y + 55), model["badge"], fill=model["color"], font=f_col_badge)

    # ==================== 3. 10 PROMPTS ROWS ====================
    for r_idx, prompt in enumerate(prompts):
        ry = grid_y_start + r_idx * (cell_h + gap)

        # Row Label Card (Left Side)
        row_box = [margin, ry, margin + row_label_w, ry + cell_h]
        draw.rounded_rectangle(row_box, radius=10, fill=panel_bg, outline=card_border, width=1)

        # Left purple indicator bar
        accent_bar = [margin + 6, ry + 12, margin + 10, ry + cell_h - 12]
        draw.rectangle(accent_bar, fill=accent_purple)

        # Row Text Content
        draw.text((margin + 20, ry + 24), prompt["title"], fill=text_white, font=f_row_title)
        draw.text((margin + 20, ry + 52), prompt["desc"], fill=accent_purple, font=f_row_desc)

        draw.text((margin + 20, ry + 100), f"Prompt #{prompt['id']}", fill=text_muted, font=f_col_spec)
        draw.text((margin + 20, ry + 120), "Resolution: 1024x1024", fill=text_muted, font=f_col_spec)
        draw.text((margin + 20, ry + 140), "Seed: Fixed per prompt", fill=text_muted, font=f_col_spec)

        # Mini legend inside row box
        mini_box = [margin + 16, ry + cell_h - 90, margin + row_label_w - 16, ry + cell_h - 18]
        draw.rounded_rectangle(mini_box, radius=6, fill=(18, 22, 32), outline=(40, 46, 62), width=1)
        draw.text((margin + 24, ry + cell_h - 80), "Metrics per image:", fill=text_muted, font=f_col_spec)
        draw.text((margin + 24, ry + cell_h - 60), "• Face: CLIP-I (0-1)", fill=(16, 185, 129), font=f_col_spec)
        draw.text((margin + 24, ry + cell_h - 42), "• Prompt: CLIP-T (0-100)", fill=(59, 130, 246), font=f_col_spec)

        # The 3 Images for this Prompt
        for c_idx, model in enumerate(models):
            cx = col_x_start + c_idx * (cell_w + gap)
            img_filename = f"prompt_{prompt['id']:02d}_{model['key']}.png"
            img_path = os.path.join(img_dir, img_filename)

            if os.path.exists(img_path):
                try:
                    cell_img = Image.open(img_path).convert("RGB")
                    cell_img = cell_img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                    canvas.paste(cell_img, (cx, ry))
                except Exception as e:
                    draw.rectangle([cx, ry, cx + cell_w, ry + cell_h], fill=(40, 20, 20))
                    draw.text((cx + 20, ry + 20), f"Error loading image:\n{e}", fill=(255, 100, 100), font=f_col_spec)
            else:
                draw.rectangle([cx, ry, cx + cell_w, ry + cell_h], fill=(30, 30, 30))
                draw.text((cx + 20, ry + 20), f"Missing: {img_filename}", fill=(255, 200, 100), font=f_col_spec)

            # Border around each image
            draw.rectangle([cx, ry, cx + cell_w, ry + cell_h], outline=card_border, width=2)

            # Metric Score Pill Overlay at the bottom
            metric_data = metrics.get((prompt['id'], model['key']))
            if metric_data:
                clip_i = metric_data['clip_i']
                clip_t = metric_data['clip_t']

                badge_h = 30
                badge_box = [cx + 8, ry + cell_h - badge_h - 8, cx + cell_w - 8, ry + cell_h - 8]
                draw.rounded_rectangle(badge_box, radius=6, fill=(10, 12, 18, 225), outline=(60, 70, 95, 240), width=1)

                dot_color = (16, 185, 129) if clip_i >= 0.72 else ((245, 158, 11) if clip_i >= 0.63 else (239, 68, 68))
                draw.ellipse([cx + 18, ry + cell_h - badge_h + 3, cx + 26, ry + cell_h - badge_h + 11], fill=dot_color)

                badge_str = f"Face: {clip_i:.3f}  |  Prompt: {clip_t:.1f}"
                draw.text((cx + 34, ry + cell_h - badge_h + 1), badge_str, fill=(255, 255, 255), font=f_score_badge)

    print(f"Saving newly formatted grid to {output_path}...")
    canvas.save(output_path, "PNG", optimize=True)
    print("New grid saved successfully!")

if __name__ == "__main__":
    main()
