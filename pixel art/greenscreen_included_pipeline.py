"""
Neon-greenscreen variant of the pixel post-process (emo girl, etc.).

Reference (do not edit): test_bg_inclusion.py
  -> cyan key, whole-image K-means after flattening bg to one color.
  -> output: results/final_assets/blue_girl_seed_7_INCLUDED.png

Why a separate file:
  FLUX often puts green spill on pale skin. Those pixels sit near the screen
  color in RGB, so the cyan pipeline trick (flatten bg + K-means on full image)
  snaps skin into the background cluster. Greenscreen needs K-means trained only
  on character pixels, and background = components that touch the image border.
"""
import cv2
import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans


def _border_touching_bg_mask(raw_bg_mask: np.ndarray) -> np.ndarray:
    """Keep screen pixels in components that touch any image edge (true backdrop)."""
    num_labels, labels = cv2.connectedComponents(raw_bg_mask.astype(np.uint8), connectivity=4)
    h, w = labels.shape
    true_bg = np.zeros((h, w), dtype=bool)
    for label in range(1, num_labels):
        component = labels == label
        touches = (
            component[0, :].any()
            or component[-1, :].any()
            or component[:, 0].any()
            or component[:, -1].any()
        )
        if touches:
            true_bg |= component
    return true_bg


def greenscreen_included_pipeline(
    input_path,
    output_path,
    offset=2,
    colors=32,
):
    print(f"Applying greenscreen included pipeline to {input_path}...")

    img = Image.open(input_path).convert("RGB")

    shifted_img = ImageChops.offset(img, offset, offset)
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    final_img = ImageChops.offset(snapped_img, -offset, -offset)

    data = np.array(final_img)
    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]

    # Saturated neon green screen (not loose enough to eat green-tinted skin).
    raw_bg_mask = (g > 200) & (g > r + 55) & (g > b + 55)
    true_bg_mask = _border_touching_bg_mask(raw_bg_mask)
    char_mask = ~true_bg_mask

    flat_color = np.median(data[true_bg_mask], axis=0).astype(np.uint8)

    char_pixels = data[char_mask]
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(char_pixels)
    data[char_mask] = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
    data[true_bg_mask] = flat_color

    Image.fromarray(data.astype(np.uint8)).save(output_path, "PNG")
    print(f"Saved {output_path}")


def save_256_native(included_1024_path, out_256_path):
    img = Image.open(included_1024_path)
    img.resize((256, 256), Image.Resampling.NEAREST).save(out_256_path, "PNG")


def main():
    out_dir = "/home/black/Projects/LORA_training/results/emo_girl_pixel"
    os.makedirs(out_dir, exist_ok=True)
    jobs = [
        ("emo_girl_v3_seed_30.png", "emo_girl_v3_seed_30"),
        ("emo_girl_v4_seed_40.png", "emo_girl_v4_seed_40"),
    ]
    for src_name, stem in jobs:
        src = os.path.join(out_dir, src_name)
        out1024 = os.path.join(out_dir, f"{stem}_INCLUDED.png")
        out256 = os.path.join(out_dir, f"{stem}_256.png")
        greenscreen_included_pipeline(src, out1024)
        save_256_native(out1024, out256)


if __name__ == "__main__":
    main()
