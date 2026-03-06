import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Path relative to THIS script file, not the working directory
SEG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_clean", "segmentations")

if not os.path.exists(SEG_PATH):
    raise FileNotFoundError(f"Segmentation path not found: {os.path.abspath(SEG_PATH)}")

unique_colors = set()
files = [f for f in os.listdir(SEG_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
print(f"Found {len(files)} segmentation files in:\n{os.path.abspath(SEG_PATH)}\n")

for file in tqdm(files):
    img_path = os.path.join(SEG_PATH, file)
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    pixels = img.reshape(-1, 3)
    colors = np.unique(pixels, axis=0)
    for color in colors:
        unique_colors.add(tuple(color))

print("\nUnique Segmentation Classes (RGB):\n")
for i, color in enumerate(sorted(unique_colors)):
    print(f"Class {i}: {color}")

print("\nTotal number of classes:", len(unique_colors))