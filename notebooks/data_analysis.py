import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# ==========================
# PATH CONFIGURATION
# ==========================

# Get the absolute path to the project root (one level up from this script)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BASE_PATH = os.path.join(PROJECT_ROOT, "data_clean")
TRAIN_IMG = os.path.join(BASE_PATH, "images")
TRAIN_MASK = os.path.join(BASE_PATH, "masks")
TRAIN_SEG = os.path.join(BASE_PATH, "segmentations")

# ==========================
# 1️⃣ Basic Dataset Stats
# ==========================

train_images = sorted(os.listdir(TRAIN_IMG))
train_masks = sorted(os.listdir(TRAIN_MASK))
train_seg = sorted(os.listdir(TRAIN_SEG))

print("Total Train Images:", len(train_images))
print("Total Train Masks:", len(train_masks))
print("Total Train Segmented:", len(train_seg))

# ==========================
# 2️⃣ Check Alignment
# ==========================

print("\nChecking filename alignment...")

mismatch = 0
for img, mask in zip(train_images, train_masks):
    if img != mask:
        mismatch += 1

print("Filename mismatches:", mismatch)

# ==========================
# 3️⃣ Image Resolution Analysis
# ==========================

print("\nChecking image resolutions...")

resolutions = []

for img_name in tqdm(train_images[:500]):  # check first 500
    img = Image.open(os.path.join(TRAIN_IMG, img_name))
    resolutions.append(img.size)

unique_res = set(resolutions)
print("Unique Resolutions Found:", unique_res)

# ==========================
# 4️⃣ Segmentation Color Analysis
# ==========================

print("\nAnalyzing segmentation colors...")

color_counter = Counter()

for seg_name in tqdm(train_seg[:300]):  # sample 300
    seg = np.array(Image.open(os.path.join(TRAIN_SEG, seg_name)))
    pixels = seg.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    
    for color in unique_colors:
        color_counter[tuple(color)] += 1

print("\nUnique Colors Found:")
for color in color_counter:
    print(color)

print("\nTotal Classes (by unique color count):", len(color_counter))

# ==========================
# 5️⃣ Class Pixel Distribution
# ==========================

print("\nComputing pixel distribution (sample)...")

pixel_distribution = Counter()

for seg_name in tqdm(train_seg[:200]):
    seg = np.array(Image.open(os.path.join(TRAIN_SEG, seg_name)))
    pixels = seg.reshape(-1, 3)
    
    for pixel in pixels:
        pixel_distribution[tuple(pixel)] += 1

print("\nTop 10 Most Frequent Colors:")
for color, count in pixel_distribution.most_common(10):
    print(color, ":", count)

# ==========================
# 6️⃣ Visual Inspection
# ==========================

sample_img = train_images[0]

img = Image.open(os.path.join(TRAIN_IMG, sample_img))
mask = Image.open(os.path.join(TRAIN_MASK, sample_img))
seg = Image.open(os.path.join(TRAIN_SEG, sample_img))

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Mask")
plt.imshow(mask)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Segmentation")
plt.imshow(seg)
plt.axis("off")

plt.show()

print("\nData analysis complete.")