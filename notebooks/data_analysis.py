import os
import numpy as np
from PIL import Image
from collections import Counter
import json

BASE_DIR = ".."

IMAGE_DIR = os.path.join(BASE_DIR, "images")
SEGM_DIR = os.path.join(BASE_DIR, "segm")

# -----------------------------
# 1. Count dataset size
# -----------------------------

images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
segm = [f for f in os.listdir(SEGM_DIR) if f.endswith(".png")]

print("Total images:", len(images))
print("Total segmentation masks:", len(segm))


# -----------------------------
# 2. Check filename alignment
# -----------------------------

image_names = set([os.path.splitext(f)[0] for f in images])
segm_names = set([os.path.splitext(f)[0] for f in segm])

missing_masks = image_names - segm_names
missing_images = segm_names - image_names

print("\nImages without segmentation:", len(missing_masks))
print("Segmentations without image:", len(missing_images))


# -----------------------------
# 3. Check image sizes
# -----------------------------

sizes = []

for img_name in images[:100]:
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = Image.open(img_path)
    sizes.append(img.size)

size_count = Counter(sizes)

print("\nImage size distribution:")
for size, count in size_count.items():
    print(size, ":", count)


# -----------------------------
# 4. Check segmentation labels
# -----------------------------

sample_mask = Image.open(os.path.join(SEGM_DIR, segm[0]))
mask_array = np.array(sample_mask)

unique_labels = np.unique(mask_array)

print("\nUnique labels in sample segmentation mask:")
print(unique_labels)


# -----------------------------
# 5. Explore captions
# -----------------------------

caption_path = os.path.join(BASE_DIR, "captions.json")

if os.path.exists(caption_path):

    with open(caption_path) as f:
        captions = json.load(f)

    print("\nTotal captions:", len(captions))

    print("\nExample caption:")
    print(list(captions.items())[0])