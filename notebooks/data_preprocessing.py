import os
import shutil

# ==========================
# PATHS
# ==========================

# Get the absolute path to the project root (one level up from this script)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RAW_BASE = os.path.join(PROJECT_ROOT, "data", "Train")
IMG_DIR = os.path.join(RAW_BASE, "Image")
MASK_DIR = os.path.join(RAW_BASE, "Mask")
SEG_DIR = os.path.join(RAW_BASE, "Segmented")

CLEAN_BASE = os.path.join(PROJECT_ROOT, "data_clean")
CLEAN_IMG = os.path.join(CLEAN_BASE, "images")
CLEAN_MASK = os.path.join(CLEAN_BASE, "masks")
CLEAN_SEG = os.path.join(CLEAN_BASE, "segmentations")

os.makedirs(CLEAN_IMG, exist_ok=True)
os.makedirs(CLEAN_MASK, exist_ok=True)
os.makedirs(CLEAN_SEG, exist_ok=True)

# ==========================
# CLEANING PROCESS
# ==========================

images = os.listdir(IMG_DIR)

valid_count = 0
skipped_count = 0

for img_name in images:
    
    base_name = os.path.splitext(img_name)[0]  # 189 from 189.jpg
    
    mask_name = img_name + ".png"              # 189.jpg.png
    seg_name = img_name + ".png"               # 189.jpg.png (as seen in the directory)
    
    img_path = os.path.join(IMG_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, mask_name)
    seg_path = os.path.join(SEG_DIR, seg_name)
    
    if os.path.exists(mask_path) and os.path.exists(seg_path):
        
        # Copy image
        shutil.copy(img_path, os.path.join(CLEAN_IMG, img_name))
        
        # Copy & rename mask
        shutil.copy(mask_path, os.path.join(CLEAN_MASK, base_name + ".png"))
        
        # Copy segmentation
        shutil.copy(seg_path, os.path.join(CLEAN_SEG, base_name + ".png"))
        
        valid_count += 1
    else:
        skipped_count += 1

print("Cleaning complete.")
print("Valid samples:", valid_count)
print("Skipped samples:", skipped_count)