# %% [markdown]
# # GCN Segmentation
#> Segmenting the GCNs from the Barhill dataset

# %%
#| default_exp seg
import os

if not os.path.exists('./data/barhill'):
    os.system('kaggle datasets download mshahoyi/barhills-processed --unzip -p ./data')
print('Data source import complete.')

# %%
if not os.path.exists('./gsam2'):
    os.system('git clone https://github.com/IDEA-Research/Grounded-SAM-2 gsam2')
    os.system('pip install -q supervision -e gsam2 -e gsam2/grounding_dino')

# %%
import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# %%
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "car. tire."
IMG_PATH = "notebooks/images/truck.jpg"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
OUTPUT_DIR = Path("outputs/test_sam2.1")
DUMP_JSON_RESULTS = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Commented out IPython magic to ensure Python compatibility.
# %cd checkpoints
!bash download_ckpts.sh
# %cd ..

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino from huggingface
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

import pathlib

image = pathlib.Path("/kaggle/input/barhills-processed/barhill/GCNs/GCN10-P1-S2/IMG_2367.JPEG")

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = "the lizard."
img_path = image
text, img_path

image = Image.open(img_path)
image

sam2_predictor.set_image(np.array(image.convert("RGB")))

inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

"""
Results is a list of dict with the following structure:
[
    {
        'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'),
        'labels': ['car', 'tire', 'tire', 'tire'],
        'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                        [1392.4701,  554.4064, 1628.6133,  777.5872],
                        [ 436.1182,  621.8940,  676.5255,  851.6897],
                        [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
    }
]
"""
results

# get the box prompt for SAM 2
input_boxes = results[0]["boxes"].cpu().numpy()

masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

img = np.array(image)
img.shape

import matplotlib.pyplot as plt

BACKGROUND_COLOR = (0, 0, 0) # Black
mask = masks[0]
current_mask = mask.astype(bool)

# Create a transparent background image (same size as original)
# Initialize with all zeros (including alpha channel = 0 for full transparency)
cutout_bgr = np.full_like(img, BACKGROUND_COLOR, dtype=np.uint8)

# Copy pixels from the original BGRA image to the cutout image
# only where the mask is True. This copies B, G, R, and A channels.
# Where the mask is True, the alpha channel from img_bgra (usually 255)
# will make the object opaque. Where the mask is False, the alpha
# channel remains 0 from the initialization, making it transparent.
cutout_bgr[current_mask] = img[current_mask]
plt.imshow(cutout_bgr)

# --- Optional: Cropping to the bounding box of the mask ---
# This creates smaller images containing only the object, reducing file size.
# Find coordinates where the mask is True
true_points = np.argwhere(current_mask)
if true_points.size > 0: # Check if the mask is not empty
    # Find min/max row and column indices
    y_min, x_min = true_points.min(axis=0)
    y_max, x_max = true_points.max(axis=0)

    # Crop the cutout image (add 1 to max indices for slicing)
    cropped_cutout = cutout_bgr[y_min:y_max+1, x_min:x_max+1]
else:
    # Handle empty mask case if necessary (e.g., skip saving)
    print(f"Warning: Mask {i} is empty, skipping cutout saving.")

plt.imshow(cropped_cutout)

# Commented out IPython magic to ensure Python compatibility.
# %cd /kaggle/working
cv2.imwrite("test.jpg", cv2.cvtColor(cropped_cutout, cv2.COLOR_BGR2RGB)) # Saves the cropped version
# print(f"Saved cutout to: {output_path}") # Optional: print each save

"""
Post-process the output of the model to get the masks, scores, and logits for visualization
"""
# convert the shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)


confidences = results[0]["scores"].cpu().numpy().tolist()
class_names = results[0]["labels"]
class_ids = np.array(list(range(len(class_names))))

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]
confidences, class_names, class_ids, labels

import matplotlib.pyplot as plt

"""
Visualize image with supervision useful API
"""
img = cv2.imread(img_path)
detections = sv.Detections(
    xyxy=input_boxes,  # (n, 4)
    mask=masks.astype(bool),  # (n, h, w)
    class_id=class_ids
)

"""
Note that if you want to use default color map,
you can set color=ColorPalette.DEFAULT
"""

# --- Annotate Boxes ---
# Note that if you want to use default color map,
# you can set color=sv.ColorPalette.DEFAULT
# If CUSTOM_COLOR_MAP is a list of hex strings:
box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame_boxes = box_annotator.annotate(scene=img.copy(), detections=detections)

# --- Annotate Labels ---
# If CUSTOM_COLOR_MAP is a list of hex strings:
label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP), text_color=sv.Color.BLACK) # Specify text color if needed
# Create labels if not already defined (example)
# labels = [f"{class_names[i]}: {scores[i]:0.2f}" for i in range(len(class_names))]
annotated_frame_labels = label_annotator.annotate(scene=annotated_frame_boxes.copy(), detections=detections, labels=labels)

# --- Save intermediate image (optional) ---
cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame_labels)

# --- Visualize intermediate image (boxes + labels) INLINE ---
print("Displaying image with boxes and labels:")
plt.figure(figsize=(10, 10)) # Adjust size as needed
plt.imshow(cv2.cvtColor(annotated_frame_labels, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# --- Annotate Masks ---
# If CUSTOM_COLOR_MAP is a list of hex strings:
mask_annotator = sv.MaskAnnotator(color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP))
# NOTE: mask_annotator modifies the 'scene' in place if you pass the same array
# It's safer to annotate on a fresh copy if you want to keep the previous step clean
# However, the supervision docs often show chaining like this. Let's assume chaining:
annotated_frame_final = mask_annotator.annotate(scene=annotated_frame_labels.copy(), detections=detections) # Use copy to avoid modifying annotated_frame_labels

# --- Save final image ---
cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame_final)

# --- Visualize final image (boxes + labels + masks) INLINE ---
print("\nDisplaying final image with boxes, labels, and masks:")
plt.figure(figsize=(15, 15)) # Adjust size as needed
plt.imshow(cv2.cvtColor(annotated_frame_final, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

"""# Segment the DS"""

ds_dir = pathlib.Path("/kaggle/input/barhills-processed/barhill")
!cp -R {ds_dir} .

ds_dir = pathlib.Path('/kaggle/working/barhill')
gcns_dir = ds_dir/'GCNs'
cropped_dir = ds_dir/'gcns-cropped'
cropped_dir.mkdir(exist_ok=True)

from tqdm import tqdm

# Create directories for each newt ID in the cropped directory
for newt_id in tqdm(os.listdir(gcns_dir)):
    newt_cropped_dir = os.path.join(cropped_dir, newt_id)
    os.makedirs(newt_cropped_dir, exist_ok=True)

    image_names = os.listdir(os.path.join(gcns_dir, newt_id))
    image_paths = [os.path.join(gcns_dir, newt_id, image) for image in image_names]
    images_cropped = []

    # Process images one by one to avoid memory issues
    for image_path, image_name in zip(image_paths, image_names):
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image {image_path}, skipping")
                continue

            # Convert to PIL for grounding model
            pil_image = Image.open(image_path)

            # Process single image with grounding model
            inputs = processor(images=pil_image, text=text, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[(image.shape[0], image.shape[1])]
            )[0]  # Get first (and only) result

            # Skip if no detections
            if len(results["boxes"]) == 0:
                print(f"No detections for {image_path}, skipping")
                continue

            # Get the highest confidence box
            confidence_scores = results["scores"]
            best_box_idx = torch.argmax(confidence_scores)
            box = results["boxes"][best_box_idx].cpu().numpy()

            # Convert box to SAM format
            sam_box = box.astype(int)

            # Generate mask with SAM
            sam2_predictor.set_image(pil_image.convert("RGB"))
            masks, _, _ = sam2_predictor.predict(
                box=sam_box,
                multimask_output=False
            )

            # Apply mask to image
            mask = masks[0]  # Get the first (and only) mask

            BACKGROUND_COLOR = (0, 0, 0) # Black
            current_mask = mask.astype(bool)

            cutout_bgr = np.full_like(image, BACKGROUND_COLOR, dtype=np.uint8)
            cutout_bgr[current_mask] = image[current_mask]

            true_points = np.argwhere(current_mask)
            if true_points.size > 0: # Check if the mask is not empty
                y_min, x_min = true_points.min(axis=0)
                y_max, x_max = true_points.max(axis=0)
                cropped_cutout = cutout_bgr[y_min:y_max+1, x_min:x_max+1]
            else:
                # Handle empty mask case if necessary (e.g., skip saving)
                print(f"Warning: Mask {i} is empty, skipping cutout saving.")

            # Create transparent background
            rgba = cv2.cvtColor(cropped_cutout, cv2.COLOR_BGR2RGB)
            images_cropped.append((pil_image.convert("RGB"), rgba))

            # Save the masked image
            output_path = os.path.join(newt_cropped_dir, image_name)
            cv2.imwrite(output_path, cv2.cvtColor(rgba, cv2.COLOR_BGR2RGB))
            print(f"processing {image_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    for i, (im, cropped) in enumerate(images_cropped):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(im)
        plt.subplot(1, 2, 2)
        plt.imshow(cropped)
    plt.show()

!rm -rf gsam2
!rm test.jpg

