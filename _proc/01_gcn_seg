# %% [markdown]
# # Newt Segmentation
# > This notebook is used to segment the newts in the Barhill dataset using the Grounded-SAM-2 model.

#%% [markdown]
# ---
# skip_showdoc: true
# ---

# %%
#|default_exp segmentation

# %% [code] 
#| eval: false
import os
if not os.path.exists("./data/barhill"):
    os.system("kaggle datasets download -d mshahoyi/barhills-processed --unzip -p ./data")

# %% [code] 
#| eval: false
try:
    import supervision
    os.chdir("gsam2")
except:
    os.system('git clone https://github.com/IDEA-Research/Grounded-SAM-2 gsam2')
    os.chdir("gsam2")
    os.system('pip install -q -e . -e grounding_dino')
    os.system('pip install -q supervision')

# %% [code] 
#|eval: false
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from PIL import Image
import argparse
import os
import cv2
import json
import torch
import pandas as pd
import pathlib
import shutil
from datetime import datetime
import subprocess
import tempfile
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import pandas as pd

# %% [code]
GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/test_sam2.1")
DUMP_JSON_RESULTS = True

# %% [code] 
# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# %% [code] 
os.chdir("checkpoints")
os.system("bash download_ckpts.sh")
os.chdir("..")

# %% [code] 
# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# %% [code] 
# build grounding dino from huggingface
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

# %% [code] 
# Go back to parent directory to access data
os.chdir("..")

# %% [code] 

image = pathlib.Path("./data/barhill/GCNs/GCN10-P1-S2/IMG_2367.JPEG")

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = "the lizard."
img_path = image
text, img_path

# %% [code] 
image = Image.open(img_path)
image

# %% [code] 
sam2_predictor.set_image(np.array(image.convert("RGB")))

inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)

# %% [code] 
with torch.no_grad():
    outputs = grounding_model(**inputs)

# %% [code] 
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

# %% [code] 
# get the box prompt for SAM 2
input_boxes = results[0]["boxes"].cpu().numpy()

masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# %% [code] 
img = np.array(image)
img.shape

# %% [code] 
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

# %% [code] 

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

# %% [code] 
# --- Annotate Masks ---
# If CUSTOM_COLOR_MAP is a list of hex strings:
mask_annotator = sv.MaskAnnotator(color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP))
# NOTE: mask_annotator modifies the 'scene' in place if you pass the same array
# It's safer to annotate on a fresh copy if you want to keep the previous step clean
# However, the supervision docs often show chaining like this. Let's assume chaining:
annotated_frame_final = mask_annotator.annotate(scene=annotated_frame_labels.copy(), detections=detections) # Use copy to avoid modifying annotated_frame_labels


# --- Visualize final image (boxes + labels + masks) INLINE ---
print("\nDisplaying final image with boxes, labels, and masks:")
plt.figure(figsize=(15, 15)) # Adjust size as needed
plt.imshow(cv2.cvtColor(annotated_frame_final, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# %% [markdown] 
# # Segment the DS

# %% [code] 
ds_dir = pathlib.Path("./data/barhill")

# %% [code] 
gcns_dir = ds_dir/'GCNs'

# %% [code] 
# Load the metadata CSV
metadata_path = "./data/barhill/gallery_and_probes.csv"
metadata_df = pd.read_csv(metadata_path)
print(f"Loaded metadata with {len(metadata_df)} rows")

# Create output directory for visualization images - as sibling to data folder
vis_output_dir = pathlib.Path("./data/segmentation_visualizations")
vis_output_dir.mkdir(exist_ok=True)

# %%
# Initialize list to store RLE masks - we'll match by image filename
rle_masks = {}
visualization_samples = []  # Store some samples to display later

# Create directories for each newt ID in the visualization directory
for newt_id in tqdm(os.listdir(gcns_dir)):  # Process first 3 newts for demo
    newt_vis_dir = vis_output_dir / newt_id
    newt_vis_dir.mkdir(exist_ok=True)
    
    image_names = os.listdir(os.path.join(gcns_dir, newt_id))
    image_paths = [os.path.join(gcns_dir, newt_id, image) for image in image_names]
    
    # Process images one by one to avoid memory issues
    for image_path, image_name in zip(image_paths[:5], image_names[:5]):  # Limit to 5 per newt for demo
        try:
            # Create a key to match with CSV (assuming CSV has full path or we can construct it)
            image_key = f"{newt_id}/{image_name}"
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image {image_path}, skipping")
                rle_masks[image_key] = None
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
                rle_masks[image_key] = None
                continue
                
            # Get the highest confidence box
            confidence_scores = results["scores"]
            best_box_idx = torch.argmax(confidence_scores)
            box = results["boxes"][best_box_idx].cpu().numpy()
            
            # Convert box to SAM format
            sam_box = box.astype(int)
            
            # Generate mask with SAM
            sam2_predictor.set_image(np.array(pil_image.convert("RGB")))
            masks, _, _ = sam2_predictor.predict(
                box=sam_box,
                multimask_output=False
            )
            
            # Get the mask and convert to RLE
            mask = masks[0]  # Get the first (and only) mask
            
            # Convert mask to RLE format using pycocotools
            # Ensure mask is in the right format (uint8, Fortran order)
            mask_uint8 = mask.astype(np.uint8, order='F')
            rle = mask_util.encode(mask_uint8)
            # Convert bytes to string for CSV storage
            rle_string = rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else str(rle['counts'])
            rle_masks[image_key] = f"{rle['size'][0]}x{rle['size'][1]}:{rle_string}"
            
            # Create visualization image (similar to the code above the for loop)
            img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Create supervision detections object
            input_boxes = np.array([box])
            detections = sv.Detections(
                xyxy=input_boxes,  # (n, 4)
                mask=mask[np.newaxis, ...].astype(bool),  # (n, h, w)
                class_id=np.array([0])
            )
            
            # Create labels
            confidence = float(confidence_scores[best_box_idx])
            labels = [f"lizard {confidence:.2f}"]
            
            # Annotate image
            box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = box_annotator.annotate(scene=img_bgr.copy(), detections=detections)
            
            label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP), text_color=sv.Color.BLACK)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            mask_annotator = sv.MaskAnnotator(color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # Save visualization image
            vis_output_path = newt_vis_dir / f"{image_name}_segmented.jpg"
            cv2.imwrite(str(vis_output_path), annotated_frame)
            
            # Store sample for display (limit to first few)
            if len(visualization_samples) < 6:
                visualization_samples.append({
                    'original': np.array(pil_image),
                    'annotated': cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                    'newt_id': newt_id,
                    'filename': image_name,
                    'confidence': confidence
                })
            
            print(f"Processed {image_path} -> RLE mask saved, visualization at {vis_output_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            rle_masks[image_key] = None
            continue

print(f"\nVisualization images saved to: {vis_output_dir.absolute()}")


# %% [code] 
# Display sample visualizations
if visualization_samples:
    fig, axes = plt.subplots(len(visualization_samples), 2, figsize=(15, 5 * len(visualization_samples)))
    if len(visualization_samples) == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(visualization_samples):
        # Original image
        axes[i, 0].imshow(sample['original'])
        axes[i, 0].set_title(f"Original: {sample['newt_id']}/{sample['filename']}")
        axes[i, 0].axis('off')
        
        # Annotated image
        axes[i, 1].imshow(sample['annotated'])
        axes[i, 1].set_title(f"Segmented (conf: {sample['confidence']:.2f})")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
else:
    print("No visualization samples to display")

# %% [code] 
# Update the metadata CSV with RLE masks
def add_rle_to_metadata(metadata_df, rle_masks):
    """Add RLE mask column to metadata DataFrame"""
    
    # Initialize the new column
    metadata_df['segmentation_mask_rle'] = None
    
    # Try to match images - this depends on how paths are stored in your CSV
    for idx, row in metadata_df.iterrows():
        # Create image key from newt_id and image_name
        image_key = f"{row['newt_id']}/{row['image_name']}"
        
        # Set the RLE mask if we found a match
        if image_key in rle_masks:
            metadata_df.at[idx, 'segmentation_mask_rle'] = rle_masks[image_key]
    
    return metadata_df

# Update the metadata
metadata_df = add_rle_to_metadata(metadata_df, rle_masks)

# Save the updated CSV
updated_csv_path = "./data/barhill/gallery_and_probes_with_masks.csv"
metadata_df.to_csv(updated_csv_path, index=False)
print(f"Updated metadata saved to {updated_csv_path}")

# Show summary
mask_count = metadata_df['segmentation_mask_rle'].notna().sum()
print(f"Added segmentation masks for {mask_count} out of {len(metadata_df)} images")

# %% [code] 
#| export
def decode_rle_mask(rle_string):
    """Decode RLE string back to binary mask"""
    import pandas as pd
    import pycocotools.mask as mask_util
    
    if rle_string is None or pd.isna(rle_string):
        return None
    
    try:
        # Parse the RLE string format: "height x width : counts"
        # Use maxsplit=1 to only split on the first colon
        parts = rle_string.split(':', 1)
        if len(parts) != 2:
            print(f"Invalid RLE format: {rle_string[:100]}...")
            return None
            
        size_part, counts_part = parts
        height, width = map(int, size_part.split('x'))
        
        # Reconstruct RLE dict
        rle_dict = {
            'size': [height, width],
            'counts': counts_part.encode('utf-8')
        }
        
        # Decode using pycocotools
        mask = mask_util.decode(rle_dict)
        return mask
    except Exception as e:
        print(f"Error decoding RLE: {e}")
        print(f"RLE string preview: {rle_string[:100]}...")
        return None

# %% [code]
#| export
def visualize_segmentation(image_path, rle_string, mode='all', figsize=(15, 5)):
    """
    Visualize image with segmentation mask applied
    
    Args:
        image_path (str): Path to the image file
        rle_string (str): RLE-encoded segmentation mask
        mode (str): Visualization mode - 'all', 'side_by_side', 'mask_only', 'overlay', 'cutout'
        figsize (tuple): Figure size for matplotlib
    
    Returns:
        tuple: (original_image, decoded_mask, masked_image) as numpy arrays
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import cv2
    
    # Load image
    if isinstance(image_path, str):
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
    else:
        # Assume it's already a numpy array or PIL image
        if hasattr(image_path, 'convert'):  # PIL Image
            image = np.array(image_path)
        else:  # numpy array
            image = image_path
    
    # Decode mask
    mask = decode_rle_mask(rle_string)
    
    if mask is None:
        print("Could not decode RLE mask")
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title("Original Image (No mask available)")
        plt.axis('off')
        plt.show()
        return image, None, None
    
    # Create masked image (cutout)
    mask_bool = mask.astype(bool)
    masked_image = np.zeros_like(image)
    masked_image[mask_bool] = image[mask_bool]
    
    # Create overlay
    overlay_image = image.copy()
    if len(image.shape) == 3:  # Color image
        # Create colored mask overlay (semi-transparent)
        overlay_image[mask_bool] = overlay_image[mask_bool] * 0.7 + np.array([255, 0, 0]) * 0.3
    
    # Visualization based on mode
    if mode == 'all':
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Segmentation Mask")
        axes[1].axis('off')
        
        axes[2].imshow(overlay_image)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        axes[3].imshow(masked_image)
        axes[3].set_title("Segmented Cutout")
        axes[3].axis('off')
        
        plt.tight_layout()
        
    elif mode == 'side_by_side':
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(masked_image)
        axes[1].set_title("Segmented")
        axes[1].axis('off')
        
        plt.tight_layout()
        
    elif mode == 'mask_only':
        plt.figure(figsize=(8, 6))
        plt.imshow(mask, cmap='gray')
        plt.title("Segmentation Mask")
        plt.axis('off')
        
    elif mode == 'overlay':
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay_image)
        plt.title("Image with Mask Overlay")
        plt.axis('off')
        
    elif mode == 'cutout':
        plt.figure(figsize=(8, 8))
        plt.imshow(masked_image)
        plt.title("Segmented Cutout")
        plt.axis('off')
    
    plt.show()
    
    return image, mask, masked_image

# %% [code]
#| export  
def visualize_segmentation_from_metadata(metadata_df, idx, base_path="", mode='all', figsize=(15, 5)):
    """
    Visualize segmentation for a specific row in metadata DataFrame
    
    Args:
        metadata_df (pd.DataFrame): Metadata DataFrame with image paths and RLE masks
        idx (int): Row index to visualize
        base_path (str): Base path to prepend to image paths if needed
        mode (str): Visualization mode - same as visualize_segmentation
        figsize (tuple): Figure size for matplotlib
    
    Returns:
        tuple: (original_image, decoded_mask, masked_image) as numpy arrays
    """
    import os
    
    row = metadata_df.iloc[idx]
    
    # Get image path
    image_path = None
    for col in ['image_path', 'path', 'file_path']:
        if col in row and pd.notna(row[col]):
            image_path = row[col]
            break
    
    if image_path is None:
        raise ValueError("No valid image path found in metadata row")
    
    # Construct full path
    full_path = os.path.join(base_path, image_path) if base_path else image_path
    
    # Get RLE string
    rle_string = row.get('segmentation_mask_rle', None)
    
    if pd.isna(rle_string):
        print(f"No segmentation mask available for row {idx}")
        rle_string = None
    
    print(f"Visualizing: {image_path}")
    if rle_string:
        print(f"RLE mask available: {len(str(rle_string))} characters")
    
    return visualize_segmentation(full_path, rle_string, mode=mode, figsize=figsize)


# %% [code]
# Example usage of the exported functions:
# Count number of masks
mask_count = metadata_df['segmentation_mask_rle'].notna().sum()
print(f"Found {mask_count} segmentation masks in metadata")

# Get first available mask if any exist
sample_rle = metadata_df['segmentation_mask_rle'].dropna().iloc[0] if mask_count > 0 else None
if sample_rle:
    # Find the corresponding image path
    sample_idx = metadata_df['segmentation_mask_rle'].dropna().index[0]
    
    # Debug: show the RLE string format
    print(f"Sample RLE string: {sample_rle[:100]}...")
    
    # Test decode function directly
    test_mask = decode_rle_mask(sample_rle)
    if test_mask is not None:
        print(f"Successfully decoded mask with shape: {test_mask.shape}")
        
        # Now try the visualization function
        print("Using exported visualization function:")
        try:
            image, mask, masked = visualize_segmentation_from_metadata(metadata_df, sample_idx, base_path="./data", mode='side_by_side')
            print(f"Returned image shape: {image.shape}")
            print(f"Returned mask shape: {mask.shape}")
            print(f"Returned masked image shape: {masked.shape}")
        except Exception as e:
            print(f"Visualization error: {e}")
    else:
        print("Failed to decode mask")
else:
    print("No segmentation masks found in metadata")

# %% [code] 
# Show directory structure
print("Directory structure:")
print(f"Data directory: {ds_dir.absolute()}")
print(f"Visualizations directory: {vis_output_dir.absolute()}")
print(f"Total visualization subdirectories created: {len(list(vis_output_dir.iterdir()))}")

# %% [markdown]
# # Create New Kaggle Dataset

# %%
# Create dataset directory under data/
dataset_name = "barhill-segmented"
dataset_dir = pathlib.Path(f"./data/{dataset_name}")
dataset_dir.mkdir(exist_ok=True)

print(f"Creating new dataset: {dataset_name}")

# Create subdirectories
original_images_dir = dataset_dir / "original_images"
sample_visualizations_dir = dataset_dir / "sample_segmentations" 
original_images_dir.mkdir(exist_ok=True)
sample_visualizations_dir.mkdir(exist_ok=True)

print("Created dataset subdirectories")

# %% [code]
# Copy original images (newt directories directly under original_images, not under GCNs)
print("Copying original images...")
shutil.copytree(gcns_dir, original_images_dir, dirs_exist_ok=True)
print(f"Copied {len(list(original_images_dir.iterdir()))} newt directories")

# Copy a sample of visualization images (limit to avoid large dataset)
print("Copying sample visualizations...")
vis_count = 0
max_vis_samples = 50  # Limit to 50 visualization images

for newt_dir in vis_output_dir.iterdir():
    if newt_dir.is_dir() and vis_count < max_vis_samples:
        sample_newt_dir = sample_visualizations_dir / newt_dir.name
        sample_newt_dir.mkdir(exist_ok=True)
        
        # Copy up to 5 images per newt
        vis_images = list(newt_dir.glob("*_segmented.jpg"))
        for img_path in vis_images[:5]:
            if vis_count < max_vis_samples:
                shutil.copy2(img_path, sample_newt_dir / img_path.name)
                vis_count += 1
            else:
                break

print(f"Copied {vis_count} visualization sample images")

# %% [code]
# Fix metadata and save
print("Fixing metadata paths and removing unnecessary columns...")

# Remove the Unnamed: 0 column if it exists
if 'Unnamed: 0' in metadata_df.columns:
    metadata_df = metadata_df.drop('Unnamed: 0', axis=1)
    metadata_df = metadata_df.drop('is_probe', axis=1)
    print("Removed 'Unnamed: 0' and 'is_probe' columns")

# Update image paths in metadata to match new structure
def fix_image_paths(metadata_df):
    """Fix image paths to point to correct locations in new dataset"""
    
    # Create a mapping of old paths to new paths
    path_mapping = {}
    
    for idx, row in metadata_df.iterrows():
        # Try different possible path column names
        old_path = None
        if 'image_path' in metadata_df.columns:
            old_path = row['image_path']
        elif 'path' in metadata_df.columns:
            old_path = row['path']
        elif 'file_path' in metadata_df.columns:
            old_path = row['file_path']
        
        if old_path and isinstance(old_path, str):
            # Extract newt_id and filename from old path
            # Old path might be like: "barhill/GCNs/GCN10-P1-S2/IMG_2367.JPEG"
            path_parts = pathlib.Path(old_path).parts
            
            # Find GCNs in the path and get the parts after it
            try:
                gcns_idx = path_parts.index('GCNs')
                newt_id = path_parts[gcns_idx + 1]
                filename = path_parts[-1]
                
                # New path format: "original_images/GCN10-P1-S2/IMG_2367.JPEG"
                new_path = f"original_images/{newt_id}/{filename}"
                
                # Update the metadata
                if 'image_path' in metadata_df.columns:
                    metadata_df.at[idx, 'image_path'] = new_path
                elif 'path' in metadata_df.columns:
                    metadata_df.at[idx, 'path'] = new_path
                elif 'file_path' in metadata_df.columns:
                    metadata_df.at[idx, 'file_path'] = new_path
                    
                path_mapping[old_path] = new_path
                
            except (ValueError, IndexError):
                print(f"Could not parse path: {old_path}")
                continue
    
    return metadata_df, path_mapping

metadata_df, path_mapping = fix_image_paths(metadata_df)
print(f"Updated {len(path_mapping)} image paths")

# Save the updated metadata CSV
metadata_dest = dataset_dir / "metadata.csv"
metadata_df.to_csv(metadata_dest, index=False)
print(f"Saved metadata CSV to: {metadata_dest}")

# %% [code]
# Verify image paths exist (assertions)
print("Verifying image paths...")
missing_files = []
verified_count = 0

for idx, row in metadata_df.iterrows():
    # Get the image path from the row
    image_path = None
    if 'image_path' in metadata_df.columns:
        image_path = row['image_path']
    elif 'path' in metadata_df.columns:
        image_path = row['path']
    elif 'file_path' in metadata_df.columns:
        image_path = row['file_path']
    
    if image_path and isinstance(image_path, str):
        full_path = dataset_dir / image_path
        if full_path.exists():
            verified_count += 1
        else:
            missing_files.append(str(full_path))

print(f"Verified {verified_count} out of {len(metadata_df)} image paths")

if missing_files:
    print(f"WARNING: {len(missing_files)} files are missing!")
    for missing in missing_files[:10]:  # Show first 10 missing files
        print(f"  Missing: {missing}")
    if len(missing_files) > 10:
        print(f"  ... and {len(missing_files) - 10} more")
else:
    print("✅ All image paths verified successfully!")

# Assert that we have a reasonable number of valid paths
assert verified_count > 0, "No valid image paths found!"
assert verified_count / len(metadata_df) > 0.5, f"Too many missing files: {len(missing_files)} missing out of {len(metadata_df)}"

print("Path verification passed!")

# %% [code]
# Create dataset info file
dataset_info = {
    "title": "Barhill Great Crested Newts - Segmented Dataset",
    "description": """
This dataset contains Great Crested Newt images from the Barhill dataset with added segmentation masks.

Contents:
- original_images/: Original newt images organized by individual ID directories
- sample_segmentations/: Sample images showing segmentation results with bounding boxes and masks
- metadata.csv: Complete metadata with RLE-encoded segmentation masks

The segmentation masks were generated using Grounded-SAM-2 model to detect and segment newts.
Each row in metadata.csv contains an 'segmentation_mask_rle' column with RLE-encoded binary masks.

Use the decode_rle_mask() function from the original notebook to convert RLE strings back to binary masks.
""",
    "created": datetime.now().isoformat(),
    "source": "Generated from Barhill dataset using Grounded-SAM-2 segmentation"
}

# Write dataset info
with open(dataset_dir / "dataset-info.json", "w") as f:
    json.dump(dataset_info, f, indent=2)

print(f"Created dataset info file")

# %% [code]
# Create README for the dataset
readme_content = f"""# Barhill Great Crested Newts - Segmented Dataset

This dataset contains Great Crested Newt images with automatically generated segmentation masks.

## Contents

### original_images/
Original newt images organized by individual newt ID directories.
Each directory contains images for one individual newt.

### sample_segmentations/
Sample images showing the segmentation results with:
- Bounding boxes around detected newts
- Confidence scores
- Segmentation masks overlaid

### metadata.csv
Complete metadata file with the following key columns:
- `segmentation_mask_rle`: RLE-encoded binary segmentation masks
- `image_path`: Path to the original image within this dataset
- Other original metadata columns from the source dataset

## Usage

To decode the RLE masks back to binary format:

```python
import pycocotools.mask as mask_util
import pandas as pd

def decode_rle_mask(rle_string):
    if pd.isna(rle_string):
        return None
    
    size_part, counts_part = rle_string.split(':')
    height, width = map(int, size_part.split('x'))
    
    rle_dict = {{
        'size': [height, width],
        'counts': counts_part.encode('utf-8')
    }}
    
    return mask_util.decode(rle_dict)

# Load metadata and decode a mask
df = pd.read_csv('metadata.csv')
mask = decode_rle_mask(df['segmentation_mask_rle'].iloc[0])
```

## Generation Details

- **Model**: Grounded-SAM-2 (IDEA-Research/grounding-dino-tiny + SAM2.1)
- **Text prompt**: "the lizard."
- **Detection threshold**: 0.4 box threshold, 0.3 text threshold
- **Segmentation**: SAM2 with highest confidence detection box as prompt

## Dataset Statistics

- Total images: {len(metadata_df)} 
- Images with segmentation masks: {mask_count}
- Sample visualizations included: {vis_count}
- Verified image paths: {verified_count}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(dataset_dir / "README.md", "w") as f:
    f.write(readme_content)

print("Created README.md")

# %% [code]
# Print dataset summary
print("\n" + "="*50)
print("DATASET CREATION COMPLETE")
print("="*50)
print(f"Dataset name: {dataset_name}")
print(f"Dataset directory: {dataset_dir.absolute()}")

total_size_mb = sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file()) / (1024*1024)
print(f"Total size: {total_size_mb:.1f} MB")

print("\nDataset contents:")
for item in dataset_dir.iterdir():
    if item.is_dir():
        file_count = len(list(item.rglob('*')))
        dir_size_mb = sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024*1024)
        print(f"  📁 {item.name}/: {file_count} files ({dir_size_mb:.1f} MB)")
    else:
        size_mb = item.stat().st_size / (1024*1024)
        print(f"  📄 {item.name}: {size_mb:.2f} MB")

# Upload to Kaggle
print("\n" + "="*50)
print("UPLOADING TO KAGGLE")
print("="*50)


# %%
try:
    # Create dataset metadata for Kaggle
    kaggle_metadata = {
        "title": "Barhill Great Crested Newts - Segmented Dataset",
        "id": "mshahoyi/barhill-newts-segmented",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["biology", "computer-vision", "segmentation", "animals", "newts"]
    }
    
    # Write Kaggle dataset metadata
    with open(dataset_dir / "dataset-metadata.json", "w") as f:
        json.dump(kaggle_metadata, f, indent=2)
    
    # Change to dataset directory and upload
    original_cwd = os.getcwd()
    os.chdir(dataset_dir)
    
    print("Checking if dataset exists...")
    
    # Check if dataset exists
    check_result = subprocess.run([
        "kaggle", "datasets", "list", 
        "--user", "mshahoyi",
        "--search", "barhill-newts-segmented"
    ], capture_output=True, text=True)
    
    dataset_exists = "barhill-newts-segmented" in check_result.stdout
    
    print(f"Current directory: {os.getcwd()}")
    print(f"Files to upload: {list(pathlib.Path('.').iterdir())}")
    
    if dataset_exists:
        print("Dataset exists, updating...")
        result = subprocess.run([
            "kaggle", "datasets", "version",
            "-p", ".",
            "-m", "Updated with segmentation masks",
            "--dir-mode", "zip"
        ], capture_output=True, text=True, check=True)
        print("✅ Dataset updated successfully!")
        print("Output:", result.stdout)
    else:
        print("Dataset does not exist, creating new...")
        result = subprocess.run([
            "kaggle", "datasets", "create",
            "-p", ".",
            "--dir-mode", "zip"
        ], capture_output=True, text=True, check=True)
        print("✅ Dataset created successfully!")
        print("Output:", result.stdout)
    
    # Change back to original directory
    os.chdir(original_cwd)
    
except Exception as e:
    print(f"❌ Error during upload: {str(e)}")
    print("You may need to upload manually using:")
    print(f"cd {dataset_dir}")
    print("kaggle datasets create/version -p . --dir-mode zip")

print(f"\nDataset available at: https://www.kaggle.com/datasets/mshahoyi/barhill-newts-segmented")# %%

# %%
#| hide
import nbdev; nbdev.nbdev_export()
