# %% [markdown]
# # Newt Segmentation
# > This notebook is used to segment the newts in the Barhill dataset using the Grounded-SAM-2 model.

# %%
#| default_exp seg

# %% [code] 
import os
if not os.path.exists("./data/barhill"):
    os.system("kaggle datasets download -d mshahoyi/barhills-processed --unzip -p ./data")

# %% [code] 
try:
    import supervision
    %cd gsam2
except:
    os.system('git clone https://github.com/IDEA-Research/Grounded-SAM-2 gsam2')
    %cd gsam2
    os.system('pip install -q -e . -e grounding_dino')
    os.system('pip install -q supervision')

# %% [code] 
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
import pandas as pd

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-05-23T16:35:06.186611Z","iopub.execute_input":"2025-05-23T16:35:06.187077Z","iopub.status.idle":"2025-05-23T16:35:06.194678Z","shell.execute_reply.started":"2025-05-23T16:35:06.187049Z","shell.execute_reply":"2025-05-23T16:35:06.193975Z"}}
"""
Hyper parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-tiny")
parser.add_argument("--text-prompt", default="car. tire.")
parser.add_argument("--img-path", default="notebooks/images/truck.jpg")
parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--output-dir", default="outputs/test_sam2.1")
parser.add_argument("--no-dump-json", action="store_true")
parser.add_argument("--force-cpu", action="store_true")

args = vars(parser.parse_args([]))

GROUNDING_MODEL = args['grounding_model']
TEXT_PROMPT = args['text_prompt']
IMG_PATH = args['img_path']
SAM2_CHECKPOINT = args['sam2_checkpoint']
SAM2_MODEL_CONFIG = args['sam2_model_config']
DEVICE = "cuda" if torch.cuda.is_available() and not args['force_cpu'] else "cpu"
OUTPUT_DIR = Path(args['output_dir'])
DUMP_JSON_RESULTS = not args['no_dump_json']

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-05-23T16:35:06.196540Z","iopub.execute_input":"2025-05-23T16:35:06.196790Z","iopub.status.idle":"2025-05-23T16:35:06.420025Z","shell.execute_reply.started":"2025-05-23T16:35:06.196765Z","shell.execute_reply":"2025-05-23T16:35:06.419457Z"}}
# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-05-23T16:35:06.421192Z","iopub.execute_input":"2025-05-23T16:35:06.421423Z","iopub.status.idle":"2025-05-23T16:35:15.939337Z","shell.execute_reply.started":"2025-05-23T16:35:06.421384Z","shell.execute_reply":"2025-05-23T16:35:15.938460Z"}}
%cd checkpoints
!bash download_ckpts.sh
%cd ..

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-05-23T16:35:15.940541Z","iopub.execute_input":"2025-05-23T16:35:15.940814Z","iopub.status.idle":"2025-05-23T16:35:19.713176Z","shell.execute_reply.started":"2025-05-23T16:35:15.940791Z","shell.execute_reply":"2025-05-23T16:35:19.712363Z"}}
# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-05-23T16:35:19.714090Z","iopub.execute_input":"2025-05-23T16:35:19.714835Z","iopub.status.idle":"2025-05-23T16:35:31.003093Z","shell.execute_reply.started":"2025-05-23T16:35:19.714803Z","shell.execute_reply":"2025-05-23T16:35:31.002444Z"}}
# build grounding dino from huggingface
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-05-23T16:35:31.003952Z","iopub.execute_input":"2025-05-23T16:35:31.004176Z","iopub.status.idle":"2025-05-23T16:35:31.010058Z","shell.execute_reply.started":"2025-05-23T16:35:31.004159Z","shell.execute_reply":"2025-05-23T16:35:31.009459Z"}}
import pathlib

image = pathlib.Path("./data/barhill/GCNs/GCN10-P1-S2/IMG_2367.JPEG")

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = "the lizard."
img_path = image
text, img_path

# %% [code] {"jupyter":{"outputs_hidden":false}}
image = Image.open(img_path)
image

# %% [code] {"jupyter":{"outputs_hidden":false}}
sam2_predictor.set_image(np.array(image.convert("RGB")))

inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)

# %% [code] {"jupyter":{"outputs_hidden":false}}
with torch.no_grad():
    outputs = grounding_model(**inputs)

# %% [code] {"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"jupyter":{"outputs_hidden":false}}
# get the box prompt for SAM 2
input_boxes = results[0]["boxes"].cpu().numpy()

masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# %% [code] {"jupyter":{"outputs_hidden":false}}
img = np.array(image)
img.shape

# %% [code] {"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"jupyter":{"outputs_hidden":false}}
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

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Segment the DS

# %% [code] {"jupyter":{"outputs_hidden":false}}
ds_dir = pathlib.Path("./data/barhill")

# %% [code] {"jupyter":{"outputs_hidden":false}}
gcns_dir = ds_dir/'GCNs'
cropped_dir = ds_dir/'gcns-cropped'
cropped_dir.mkdir(exist_ok=True)

# %% [code] {"jupyter":{"outputs_hidden":false}}
from tqdm import tqdm

# Create directories for each newt ID in the cropped directory
for newt_id in tqdm(os.listdir(gcns_dir)[:1]):
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

# %% [code] {"jupyter":{"outputs_hidden":false}}
import pandas as pd

# Load the metadata CSV
metadata_path = "./data/barhill/gallery_and_probes.csv"
metadata_df = pd.read_csv(metadata_path)
print(f"Loaded metadata with {len(metadata_df)} rows")

# Create output directory for visualization images - as sibling to data folder
vis_output_dir = pathlib.Path("./segmentation_visualizations")
vis_output_dir.mkdir(exist_ok=True)

# %% [code] {"jupyter":{"outputs_hidden":false}}
from tqdm import tqdm
import matplotlib.pyplot as plt

# Initialize list to store RLE masks - we'll match by image filename
rle_masks = {}
visualization_samples = []  # Store some samples to display later

# Create directories for each newt ID in the visualization directory
for newt_id in tqdm(os.listdir(gcns_dir)[:3]):  # Process first 3 newts for demo
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

# %% [code] {"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Update the metadata CSV with RLE masks
def add_rle_to_metadata(metadata_df, rle_masks):
    """Add RLE mask column to metadata DataFrame"""
    
    # Initialize the new column
    metadata_df['segmentation_mask_rle'] = None
    
    # Try to match images - this depends on how paths are stored in your CSV
    for idx, row in metadata_df.iterrows():
        # Adjust this matching logic based on your CSV structure
        # Common patterns:
        
        # If CSV has full path:
        if 'image_path' in metadata_df.columns:
            image_key = row['image_path']
        # If CSV has separate columns for newt_id and filename:
        elif 'newt_id' in metadata_df.columns and 'filename' in metadata_df.columns:
            image_key = f"{row['newt_id']}/{row['filename']}"
        # If CSV has just filename and you need to find the newt_id:
        else:
            # You might need to adjust this based on your actual CSV structure
            filename = row.get('filename', row.get('image_name', ''))
            # Find which newt_id directory contains this file
            for key in rle_masks.keys():
                if filename in key:
                    image_key = key
                    break
            else:
                continue
        
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

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Optional: Function to decode RLE masks back to binary masks
def decode_rle_mask(rle_string):
    """Decode RLE string back to binary mask"""
    if rle_string is None or pd.isna(rle_string):
        return None
    
    try:
        # Parse the RLE string format: "height x width : counts"
        size_part, counts_part = rle_string.split(':')
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
        return None

# Example usage:
sample_rle = metadata_df['segmentation_mask_rle'].dropna().iloc[0] if mask_count > 0 else None
if sample_rle:
    decoded_mask = decode_rle_mask(sample_rle)
    if decoded_mask is not None:
        print(f"Successfully decoded mask with shape: {decoded_mask.shape}")
        
        # Display the decoded mask
        plt.figure(figsize=(8, 6))
        plt.imshow(decoded_mask, cmap='gray')
        plt.title("Decoded RLE Mask Example")
        plt.axis('off')
        plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Show directory structure
print("Directory structure:")
print(f"Data directory: {ds_dir.absolute()}")
print(f"Visualizations directory: {vis_output_dir.absolute()}")
print(f"Total visualization subdirectories created: {len(list(vis_output_dir.iterdir()))}")
