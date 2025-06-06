"""This notebook is used to segment the newts in the Barhill dataset using the Grounded-SAM-2 model."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_gcn_seg.ipynb.

# %% auto 0
__all__ = ['decode_rle_mask', 'visualize_segmentation', 'visualize_segmentation_from_metadata']

# %% ../nbs/01_gcn_seg.ipynb 29
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

# %% ../nbs/01_gcn_seg.ipynb 30
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

# %% ../nbs/01_gcn_seg.ipynb 31
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
