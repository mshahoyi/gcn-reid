"""This notebook is used to segment the newts in the Barhill dataset using the Grounded-SAM-2 model."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_gcn_seg.ipynb.

# %% auto 0
__all__ = ['visualize_bbox', 'decode_rle_mask', 'visualize_segmentation']

# %% ../nbs/01_gcn_seg.ipynb 15
def visualize_bbox(image_path, bbox, label):
    import cv2
    import numpy as np
    import supervision as sv
    from utils.supervision_utils import CUSTOM_COLOR_MAP
    import matplotlib.pyplot as plt
    
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=np.array(bbox).reshape(1, 4),
        mask=None,  # (n, h, w)
        class_id=np.array([0])
    )


    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame_boxes = box_annotator.annotate(scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP), text_color=sv.Color.BLACK) # Specify text color if needed
    annotated_frame_labels = label_annotator.annotate(scene=annotated_frame_boxes.copy(), detections=detections, labels=[label])

    plt.imshow(cv2.cvtColor(annotated_frame_labels, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# %% ../nbs/01_gcn_seg.ipynb 21
def decode_rle_mask(rle_string):
    """Decode RLE string back to binary mask"""
    import pandas as pd
    import pycocotools.mask as mask_util
    
    parts = rle_string.split(':', 1)
    if len(parts) != 2:
        print(f"Invalid RLE format: {rle_string[:100]}...")
        return None
        
    size_part, counts_part = parts
    height, width = map(int, size_part.split('x'))
    
    rle_dict = {
        'size': [height, width],
        'counts': counts_part.encode('utf-8')
    }
    
    mask = mask_util.decode(rle_dict)
    return mask

# %% ../nbs/01_gcn_seg.ipynb 23
def visualize_segmentation(image_path, mask, bbox, label):
    import cv2
    import numpy as np
    import supervision as sv
    from utils.supervision_utils import CUSTOM_COLOR_MAP
    import matplotlib.pyplot as plt
    
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=np.array(bbox).reshape(1, 4),
        mask=mask.astype(bool),
        class_id=np.array([0])
    )

    mask_annotator = sv.MaskAnnotator(color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame_final = mask_annotator.annotate(scene=img.copy(), detections=detections)

    if bbox is not None:
        box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame_boxes = box_annotator.annotate(scene=annotated_frame_final.copy(), detections=detections)
        
        label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP), text_color=sv.Color.BLACK) # Specify text color if needed
        annotated_frame_labels = label_annotator.annotate(scene=annotated_frame_boxes.copy(), detections=detections, labels=[label])

        annotated_frame_final = mask_annotator.annotate(scene=annotated_frame_labels.copy(), detections=detections) # Use copy to avoid modifying annotated_frame_labels

    plt.imshow(cv2.cvtColor(annotated_frame_final, cv2.COLOR_BGR2RGB))
    plt.axis('off')
