# %% [markdown]
# # Attribution--the attribution toolkit
#> Notebook for attribution testing

# %%
#| default_exp attribution

#%%
#| eval: false
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import kaggle
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import pandas as pd
import cv2
from captum.attr import IntegratedGradients, Occlusion, FeaturePermutation
from scipy.ndimage import gaussian_filter
# from captum.attr._utils.masking import_mask # Corrected import

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

#%%
# Download dataset from Kaggle only if it does not exist
dataset_dir = './data/barhill'
if not os.path.exists(dataset_dir):
    kaggle.api.dataset_download_files('mshahoyi/barhills-processed', path='./data', unzip=True)
    print("Dataset downloaded and unzipped.")
else:
    print("Dataset already exists. Skipping download.")

#%%
# Load the pretrained MegaDescriptor model
model_name = 'hf-hub:BVRA/MegaDescriptor-T-224'
model = timm.create_model(model_name, num_classes=0, pretrained=True).to(device)
model.eval()

#%%
# Define image transformations
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#%%
# Load metadata
metadata_path = './data/barhill/gallery_and_probes.csv'
df = pd.read_csv(metadata_path)

#%%
# Select two random newts
unique_newts = df['newt_id'].unique()
np.random.seed(42)
random_newts = np.random.choice(unique_newts, 2, replace=False)
# random_newts[0] and random_newts[1] are guaranteed to be different newt IDs
# (assuming there are at least 2 unique newts in your dataset).
print(f"Selected newts: {random_newts}")

#%%
# Get sample images for each newt
newt1_images = df[df['newt_id'] == random_newts[0]]['image_path'].values[:2]
newt2_images = df[df['newt_id'] == random_newts[1]]['image_path'].values[:2]

print(f"Newt 1 images: {newt1_images}")
print(f"Newt 2 images: {newt2_images}")

#%%
# Plot the selected images for visual confirmation
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("Selected Images for Analysis", fontsize=16)

# Load and preprocess images
def get_full_image_path(rel_path):
    return os.path.join('./data', rel_path)


# Newt 1, Image 1
img_n1_i1 = Image.open(get_full_image_path(newt1_images[0])).convert('RGB')
axes[0, 0].imshow(img_n1_i1)
axes[0, 0].set_title(f"Newt {random_newts[0]} - Image 1\n{newt1_images[0]}")
axes[0, 0].axis('off')

# Newt 1, Image 2
if len(newt1_images) > 1:
    img_n1_i2 = Image.open(get_full_image_path(newt1_images[1])).convert('RGB')
    axes[0, 1].imshow(img_n1_i2)
    axes[0, 1].set_title(f"Newt {random_newts[0]} - Image 2\n{newt1_images[1]}")
    axes[0, 1].axis('off')
else:
    axes[0, 1].axis('off') # Hide subplot if no second image
    axes[0, 1].text(0.5, 0.5, 'No second image', ha='center', va='center')


# Newt 2, Image 1
img_n2_i1 = Image.open(get_full_image_path(newt2_images[0])).convert('RGB')
axes[1, 0].imshow(img_n2_i1)
axes[1, 0].set_title(f"Newt {random_newts[1]} - Image 1\n{newt2_images[0]}")
axes[1, 0].axis('off')

# Newt 2, Image 2
if len(newt2_images) > 1:
    img_n2_i2 = Image.open(get_full_image_path(newt2_images[1])).convert('RGB')
    axes[1, 1].imshow(img_n2_i2)
    axes[1, 1].set_title(f"Newt {random_newts[1]} - Image 2\n{newt2_images[1]}")
    axes[1, 1].axis('off')
else:
    axes[1, 1].axis('off') # Hide subplot if no second image
    axes[1, 1].text(0.5, 0.5, 'No second image', ha='center', va='center')

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.show()

#%%
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img)
    return img, input_tensor.unsqueeze(0)

#%%
newt1_img1, newt1_tensor1 = load_and_preprocess_image(get_full_image_path(newt1_images[0]))
newt1_img2, newt1_tensor2 = load_and_preprocess_image(get_full_image_path(newt1_images[1]))
newt2_img1, newt2_tensor1 = load_and_preprocess_image(get_full_image_path(newt2_images[0]))
newt2_img2, newt2_tensor2 = load_and_preprocess_image(get_full_image_path(newt2_images[1]))


#%%
class SimilarityModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
    def forward(self, x1, x2):
        # Get features from both images
        features1 = self.backbone(x1)
        features2 = self.backbone(x2)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(features1, features2)
        return similarity

#%%
# Create the similarity model
similarity_model = SimilarityModel(model).to(device)
similarity_model.eval()

#%% [markdown]
# ## Occlusion Sensitivity

#%%
#| export
def my_occlusion_sensitivity(model, image1, image2, patch_size=16, stride=8, occlusion_value=0, device=None):
    """
    Perform occlusion sensitivity test on the first image to see which regions
    affect similarity with the second image.
    
    Args:
        model: The similarity model
        image1: First image tensor (to be occluded) - shape [1, C, H, W]
        image2: Second image tensor - shape [1, C, H, W]
        patch_size: Size of the occlusion patch
        stride: Stride for moving the occlusion patch
        occlusion_value: Value used for occlusion (default: 0)
        
    Returns:
        Sensitivity map showing which regions, when occluded, affect similarity the most
    """

    import torch

    # Move tensors to the right device
    if device is not None:
        image1 = image1.to(device)
        image2 = image2.to(device)
    
    # Get the original similarity score
    with torch.no_grad():
        original_similarity = model(image1, image2).item()
    
    # Get image dimensions
    _, c, h, w = image1.shape
    
    # Initialize sensitivity map
    sensitivity_map = torch.zeros((h, w), device='cpu')
    
    # Compute number of patches
    n_h_patches = (h - patch_size) // stride + 1
    n_w_patches = (w - patch_size) // stride + 1
    
    # Create progress counter
    total_patches = n_h_patches * n_w_patches
    patch_count = 0
    
    # Slide the occlusion window over the image
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            # Create a copy of the image
            occluded_image = image1.clone()
            
            # Apply occlusion
            occluded_image[0, :, i:i+patch_size, j:j+patch_size] = occlusion_value
            
            # Compute the similarity with occlusion
            with torch.no_grad():
                occluded_similarity = model(occluded_image, image2).item()
            
            # Calculate the difference (sensitivity)
            sensitivity = original_similarity - occluded_similarity
            
            # Update the sensitivity map
            sensitivity_map[i:i+patch_size, j:j+patch_size] += sensitivity
            
            # Update progress counter
            patch_count += 1
            if patch_count % 10 == 0:
                print(f"Processed {patch_count}/{total_patches} patches", end='\r')
    
    print(f"\nCompleted occlusion testing - {total_patches} patches processed.")
    
    # Normalize the sensitivity map for visualization
    if sensitivity_map.max() > 0:
        sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min())
    
    return sensitivity_map.numpy()


#%%
def occlusion_sensitivity(model, image1, image2, patch_size=16, stride=8, occlusion_value=0):
    """
    Perform occlusion sensitivity test on the first image to see which regions
    affect similarity with the second image using Captum.
    
    Args:
        model: The similarity model
        image1: First image tensor (to be occluded) - shape [1, C, H, W]
        image2: Second image tensor - shape [1, C, H, W]
        patch_size: Size of the occlusion patch
        stride: Stride for moving the occlusion patch
        occlusion_value: Value used for occlusion (default: 0)
        
    Returns:
        Sensitivity map showing which regions, when occluded, affect similarity the most
    """    
    # Move tensors to the right device
    image1 = image1.to(device)
    image2 = image2.to(device)
    
    # Create a wrapper function for the model that takes a single input
    # This needs to be an nn.Module for Captum's hooks
    class ModelWrapper(nn.Module):
        def __init__(self, similarity_model_instance, fixed_image_tensor):
            super().__init__()
            self.similarity_model_instance = similarity_model_instance
            self.fixed_image_tensor = fixed_image_tensor
            self.similarity_model_instance.eval() # Ensure eval mode
        
        def forward(self, x):
            return self.similarity_model_instance(x, self.fixed_image_tensor)

    wrapped_model_for_captum = ModelWrapper(model, image2).to(device)
    wrapped_model_for_captum.eval()
    
    # Initialize the Occlusion attribution method
    occlusion_attr = Occlusion( # Renamed to avoid conflict with captum.attr.Occlusion
        wrapped_model_for_captum
    )
    
    # Compute attributions
    attributions = occlusion_attr.attribute(
        image1,
        strides=(3, stride, stride),  # (channels, height, width)
        sliding_window_shapes=(3, patch_size, patch_size),
        baselines=occlusion_value,
        target=None,  # Corrected: Use None for scalar output per batch item
    )
    
    # Convert attributions to sensitivity map
    # The output of occlusion.attribute is typically [N, C, H, W]
    # We want to see the impact, so taking the absolute difference or sum can be useful.
    # Here, let's consider the sum of attributions across channels.
    # A common way to interpret occlusion is that a large magnitude (positive or negative)
    # in attribution for a region means occluding it changed the output significantly.
    # The sign indicates direction. If baseline is 0, and output drops, attribution might be negative.
    # Let's sum attributions and then take absolute for magnitude of change.
    # Or, if we want to see "what makes the score drop", we might not take abs if original_score - perturbed_score is calculated.
    # Captum's Occlusion gives attribution of occluded region towards output.
    # A simple way to get a per-pixel map is to average over the channel dimension.
    
    sensitivity_map = attributions.squeeze(0).abs().mean(dim=0).cpu().detach() # .detach() is good practice
    
    # Normalize the sensitivity map for visualization
    if sensitivity_map.max() > 0:
        sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min())
    
    return sensitivity_map.numpy()

#%%
def visualize_occlusion_sensitivity(image, sensitivity_map, title):
    """
    Visualize the occlusion sensitivity map overlaid on the original image.
    
    Args:
        image: Original PIL image
        sensitivity_map: The computed sensitivity map
        title: Title for the plot
    """
    # Resize sensitivity map to match image dimensions
    resized_map = cv2.resize(sensitivity_map, (image.size[0], image.size[1]))
    
    # Convert PIL image to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Create a heatmap visualization
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay the heatmap on the image
    overlay = 0.7 * img_array + 0.3 * heatmap
    overlay = overlay / np.max(overlay)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(resized_map, cmap='jet')
    plt.title("Sensitivity Map")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay - {title}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#%%
# Perform occlusion sensitivity testing on same newt pair
print("Performing occlusion sensitivity test for same newt...")
sensitivity_map_same = occlusion_sensitivity(
    similarity_model, 
    newt1_tensor1, 
    newt1_tensor2, 
    patch_size=16, 
    stride=8
)


# Visualize occlusion sensitivity results
visualize_occlusion_sensitivity(
    newt1_img1,
    sensitivity_map_same,
    f"Same Newt {random_newts[0]} - Regions Important for Similarity"
)

#%%
# Perform MY occlusion sensitivity testing on same newt pair
print("Performing MY occlusion sensitivity test for same newts...")
sensitivity_map_same = my_occlusion_sensitivity(
    similarity_model,
    newt1_tensor1,  
    newt1_tensor2,  
    patch_size=16,
    stride=8
)

# Visualize occlusion sensitivity results
visualize_occlusion_sensitivity(
    newt1_img1,
    sensitivity_map_same,
    f"Same Newt {random_newts[0]} - Regions Important for Similarity"
)



#%%
# Perform occlusion sensitivity testing on different newt pair
print("Performing occlusion sensitivity test for different newts...")
sensitivity_map_diff = occlusion_sensitivity(
    similarity_model,
    newt1_tensor1,  # Image of newt_ID_A (this one will be occluded)
    newt2_tensor1,  # Image of newt_ID_B (this is the reference for similarity)
    patch_size=16,
    stride=8
)

# Visualize occlusion sensitivity results
visualize_occlusion_sensitivity(
    newt1_img1,
    sensitivity_map_diff,
    f"Different Newts {random_newts[0]} vs {random_newts[1]} - Regions Important for Similarity"
)

#%%
# Perform MY occlusion sensitivity testing on different newt pair
print("Performing MY occlusion sensitivity test for different newts...")
sensitivity_map_diff = my_occlusion_sensitivity(
    similarity_model,
    newt1_tensor1,  # Image of newt_ID_A (this one will be occluded)
    newt2_tensor1,  # Image of newt_ID_B (this is the reference for similarity)
    patch_size=16,
    stride=8
)

# Visualize occlusion sensitivity results
visualize_occlusion_sensitivity(
    newt1_img1,
    sensitivity_map_diff,
    f"Different Newts {random_newts[0]} vs {random_newts[1]} - Regions Important for Similarity"
)


# %% [markdown]
# ## Integrated Gradients

#%%
def integrated_gradients_similarity(model_instance, image1_tensor, image2_tensor, n_steps=50, target_output_idx=None):
    """
    Compute Integrated Gradients for the first image with respect to the similarity
    score with the second image.
    
    Args:
        model_instance: The SimilarityModel instance (which is an nn.Module).
        image1_tensor: Tensor of the first image (to attribute). Shape [1, C, H, W].
        image2_tensor: Tensor of the second image (fixed reference). Shape [1, C, H, W].
        n_steps: Number of steps for the integration.
        target_output_idx: If model outputs multiple values, specify index. For scalar output, can be None.

    Returns:
        Attributions for image1_tensor.
    """
    model_instance.eval() # Ensure the main model is in eval mode
    image1_tensor = image1_tensor.to(device)
    image2_tensor = image2_tensor.to(device)

    # Ensure tensors require gradients
    image1_tensor.requires_grad_()

    # Define a wrapper nn.Module for Captum
    class ModelWrapper(nn.Module):
        def __init__(self, similarity_model_instance, fixed_image_tensor):
            super().__init__()
            self.similarity_model_instance = similarity_model_instance
            self.fixed_image_tensor = fixed_image_tensor
            # Ensure the passed model instance is also in eval mode if it wasn't already
            self.similarity_model_instance.eval() 
        
        def forward(self, img1_input):
            return self.similarity_model_instance(img1_input, self.fixed_image_tensor)

    wrapped_model = ModelWrapper(model_instance, image2_tensor).to(device)
    wrapped_model.eval() 

    ig = IntegratedGradients(wrapped_model)
    
    baseline = torch.zeros_like(image1_tensor).to(device)
    
    attributions = ig.attribute(image1_tensor,
                                baselines=baseline,
                                target=target_output_idx, 
                                n_steps=n_steps,
                                return_convergence_delta=False) 
    return attributions

#%%
def visualize_integrated_gradients(image_pil, attributions_tensor, title):
    """
    Visualize Integrated Gradients attributions.
    """
    # Convert attributions to numpy array and take the sum across color channels
    attributions_np = attributions_tensor.squeeze().cpu().detach().numpy()
    attributions_np = np.transpose(attributions_np, (1, 2, 0))
    attribution_map = np.sum(np.abs(attributions_np), axis=2) # Sum absolute attributions across channels
    
    # Normalize the attribution map for visualization
    if np.max(attribution_map) > 0:
        attribution_map = (attribution_map - np.min(attribution_map)) / (np.max(attribution_map) - np.min(attribution_map))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    
    axes[0].imshow(image_pil)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    im = axes[1].imshow(attribution_map, cmap='inferno') # 'inferno' or 'viridis' are good choices
    axes[1].set_title("Integrated Gradients Attribution")
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#%%
# Perform Integrated Gradients for the same newt pair
# We are interested in how image1 contributes to similarity with image2
print("Performing Integrated Gradients for same newt (image1 vs image2)...")
attributions_same_newt_img1 = integrated_gradients_similarity(
    similarity_model,
    newt1_tensor1, # Image to attribute
    newt1_tensor2  # Fixed reference image
)
visualize_integrated_gradients(
    newt1_img1, # PIL image corresponding to newt1_tensor1
    attributions_same_newt_img1,
    f"IG: Newt {random_newts[0]} (Img 1) vs Newt {random_newts[0]} (Img 2)"
)

# Perform Integrated Gradients for the different newt pair
# We are interested in how image1 (from newt A) contributes to similarity with image1 (from newt B)
print("\nPerforming Integrated Gradients for different newts (newt A img1 vs newt B img1)...")
attributions_diff_newt_img1 = integrated_gradients_similarity(
    similarity_model,
    newt1_tensor1, # Image to attribute (from first newt)
    newt2_tensor1  # Fixed reference image (from second newt)
)
visualize_integrated_gradients(
    newt1_img1, # PIL image corresponding to newt1_tensor1
    attributions_diff_newt_img1,
    f"IG: Newt {random_newts[0]} (Img 1) vs Newt {random_newts[1]} (Img 1)"
)

# %% [markdown]
# ## Blur Perturbation

#%%
def blur_perturbation_similarity(model_instance, image1_tensor, image2_tensor, patch_size=16, stride=8, blur_sigma=5):
    """
    Perform perturbation-based saliency by blurring patches of image1 and observing
    the change in similarity with image2.
    
    Args:
        model_instance: The SimilarityModel instance.
        image1_tensor: Tensor of the first image (to be perturbed). Shape [1, C, H, W].
        image2_tensor: Tensor of the second image (fixed reference). Shape [1, C, H, W].
        patch_size: Size of the patch to blur.
        stride: Stride for moving the patch.
        blur_sigma: Sigma for Gaussian blur.
        
    Returns:
        Sensitivity map (higher values mean blurring that region decreased similarity more).
    """
    model_instance.eval()
    image1_tensor_cpu = image1_tensor.cpu() # Work with CPU tensor for easier numpy conversion and blurring
    image2_tensor_dev = image2_tensor.to(device) # Keep image2 on device for model input

    # Get the original similarity score
    with torch.no_grad():
        original_similarity = model_instance(image1_tensor.to(device), image2_tensor_dev).item()
    
    # Get image dimensions
    _, c, h, w = image1_tensor_cpu.shape
    
    # Initialize sensitivity map
    sensitivity_map = torch.zeros((h, w), device='cpu')
    
    # Create a blurred version of the entire image1 (used for replacing patches)
    # Convert tensor to numpy for blurring: (C, H, W)
    image1_numpy = image1_tensor_cpu.squeeze(0).numpy() 
    blurred_image1_numpy = np.zeros_like(image1_numpy)
    for channel_idx in range(c):
        blurred_image1_numpy[channel_idx, :, :] = gaussian_filter(image1_numpy[channel_idx, :, :], sigma=blur_sigma)
    
    # Compute number of patches
    n_h_patches = (h - patch_size) // stride + 1
    n_w_patches = (w - patch_size) // stride + 1
    
    total_patches = n_h_patches * n_w_patches
    patch_count = 0
    
    print(f"Starting blur perturbation: {total_patches} patches to process...")
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            perturbed_image_numpy = image1_numpy.copy()
            
            # Replace the patch with the corresponding patch from the blurred image
            perturbed_image_numpy[:, i:i+patch_size, j:j+patch_size] = \
                blurred_image1_numpy[:, i:i+patch_size, j:j+patch_size]
            
            # Convert back to tensor and move to device
            perturbed_image_tensor = torch.from_numpy(perturbed_image_numpy).unsqueeze(0).to(device)
            
            with torch.no_grad():
                perturbed_similarity = model_instance(perturbed_image_tensor, image2_tensor_dev).item()
            
            sensitivity = original_similarity - perturbed_similarity
            sensitivity_map[i:i+patch_size, j:j+patch_size] += sensitivity # Accumulate if patches overlap
            
            patch_count += 1
            if patch_count % 20 == 0 or patch_count == total_patches:
                print(f"Processed {patch_count}/{total_patches} patches...", end='\r')
    
    print(f"\nCompleted blur perturbation.")
    
    # Normalize the sensitivity map
    if sensitivity_map.abs().max() > 0: # Check against absolute max to handle negative sensitivities too
         # Center around 0 then scale, or just scale positive changes
        if sensitivity_map.max() > sensitivity_map.min() and not (sensitivity_map.max() == 0 and sensitivity_map.min() == 0) :
            sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min())
        elif sensitivity_map.max() > 0 : # if all values are same and positive
             sensitivity_map = sensitivity_map / sensitivity_map.max()


    return sensitivity_map.numpy()


#%%
# We can reuse visualize_occlusion_sensitivity, let's call it visualize_perturbation_map
# or just use it as is if the title parameter is sufficient.
# For consistency, I'll use the existing visualize_occlusion_sensitivity function.

# Perform Blur Perturbation for the same newt pair
print("Performing Blur Perturbation for same newt (image1 vs image2)...")
blur_map_same_newt = blur_perturbation_similarity(
    similarity_model,
    newt1_tensor1, 
    newt1_tensor2,
    patch_size=24, # Larger patch might be more informative for blur
    stride=12,
    blur_sigma=5
)
visualize_occlusion_sensitivity( # Reusing the visualization function
    newt1_img1,
    blur_map_same_newt,
    f"Blur Perturbation: Newt {random_newts[0]} (Img 1) vs Newt {random_newts[0]} (Img 2)"
)

# Perform Blur Perturbation for the different newt pair
print("\nPerforming Blur Perturbation for different newts (newt A img1 vs newt B img1)...")
blur_map_diff_newt = blur_perturbation_similarity(
    similarity_model,
    newt1_tensor1, 
    newt2_tensor1,
    patch_size=24,
    stride=12,
    blur_sigma=5
)
visualize_occlusion_sensitivity( # Reusing the visualization function
    newt1_img1,
    blur_map_diff_newt,
    f"Blur Perturbation: Newt {random_newts[0]} (Img 1) vs Newt {random_newts[1]} (Img 1)"
)


# %%
#| hide
import nbdev; nbdev.nbdev_export()