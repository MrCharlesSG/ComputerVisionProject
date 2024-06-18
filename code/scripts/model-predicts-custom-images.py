import os
import re
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.patches as mpatches
from torchvision import models
import custom_unet_models

model_path = '../models/CustomUNet2-10E/model-9.pth'
image_dir = '../datasets/custom-dataset/images'
mask_dir = '../datasets/custom-dataset/masks'

# Load the model
model = custom_unet_models.CustomUNet2(num_classes=3)
model.load_state_dict(torch.load(model_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Preprocess the image
def preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to the input size expected by the model
        transforms.ToTensor(),         # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess_transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Postprocess the output to visualize the segmentation map
def postprocess(output):
    output = output.squeeze(0)  # Remove batch dimension
    output = output.argmax(0)   # Get the predicted class for each pixel
    return output.numpy()

# Define color map and class names
color_map = {
    0: (0, 0, 0),        # Class 0: Black
    1: (255, 0, 0),      # Class 1: Red
    2: (0, 255, 0),      # Class 2: Green
    # Add more classes and colors as needed
}

class_names = {
    0: 'Background',
    1: 'No Damage',
    2: 'Severe Damage',
}

# Apply the color map to the segmentation map
def apply_color_map(segmentation_map, color_map):
    height, width = segmentation_map.shape
    colored_map = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        colored_map[segmentation_map == class_id] = color
    return colored_map

# Visualize the original image, segmentation map, and the legend
def visualize(image_path, colored_segmentation_map, actual_mask, category, class_names, color_map):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create the legend
    patches = [mpatches.Patch(color=np.array(color)/255, label=class_names[class_id])
               for class_id, color in color_map.items()]

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original Image')

    plt.subplot(1, 4, 2)
    plt.imshow(colored_segmentation_map)
    plt.title('Segmentation Map')

    plt.subplot(1, 4, 3)
    plt.imshow(actual_mask, cmap='gray')
    plt.title(category)

    plt.subplot(1, 4, 4)
    plt.axis('off')
    plt.legend(handles=patches, loc='center')
    plt.title('Legend')

    plt.show()

# Load the actual mask
def load_actual_mask(mask_path):
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)
    return mask

# Extract category from mask filename
def extract_category_from_filename(mask_filename):
    match = re.search(r'tag-(.*?)-\d+\.png', mask_filename)
    if match:
        return match.group(1).replace('-', ' ')
    return 'Unknown'

# Match images with their corresponding masks
def find_matching_mask(image_path, mask_dir):
    image_number = re.findall(r'\d+', os.path.basename(image_path))[0]
    if image_number == '9':
        print(image_number)
    for mask_filename in os.listdir(mask_dir):
        if f"task-{image_number}" in mask_filename:
            return os.path.join(mask_dir, mask_filename), extract_category_from_filename(mask_filename)
    return None, None

# Process all images in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        mask_path, category = find_matching_mask(image_path, mask_dir)
        if mask_path:
            input_image = preprocess(image_path)
            actual_mask = load_actual_mask(mask_path)

            # Perform prediction
            with torch.no_grad():
                output = model(input_image)

            # Postprocess the output
            segmentation_map = postprocess(output)
            colored_segmentation_map = apply_color_map(segmentation_map, color_map)

            # Visualize the results
            visualize(image_path, colored_segmentation_map, actual_mask, category, class_names, color_map)
