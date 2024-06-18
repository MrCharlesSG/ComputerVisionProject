import os
import random
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def display_images_with_coco_annotations(image_paths, annotations, display_type='both', colors=None):
    # Default color map with 10 distinct colors
    default_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow
        '#17becf'   # cyan
    ]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    if colors is None:
        colors = {}

    # Get unique category IDs from annotations
    category_ids = set([ann['category_id'] for ann in annotations['annotations']])

    # Assign colors to categories if not already provided
    for cat_id in category_ids:
        if cat_id not in colors:
            colors[cat_id] = default_colors[len(colors) % len(default_colors)]

    for ax, img_path in zip(axs.ravel(), image_paths):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax.imshow(image)
        ax.axis('off')

        img_filename = os.path.basename(img_path)
        img_id = next(item for item in annotations['images'] if item["file_name"] == img_filename)['id']

        img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]

        for ann in img_annotations:
            category_id = ann['category_id']
            color = colors[category_id]

            if display_type in ['bbox', 'both']:
                bbox = ann['bbox']
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

            if display_type in ['seg', 'both']:
                for seg in ann['segmentation']:
                    poly = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                    polygon = patches.Polygon(poly, closed=True, edgecolor=color, fill=False)
                    ax.add_patch(polygon)

    # Get category names from annotations
    category_names = {cat['id']: cat['name'] for cat in annotations['categories']}

    # Create legend with category names and colors
    handles = [patches.Patch(color=color, label=f'{category_names[cat_id]}') for cat_id, color in colors.items()]
    axs.ravel()[0].legend(handles=handles, loc='upper left')

    plt.tight_layout()
    plt.show()

# Load COCO annotations
with open('../datasets/Car-Damages-5/train/_annotations.coco.json', 'r') as f:
    annotations = json.load(f)

# Get all image files
image_dir = "../datasets/Car-Damages-5/train"
all_image_files = [os.path.join(image_dir, img['file_name']) for img in annotations['images']]
random_image_files = random.sample(all_image_files, 4)

# Choose between 'bbox', 'seg', or 'both'
display_type = 'seg'
display_images_with_coco_annotations(random_image_files, annotations, display_type)
