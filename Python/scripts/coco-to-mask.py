#https://youtu.be/SQng3eIEw-k
"""

Convert coco json masks to labeled masks and copy original images to place
them along with the masks.

Dataset from: https://github.com/sartorius-research/LIVECell/tree/main
Note that the dataset comes with:
Creative Commons Attribution - NonCommercial 4.0 International Public License
In summary, you are good to use it for research purposes but for commercial
use you need to investigate whether trained models using this data must also comply
with this license - it probably does apply to any derivative work so please be mindful.

You can directly download from the source github page. Links below.

Training json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json
Validation json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json
Test json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json
Images: Download images.zip by following the link: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip

If these links do not work, follow the instructions on their github page.


"""

import json
import numpy as np
import skimage
import tifffile
import os
import shutil

from PIL import Image
from matplotlib import pyplot as plt


def create_mask(image_info, annotations, output_folder):
    # Create an empty mask as a numpy array
    mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint16)

    for ann in annotations:
        if ann['image_id'] == image_info['id']:
            # Extract segmentation polygon
            category_id = ann['category_id']  # Get the class ID for the annotation
            for seg in ann['segmentation']:
                # Convert polygons to a binary mask and add it to the main mask
                try:
                    rr, cc = skimage.draw.polygon(np.array(seg[1::2]), np.array(seg[0::2]))
                    mask_np[rr, cc] = category_id  # Assign the category ID instead of object number
                except Exception as e:
                    print(f"Error processing annotation {ann['id']}: {e}")

    # Save the numpy array as a TIFF using tifffile library
    mask_path = os.path.join(output_folder, image_info['file_name'].replace('.tif', '_mask.tif'))
    tifffile.imsave(mask_path, mask_np)
    print(f"Saved mask for {image_info['file_name']} to {mask_path}")


def main(json_file, mask_output_folder, image_output_folder, original_image_dir):
    # Load COCO JSON annotations
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    # Ensure the output directories exist
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    for img in images:
        # Create the masks
        create_mask(img, annotations, mask_output_folder)

        # Copy original images to the specified folder
        original_image_path = os.path.join(original_image_dir, img['file_name'])

        new_image_path = os.path.join(image_output_folder, os.path.basename(original_image_path))
        shutil.copy2(original_image_path, new_image_path)
        print(f"Copied original image to {new_image_path}")


if __name__ == '__main__':
    data_splits = ['train', 'test', 'valid']
    for split in data_splits:
        original_image_dir = f'Car-Damages-5/{split}/'
        json_file = f'Car-Damages-5/{split}/_annotations.coco.json'
        mask_output_folder = f'dataset3/{split}/masks'
        image_output_folder = f'dataset3/{split}/images'
        main(json_file, mask_output_folder, image_output_folder, original_image_dir)
