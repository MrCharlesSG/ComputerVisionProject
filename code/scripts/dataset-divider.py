import os
import shutil
from pathlib import Path

def copy_percentage_of_files(dataset_dir, destination_dir, percentage):
    def copy_files(src_dir, dest_dir, num_files_to_copy):
        files = sorted([f for f in Path(src_dir).iterdir() if f.is_file()])  # Sort files to ensure order
        files_to_copy = files[:num_files_to_copy]  # Select the 5-epochs num_files_to_copy files

        print(f"Copying {num_files_to_copy} files from {src_dir} to {dest_dir}")

        for file in files_to_copy:
            shutil.copy(file, dest_dir)
            print(f"Copied {file} to {dest_dir}")

    def process_directory(subset):
        subset_src_dir = os.path.join(dataset_dir, subset)
        subset_dest_dir = os.path.join(destination_dir, subset)

        for subfolder in ['images', 'masks']:
            src_dir = os.path.join(subset_src_dir, subfolder)
            dest_dir = os.path.join(subset_dest_dir, subfolder)
            os.makedirs(dest_dir, exist_ok=True)

            total_files = len([f for f in Path(src_dir).iterdir() if f.is_file()])
            num_files_to_copy = int(total_files * (percentage / 100))

            print(f"Processing {subset}/{subfolder} - Total files: {total_files}, Files to copy: {num_files_to_copy}")
            copy_files(src_dir, dest_dir, num_files_to_copy)

    for subset in ['test', 'train', 'valid']:
        process_directory(subset)

# Example usage
source_dir = "../datasets/dataset"
percentage = 40
destination_dir = "dataset-" + percentage.__str__()

copy_percentage_of_files(source_dir, destination_dir, percentage)
