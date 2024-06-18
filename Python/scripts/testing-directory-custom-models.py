import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from custom_unet_models import CustomUNet2
from datasetseg import SegmentationDataset  # Assuming this is your custom dataset loader
from hyperparams import criterion, image_dir, mask_dir, batch_size
from metrics_functions import dice_score, iou, pixel_accuracy


def write_test_csv_stats_for_training(csv_file_path, model,
                                      test_loss, test_dice, test_iou, test_accuracy):
    with open(csv_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([model, test_loss, test_dice, test_iou, test_accuracy])


def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_accuracy = 0
    total_iou = 0

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device).long()
            masks_squeezed = masks.squeeze(1)
            outputs = model(images)
            loss = criterion(outputs, masks_squeezed)
            total_loss += loss.item()

            # Compute metrics
            predicted_masks = torch.argmax(outputs, dim=1)
            dice = dice_score(masks_squeezed, predicted_masks)
            iou_score = iou(masks_squeezed, predicted_masks)
            accuracy = pixel_accuracy(masks_squeezed, predicted_masks)

            total_dice += dice
            total_iou += iou_score
            total_accuracy += accuracy

    num_samples = len(test_loader)
    return total_loss / num_samples, total_dice / num_samples, total_iou / num_samples, total_accuracy / num_samples


def evaluate_models_in_directory(model_directory, test_loader, criterion, device, csv_file_path):
    model_files = Path(model_directory).glob('*.pth')

    for model_file in model_files:
        model_name = model_file.stem  # Get the filename without extension
        print(f"Evaluating model: {model_name}")

        # Load model checkpoint
        model = CustomUNet2(num_classes=3)
        model.load_state_dict(torch.load(model_file))
        model.to(device)

        # Test the model
        test_loss, test_dice, test_iou, test_accuracy = test(model, test_loader, criterion, device)

        # Write results to CSV
        write_test_csv_stats_for_training(csv_file_path, model_name, test_loss, test_dice, test_iou, test_accuracy)


# Configuration
model_directory = 'models/CustomUNet2-10E'
test_dir = '../datasets/dataset-40/test'
file_path = Path('../metrics/test/test-CustomUNet2-10E.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare the test dataset and data loader
test_dataset = SegmentationDataset(test_dir, image_dir, mask_dir)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
if not file_path.exists():
    file_path.touch()

    write_test_csv_stats_for_training(file_path, "Model", "Test Loss", "Test Dice", "Test IoU",
                                      "Test Accuracy")
# Evaluate models in directory
evaluate_models_in_directory(model_directory, test_loader, criterion, device, file_path)
