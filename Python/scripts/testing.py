import csv
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import models

from datasetseg import SegmentationDataset
from hyperparams import criterion, num_classes, image_dir, mask_dir, batch_size
from metrics_functions import dice_score, iou, pixel_accuracy

import torch


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

    print(f'Start Evaluation')
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device).long()
            masks_squeezed = masks.squeeze(1)
            outputs = model(images)['out']
            loss = criterion(outputs, masks_squeezed)
            total_loss += loss.item()

            # Compute Dice Score
            predicted_masks = torch.argmax(outputs, dim=1)
            dice = dice_score(masks_squeezed, predicted_masks)
            iou_train = iou(masks_squeezed, predicted_masks)
            accuracy_train = pixel_accuracy(masks_squeezed, predicted_masks)

            total_dice += dice
            total_iou += iou_train
            total_accuracy += accuracy_train
            print(f"Step [{i + 1}/{len(test_loader)}], Loss: {total_loss:.4f}"
                  f" Dice: {total_dice}, IoU: {total_iou}, Accuracy: {total_accuracy}")

    return total_loss / len(test_loader), total_dice / len(test_loader), total_iou / len(
        test_loader), total_accuracy / len(test_loader)


model_weights = 'models/fcn_resnet50-5e-SGD.pth'
model_name = 'fcn_resnet50-preTrained' #model_weights.split('/')[-1].split('.')[0]
test_dir = '../datasets/dataset-40/test'
file_path = Path('../metrics/test/test-SoAM-first.csv')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained DeepLabV3 model
model = models.segmentation.fcn_resnet50(pretrained=True, num_classes=num_classes)
#model.load_state_dict(torch.load(model_weights))
model.to(device)

# Prepare the testidation/test dataset and data loader
# You need to replace `test_dataset` with your actual testidation/test dataset
test_dataset = SegmentationDataset(test_dir, image_dir, mask_dir)
train_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Test the model
test_loss, test_dice, test_iou, test_accuracy = test(model, train_data_loader, criterion, device)

if not file_path.exists():
    file_path.touch()

    write_test_csv_stats_for_training(file_path, "Model", "Test Loss", "Test Dice", "Test IoU",
                                      "Test Accuracy")

write_test_csv_stats_for_training(file_path, model_name, test_loss, test_dice, test_iou, test_accuracy)
