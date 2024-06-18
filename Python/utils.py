import csv
import os

import torch
from torchmetrics import F1Score
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import F1Score


def create_directory(directory_path):
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def create_csv_file_for_training(directory_path, filename_prefix='stats'):
    counter = 0
    while True:
        filename = f'{filename_prefix}_{counter}.csv'
        file_path = Path(directory_path) / filename
        if not file_path.exists():
            break
        counter += 1

    file_path.touch()

    write_csv_stats_for_training(file_path, "Epoch", "Train Loss", "Train Dice", "Train IoU",
                                 "Train Accuracy", "Val Loss", "Val Dice", "Val IoU",
                                 "Val Accuracy")

    return file_path


def write_csv_stats_for_training(csv_file_path, epoch, train_loss, train_dice, train_iou, train_accuracy,
                                 val_loss, val_dice, val_iou, val_accuracy):
    with open(csv_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([epoch, train_loss, train_dice, train_iou, train_accuracy, val_loss, val_dice, val_iou, val_accuracy])


def calculate_pixel_accuracy(outputs, masks):
    with torch.no_grad():
        outputs = torch.argmax(outputs, dim=1)
        correct = torch.eq(outputs, masks).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def save_test_stats(test_loss, test_pixel_accuracy, test_f1, name):
    file_path = "models/test-SoAM-first.csv"

    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['Test Loss', 'Test Pixel Accuracy', 'Test F1 Score', 'Name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writeheader()

        writer.writerow({'Test Loss': test_loss, 'Test Pixel Accuracy': test_pixel_accuracy,
                         'Test F1 Score': test_f1, 'Name': name})
